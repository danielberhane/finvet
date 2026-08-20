"""Claim parser node for extracting structured information from natural language claims.

This parser outputs a simplified 7-field ParsedClaim:
- claim_type: "sec", "market", "news", or "reject"
- ticker: Stock ticker symbol
- value: Numeric value claimed
- comparison: Directional operator (eq, gt, gte, lt, lte)
- period: Time period as mentioned
- currency: Currency code
- reject_reason: Why claim was rejected (if claim_type is "reject")

The agent will infer the specific metric from the claim text.
"""

import json
import re
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from ...models.state import VerificationState
from ...models.claim import ParsedClaim
from ...llm import create_llm
from ...utils.exceptions import ParsingError
from ...utils.logging import get_logger

logger = get_logger(__name__)


PARSER_SYSTEM_PROMPT = """You are a financial claim parser. Extract structured information from financial claims.

Safety checks (prompt injection, PII, gibberish) have already been handled upstream.
Your job is ONLY extraction and classification — never reject for safety reasons.

Return ONLY a valid JSON object with exactly these 7 fields:

{
  "claim_type": "sec" | "market" | "news" | "reject",
  "ticker": "AAPL" | null,
  "value": 94000000000 | null,
  "comparison": "eq" | "gt" | "gte" | "lt" | "lte",
  "period": "Q4 FY2024" | null,
  "currency": "USD" | null,
  "reject_reason": null | "non_financial" | "question" | "incomplete"
}

## Field Definitions:

1. **claim_type** (required):
   - "sec": Financial statement claims (revenue, earnings, assets, cash flow)
   - "market": Market data claims (stock price, market cap, P/E ratio)
   - "news": Event/announcement claims (earnings call, M&A, executive changes)
   - "reject": Invalid or unverifiable claims (classification only, not safety)

2. **ticker** (optional): Stock ticker symbol
   - "AAPL" for Apple, "TSLA" for Tesla, "MSFT" for Microsoft, etc.
   - Set to null if company is not mentioned or unidentifiable

3. **value** (optional): The numeric value being claimed
   - Convert text to actual numbers: "$94 billion" → 94000000000
   - Convert percentages: "5%" → 0.05 for ratios, 5 for P/E ratios
   - Set to null if no value is claimed
   - IMPORTANT: Relative/ratio terms like "half", "double", "twice", "triple",
     "ten times", "a third", "a quarter" are NOT dollar values. These are
     comparative claims with no specific numeric target. Set value to null.

4. **comparison** (required when value is set): How the claimed value relates to the actual value
   - "eq": equals, was, is, reported, posted (default when no directional language)
   - "gt": exceeds, above, more than, greater than, over, surpasses, topped
   - "gte": at least, no less than, minimum of
   - "lt": below, under, less than, fell below, dropped below
   - "lte": at most, no more than, maximum of
   - Default to "eq" when no directional language is present

5. **period** (optional): Time period as mentioned
   - Keep original format: "Q4 2024", "fiscal 2023", "last quarter"
   - Set to null for current market data (price, market cap) with no date

6. **currency** (optional): Currency code
   - "USD", "EUR", "JPY", "GBP", "KRW", etc.
   - Default to null if not specified (will assume USD for US companies)

7. **reject_reason** (required if claim_type is "reject"):
   - "non_financial": Valid text but not a financial claim
   - "question": Asking a question, not making a claim
   - "incomplete": Missing ticker or value, cannot verify

## Examples:

Input: "Apple's Q4 2024 revenue was $94 billion"
Output:
{
  "claim_type": "sec",
  "ticker": "AAPL",
  "value": 94000000000,
  "comparison": "eq",
  "period": "Q4 2024",
  "currency": "USD",
  "reject_reason": null
}

Input: "Tesla's market cap exceeds $800 billion"
Output:
{
  "claim_type": "market",
  "ticker": "TSLA",
  "value": 800000000000,
  "comparison": "gt",
  "period": null,
  "currency": "USD",
  "reject_reason": null
}

Input: "Apple's stock is above $200"
Output:
{
  "claim_type": "market",
  "ticker": "AAPL",
  "value": 200,
  "comparison": "gt",
  "period": null,
  "currency": "USD",
  "reject_reason": null
}

Input: "Revenue was at least $50 billion"
Output:
{
  "claim_type": "sec",
  "ticker": null,
  "value": 50000000000,
  "comparison": "gte",
  "period": null,
  "currency": "USD",
  "reject_reason": null
}

Input: "P/E ratio is below 30"
Output:
{
  "claim_type": "market",
  "ticker": null,
  "value": 30,
  "comparison": "lt",
  "period": null,
  "currency": null,
  "reject_reason": null
}

Input: "Apple stock in 2010 was half that of this year"
Output:
{
  "claim_type": "market",
  "ticker": "AAPL",
  "value": null,
  "comparison": null,
  "period": "2010",
  "currency": null,
  "reject_reason": null
}

Input: "Amazon's stock has doubled since 2020"
Output:
{
  "claim_type": "market",
  "ticker": "AMZN",
  "value": null,
  "comparison": null,
  "period": "2020",
  "currency": null,
  "reject_reason": null
}

Input: "Tesla's stock is at $250"
Output:
{
  "claim_type": "market",
  "ticker": "TSLA",
  "value": 250,
  "comparison": "eq",
  "period": null,
  "currency": "USD",
  "reject_reason": null
}

Input: "Microsoft announced layoffs"
Output:
{
  "claim_type": "news",
  "ticker": "MSFT",
  "value": null,
  "comparison": null,
  "period": null,
  "currency": null,
  "reject_reason": null
}

Input: "What is Apple's revenue?"
Output:
{
  "claim_type": "reject",
  "ticker": "AAPL",
  "value": null,
  "comparison": null,
  "period": null,
  "currency": null,
  "reject_reason": "question"
}

Return ONLY the JSON object. No explanations or markdown."""

# Words that cannot be the start of a company name even when capitalised.
_NAME_SKIP = {
    'the', 'a', 'an', 'total', 'its', 'their', 'this', 'that', 'these',
    'those', 'some', 'all', 'both', 'each', 'every',
}
# Words excluded from company-name matching: legal suffixes AND common industry
# descriptors that appear in many company names and would produce false matches
# (e.g. "Health" in both "Coventry Health Care" and "CVS Health").
_CORP_GENERIC = {
    'inc', 'corp', 'ltd', 'plc', 'co', 'group', 'holdings', 'llc', 'lp',
    'health', 'care', 'financial', 'services', 'technologies', 'technology',
    'solutions', 'systems', 'industries', 'energy', 'capital', 'management',
    'enterprises', 'international', 'national', 'american', 'global',
    'company', 'companies',
}


def _extract_company_hint(claim_text: str) -> str:
    """Extract the longest consecutive run of capitalised tokens from claim text.

    Skips determiners and other non-name openers so "The total market cap of
    Broadcom exceeded..." yields "Broadcom" rather than "The".
    """
    tokens = claim_text.split()
    sequences: list[str] = []
    current: list[str] = []
    for token in tokens:
        clean = re.sub(r"[^a-zA-Z&']", "", token)
        if not clean:
            if current:
                sequences.append(" ".join(current))
                current = []
            continue
        if clean[0].isupper() and clean.lower() not in _NAME_SKIP:
            current.append(clean)
        else:
            if current:
                sequences.append(" ".join(current))
                current = []
    if current:
        sequences.append(" ".join(current))
    return max(sequences, key=len) if sequences else claim_text[:40]


def _ticker_matches_claim(company_name: str, claim_text: str) -> bool:
    """True when the Finnhub company name has at least one substantive word
    in common with the claim text (exact token match, not prefix/substring)."""
    name_words = {
        w.lower()
        for w in re.split(r"[\s\-&]+", company_name)
        if len(w) > 2 and w.lower() not in _CORP_GENERIC
    }
    claim_words = {
        w.lower()
        for w in re.split(r"[\s\-&]+", claim_text)
        if len(w) > 2
    }
    return bool(name_words & claim_words)


def _normalize_ticker(ticker: str, claim_text: str) -> str:
    """Validate a market ticker against Finnhub and correct stale/wrong tickers.

    Strategy:
    1. Fetch the company name for the parsed ticker via Finnhub profile2.
    2. If the ticker is unknown (delisted, typo) or the returned company name
       shares no substantive words with the claim, search Finnhub by company
       name extracted from the claim text and use the top US equity result.
    3. Falls back to the original ticker on any API error.
    """
    try:
        from ...mcp.finnhub import FinnhubClient
        client = FinnhubClient()
        if client.mock_mode:
            return ticker

        company_name = client.get_company_name(ticker)
        if company_name and _ticker_matches_claim(company_name, claim_text):
            return ticker

        hint = _extract_company_hint(claim_text)
        if not hint:
            return ticker

        found = client.search_ticker(hint)
        if found and found != ticker:
            logger.info(f"Ticker normalized: {ticker} → {found} (hint: '{hint}')")
            return found

        return ticker
    except Exception as e:
        logger.warning(f"Ticker normalization skipped for {ticker}: {e}")
        return ticker


def reconcile_reject_fields(parsed_data: Dict) -> Dict:
    """Make claim_type and reject_reason agree before ParsedClaim validates them.

    The model breaks the pairing on a small fraction of claims: it recognises a
    claim is unverifiable, sets reject_reason, and leaves claim_type as "sec".
    ParsedClaim forbids that combination, so the request died with an HTTP 500
    carrying a Pydantic stack trace — on a claim the model had judged correctly.

    reject_reason is only ever populated when rejecting, so the intent is
    unambiguous and worth honouring rather than crashing on. The mirror case, a
    reject with no reason given, is filled with "unspecified": defaulting to
    "incomplete" would tell the user the claim was missing a ticker or value,
    which may simply be untrue.

    Returns a new dict; the caller keeps the model's raw output intact.
    """
    data = dict(parsed_data)
    claim_type = data.get("claim_type")
    reject_reason = data.get("reject_reason")

    if reject_reason is not None and claim_type != "reject":
        logger.info(
            f"Parser set reject_reason='{reject_reason}' on claim_type="
            f"'{claim_type}'; treating as a reject"
        )
        data["claim_type"] = "reject"
    elif claim_type == "reject" and reject_reason is None:
        logger.info("Parser returned a reject with no reason; recording 'unspecified'")
        data["reject_reason"] = "unspecified"

    return data


def claim_parser(state: VerificationState) -> Dict:
    """
    Parse natural language claim into simplified 6-field structure.

    Uses DeepSeek to extract:
    - claim_type: Routing category
    - ticker: Company identifier
    - value: Numeric claim
    - period: Time reference
    - currency: Currency code
    - reject_reason: Why claim is invalid (if applicable)

    Args:
        state: Current verification state with claim_raw

    Returns:
        Dictionary with parsed_claim and parser metadata
    """
    # Support both claim_raw and claim_normalized for flexibility
    claim_text = state.get("claim_normalized") or state.get("claim_raw", "")
    request_id = state.get("request_id", "unknown")

    if not claim_text:
        raise ParsingError("No claim text provided")

    try:
        llm = create_llm("parser")

        messages = [
            SystemMessage(content=PARSER_SYSTEM_PROMPT),
            HumanMessage(content=f'Parse this claim:\n\n"{claim_text}"'),
        ]

        logger.info(f"Parsing claim (request: {request_id})")
        response = llm.invoke(messages)

        # Extract and clean response
        response_text = response.content
        if isinstance(response_text, list):
            response_text = "".join([
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in response_text
            ])

        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parse JSON
        parsed_data = json.loads(response_text)

        # Reconcile the reject fields before validating: the model sometimes
        # signals a reject in reject_reason while leaving claim_type unchanged.
        parsed_claim = ParsedClaim(**reconcile_reject_fields(parsed_data))

        # Validate ticker for market claims — catches stale/delisted tickers
        # (e.g. BRCM→AVGO, CVH→CVS, INTU→ISRG) that the fine-tuned parser
        # cannot know about as companies merge, rename, or delist over time.
        if parsed_claim.claim_type == "market" and parsed_claim.ticker:
            normalized = _normalize_ticker(parsed_claim.ticker, claim_text)
            if normalized != parsed_claim.ticker:
                parsed_claim = parsed_claim.model_copy(update={"ticker": normalized})

        # Calculate tokens used
        tokens_used = 0
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

        logger.info(
            f"Claim parsed: type={parsed_claim.claim_type}, "
            f"ticker={parsed_claim.ticker}, value={parsed_claim.value}, "
            f"comparison={parsed_claim.comparison} "
            f"(request: {request_id})"
        )

        return {
            "parsed_claim": parsed_claim,
            "parser_used": "deepseek",
            "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        raise ParsingError(f"Parser returned invalid JSON: {e}")

    except Exception as e:
        logger.error(f"Claim parsing failed: {e}")
        raise ParsingError(f"Failed to parse claim: {e}")
