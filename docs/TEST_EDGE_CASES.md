# FinVet Test Edge Cases

Edge cases discovered during testing that expose limitations, failure modes, or interesting behaviors in the verification pipeline. Use these for regression testing and evaluation framework development.

---

## EC-001: Pre-IPO / Entity Restructuring (Google 2000 Revenue)

**Claim:** "Google annual revenue in 2000 exceeded 34 billion"

**Expected verdict:** REFUTES (Google's 2000 revenue was ~$19 million, not $34 billion)

**Actual verdict:** NOT_ENOUGH_INFO (0.65 confidence, triggered HITL)

**Root cause:** Three compounding issues:
1. **Entity restructuring:** Alphabet Inc. (CIK `0001652044`) was created in October 2015. Pre-2015 filings are under Google Inc. (CIK `0001288776`). The agent only searched Alphabet's CIK.
2. **Pre-IPO data:** Google went public in August 2004. Year 2000 financials are only available in the S-1 registration statement filed for the IPO, not in regular 10-K filings.
3. **Filing limit:** `get_recent_filings(limit=20)` only returns ~20 years of annual filings from the current entity, missing older filings entirely.

**Affected components:**
- `sec_tools.py` — `get_company_info` returns only the current entity CIK
- `sec_edgar.py` — no predecessor entity resolution
- `base.py` — agent cannot discover historical CIK changes

**Potential fixes:**
- Add predecessor entity lookup (SEC EDGAR has company history data)
- Search for S-1/S-1A filings for pre-IPO historical financials
- Fall back to news agent for corroboration when SEC data is unavailable
- Add a knowledge cutoff disclosure: "SEC EDGAR data for this entity only available from [year]"

**Category:** Data coverage gap

---

## EC-002: Non-Deterministic Claim Parsing

**Claim:** "Trump introduced 100% tariff on Canadian steel"

**Expected:** Consistent classification across runs (likely `news` or `reject`)

**Actual:** Sometimes classified as `reject` (reason: `non_financial`), other times as `news`. Same claim, same model, temperature=0.

**Root cause:** LLM non-determinism. DeepSeek at temperature=0 is not fully deterministic due to batched inference and floating-point non-determinism on GPU. Borderline claims that sit between two categories are most affected.

**Affected components:**
- `claim_parser.py` — relies on LLM classification with no fallback or consistency check

**Potential fixes:**
- Run parser N times and take majority vote (expensive)
- Add explicit examples of political/trade claims to the parser prompt
- Add a confidence score to the parsing step (not just verdict)
- Fine-tune a smaller classifier model for routing (deterministic)

**Category:** Parser consistency

---

## EC-003: Fiscal Year vs Calendar Year Mismatch

**Claim:** "Apple's FY2024 revenue was $391 billion"

**Expected:** Agent correctly identifies Apple's fiscal year ends September 28, not December 31.

**Actual:** Period resolver assumes calendar year (Jan 1 - Dec 31). Agent compensates by calling `get_company_info` to discover `fiscal_year_end: "0928"`, then fetches the correct filing. Works most of the time, but can fail if the agent doesn't call `get_company_info` first or misinterprets the fiscal year end.

**Root cause:** `period_resolver.py` always maps "FY2024" to calendar year 2024-01-01 to 2024-12-31. The agent must independently correct this.

**Affected components:**
- `period_resolver.py` — no fiscal year awareness
- `base.py` — depends on agent calling `get_company_info` and reasoning correctly

**Potential fixes:**
- Period resolver could call `get_company_info` itself to get fiscal year end
- Cache fiscal year ends for common tickers
- Add fiscal year end to the agent context automatically

**Category:** Period resolution

---

## EC-004: Q4 Claims (No Standalone Q4 Filing)

**Claim:** "Apple's Q4 2024 revenue was $94 billion"

**Expected:** The claim is declined. Q4 derivation is not supported.

**Actual:** Matches expected. Earlier releases instructed the agent to derive
Q4 as annual minus the nine-month cumulative, and results were mixed --
sometimes the subtraction was performed, sometimes the annual figure was used
as Q4, sometimes NOT_ENOUGH_INFO.

**Root cause:** Q4 is never filed separately. Companies file 10-Q for Q1-Q3 and
a 10-K for the full year, so Q4 exists only as a difference. The subtraction is
valid only when both figures come from the same restatement generation, cover
the same entity scope, and share a fiscal calendar -- none of which was
checked. A derived number also carries no filing, no accession, and no XBRL
concept, so it cannot be located in a source or audited afterwards.

**Resolution:** The derivation instructions were removed from the
verdict-extraction prompt (`agents/base.py`) and the `search_filings` docstring
(`tools/sec_tools.py`); the docstring now states the limitation. See
`RELEASE_A_DECISIONS.md`, D1.

**To support it later:** deterministic derivation with restatement-generation
matching, entity-scope checks, and a provenance record naming both source
filings -- not a prompt instruction asking the model to subtract.

**Category:** Multi-step reasoning

---

## EC-005: Currency Conversion Claims

**Claim:** "Samsung's revenue exceeded 300 trillion won in 2024"

**Expected:** Agent handles non-USD currency claims correctly.

**Status:** Not yet tested. SEC EDGAR only covers US-listed companies. Samsung trades on the Korea Exchange and files with Korean regulators, not the SEC. The claim should either be routed to the news agent or rejected as unverifiable via SEC.

**Potential issues:**
- Parser may not extract currency correctly for non-USD amounts
- No tools currently support non-SEC financial data
- "trillion won" parsing: 300 * 10^12 KRW

**Category:** Data coverage gap, international

---

## EC-006: Relative/Comparative Claims (No Absolute Value)

**Claim:** "Apple stock in 2010 was half that of this year"

**Expected verdict:** Requires fetching 2010 prices AND current prices, then comparing.

**Actual:** Parser correctly sets `value: null` and `comparison: null`. Agent receives a claim with no specific number to verify against, making standard verdict override impossible.

**Root cause:** The pipeline is optimized for claims with explicit numeric values. Relative claims ("half", "double", "triple") have no absolute target to compare against.

**Affected components:**
- `claim_parser.py` — correctly sets value=null, but downstream nodes expect a value
- `base.py` — Python verdict override requires both `claimed_val` and `retrieved_value`
- Consensus — magnitude-based adjustments don't apply

**Potential fixes:**
- Parser resolves relative claims into absolute values when possible
- Agent fetches both values and computes the ratio
- Add a "ratio_comparison" mode to the verdict override logic

**Category:** Claim type limitation

---

## EC-007: Stale Market Data (Price Claims)

**Claim:** "Tesla's stock price is above $300"

**Expected:** Verdict depends on current market price.

**Actual:** Finnhub free tier has 15-20 minute delay. If the claim is submitted during market hours and the price is near $300, the delayed data may produce an incorrect verdict.

**Root cause:** Market data is not real-time. The `get_stock_quote` tool returns delayed prices.

**Affected components:**
- `market_tools.py` — `get_stock_quote` returns delayed data
- `mcp/finnhub.py` — Finnhub free tier limitation

**Potential fixes:**
- Add a disclosure: "Market data delayed 15-20 minutes"
- Widen tolerance for current-price claims
- Use multiple data sources for price corroboration

**Category:** Data freshness

---

## EC-008: XBRL Extraction Failure — Company-Specific (Coca-Cola)

**Claim 1:** "Coca cola's annual revenue in 2000 exceeded 50 billion"
**Claim 2:** "Coca Cola revenue in 2015 exceeded 50 billion"

**Expected verdicts:**
- Claim 1: REFUTES (2000 revenue was ~$20.5B)
- Claim 2: REFUTES (2015 revenue was ~$44.3B)

**Actual verdict:** NOT_ENOUGH_INFO (0.35 confidence, triggered HITL) — for BOTH claims

**Root cause (UPDATED):** Initially attributed to pre-XBRL era (pre-2009), but testing Claim 2 (2015, well after XBRL mandate) produces the same failure. The MCP server's XBRL extraction cannot parse Coca-Cola's filings **regardless of year**. The `get_income_statement` tool returns `items=[]` for both the 2000 and 2015 10-K filings. This points to a **company-specific XBRL parsing issue**, not a temporal one:
1. Coca-Cola may use non-standard XBRL taxonomy extensions for revenue concepts
2. The MCP server's regex patterns for concept extraction may not match Coca-Cola's inline XBRL format
3. The filing structure may differ from the companies the extractor was tested with (Apple, Microsoft, etc.)

**Test pair value:** These two claims together prove the issue is NOT pre-XBRL vs post-XBRL. It's company-specific extraction failure.

**Affected components:**
- `../SEC-MCP/sec-edgar-mcp/` — `_extract_xbrl_concept_value()` fails on Coca-Cola's XBRL format
- `mcp/sec_edgar.py` — XBRL extraction returns empty, no fallback
- `rag/` — filing text not ingested into vector store as backup
- `base.py` — no fallback when XBRL extraction returns empty but filing exists

**Potential fixes:**
- Debug the MCP server against Coca-Cola's actual 10-K XBRL to find which concepts/patterns fail
- Add broader XBRL concept aliases (e.g., `us-gaap:Revenues` vs `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`)
- Ingest filing text into RAG as fallback when XBRL returns empty
- Fall back to news agent for corroboration
- Add HTML table parsing as last resort

**Category:** XBRL extraction failure (company-specific)

---

## Template for New Edge Cases

```
## EC-XXX: [Short Title]

**Claim:** "[exact claim text]"

**Expected verdict:** [what should happen]

**Actual verdict:** [what actually happened]

**Root cause:** [why]

**Affected components:**
- [file] — [what's wrong]

**Potential fixes:**
- [fix 1]
- [fix 2]

**Category:** [parser consistency | data coverage gap | period resolution | multi-step reasoning | claim type limitation | data freshness | guardrail gap]
```
