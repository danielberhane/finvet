"""SEC EDGAR MCP server adapter — typed methods backed by the real MCP server."""

from datetime import date
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..utils.logging import get_logger
from .mcp_client import MCPClient, MCPError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models (contract with tools/sec_tools.py — do NOT change fields)
# ---------------------------------------------------------------------------


class CompanyInfo(BaseModel):
    """Company information from SEC EDGAR."""
    cik: str = Field(..., description="SEC Central Index Key")
    name: str = Field(..., description="Official company name")
    ticker: Optional[str] = Field(None, description="Primary ticker symbol")
    fiscal_year_end: str = Field(..., description="Fiscal year end (MMDD format)")
    sic: Optional[str] = Field(None, description="Standard Industrial Classification code")
    sic_description: Optional[str] = Field(None, description="Industry description")


class Filing(BaseModel):
    """SEC filing information."""
    accession_number: str = Field(..., description="SEC accession number")
    filing_date: str = Field(..., description="Date the filing was submitted")
    period_of_report: str = Field(..., description="Period covered by the report")
    form_type: str = Field(..., description="Type of form (10-K, 10-Q, 8-K, etc.)")
    file_number: Optional[str] = Field(None, description="SEC file number")
    url: Optional[str] = Field(None, description="URL to the filing")


class FinancialItem(BaseModel):
    """Financial line item from SEC filing."""
    line_item: str = Field(..., description="Name of the line item")
    concept: Optional[str] = Field(None, description="XBRL concept name")
    value: float = Field(..., description="Numeric value")
    units: str = Field(default="USD", description="Currency and scale (e.g., 'USD')")
    period: str = Field(..., description="Period the value applies to")
    period_end: Optional[str] = Field(None, description="End date of the period (YYYY-MM-DD)")
    decimals: Optional[str] = Field(None, description="Precision indicator")
    context_ref: Optional[str] = Field(None, description="XBRL context reference")
    consolidated: Optional[bool] = Field(
        None,
        description="True if confirmed as an entity-wide fact, False if unverified, "
                    "None if the concept is not consolidation-sensitive",
    )


# ---------------------------------------------------------------------------
# XBRL concept lists by statement type
# ---------------------------------------------------------------------------

CONCEPTS_BY_TYPE = {
    "income": [
        # Revenue — companies use different concepts depending on era and industry.
        # ASC 606 (post-2018): RevenueFromContractWithCustomer...
        # General: Revenues
        # Pre-2018: SalesRevenueNet, SalesRevenueGoodsNet
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "GrossProfit",
        "OperatingIncomeLoss",
        "NetIncomeLoss",
        "ProfitLoss",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        # Both in the claim parser's sec metric whitelist; their absence made
        # R&D and interest-expense claims structurally unverifiable (20 of 200
        # real-sourced sec rows fell through to human review as PENDING).
        "ResearchAndDevelopmentExpense",
        "InterestExpense",
    ],
    "balance": [
        "Assets",
        "AssetsCurrent",
        "Liabilities",
        "LiabilitiesCurrent",
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "PropertyPlantAndEquipmentNet",
    ],
    "cashflow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInFinancingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsOfDividends",
    ],
}


# Concepts a filing commonly tags BOTH entity-wide and broken out by product,
# segment or geography. The MCP server returns whichever XBRL fact it meets
# first, so for these the dimensioned member can shadow the consolidated total
# (Apple FY2024: Products $294.866B instead of net sales $391.035B). For these
# concepts the entity-wide value is re-read from SEC's companyconcept API,
# which exposes undimensioned facts only.
CONSOLIDATION_SENSITIVE_CONCEPTS = frozenset({
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "OperatingIncomeLoss",
})

SEC_COMPANY_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SECEdgarClient:
    """Typed adapter for the SEC EDGAR MCP server (streamable-http)."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.sec_edgar_mcp_url
        self._mcp = MCPClient(self.base_url)
        self._concept_cache: Dict[tuple, Optional[Dict]] = {}

    # -- company info --------------------------------------------------------

    def get_company_info(self, identifier: str) -> CompanyInfo:
        """Look up company metadata (CIK, fiscal year end, SIC, etc.)."""
        result = self._mcp.call_tool("get_company_info", {"identifier": identifier})

        if isinstance(result, dict):
            if not result.get("success", True):
                raise MCPError(f"SEC EDGAR error: {result.get('error', 'Unknown')}")
            if "company" in result:
                data = result["company"]
                if "cik" in data and isinstance(data["cik"], int):
                    data["cik"] = f"{data['cik']:010d}"
                return CompanyInfo(**data)
        return CompanyInfo(**result)

    # -- filings -------------------------------------------------------------

    def get_recent_filings(
        self,
        identifier: str,
        form_type: str,
        limit: int = 10,
    ) -> List[Filing]:
        """List recent SEC filings for a company."""
        result = self._mcp.call_tool(
            "get_recent_filings",
            {"identifier": identifier, "form_type": form_type, "limit": limit},
        )

        if isinstance(result, dict) and "filings" in result:
            return [Filing(**f) for f in result["filings"]]
        if isinstance(result, list):
            return [Filing(**f) for f in result]
        return []

    # -- financials via XBRL concepts ----------------------------------------

    def get_financials(
        self,
        identifier: str,
        accession_number: Optional[str] = None,
        statement_type: str = "income",
        period: str = "quarterly",
        period_end: Optional[str] = None,
    ) -> List[FinancialItem]:
        """Extract financial data from a specific filing using XBRL concepts.

        Uses get_xbrl_concepts on the MCP server (not get_financials) because
        get_xbrl_concepts accepts an accession_number for period-specific data
        while the server's get_financials only returns the latest filing.

        The MCP tool takes no period argument and returns a single, arbitrary
        XBRL context per concept, so a filing's comparative years come back
        interchangeably. Pass `period_end` (from the resolved canonical period)
        to re-read each concept from SEC's companyconcept feed, which carries
        every fact with its own start and end dates and can therefore be
        targeted. Without it the legacy behaviour is preserved.
        """
        # Normalise cashflow → cashflow key
        key = "cashflow" if statement_type in ("cash", "cashflow") else statement_type
        concepts = CONCEPTS_BY_TYPE.get(key, CONCEPTS_BY_TYPE["income"])

        args = {"identifier": identifier, "concepts": concepts}
        if accession_number:
            args["accession_number"] = accession_number

        result = self._mcp.call_tool("get_xbrl_concepts", args)
        items = self._parse_xbrl_result(result)

        ref = result if isinstance(result, dict) else {}
        cik = ref.get("cik") or identifier
        accn = ref.get("accession_number") or accession_number

        if period_end:
            return self._resolve_period(items, cik, accn, period_end, period)

        return self._resolve_consolidated(items, cik=cik, accession_number=accn)

    # -- period-targeted resolution ------------------------------------------

    def _resolve_period(
        self,
        items: List[FinancialItem],
        cik: Optional[Any],
        accession_number: Optional[str],
        period_end: str,
        period: str,
    ) -> List[FinancialItem]:
        """Re-read every concept for the requested period end and duration.

        companyconcept exposes undimensioned facts only, so this subsumes the
        consolidated-fact overlay: a value confirmed here is both entity-wide
        and for the period actually asked about. Anything that cannot be
        confirmed keeps the filing value but is flagged rather than trusted.
        """
        for item in items:
            item.consolidated = False
            if cik is None:
                continue

            payload = self._fetch_company_concept(cik, item.line_item)
            if not payload:
                continue

            value = _select_fact_for_period(
                payload, accession_number, period_end, period
            )
            if value is None:
                # The requested period matched nothing — the fiscal/calendar
                # misalignment case: period_resolver maps "fiscal 2024" to
                # 2024-12-31 while e.g. Apple's year ends 2024-09-28. Fall
                # back to the pre-period-fix consolidation logic for the
                # item's OWN period, or the segment-shadowing correction is
                # silently lost with it (live regression: Apple $294.866B
                # Products reached the verdict and refuted a true claim).
                if item.line_item not in CONSOLIDATION_SENSITIVE_CONCEPTS:
                    # None keeps its meaning: not consolidation-sensitive.
                    item.consolidated = None
                    continue
                fallback = _select_entity_wide_fact(
                    payload, accession_number, item.period_end
                )
                if fallback is None:
                    logger.info(
                        f"{item.line_item}: no entity-wide fact for requested "
                        f"period {period_end} ({period}) nor for the filing's "
                        f"own {item.period_end}; keeping unverified value "
                        f"{item.value:,.0f}"
                    )
                    continue
                if fallback != item.value:
                    logger.info(
                        f"{item.line_item}: requested period {period_end} "
                        f"unmatched; entity-wide {fallback:,.0f} for the "
                        f"filing's own {item.period_end} supersedes "
                        f"{item.value:,.0f}"
                    )
                item.value = fallback
                item.consolidated = True
                continue

            if value != item.value or item.period_end != period_end:
                logger.info(
                    f"{item.line_item}: filing fact {item.value:,.0f} "
                    f"(period {item.period_end}) superseded by entity-wide "
                    f"{value:,.0f} for requested period {period_end}"
                )
            item.value = value
            item.period_end = period_end
            item.period = period_end
            item.consolidated = True

        return items

    # -- consolidated-fact resolution ----------------------------------------

    def _resolve_consolidated(
        self,
        items: List[FinancialItem],
        cik: Optional[Any],
        accession_number: Optional[str],
    ) -> List[FinancialItem]:
        """Re-read consolidation-sensitive concepts from SEC's undimensioned feed.

        A value is only marked consolidated=True once an entity-wide fact for the
        same period has confirmed it. Anything we cannot confirm stays at the
        filing value but is flagged False rather than silently trusted.
        """
        for item in items:
            if item.line_item not in CONSOLIDATION_SENSITIVE_CONCEPTS:
                continue

            item.consolidated = False
            if cik is None:
                continue

            payload = self._fetch_company_concept(cik, item.line_item)
            if not payload:
                continue

            value = _select_entity_wide_fact(payload, accession_number, item.period_end)
            if value is None:
                continue

            if value != item.value:
                logger.info(
                    f"{item.line_item}: filing fact {item.value:,.0f} "
                    f"(context {item.context_ref}) superseded by entity-wide "
                    f"{value:,.0f} for period {item.period_end}"
                )
            item.value = value
            item.consolidated = True

        return items

    def _fetch_company_concept(self, cik: Any, concept: str) -> Optional[Dict]:
        """Fetch one concept from data.sec.gov. Returns None on any failure."""
        key = (str(cik), concept)
        if key in self._concept_cache:
            return self._concept_cache[key]

        url = SEC_COMPANY_CONCEPT_URL.format(
            cik=str(cik).lstrip("0").zfill(10), concept=concept
        )
        if settings.sec_user_agent_is_placeholder:
            logger.warning(
                "SEC_EDGAR_USER_AGENT is still the placeholder contact. SEC Fair Access "
                "requires a real name and email on automated requests; set it in .env."
            )

        payload: Optional[Dict] = None
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": settings.sec_edgar_user_agent},
                timeout=10.0,
            )
            if resp.status_code == 200:
                payload = resp.json()
            elif resp.status_code != 404:
                logger.warning(f"companyconcept {concept}: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"companyconcept {concept} unavailable: {e}")

        self._concept_cache[key] = payload
        return payload

    @staticmethod
    def _parse_xbrl_result(result) -> List[FinancialItem]:
        """Convert get_xbrl_concepts response into FinancialItem list."""
        items: List[FinancialItem] = []

        if not isinstance(result, dict):
            return items

        concepts = result.get("concepts", {})
        filing_ref = result.get("filing_reference", {})
        filing_date = filing_ref.get("filing_date", "")

        # Server returns concepts as a dict keyed by concept name
        if isinstance(concepts, dict):
            entries = [
                (name, data) for name, data in concepts.items()
                if isinstance(data, dict)
            ]
        elif isinstance(concepts, list):
            entries = [(c.get("concept", ""), c) for c in concepts if isinstance(c, dict)]
        else:
            return items

        for concept_name, data in entries:
            value = data.get("value")
            if value is None:
                continue
            try:
                value = float(value)
            except (ValueError, TypeError):
                logger.debug(f"Skipping non-numeric concept {concept_name}: value={value!r}")
                continue

            items.append(FinancialItem(
                line_item=concept_name,
                concept=f"us-gaap:{concept_name}" if concept_name else None,
                value=value,
                units=data.get("unit") or "USD",
                period=data.get("period", ""),
                period_end=data.get("period") or filing_date,
                decimals=data.get("decimals"),
                context_ref=data.get("context"),
            ))

        return items

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        self._mcp.close()


def _select_entity_wide_fact(
    payload: Dict,
    accession_number: Optional[str],
    period_end: Optional[str],
) -> Optional[float]:
    """Pick the entity-wide fact matching this filing's reporting period.

    companyconcept carries only undimensioned facts, but a 10-K reports three
    comparative years, so the period end is what disambiguates them. Facts from
    the filing under inspection win; otherwise any filing reporting the same
    period end is acceptable.
    """
    if not period_end:
        return None

    units = (payload.get("units") or {}).get("USD") or []
    matches = [f for f in units if f.get("end") == period_end and f.get("val") is not None]
    if not matches:
        return None

    same_filing = [f for f in matches if f.get("accn") == accession_number]
    chosen = same_filing or matches
    try:
        return float(chosen[0]["val"])
    except (TypeError, ValueError):
        return None


# Expected fact duration in days, by period kind. A single end date does not
# identify a fact: a 10-Q tags both the three-month quarter and the cumulative
# year-to-date figure with the same end, and picking the wrong one silently
# substitutes one for the other.
_PERIOD_DURATION_DAYS = {
    "quarterly": (75, 105),
    "half_year": (165, 195),
    "annual": (330, 400),
}


def _fact_duration_days(fact: Dict) -> Optional[int]:
    """Days a duration fact covers, or None for an instant (balance-sheet) fact."""
    start, end = fact.get("start"), fact.get("end")
    if not start or not end:
        return None
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except (TypeError, ValueError):
        return None


def _select_fact_for_period(
    payload: Dict,
    accession_number: Optional[str],
    period_end: Optional[str],
    period: str,
) -> Optional[float]:
    """Pick the entity-wide fact for a specific period end AND duration.

    companyconcept carries only undimensioned facts, but several of them can
    share an end date at different durations. Both must match. Facts from the
    filing under inspection win ties; otherwise any filing reporting the same
    period is acceptable. Returns None rather than guessing.
    """
    if not period_end:
        return None

    units = payload.get("units") or {}
    facts = [f for unit in units.values() for f in unit if isinstance(f, dict)]
    matches = [f for f in facts if f.get("end") == period_end and f.get("val") is not None]
    if not matches:
        return None

    window = _PERIOD_DURATION_DAYS.get(period)
    if window:
        lo, hi = window
        sized = [
            f for f in matches
            if (d := _fact_duration_days(f)) is None or lo <= d <= hi
        ]
        # An instant fact has no duration to check; a duration fact outside the
        # window is the wrong span and must not be substituted.
        if not sized:
            return None
        matches = sized

    same_filing = [f for f in matches if f.get("accn") == accession_number]
    chosen = same_filing or matches
    try:
        return float(chosen[0]["val"])
    except (TypeError, ValueError):
        return None
