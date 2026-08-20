"""Metric vocabulary — vendored from the fine-tuned claim parser.

Source: ~/Projects/Active/claim_parser_fine-tuned/scripts/vocab.py
Vendored: 2026-08-20 (separate repos — copied, not imported).
tests/unit/test_metrics_vocab.py compares this copy against the source when the
sibling repo is present, so drift is caught rather than accumulated.

Four structures:

- METRIC_WHITELIST — the 74 canonical metrics, scoped by claim_type with zero
  cross-class overlap, so a claim_type fully determines the candidate set.
- METRIC_REMAPS — aliases a model plausibly emits, mapped to canonical names.
- METRIC_HARD_DROPS — names that look like metrics but must resolve to null:
  segment and product-line figures (iphone_revenue), KPIs (subscriber_count),
  and corporate events (buyback). Resolving these to a parent metric is the
  segment-shadowing bug class.
- SERVABLE_METRICS — ours, not the parser project's: the subset FinVet can
  actually retrieve today. The whole whitelist is carried for conformance with
  the parser and its gold sets; this set is what turns "valid metric, no data
  source" into a disclosed limitation instead of a silent NOT_ENOUGH_INFO.
"""

METRIC_WHITELIST: dict[str, frozenset[str]] = {
    "sec": frozenset({
        "adjusted_ebitda", "book_value_per_share", "capex",
        "cost_of_revenue", "diluted_eps", "ebitda", "eps", "free_cash_flow",
        "gross_margin", "gross_profit", "interest_expense", "net_income",
        "net_margin", "operating_cash_flow", "operating_income",
        "operating_margin", "research_and_development", "revenue",
        "shareholders_equity", "total_assets", "total_debt",
        "total_liabilities",
    }),
    "market": frozenset({
        "52_week_high", "52_week_low", "beta", "closing_price",
        "debt_to_equity", "dividend_yield", "enterprise_value",
        "ev_to_ebitda", "forward_pe", "intraday_high", "intraday_low",
        "market_cap", "opening_price", "pe_ratio", "price_change_absolute",
        "price_change_percent", "price_to_book", "price_to_sales",
        "shares_outstanding", "short_interest", "volume",
    }),
    "news": frozenset({
        "acquisition_value", "consensus_rating", "consumer_confidence",
        "core_pce", "cpi_inflation", "earnings_growth_forecast",
        "estimated_eps", "estimated_revenue", "federal_funds_rate",
        "fine_amount", "gdp_growth", "housing_starts", "investment_value",
        "layoffs", "options_exercised", "ownership_percentage", "pmi",
        "price_target", "retail_sales_growth", "revenue_growth_forecast",
        "settlement_amount", "shares_bought", "shares_bought_value",
        "shares_sold", "shares_sold_value", "spinoff", "tariff_rate",
        "tax_rate", "trade_deficit", "unemployment_rate", "wage_growth",
    }),
    "reject": frozenset(),
}

METRIC_REMAPS: dict[str, dict[str, str]] = {
    "sec": {
        "net_interest_expense": "interest_expense",
        "r_and_d_expense": "research_and_development",
        "research_and_development_expense": "research_and_development",
    },
    "market": {
        "enterprise_value_to_ebitda": "ev_to_ebitda",
        "ev_ebitda": "ev_to_ebitda",
        "fifty_two_week_high": "52_week_high",
        "fifty_two_week_low": "52_week_low",
        "forward_pe_ratio": "forward_pe",
        "price_to_book_ratio": "price_to_book",
        "price_to_sales_ratio": "price_to_sales",
        "stock_price": "closing_price",
    },
    "news": {
        "acquisition": "acquisition_value",
        "analyst_earnings_estimate": "estimated_eps",
        "analyst_price_target": "price_target",
        "analyst_rating": "consensus_rating",
        "analyst_revenue_estimate": "estimated_revenue",
        "capital_investment": "investment_value",
        "earnings_estimate": "estimated_eps",
        "fine": "fine_amount",
        "inflation_rate": "cpi_inflation",
        "insider_ownership": "ownership_percentage",
        "investment": "investment_value",
        "legal_settlement": "settlement_amount",
        "merger": "acquisition_value",
        "ownership": "ownership_percentage",
        "policy_rate": "federal_funds_rate",
        "regulatory_fine": "fine_amount",
        "retail_sales_change": "retail_sales_growth",
        "revenue_estimate": "estimated_revenue",
        "settlement": "settlement_amount",
        "stock_options_exercise": "options_exercised",
    },
    "reject": {},
}

METRIC_HARD_DROPS: frozenset[str] = frozenset({
    "advertising_revenue", "audit_review", "aws_revenue", "bankruptcy",
    "buyback", "cash_and_equivalents", "ceo_departure", "cloud_revenue",
    "contribution_margin", "data_center_gpu_share", "data_center_revenue",
    "delisting", "deliveries", "delivery_forecast", "dividend_increase",
    "downgrade", "executive_departure", "executive_resignation",
    "expense_allocation_review", "filing_error_warning", "gaming_revenue",
    "gross_bookings", "gross_merchandise_volume", "iphone_revenue", "ipo",
    "lawsuit", "leadership_change", "leadership_transition",
    "merger_breakup_fee", "monthly_active_users", "nights_booked",
    "operating_expenses", "operating_margin", "partnership",
    "product_launch", "regulatory_action", "restatement", "restructuring",
    "revenue_recognition_review", "search_revenue", "segment_revenue",
    "services_revenue", "strategic_partnership", "subscriber_count",
    "subscriber_forecast", "subscription_revenue", "total_payment_volume",
    "transaction_margin", "upgrade",
})

# What FinVet can retrieve today. sec: the concepts requested from SEC EDGAR
# in mcp/sec_edgar.py CONCEPTS_BY_TYPE (total_debt is partial — LongTermDebt
# only). Derived metrics (margins, free_cash_flow) and non-GAAP (ebitda) are
# deliberately absent pending the stage-05 derivation decision. market: the
# fields Finnhub's quote and company-overview endpoints return. news: FinVet
# has no structured news or macro source, so nothing is servable.
SERVABLE_METRICS: dict[str, frozenset[str]] = {
    "sec": frozenset({
        "capex", "cost_of_revenue", "diluted_eps", "eps", "gross_profit",
        "interest_expense", "net_income", "operating_cash_flow",
        "operating_income", "research_and_development", "revenue",
        "shareholders_equity", "total_assets", "total_debt",
        "total_liabilities",
    }),
    "market": frozenset({
        "52_week_high", "52_week_low", "closing_price", "dividend_yield",
        "intraday_high", "intraday_low", "market_cap", "opening_price",
        "pe_ratio", "price_change_absolute", "price_change_percent",
        "volume",
    }),
    "news": frozenset(),
    "reject": frozenset(),
}


# sec metric -> the XBRL concepts that carry it, mirroring CONCEPTS_BY_TYPE in
# mcp/sec_edgar.py (kept in step by tests). Used by the retrieved-value
# fallback so a concept lookup replaces the old pick-the-number-nearest-the-
# claim search — a confirmation bias that sat directly under the verdict
# override. Derived metrics are deliberately absent: they have no single
# concept, and the fallback declines rather than guesses.
METRIC_TO_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps": ("EarningsPerShareBasic",),
    "diluted_eps": ("EarningsPerShareDiluted",),
    "research_and_development": ("ResearchAndDevelopmentExpense",),
    "interest_expense": ("InterestExpense",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "shareholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "total_debt": ("LongTermDebt", "LongTermDebtNoncurrent"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
}
