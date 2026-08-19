"""Tests for consolidated-fact resolution in the SEC EDGAR adapter.

Regression cover for the segment-shadowing bug: a filing's XBRL tags the same
revenue concept at the consolidated level and again per product/segment, and the
MCP server returns whichever fact it encounters first. For Apple's FY2024 10-K
that is the Products member ($294.866B), not consolidated net sales ($391.035B),
which turned a true claim into a confident REFUTES.
"""

from unittest.mock import patch

import pytest

from finvet.mcp.sec_edgar import (
    CONSOLIDATION_SENSITIVE_CONCEPTS,
    SECEdgarClient,
)

APPLE_ACCN = "0000320193-24-000123"
PRODUCTS_REVENUE = 294_866_000_000.0
CONSOLIDATED_REVENUE = 391_035_000_000.0
NET_INCOME = 93_736_000_000.0

# What the MCP server actually returns for Apple's FY2024 10-K: revenue carries
# a dimensioned context (c-13), net income an undimensioned one (c-1).
MCP_RESPONSE = {
    "success": True,
    "cik": 320193,
    "filing_date": "2024-11-01",
    "accession_number": APPLE_ACCN,
    "concepts": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "value": PRODUCTS_REVENUE,
            "unit": "USD",
            "context": "c-13",
            "period": "2024-09-28",
        },
        "NetIncomeLoss": {
            "value": NET_INCOME,
            "unit": "USD",
            "context": "c-1",
            "period": "2024-09-28",
        },
    },
    "filing_reference": {"filing_date": "2024-11-01"},
}

# Shape of data.sec.gov companyconcept — undimensioned facts only. A 10-K
# carries three comparative years, so period matching is load-bearing.
COMPANY_CONCEPT = {
    "units": {
        "USD": [
            {"start": "2021-09-26", "end": "2022-09-24", "val": 394_328_000_000,
             "accn": APPLE_ACCN, "fy": 2024, "fp": "FY", "form": "10-K"},
            {"start": "2022-09-25", "end": "2023-09-30", "val": 383_285_000_000,
             "accn": APPLE_ACCN, "fy": 2024, "fp": "FY", "form": "10-K"},
            {"start": "2023-10-01", "end": "2024-09-28", "val": CONSOLIDATED_REVENUE,
             "accn": APPLE_ACCN, "fy": 2024, "fp": "FY", "form": "10-K"},
        ]
    }
}


def _items_by_concept(items):
    return {i.line_item: i for i in items}


class TestConsolidationSensitiveConcepts:

    def test_revenue_concepts_are_marked_sensitive(self):
        assert "RevenueFromContractWithCustomerExcludingAssessedTax" in CONSOLIDATION_SENSITIVE_CONCEPTS
        assert "Revenues" in CONSOLIDATION_SENSITIVE_CONCEPTS

    def test_net_income_is_not_sensitive(self):
        """NetIncomeLoss is not segment-broken in practice; leave it alone."""
        assert "NetIncomeLoss" not in CONSOLIDATION_SENSITIVE_CONCEPTS


class TestConsolidatedOverlay:

    def test_dimensioned_revenue_is_replaced_with_consolidated_total(self):
        """The bug: $294.866B (Products) must become $391.035B (consolidated)."""
        client = SECEdgarClient()
        with patch.object(client._mcp, "call_tool", return_value=MCP_RESPONSE), \
             patch.object(client, "_fetch_company_concept", return_value=COMPANY_CONCEPT):
            items = _items_by_concept(client.get_financials("AAPL", APPLE_ACCN, "income"))

        revenue = items["RevenueFromContractWithCustomerExcludingAssessedTax"]
        assert revenue.value == CONSOLIDATED_REVENUE
        assert revenue.consolidated is True

    def test_period_end_selects_the_right_comparative_year(self):
        """A 10-K holds three years; the filing's own period must win."""
        client = SECEdgarClient()
        with patch.object(client._mcp, "call_tool", return_value=MCP_RESPONSE), \
             patch.object(client, "_fetch_company_concept", return_value=COMPANY_CONCEPT):
            items = _items_by_concept(client.get_financials("AAPL", APPLE_ACCN, "income"))

        assert items["RevenueFromContractWithCustomerExcludingAssessedTax"].value != 383_285_000_000.0

    def test_non_sensitive_concepts_are_untouched(self):
        client = SECEdgarClient()
        with patch.object(client._mcp, "call_tool", return_value=MCP_RESPONSE), \
             patch.object(client, "_fetch_company_concept", return_value=COMPANY_CONCEPT):
            items = _items_by_concept(client.get_financials("AAPL", APPLE_ACCN, "income"))

        net_income = items["NetIncomeLoss"]
        assert net_income.value == NET_INCOME
        assert net_income.consolidated is None

    def test_unverified_value_is_flagged_not_silently_trusted(self):
        """If SEC is unreachable we keep the filing value but never claim it is consolidated."""
        client = SECEdgarClient()
        with patch.object(client._mcp, "call_tool", return_value=MCP_RESPONSE), \
             patch.object(client, "_fetch_company_concept", return_value=None):
            items = _items_by_concept(client.get_financials("AAPL", APPLE_ACCN, "income"))

        revenue = items["RevenueFromContractWithCustomerExcludingAssessedTax"]
        assert revenue.value == PRODUCTS_REVENUE
        assert revenue.consolidated is False

    def test_no_matching_accession_leaves_value_flagged(self):
        other = {"units": {"USD": [
            {"start": "2022-09-25", "end": "2023-09-30", "val": 383_285_000_000,
             "accn": "0000320193-23-000106", "fy": 2023, "fp": "FY", "form": "10-K"},
        ]}}
        client = SECEdgarClient()
        with patch.object(client._mcp, "call_tool", return_value=MCP_RESPONSE), \
             patch.object(client, "_fetch_company_concept", return_value=other):
            items = _items_by_concept(client.get_financials("AAPL", APPLE_ACCN, "income"))

        assert items["RevenueFromContractWithCustomerExcludingAssessedTax"].consolidated is False
