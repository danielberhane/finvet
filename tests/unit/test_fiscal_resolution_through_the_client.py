"""The statement a claim is checked against, built from SEC's feed.

Driven through `SECEdgarClient.get_financials`, the call every statement tool
makes. Payloads are the recorded companyfacts fixtures, served in place of the
one HTTP download; nothing else is patched.

The property that matters most is the second class: the agent chooses which
filing to open, and Bank of America's quarterly revenue is one figure in one
filing and another in the next. A number that depends on that choice is a
number the model picked.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from finvet.mcp.sec_edgar import SECDataUnavailable, SECEdgarClient

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sec_companyfacts"
CONTRACT = "RevenueFromContractWithCustomerExcludingAssessedTax"
CIK = {"HD": "0000354950", "FDX": "0001048911", "GE": "0000040545",
       "BAC": "0000070858", "JPM": "0000019617"}


def _raw(ticker):
    return json.loads((FIXTURES / f"{ticker}.json").read_text())


def _statement(ticker, statement, year, period, accession="0000000000-00-000000",
               client=None):
    client = client or SECEdgarClient()
    kind = "annual" if period == "FY" else "quarterly"
    with patch.object(client, "_download_company_facts", return_value=_raw(ticker)), \
         patch.object(client._mcp, "call_tool",
                      side_effect=AssertionError("the fiscal path opens no filing")):
        items = client.get_financials(
            CIK[ticker], accession, statement, period=kind,
            period_end=f"{year}-12-31", fiscal_year=year, fiscal_period=period)
    return {i.line_item: i for i in items}


class TestTheStatementForAFiscalPeriod:

    def test_home_depot_fiscal_2025_revenue(self):
        revenue = _statement("HD", "income", 2025, "FY")[CONTRACT]
        assert revenue.value == 164_683_000_000
        assert (revenue.period_start, revenue.period_end) == ("2025-02-03", "2026-02-01")
        assert revenue.consolidated is True
        assert (revenue.fiscal_year, revenue.fiscal_period) == (2025, "FY")

    def test_fedex_net_income_comes_from_the_annual_report(self):
        income = _statement("FDX", "income", 2026, "FY")["NetIncomeLoss"]
        assert income.value == 4_433_000_000
        assert income.source_form == "10-K"

    def test_a_balance_sheet_total(self):
        assets = _statement("JPM", "balance", 2025, "FY")["Assets"]
        assert assets.value == 4_424_900_000_000
        assert assets.period_end == "2025-12-31"

    def test_a_restated_quarter_carries_what_it_replaced(self):
        revenue = _statement("BAC", "income", 2025, "Q2")["Revenues"]
        assert revenue.value == 27_443_000_000
        assert revenue.superseded_values == [26_463_000_000]

    def test_a_concept_that_cannot_be_established_is_left_out(self):
        """GE files contract revenue for the fourth quarter only. There is no
        annual figure to offer, and a line with no verified value is not
        offered with an unverified one."""
        statement = _statement("GE", "income", 2023, "FY")
        assert CONTRACT not in statement
        assert statement["Revenues"].value == 35_348_000_000


class TestTheFilingOpenedDoesNotChangeTheNumber:

    @pytest.mark.parametrize("ticker,statement,year,period", [
        ("BAC", "income", 2025, "Q2"),
        ("GE", "income", 2023, "FY"),
        ("HD", "income", 2025, "FY"),
    ])
    def test_two_filings_one_statement(self, ticker, statement, year, period):
        first = _statement(ticker, statement, year, period,
                           accession="0000070858-25-000268")
        second = _statement(ticker, statement, year, period,
                            accession="0000070858-26-000394")
        assert ({k: v.value for k, v in first.items()}
                == {k: v.value for k, v in second.items()})


class TestTheFeed:

    def test_one_download_serves_every_statement_of_an_issuer(self):
        client = SECEdgarClient()
        with patch.object(client, "_download_company_facts",
                          return_value=_raw("HD")) as download, \
             patch.object(client._mcp, "call_tool", side_effect=AssertionError):
            for statement in ("income", "balance", "cashflow", "income"):
                client.get_financials(CIK["HD"], None, statement, period="annual",
                                      period_end="2025-12-31",
                                      fiscal_year=2025, fiscal_period="FY")
        assert download.call_count == 1

    def test_an_unavailable_feed_is_an_error_not_an_empty_statement(self):
        client = SECEdgarClient()
        with patch.object(client, "_download_company_facts", return_value=None), \
             patch.object(client._mcp, "call_tool", side_effect=AssertionError):
            with pytest.raises(SECDataUnavailable):
                client.get_financials(CIK["HD"], None, "income", period="annual",
                                      period_end="2025-12-31",
                                      fiscal_year=2025, fiscal_period="FY")

    def test_without_a_fiscal_period_the_filing_is_still_read(self):
        """A claim naming no period has no fiscal label to resolve, and keeps
        the path it had."""
        client = SECEdgarClient()
        mcp = {"success": True, "cik": 354950, "accession_number": "a",
               "concepts": {"NetIncomeLoss": {"value": 1.0, "unit": "USD",
                                              "context": "c", "period": "2026-02-01"}},
               "filing_reference": {"filing_date": "2026-03-18"}}
        with patch.object(client, "_download_company_facts",
                          return_value=_raw("HD")), \
             patch.object(client._mcp, "call_tool", return_value=mcp) as call:
            client.get_financials(CIK["HD"], "a", "income", period="annual")
        assert call.called
