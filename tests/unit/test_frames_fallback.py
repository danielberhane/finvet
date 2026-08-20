"""Tests for the frames-endpoint fallback in period-targeted retrieval.

The live defect (Ford, diluted EPS, 2Q 2024): the MCP server returned the
prior-year comparative from the right filing, and SEC's companyconcept feed —
the overlay's only entity-wide source — carries no facts at all for that
company and concept ("units": {"USD/shares": {}}). The frames endpoint, keyed
by concept + calendar frame, has the filed value. When companyconcept cannot
resolve the requested period, frames is the third door into the same primary
source; only when frames also misses does the old keep-but-flag path apply.
"""

from unittest.mock import patch

from finvet.mcp.sec_edgar import SECEdgarClient, frame_for

ACCN = "0000037996-24-000144"

# The MCP server's actual answer for Ford's Q2 2024 10-Q: the comparative.
MCP_COMPARATIVE = {
    "success": True,
    "cik": "0000037996",
    "accession_number": ACCN,
    "concepts": {
        "EarningsPerShareDiluted": {
            "value": 0.47, "unit": "USD/shares",
            "context": "c-7", "period": "2023-06-30",
        }
    },
    "filing_reference": {"filing_date": "2024-07-25"},
}

# companyconcept's actual answer for Ford + EarningsPerShareDiluted: nothing.
EMPTY_COMPANYCONCEPT = {"cik": 37996, "units": {"USD/shares": {}}}

FORD_FRAME_FACTS = {
    37996: {"cik": 37996, "entityName": "Ford Motor Co", "val": 0.46,
            "start": "2024-04-01", "end": "2024-06-30",
            "accn": "0000037996-25-000147"},
    320193: {"cik": 320193, "entityName": "Apple Inc.", "val": 1.40,
             "start": "2024-03-31", "end": "2024-06-29",
             "accn": "0000320193-24-000081"},
}


class TestFrameFor:

    def test_quarterly_maps_to_calendar_quarter(self):
        assert frame_for("2024-06-30", "quarterly") == "CY2024Q2"

    def test_annual_maps_to_calendar_year(self):
        assert frame_for("2024-12-31", "annual") == "CY2024"

    def test_instant_date_maps_to_instantaneous_frame(self):
        assert frame_for("2024-12-31", "date") == "CY2024Q4I"

    def test_half_year_has_no_frame(self):
        """SEC frames carry no six-month bucket; fabricating one would fetch
        the wrong duration."""
        assert frame_for("2024-06-30", "half_year") is None

    def test_unparseable_date_has_no_frame(self):
        assert frame_for("sometime", "quarterly") is None
        assert frame_for(None, "annual") is None


class TestFramesFallbackSelection:

    def _client(self):
        return SECEdgarClient()

    def test_matches_the_company_by_cik(self):
        client = self._client()
        with patch.object(client, "_fetch_frame_facts", return_value=FORD_FRAME_FACTS):
            got = client._frames_fallback("0000037996", "EarningsPerShareDiluted",
                                          "2024-06-30", "quarterly")
        assert got == (0.46, "2024-06-30")

    def test_zero_padded_and_bare_ciks_are_the_same_company(self):
        client = self._client()
        with patch.object(client, "_fetch_frame_facts", return_value=FORD_FRAME_FACTS):
            assert client._frames_fallback(37996, "EarningsPerShareDiluted",
                                           "2024-06-30", "quarterly") == (0.46, "2024-06-30")

    def test_company_absent_from_the_frame_returns_none(self):
        client = self._client()
        with patch.object(client, "_fetch_frame_facts", return_value=FORD_FRAME_FACTS):
            assert client._frames_fallback("0000000123", "EarningsPerShareDiluted",
                                           "2024-06-30", "quarterly") is None

    def test_unframeable_period_returns_none_without_fetching(self):
        client = self._client()
        with patch.object(client, "_fetch_frame_facts") as fetch:
            got = client._frames_fallback("0000037996", "EarningsPerShareDiluted",
                                          "2024-06-30", "half_year")
        assert got is None
        fetch.assert_not_called()


class TestGetFinancialsUsesFramesWhenCompanyconceptIsEmpty:

    def _client(self):
        return SECEdgarClient()

    def test_the_ford_case_retrieves_the_filed_value(self):
        """Comparative from the filing + empty companyconcept must not stand
        when frames carries the requested period's fact."""
        client = self._client()
        with patch.object(client._mcp, "call_tool", return_value=MCP_COMPARATIVE), \
             patch.object(client, "_fetch_company_concept",
                          return_value=EMPTY_COMPANYCONCEPT), \
             patch.object(client, "_fetch_frame_facts", return_value=FORD_FRAME_FACTS):
            items = client.get_financials(
                "0000037996", ACCN, "income", period="quarterly",
                period_end="2024-06-30",
            )
        item = next(i for i in items if i.line_item == "EarningsPerShareDiluted")
        assert item.value == 0.46
        assert item.period_end == "2024-06-30"
        assert item.consolidated is True

    def test_frames_miss_keeps_the_old_flagged_behaviour(self):
        client = self._client()
        with patch.object(client._mcp, "call_tool", return_value=MCP_COMPARATIVE), \
             patch.object(client, "_fetch_company_concept",
                          return_value=EMPTY_COMPANYCONCEPT), \
             patch.object(client, "_fetch_frame_facts", return_value={}):
            items = client.get_financials(
                "0000037996", ACCN, "income", period="quarterly",
                period_end="2024-06-30",
            )
        item = next(i for i in items if i.line_item == "EarningsPerShareDiluted")
        assert item.value == 0.47
        assert item.period_end == "2023-06-30"
        assert item.consolidated is not True

    def test_companyconcept_hit_never_consults_frames(self):
        """frames is a fallback, not a second opinion: the companyconcept fact
        already matches end date AND duration, which frames cannot refine."""
        payload = {"units": {"USD/shares": [
            {"start": "2024-04-01", "end": "2024-06-30", "val": 0.46,
             "accn": ACCN, "form": "10-Q"},
        ]}}
        client = self._client()
        with patch.object(client._mcp, "call_tool", return_value=MCP_COMPARATIVE), \
             patch.object(client, "_fetch_company_concept", return_value=payload), \
             patch.object(client, "_fetch_frame_facts") as frames:
            items = client.get_financials(
                "0000037996", ACCN, "income", period="quarterly",
                period_end="2024-06-30",
            )
        assert next(i for i in items if i.line_item == "EarningsPerShareDiluted").value == 0.46
        frames.assert_not_called()
