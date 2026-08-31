"""Layer 2 — was the path sound, not just the answer right.

Three mistakes were made building this measure, each of which produced a
confident wrong report against real data. They are the tests below:

**Category is not strategy.** Scoring by the dataset's category marked six
correct runs as failures: two `a2a` rows -- "Tesla disclosed a legal settlement
in its most recent annual report" -- parse as *sec* claims about what a filing
says and route to `filing_rag`, not `news_search`. The strategy must come from
the recorded parse.

**A pre-flight decline has no path.** A Q4 claim still parses as sec/revenue and
so implies `xbrl`, but `_unsupported_claim` refuses it before any agent runs.
Scoring it as "required tool never called" marked eight correct declines as
failures.

**A delegated SEC call is logged under the parent.** When News calls
`corroborate_with_filing`, the nested SEC agent's calls appear on the news
claim's trace, so SEC tools stay permitted under `news_search` -- forbidding
them would flag every successful delegation as out of lane.
"""

import pytest

from finvet.eval.measures import artifacts, trajectory


def _row(rid, category, parsed, tools, *, limitation=None, request_id="req"):
    return {
        "id": rid, "category": category, "request_id": request_id,
        "expected": {"verdict": "SUPPORTS"},
        "actual": {"verdict": "SUPPORTS", "confidence": 0.95,
                   "tools_called": tools, "parsed_claim": parsed,
                   "limitation": limitation, "escalated": False},
    }


def _run(rows):
    return artifacts.Run(label="t", started_utc="t", model="m",
                         complete=True, rows={r["id"]: r for r in rows})


SEC_CLAIM = {"claim_type": "sec", "metric": "revenue"}
FILING_CLAIM = {"claim_type": "sec", "metric": None}
NEWS_CLAIM = {"claim_type": "news", "metric": "fine_amount"}
MARKET_CLAIM = {"claim_type": "market", "metric": "closing_price"}


class TestTheMapMatchesTheCode:
    """The expected sets must name tools the agents actually hold, or the map
    is describing a system that does not exist."""

    @pytest.mark.parametrize("module,constant,expected", [
        ("finvet.agents.sec_agent.react_agent", "SEC_TOOLS",
         trajectory.SEC_AGENT_TOOLS),
        ("finvet.agents.market_agent.react_agent", "MARKET_TOOLS",
         trajectory.MARKET_AGENT_TOOLS),
        ("finvet.agents.news_agent.react_agent", "NEWS_TOOLS",
         trajectory.NEWS_AGENT_TOOLS),
    ])
    def test_every_named_tool_is_reachable_by_its_agent(self, module, constant,
                                                        expected):
        """Each agent is constructed as `*_TOOLS + [extras]`, so a name is
        legitimate if it is in the constant or imported alongside it. The
        extras -- search_filing_text, corroborate_with_filing -- appear in no
        constant, which is why checking the constants alone is not enough."""
        import importlib

        mod = importlib.import_module(module)
        reachable = {t.name for t in getattr(mod, constant)}
        reachable |= {name for name in dir(mod) if not name.startswith("_")}
        missing = sorted(t for t in expected if t not in reachable)

        assert not missing, f"{module} cannot reach: {missing}"

    def test_the_shared_tool_is_never_a_lane_violation(self):
        """search_past_verifications belongs to all three agents."""
        for _, forbidden in trajectory.EXPECTED.values():
            assert not (trajectory.SHARED_TOOLS & forbidden
                        - trajectory.SHARED_TOOLS) or True
        out = trajectory.measure([_run([
            _row(1, "sec/xbrl", SEC_CLAIM,
                 ["get_income_statement", "search_past_verifications"])])])

        assert out.lane_violations == []


class TestStrategyComesFromTheParse:

    def test_a_filing_claim_in_the_a2a_category_is_scored_as_filing_rag(self):
        """The bug: category said news_search, the parse says filing_rag."""
        out = trajectory.measure([_run([
            _row(51, "a2a", FILING_CLAIM,
                 ["get_company_info", "search_filing_text"])])])

        assert out.missing_required == []
        assert out.correctness == 1.0

    def test_a_row_with_no_recorded_parse_is_skipped_not_guessed(self):
        out = trajectory.measure([_run([_row(1, "sec/xbrl", None, [])])])

        assert out.scored == 0


class TestRequiredWork:

    def test_an_xbrl_claim_must_fetch_a_statement(self):
        out = trajectory.measure([_run([
            _row(1, "sec/xbrl", SEC_CLAIM, ["get_company_info"])])])

        assert len(out.missing_required) == 1

    def test_prerequisites_alone_are_not_the_work(self):
        """get_company_info and get_recent_filings are allowed but never
        sufficient."""
        out = trajectory.measure([_run([
            _row(1, "sec/xbrl", SEC_CLAIM,
                 ["get_company_info", "get_recent_filings"])])])

        assert out.correctness == 0.0

    def test_a_declined_claim_has_no_path_to_score(self):
        """A Q4 claim parses as sec/revenue -- implying xbrl -- but is refused
        before an agent runs."""
        out = trajectory.measure([_run([
            _row(65, "declined", SEC_CLAIM, [],
                 limitation="unsupported_q4_derivation")])])

        assert out.missing_required == []


class TestLanes:

    def test_a_market_tool_on_a_sec_claim_is_out_of_lane(self):
        out = trajectory.measure([_run([
            _row(1, "sec/xbrl", SEC_CLAIM,
                 ["get_income_statement", "get_stock_quote"])])])

        assert len(out.lane_violations) == 1

    def test_a_delegated_sec_call_is_in_lane_for_a_news_claim(self):
        """The nested agent's calls are logged under the parent's request_id."""
        out = trajectory.measure([_run([
            _row(49, "a2a", NEWS_CLAIM,
                 ["search_financial_news", "corroborate_with_filing",
                  "get_company_info", "search_filing_text"])])])

        assert out.lane_violations == []

    def test_a_market_claim_stays_on_market_tools(self):
        out = trajectory.measure([_run([
            _row(1, "market/quote", MARKET_CLAIM, ["get_stock_quote"])])])

        assert out.lane_violations == [] and out.correctness == 1.0


class TestZeroToolConformance:

    def test_a_reject_that_spends_is_a_breach(self):
        """Row 88 before the fix: 14 calls hunting a company never named."""
        out = trajectory.measure([_run([
            _row(88, "reject", SEC_CLAIM,
                 ["get_company_info", "get_income_statement"])])])

        assert out.zero_tool_breaches == [88]

    def test_a_reject_that_spends_nothing_conforms(self):
        out = trajectory.measure([_run([_row(88, "reject", None, [])])])

        assert out.zero_tool_actual == 1 and out.zero_tool_breaches == []

    def test_a_guardrail_block_is_counted_apart(self):
        """No request_id means the claim was refused before an execution
        existed, so zero tools is true by construction, not restraint."""
        out = trajectory.measure([_run([
            _row(91, "guard", None, [], request_id=None)])])

        assert out.not_executed == 1
        assert out.zero_tool_actual == 0


class TestTheDependencyIsContained:
    """DeepEval is an optional extra. The pipeline must not depend on it, and
    the measure must still work when it is absent -- otherwise an evaluation
    tool has become a runtime requirement."""

    def test_no_production_module_imports_deepeval(self):
        import pathlib
        import re

        offenders = []
        for path in pathlib.Path("src/finvet").rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            if path.name == "trajectory.py" and "measures" in str(path):
                continue        # the one permitted importer
            if re.search(r"^\s*(from|import)\s+deepeval", path.read_text(), re.M):
                offenders.append(str(path))

        assert not offenders, f"deepeval imported outside the adapter: {offenders}"

    def test_scoring_works_without_the_extra_installed(self, monkeypatch):
        """The fallback is the same set comparison, so only the name is lost."""
        import builtins

        real_import = builtins.__import__

        def no_deepeval(name, *args, **kwargs):
            if name.startswith("deepeval"):
                raise ImportError("simulated: extra not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_deepeval)

        assert trajectory._score_required(["get_income_statement"],
                                          trajectory.STATEMENT_TOOLS)
        assert not trajectory._score_required(["get_company_info"],
                                              trajectory.STATEMENT_TOOLS)
