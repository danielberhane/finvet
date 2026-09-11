"""Public documentation may not describe a contract the code does not have.

Every value named here is one a reader would act on: an API status they might
branch on, a delegation outcome they might expect, an integrity verdict they
might trust. When the code's vocabulary changes and a document keeps the old
word, the document becomes a confident description of a system that no longer
exists — and nothing fails.

**On phrasing checks.** An earlier draft of this guard grepped for
"tamper-proof" and flagged README as violating the honesty rule. It was
matching *"it is **not** tamper-proof"* — the disclaimer the rule exists to
require. A check that cannot tell a claim from its denial fails on correct
documentation and trains people to disable it, so the assertions below match
advertisements specifically, and each carries the negation it must tolerate.
"""

import re
from pathlib import Path

import pytest

# SECURITY.md joined this list when it was written. The deployment warnings it
# carries -- no authentication, the development database password, the bind
# override -- used to live in the README, and moving them out silently took
# them outside this guard's view. A reader finds them either way; the guard
# should too.
PUBLIC_DOCS = [Path("README.md"), Path("docs/ARCHITECTURE.md"),
               Path("SECURITY.md")]


def _text():
    return "\n".join(p.read_text() for p in PUBLIC_DOCS if p.exists())


class TestDocumentedVocabulariesExistInCode:

    def test_every_documented_terminal_status_is_real(self):
        import typing

        from finvet.api.execution import TerminalStatus

        declared = set(typing.get_args(TerminalStatus))
        # Only consider words presented as status values, i.e. in backticks.
        quoted = set(re.findall(r'`(success|pending_review|rejected|'
                                r'guardrail_blocked|error|reviewed|'
                                r'[a-z_]*_review[a-z_]*)`', _text()))
        # These are legitimate non-TerminalStatus terms the docs also use.
        allowed = declared | {"reviewed", "pending_review"}
        assert quoted <= allowed, f"undocumented statuses: {quoted - allowed}"

    def test_every_documented_a2a_status_is_real(self):
        import typing

        from finvet.models.a2a import A2AStatus

        declared = set(typing.get_args(A2AStatus))
        mentioned = set(re.findall(r'\b([A-Z][A-Z_]{4,})\b', _text()))

        # The shape test is derived from the statuses themselves rather than
        # kept as a hand-written list. The list version froze: FOUND_UNCERTIFIED
        # was added to the code and ended in a suffix nobody had thought to
        # add, so this test could not see the name at all. Deriving it means
        # the next status extends the detector on its own.
        #
        # UNDISCLOSED_MATERIAL_CLAIM is carried explicitly: it is a status
        # Release A *removed*, and naming it in the docs must still fail.
        #
        # Residual limit, stated rather than implied: an invented status whose
        # suffix matches no real one (say FOUND_UNVERIFIED) is still invisible
        # here. Catching that needs a different check than a name heuristic.
        suffixes = tuple(f"_{s.rsplit('_', 1)[-1]}" for s in declared if "_" in s)
        single_words = {s for s in declared if "_" not in s}
        a2a_shaped = {m for m in mentioned
                      if m.endswith(suffixes)
                      or m in single_words
                      or m.endswith("_CLAIM")}

        assert a2a_shaped <= declared, (
            f"docs name A2A statuses the code does not have: "
            f"{sorted(a2a_shaped - declared)}")

    def test_every_documented_integrity_status_is_real(self):
        from finvet.api.routes.audit import NON_TERMINAL_REVIEW_VERDICTS

        text = _text()
        for status in ("verified", "failed", "unavailable"):
            assert f"`{status}`" in text, (
                f"the integrity status {status!r} is not documented")
        for reason in NON_TERMINAL_REVIEW_VERDICTS.values():
            assert reason in text, f"reason {reason!r} is undocumented"


class TestNoRemovedCapabilityIsAdvertised:

    @pytest.mark.parametrize("term", [
        "UNDISCLOSED_MATERIAL_CLAIM",
        "unsupported_material_claim",
    ])
    def test_the_removed_materiality_status_is_not_named(self, term):
        assert term not in _text()

    def test_q4_derivation_is_not_advertised(self):
        """Describing it as unsupported is required; describing how to do it
        is the violation."""
        for match in re.findall(r"[^.]*Q4[^.]*\.", _text()):
            if re.search(r"annual\s*[-−]\s*|minus a nine-month|derives Q4",
                         match, re.I):
                assert re.search(r"not supported|unsupported|declin|cannot",
                                 match, re.I), (
                    f"Q4 derivation described without saying it is "
                    f"unsupported: {match.strip()[:120]}")


class TestHonestyClaimsAreDenialsNotAdvertisements:
    """The check that must tolerate its own subject matter."""

    @pytest.mark.parametrize("term", ["tamper-proof", "immutable ledger",
                                      "production-ready"])
    def test_any_mention_is_a_denial(self, term):
        for line in _text().splitlines():
            if term not in line.lower():
                continue
            assert re.search(r"\bnot\b|\bnever\b|\bcannot\b|\bdo(es)? not\b",
                             line, re.I), (
                f"{term!r} appears as a claim rather than a denial: "
                f"{line.strip()[:120]}")

    def test_the_checksum_scope_is_stated_honestly(self):
        text = _text().lower()
        assert "tamper-proof" in text, (
            "the integrity section should say what the checksum is *not*")


class TestDocumentedDefaultsMatchTheCode:

    def test_the_documented_memory_default_is_the_real_one(self):
        from finvet.config.settings import Settings

        actual = Settings(_env_file=None,
                          tavily_api_key="x",
                          postgres_password="x").enable_claim_memory
        assert actual is False
        assert re.search(r"ENABLE_CLAIM_MEMORY.*\|.*false", _text(), re.I), (
            "the architecture env table does not show the real default")

    def test_the_documented_relevance_floor_is_the_real_one(self):
        from finvet.config.constants import RAG_MIN_VECTOR_SIMILARITY

        assert str(RAG_MIN_VECTOR_SIMILARITY) in _text(), (
            f"docs do not name the shipped threshold "
            f"{RAG_MIN_VECTOR_SIMILARITY}")

    def test_the_documented_python_range_matches_pyproject(self):
        import tomllib

        spec = tomllib.load(open("pyproject.toml", "rb"))["project"][
            "requires-python"]
        assert spec == ">=3.11,<3.14"
        assert "3.11" in _text() and "3.13" in _text()


class TestTheStackIsNotPublishedToTheNetwork:
    """`"5432:5432"` publishes on 0.0.0.0.

    That put a Postgres carrying a development default password, an API with no
    authentication, and an unauthenticated Ollama on whatever network the host
    was joined to. The README tells a reader to run `docker compose up`, so the
    default is what almost everyone gets.

    Loopback is the default and `FINVET_BIND_ADDR` is the deliberate way out.
    Asserted here because a future edit adding a service is exactly where this
    regresses, and nothing else would notice.
    """

    def _published_ports(self):
        import yaml

        compose = yaml.safe_load(open("docker-compose.yml"))
        for name, service in (compose.get("services") or {}).items():
            for entry in (service.get("ports") or []):
                yield name, entry

    def test_every_published_port_binds_to_loopback(self):
        offenders = [
            f"{name}: {entry}"
            for name, entry in self._published_ports()
            if not str(entry).startswith("${FINVET_BIND_ADDR:-127.0.0.1}:")
        ]

        assert not offenders, (
            "these publish on every interface; prefix them with "
            f"${{FINVET_BIND_ADDR:-127.0.0.1}}: — {offenders}")

    def test_the_stack_actually_publishes_something(self):
        """The control. A guard over an empty list passes for the wrong
        reason."""
        assert len(list(self._published_ports())) >= 4

    def test_the_escape_hatch_is_documented(self):
        assert "FINVET_BIND_ADDR" in _text(), (
            "the override is not documented, so the only way a reader finds "
            "it is by reading the compose file")


class TestTheTrustBoundaryIsDescribedAsItIs:
    """The README said filing text "can never become the number a verdict rests
    on". That was true until penalties gained a deterministic path: for
    fine_amount and settlement_amount, `_penalty_observation` returns a
    TrustedObservation whose tool is `search_filing_text`.

    The distinction that survived is *who reads the filing* -- Python, not the
    model -- so the docs must state that rather than an absolute that one run
    falsifies."""

    def test_the_claim_is_qualified_wherever_it_appears(self):
        """Written the way this file's other phrasing checks are, and for the
        reason its docstring gives: a check that cannot tell a claim from its
        qualified form fails on correct documentation.

        "a *model's reading* of filing text can never be the number" is true.
        The same sentence without that qualifier is not."""
        # Whitespace-normalised, not line-by-line: the docs are hard-wrapped,
        # so the qualifier and the phrase it qualifies routinely land on
        # different lines. A line-based check reported the corrected sentence
        # as a violation.
        flat = " ".join(_text().split())
        for match in re.finditer(r"never become the number", flat, re.I):
            window = flat[max(0, match.start() - 160): match.end() + 40]
            assert re.search(r"model", window, re.I), (
                f"an unqualified claim that prose can never carry a number: "
                f"...{window[-140:]}")

    def test_the_deterministic_exception_is_disclosed(self):
        """Understating is its own inaccuracy: the capability is what makes a
        $1T fine claim answerable, and a reader should know it exists."""
        text = _text()

        assert "fine_amount" in text and "Legal Proceedings" in text, (
            "the docs do not mention that Python extracts penalty amounts "
            "from filing text")

    def test_the_extraction_is_attributed_to_python_not_the_model(self):
        text = _text().lower()

        assert "python" in text, (
            "the docs must say who reads the filing, since that is the whole "
            "distinction")
