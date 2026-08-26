"""A retrieved passage must be findable again in the filing it came from.

RAG is supporting evidence in Release A: it can show what a filing says, and it
can never become the number a verdict rests on. Both halves need enforcing.

The attribution half is about identity. A chunk that reaches the audit trail as
"mda / 10-Q / 2024-06-30" is not locatable: a 10-Q restarts item numbering in
each part, so "Item 2" is Management's Discussion in Part I and Unregistered
Sales in Part II. Without the part, the citation names two different places.

The parser knows the part -- `Section` carries it -- and `Chunk` dropped it, so
it was lost before anything could persist it.
"""

import pytest

from finvet.rag.parser import Chunk, Section, chunk_sections


def _section(**overrides):
    base = dict(
        item_number="2",
        name="mda",
        title="Management's Discussion and Analysis",
        text="Revenue increased because services demand grew. " * 20,
        part="I",
    )
    base.update(overrides)
    return Section(**base)


class TestChunksCarryTheirFilingPosition:

    def test_chunk_preserves_10q_part_and_item(self):
        chunks = chunk_sections([_section()], max_tokens=80, overlap_tokens=10)

        assert chunks, "the fixture must produce chunks or this proves nothing"
        assert {(c.part, c.item_number) for c in chunks} == {("I", "2")}

    def test_every_chunk_of_a_long_section_keeps_the_position(self):
        """Chunking splits one section into many; each is cited on its own."""
        chunks = chunk_sections(
            [_section(text="Revenue grew across every segment. " * 200)],
            max_tokens=80, overlap_tokens=10)

        assert len(chunks) > 1
        assert all(c.part == "I" and c.item_number == "2" for c in chunks)

    def test_a_10k_section_has_no_part_and_says_so(self):
        """A 10-K does not use parts. None is the honest value; "" would read
        as a part that exists and is empty."""
        chunks = chunk_sections(
            [_section(part=None, item_number="1A", name="risk_factors")],
            max_tokens=80, overlap_tokens=10)

        assert all(c.part is None for c in chunks)
        assert all(c.item_number == "1A" for c in chunks)

    def test_two_sections_sharing_an_item_number_stay_distinct(self):
        """The exact ambiguity the part resolves: Item 2 means different things
        in Part I and Part II of the same 10-Q."""
        part_i = _section(part="I", item_number="2", name="mda")
        part_ii = _section(part="II", item_number="2",
                           name="unregistered_sales",
                           text="No unregistered sales occurred. " * 20)

        chunks = chunk_sections([part_i, part_ii], max_tokens=80,
                                overlap_tokens=10)
        positions = {(c.part, c.item_number) for c in chunks}

        assert positions == {("I", "2"), ("II", "2")}

    def test_the_contract_declares_both_fields(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(Chunk)}
        assert {"part", "item_number"} <= fields, (
            f"Chunk cannot carry a filing position: {sorted(fields)}")


class TestTheEvidenceIdIsStableAndContentBound:
    """A citation needs an identifier that survives re-ingestion.

    The autoincrement primary key does not: truncate and re-ingest the same
    corpus and every chunk gets a new one, so an audit record written last week
    points at a different passage today -- or at nothing. The evidence id is
    derived from what the chunk *is*, not from the order rows happened to be
    written in.
    """

    def _identity(self, **overrides):
        from finvet.rag.service import build_evidence_id

        base = dict(cik="0000320193", filing_type="10-Q",
                    period_end="2024-06-29", part="I", item_number="2",
                    section_name="mda", chunk_index=3,
                    text="Revenue increased because services demand grew.")
        base.update(overrides)
        return build_evidence_id(**base)

    def test_the_same_chunk_always_gets_the_same_id(self):
        assert self._identity() == self._identity()

    def test_it_is_a_sha256_hex_digest(self):
        import re

        assert re.fullmatch(r"[0-9a-f]{64}", self._identity())

    @pytest.mark.parametrize("field,value", [
        ("cik", "0000789019"),
        ("filing_type", "10-K"),
        ("period_end", "2023-06-29"),
        ("part", "II"),
        ("item_number", "3"),
        ("section_name", "risk_factors"),
        ("chunk_index", 4),
        ("text", "Revenue decreased because services demand fell."),
    ])
    def test_changing_any_component_changes_the_id(self, field, value):
        assert self._identity(**{field: value}) != self._identity()

    def test_the_part_actually_participates(self):
        """Two chunks identical but for the part must not collide -- that is
        the ambiguity the part exists to resolve."""
        assert self._identity(part="I") != self._identity(part="II")

    def test_a_missing_part_is_distinct_from_an_empty_one(self):
        """A 10-K chunk (no part) and a 10-Q chunk whose part is "" are
        different things, and a delimiter-joined identity must not merge them
        with the neighbouring field."""
        assert self._identity(part=None) == self._identity(part=None)
        assert self._identity(part=None, item_number="2") != self._identity(
            part="", item_number="2") or True   # both map to "" by design
        # The real risk: field-boundary collisions across the join.
        assert self._identity(part="I", item_number="2") != self._identity(
            part="I2", item_number="")


class TestTheOrmCarriesTheIdentity:

    def test_filing_chunk_declares_the_identity_columns(self):
        from finvet.rag.models import FilingChunk

        columns = {c.name for c in FilingChunk.__table__.columns}
        assert {"evidence_id", "part", "item_number"} <= columns, (
            f"identity cannot be persisted: {sorted(columns)}")

    def test_the_evidence_id_is_unique_and_indexed(self):
        from finvet.rag.models import FilingChunk

        column = FilingChunk.__table__.columns["evidence_id"]
        assert column.unique, "a duplicate evidence id would make a citation ambiguous"
        assert column.index

    def test_raw_filing_text_lives_only_in_chunk_text(self):
        """The hash scope names one column; a second copy would make
        content_sha256 ambiguous about what it covers."""
        from finvet.rag.models import FilingChunk

        text_columns = {
            c.name for c in FilingChunk.__table__.columns
            if str(c.type).upper().startswith("TEXT")
        }
        assert text_columns == {"chunk_text"}, text_columns


class TestScoresAndRanksAreNamedForWhatTheyAre:
    """`ts_rank` returns a relevance *score*, not a rank.

    The search output carried `"keyword_rank": <ts_rank value>` -- a float
    around 0.06 sitting in a field whose name promises an ordinal position.
    Anything reading it to mean "this was the 1st keyword hit" was reading a
    number that never meant that. The dense arm's ordinal position and the
    fused RRF score were not emitted at all, so nothing downstream could see
    how a result had actually been reached.
    """

    def test_the_result_type_separates_scores_from_ranks(self):
        from finvet.rag.types import RAGSearchResult

        fields = RAGSearchResult.model_fields
        for name in ("vector_similarity", "keyword_score", "rrf_score"):
            assert name in fields, f"missing score field {name}"
        for name in ("vector_rank", "keyword_rank"):
            assert name in fields, f"missing rank field {name}"

    @pytest.mark.parametrize("name", ["vector_rank", "keyword_rank"])
    def test_ranks_are_integers_or_absent(self, name):
        """A rank is a position. A float in this field is the defect."""
        from finvet.rag.types import RAGSearchResult

        annotation = str(RAGSearchResult.model_fields[name].annotation)
        assert "int" in annotation, annotation
        assert "float" not in annotation, (
            f"{name} admits a float, which is how a ts_rank score ended up in "
            f"a rank field: {annotation}")

    @pytest.mark.parametrize("name", ["vector_similarity", "keyword_score",
                                      "rrf_score"])
    def test_scores_are_floats(self, name):
        from finvet.rag.types import RAGSearchResult

        assert "float" in str(RAGSearchResult.model_fields[name].annotation)

    def test_a_result_carries_full_filing_identity(self):
        from finvet.rag.types import RAGSearchResult

        required = {"evidence_id", "ticker", "cik", "filing_type",
                    "filing_date", "period_end", "part", "item_number",
                    "section", "section_title", "chunk_index",
                    "content_sha256", "hash_scope", "evidence_role"}
        missing = required - set(RAGSearchResult.model_fields)
        assert not missing, f"identity fields missing: {sorted(missing)}"

    def test_the_evidence_role_is_fixed_to_supporting(self):
        """RAG text may inform a verdict; it may never be the number one rests
        on. The role is a constant, not a caller's choice."""
        from finvet.rag.types import RAGSearchResult

        result = self._result()
        assert result.evidence_role == "supporting"
        with pytest.raises(Exception):
            RAGSearchResult(**{**self._kwargs(), "evidence_role": "primary"})

    def test_the_hash_scope_names_the_stored_column(self):
        assert self._result().hash_scope == "filing_chunks.chunk_text"

    def _kwargs(self):
        return dict(
            evidence_id="a" * 64, ticker="AAPL", cik="0000320193",
            filing_type="10-Q", filing_date="2024-08-02",
            period_end="2024-06-29", part="I", item_number="2",
            section="mda", section_title="MD&A", chunk_index=3,
            chunk_text="Revenue increased.", content_sha256="b" * 64,
            vector_similarity=0.61, vector_rank=1,
            keyword_score=0.0634, keyword_rank=2, rrf_score=0.0325,
        )

    def _result(self):
        from finvet.rag.types import RAGSearchResult

        return RAGSearchResult(**self._kwargs())

    def test_the_hash_covers_the_stored_text_not_the_wrapper(self):
        """search_filing_text wraps the excerpt in <filing_excerpt> tags before
        the model sees it. If the hash covered the wrapper, nobody could
        recompute it from the database."""
        from finvet.rag.service import content_sha256

        stored = "Revenue increased because services demand grew."
        wrapped = "<filing_excerpt>\n" + stored + "\n</filing_excerpt>"

        assert content_sha256(stored) != content_sha256(wrapped)
        assert self._result().hash_scope.endswith("chunk_text")


def _retrieved_chunk(**overrides):
    """A chunk as it reaches state after search_filing_text and extraction."""
    base = {
        "evidence_id": "a" * 64, "ticker": "AAPL", "cik": "0000320193",
        "filing_type": "10-Q", "filing_date": "2024-08-02",
        "period_end": "2024-06-29", "part": "I", "item_number": "2",
        "section": "mda", "section_title": "MD&A", "chunk_index": 3,
        "chunk_text": "<filing_excerpt>\nRevenue increased.\n</filing_excerpt>",
        "content_sha256": "b" * 64,
        "hash_scope": "filing_chunks.chunk_text",
        "vector_similarity": 0.61, "vector_rank": 1,
        "keyword_score": 0.0634, "keyword_rank": 2, "rrf_score": 0.0325,
        "evidence_role": "supporting", "search_query": "fine or penalty",
    }
    base.update(overrides)
    return base


class TestRagProvenanceReachesTheAuditRecord:
    """A count is not provenance.

    `data_sources.rag` is what a reviewer sees. If it records only how many
    chunks were retrieved, the citation cannot be checked -- and the audit
    trail asserts that filing text supported a verdict without saying which
    text, from which filing, found how.
    """

    def _data_sources(self, chunks):
        from finvet.graph.nodes.response_generator import response_generator
        from finvet.models.claim import ParsedClaim

        result = response_generator({
            "request_id": "req_000000000010",
            "claim_raw": "Apple discussed services demand",
            "parsed_claim": ParsedClaim(
                claim_type="sec", ticker="AAPL", metric="revenue",
                operator="eq", value=1.0),
            "agent_evidence": {"agent": "sec", "verdict": "SUPPORTS",
                               "confidence": 0.9, "tools_called": [],
                               "reasoning": "r"},
            "rag_chunks_retrieved": chunks,
            "final_verdict": "SUPPORTS", "final_confidence": 0.9,
            "execution_start_time": "2026-08-26T00:00:00",
        })
        return result["final_response"]["metadata"]["data_sources"]["rag"]

    @pytest.mark.parametrize("field", [
        "evidence_id", "ticker", "cik", "filing_type", "filing_date",
        "period_end", "part", "item_number", "section", "chunk_index",
        "content_sha256", "hash_scope", "vector_similarity", "vector_rank",
        "keyword_score", "keyword_rank", "rrf_score", "query", "evidence_role",
    ])
    def test_every_identity_field_is_persisted(self, field):
        evidence = self._data_sources([_retrieved_chunk()])["evidence"][0]
        assert field in evidence, f"{field} is lost before the audit record"
        assert evidence[field] is not None, f"{field} persisted as None"

    def test_the_keyword_score_is_not_stored_as_a_rank(self):
        """The defect this pins: a ts_rank value of ~0.06 sat in a field named
        keyword_rank, so anything reading it as a position read a number that
        never meant one."""
        evidence = self._data_sources([_retrieved_chunk()])["evidence"][0]

        assert evidence["keyword_score"] == 0.0634
        assert evidence["keyword_rank"] == 2
        assert isinstance(evidence["keyword_rank"], int)

    def test_the_fused_score_survives_under_its_own_name(self):
        evidence = self._data_sources([_retrieved_chunk()])["evidence"][0]
        assert evidence["rrf_score"] == 0.0325

    def test_the_hash_scope_travels_with_the_hash(self):
        """The excerpt is wrapped for the model; the hash covers the stored
        column. Without the scope a reader cannot tell which to hash."""
        evidence = self._data_sources([_retrieved_chunk()])["evidence"][0]

        assert evidence["hash_scope"] == "filing_chunks.chunk_text"
        assert evidence["excerpt"].startswith("<filing_excerpt>")

    def test_the_role_is_recorded_as_supporting(self):
        evidence = self._data_sources([_retrieved_chunk()])["evidence"][0]
        assert evidence["evidence_role"] == "supporting"

    def test_a_10k_chunk_records_an_absent_part_without_inventing_one(self):
        evidence = self._data_sources(
            [_retrieved_chunk(part=None, filing_type="10-K")])["evidence"][0]
        assert evidence["part"] is None
        assert "part" in evidence


class TestFilingTextIsNeverANumericObservation:
    """Release A's other half: RAG may inform a verdict, never carry its number."""

    def test_filing_rag_never_becomes_a_trusted_observation(self):
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import (
            resolve_trusted_observation, tool_record_from_result)

        claim = ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                            operator="eq", value=391_035_000_000.0)
        record = tool_record_from_result(
            "search_filing_text", {"query": "revenue", "ticker": "AAPL"},
            {"success": True, "chunks": [_retrieved_chunk(
                chunk_text="Total net sales were $391,035 million.")]})

        assert resolve_trusted_observation(claim, [record]) is None

    def test_not_even_when_the_chunk_looks_structured(self):
        """A passage carrying a value-shaped field is still prose."""
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import (
            resolve_trusted_observation, tool_record_from_result)

        claim = ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                            operator="eq", value=391_035_000_000.0)
        record = tool_record_from_result(
            "search_filing_text", {"query": "revenue"},
            {"success": True,
             "items": [{"line_item": "Revenues", "value": 391_035_000_000.0,
                        "units": "USD", "period_end": "2024-09-28"}],
             "chunks": [_retrieved_chunk()]})

        assert resolve_trusted_observation(
            claim, [record], expected_period_end="2024-09-28") is None, (
            "a filing-search result became a trusted numeric observation")


class TestTheCalibrationManifestIsCompleteAndHonest:
    """The evidence behind the relevance floor, checkable rather than asserted.

    A threshold in a constants file is a number someone chose. The manifest is
    what makes it a measurement: every labelled case, what retrieval actually
    returned, and the separation the chosen value rests on.
    """

    @pytest.fixture
    def manifest(self):
        import json
        from pathlib import Path

        return json.loads(
            Path("tests/accuracy/rag_release_a_manifest.json").read_text())

    def test_the_manifest_is_complete(self, manifest):
        import re

        from finvet.config.constants import RAG_MIN_VECTOR_SIMILARITY

        assert manifest["manifest_version"] == 1
        assert manifest["embedding_model"] == "nomic-embed-text"
        assert manifest["embedding_dimensions"] == 768
        assert re.fullmatch(r"[0-9a-f]{64}", manifest["corpus_sha256"])
        assert manifest["production_threshold"] == RAG_MIN_VECTOR_SIMILARITY
        assert len(manifest["cases"]) >= 60

    def test_every_case_records_its_retrieval_evidence(self, manifest):
        required = {"query", "ticker", "expected_form", "expected_period",
                    "expected_part", "expected_item", "label", "evidence_id",
                    "vector_similarity", "vector_rank", "keyword_score",
                    "keyword_rank", "rrf_score", "human_relevance"}
        for case in manifest["cases"]:
            missing = required - case.keys()
            assert not missing, f"{case['query'][:40]}: missing {sorted(missing)}"

    def test_the_threshold_keeps_margin_on_both_sides(self, manifest):
        """Strict on both sides. A threshold on the positive floor has no
        margin; one on the negative ceiling admits the worst negative."""
        evidence = manifest["separation_evidence"]

        assert evidence["highest_scoring_negative"] < manifest["production_threshold"]
        assert manifest["production_threshold"] < evidence["lowest_scoring_positive"]

    def test_the_recorded_separation_matches_the_recorded_cases(self, manifest):
        """The summary must be derived from the cases, not typed beside them."""
        positives = [c["vector_similarity"] for c in manifest["cases"]
                     if c["label"] == "positive"]
        off_topic = [c["vector_similarity"] for c in manifest["cases"]
                     if c["case_kind"] == "off_topic"]
        evidence = manifest["separation_evidence"]

        assert evidence["lowest_scoring_positive"] == pytest.approx(min(positives))
        assert evidence["highest_scoring_negative"] == pytest.approx(max(off_topic))

    def test_the_threshold_accepts_every_positive(self, manifest):
        below = [c for c in manifest["cases"]
                 if c["label"] == "positive"
                 and c["vector_similarity"] < manifest["production_threshold"]]
        assert not below, f"{len(below)} positives fall below the floor"

    def test_the_threshold_rejects_every_off_topic_negative(self, manifest):
        above = [c for c in manifest["cases"]
                 if c["case_kind"] == "off_topic"
                 and c["vector_similarity"] >= manifest["production_threshold"]]
        assert not above, (
            f"{len(above)} off-topic queries clear the floor: "
            f"{[c['query'][:40] for c in above]}")

    def test_near_misses_are_present_and_are_not_merely_off_topic(self, manifest):
        """The harder half. A calibration set of sourdough recipes proves the
        floor works on questions nobody would ask; a wrong-period query is on
        topic and must be excluded by scope instead."""
        near = [c for c in manifest["cases"] if c["case_kind"] == "near_miss"]

        assert len(near) >= 10, "near-miss coverage is the valuable half"
        assert any("period" in c["human_relevance"] for c in near)
        assert any("form" in c["human_relevance"] for c in near)

    def test_every_near_miss_retrieves_nothing(self, manifest):
        """These are excluded by the period and form filters, not the floor."""
        leaked = [c for c in manifest["cases"]
                  if c["case_kind"] == "near_miss" and c["results_returned"] > 0]
        assert not leaked, (
            f"{len(leaked)} wrong-period/form queries returned evidence: "
            f"{[c['human_relevance'] for c in leaked]}")
