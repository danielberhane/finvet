# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ragas==0.4.3",
#     "langchain-community>=0.3,<0.4",
#     "openai>=1.40",
#     "httpx>=0.27",
# ]
# ///
"""RAGAS evaluation of FinVet's answers on claims about filing text.

What it measures
    On 50 claims about what a filing says -- the 12 `sec/qualitative` claims of
    the golden set (ids 37-48) and 38 more (ids 2001-2038: 16 supported, 12
    contradicted, 10 not in the filing), kept privately and pinned by hash --
    FinVet's model reads retrieved filing passages and writes the reasoning
    that is the answer. RAGAS scores that reasoning on three questions:

      faithfulness       share of its statements the passages support
      answer relevancy   whether it addresses the claim
      context precision  whether the useful passages were ranked first

Who does what
    generator  FinVet's configured model (DeepSeek), reached through /verify --
               the thing being graded is FinVet exactly as it runs
    judge      Qwen3.8 on the GMU ORC gateway: a different model family, so
               the grader has no reason to favour the generator's writing
    embedder   nomic-embed-text on local Ollama, for answer relevancy only

What is fed to RAGAS, and why
    response   the model's reasoning alone. FinVet's `explanation` appends a
               block written by code ("- Retrieved value", "- Difference from
               claimed value", "- Data source"); grading it would score
               FinVet's formatting as the model's words.
    contexts   the retrieved passages with FinVet's <filing_excerpt> wrapper
               removed. Other tool outputs the model saw are left out, which
               can only lower faithfulness, never raise it.
    excluded   a response whose reasoning was written by code (the
               deterministic path) or that retrieved no passages.

Controls
    Every judge request turns Qwen's reasoning mode off (it can leave the JSON
    empty) and opts out of the gateway's response cache (it would replay the
    first pass into the second). Each answer is judged twice; a faithfulness
    difference above 0.10 is flagged.

Outputs
    public   docs/eval/runs/rag-ragas-<date>.json -- scores, verdicts, passage
             hashes. No claim text, reasoning text or judge reasons: all three
             restate held-out claims.
    private  --record (must be under notes/, which git ignores) -- claim,
             reasoning, passages and judge reasons, for the human check.

Two phases, because the generator and the judge are reached over different
networks (the GMU VPN blocks DeepSeek's replies; the judge is only reachable on
it). Answers are frozen before grading, so the grader cannot affect them and
they can be re-graded later without asking FinVet again.

    # 1. answer -- VPN off; FinVet API up on DeepSeek
    FINVET_GOLDEN_DIR=... uv run scripts/eval_rag_ragas.py answer \\
        --extra-claims notes/rag_ragas_claims_v1.jsonl \\
        --record notes/rag-ragas-record.json

    # 2. grade -- VPN on; Ollama up
    LITELLM_API_KEY=... uv run scripts/eval_rag_ragas.py grade \\
        --record notes/rag-ragas-record.json

The pure helpers at the top import nothing outside the standard library, so
tests/unit/test_rag_ragas_inputs.py can test them from the project environment.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

CLAIM_IDS = list(range(37, 49))
# The 38 further claims, frozen 2026-10-01 before any answer was collected.
EXTRA_CLAIMS_SHA256 = "46402ac00989873b620dc0bf7b31dd02fae1c2af9514e665f5e526e0b2996a2c"
JUDGE_MODEL = "Qwen3.8"
JUDGE_BASE_URL = "https://llm.orc.gmu.edu/v1"
EMBED_MODEL = "nomic-embed-text"
EMBED_BASE_URL = "http://localhost:11434/v1"
# Sent with every judge request; both verified against the gateway on 2026-09-30.
JUDGE_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "cache": {"no-cache": True, "no-store": True},
}
STABILITY_FLAG = 0.10

# The lines FinVet's response generator appends after the model's reasoning
# (src/finvet/graph/nodes/response_generator.py, _build_explanation).
CODE_SUMMARY_PREFIXES = ("- Retrieved value:", "- Difference from claimed value:", "- Data source:")
# The opening of reasoning written by code, not a model (agents/base.py,
# deterministic_reasoning).
CODE_WRITTEN_REASONING = "Verdict determined by direct comparison, without a model."
EXCERPT_OPEN, EXCERPT_CLOSE = "<filing_excerpt>", "</filing_excerpt>"


# --- pure helpers (standard library only) ------------------------------------

def model_reasoning(explanation: str) -> str:
    """The model's reasoning, with the block FinVet's code appends removed.

    `_build_explanation` joins the reasoning and the code-built summary with a
    blank line; the summary is the last part and every line starts with one of
    CODE_SUMMARY_PREFIXES. Only that exact trailing block is removed.
    """
    parts = (explanation or "").split("\n\n")
    last = parts[-1].strip().splitlines()
    if last and all(line.startswith(CODE_SUMMARY_PREFIXES) for line in last):
        parts = parts[:-1]
    text = "\n\n".join(parts).strip()
    return "" if text == "No detailed explanation available." else text


def passage_texts(response: dict) -> list[str]:
    """The retrieved filing passages, without FinVet's delimiter tags."""
    evidence = (((response.get("metadata") or {}).get("data_sources") or {})
                .get("rag") or {}).get("evidence") or []
    out = []
    for item in evidence:
        text = (item.get("excerpt") or "").strip()
        if text.startswith(EXCERPT_OPEN) and text.endswith(EXCERPT_CLOSE):
            text = text[len(EXCERPT_OPEN):-len(EXCERPT_CLOSE)].strip()
        if text:
            out.append(text)
    return out


def passage_identities(response: dict) -> list[dict]:
    evidence = (((response.get("metadata") or {}).get("data_sources") or {})
                .get("rag") or {}).get("evidence") or []
    return [{k: item.get(k) for k in ("content_sha256", "section", "filing_type", "period_end")}
            for item in evidence]


def is_code_written(reasoning: str) -> bool:
    return reasoning.startswith(CODE_WRITTEN_REASONING)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- collect: FinVet answers the claims ---------------------------------------

def load_claims(golden_dir: Path, extra_path: Path) -> list[dict]:
    """The 12 golden qualitative claims plus the 38 frozen further claims."""
    rows = [json.loads(line) for line in (golden_dir / "golden_c.jsonl").read_text().splitlines()
            if line.strip()]
    golden = {r["id"]: r for r in rows if r.get("id") in CLAIM_IDS}
    missing = sorted(set(CLAIM_IDS) - set(golden))
    if missing:
        raise SystemExit(f"golden claims {missing} not found")
    raw = extra_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXTRA_CLAIMS_SHA256:
        raise SystemExit(f"{extra_path} does not match the frozen hash; refusing to run on altered claims")
    extra = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    claims = [{**golden[i], "kind": "supported", "origin": "golden_c"} for i in CLAIM_IDS]
    claims += [{**r, "origin": "rag_ragas_claims_v1"} for r in extra]
    if len(claims) != 50 or len({c["id"] for c in claims}) != 50:
        raise SystemExit("expected 50 distinct claims")
    return claims


def private_path(path: str) -> Path:
    p = Path(path)
    if "notes" not in p.parts:
        raise SystemExit("--record must be under notes/ (gitignored): it holds held-out claim text")
    return p


# --- phase 1: answer -- FinVet answers every claim ----------------------------

def answer(claims: list[dict], api: str) -> tuple[dict, list[dict]]:
    import httpx

    health = httpx.get(f"{api}/health", timeout=10).json()
    generator = {"version": health.get("version"),
                 "models": {role: cfg.get("model") for role, cfg in (health.get("llm") or {}).items()}}
    records = []
    for claim in claims:
        resp = httpx.post(f"{api}/verify", json={"claim": claim["claim"]}, timeout=300)
        body = resp.json()
        reasoning = model_reasoning(body.get("explanation") or "")
        passages = passage_texts(body)
        rec = {"id": claim["id"], "kind": claim["kind"], "origin": claim["origin"],
               "claim": claim["claim"],
               "expected_verdict": (claim.get("expected") or {}).get("verdict"),
               "http": resp.status_code, "verdict": body.get("verdict"),
               "confidence": body.get("confidence"),
               "reasoning": reasoning, "passages": passages,
               "passage_identities": passage_identities(body)}
        if not passages:
            rec["excluded"] = "no passages retrieved"
        elif not reasoning:
            rec["excluded"] = "no model reasoning in the response"
        elif is_code_written(reasoning):
            rec["excluded"] = "reasoning written by code (deterministic path)"
        records.append(rec)
        print(f"  answered {rec['id']} ({rec['kind']}): http {resp.status_code} verdict {rec['verdict']} "
              f"passages {len(passages)}" + (f"  EXCLUDED: {rec['excluded']}" if "excluded" in rec else ""))
    return generator, records


# --- phase 2: grade -- RAGAS with Qwen as judge ---------------------------------

def build_scorers():
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        Faithfulness,
    )

    key = os.environ.get("LITELLM_API_KEY", "")
    if not key:
        raise SystemExit("LITELLM_API_KEY is unset; the judge needs it")
    judge = llm_factory(JUDGE_MODEL, client=AsyncOpenAI(api_key=key, base_url=JUDGE_BASE_URL),
                        temperature=0, max_tokens=4096, extra_body=JUDGE_EXTRA_BODY)
    embedder = embedding_factory("openai", model=EMBED_MODEL,
                                 client=AsyncOpenAI(api_key="ollama", base_url=EMBED_BASE_URL))
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embedder),
        "context_precision": ContextPrecisionWithoutReference(llm=judge),
    }


async def judge_once(scorers: dict, rec: dict) -> dict:
    claim, reasoning, passages = rec["claim"], rec["reasoning"], rec["passages"]
    faith = await scorers["faithfulness"].ascore(user_input=claim, response=reasoning,
                                                 retrieved_contexts=passages)
    relev = await scorers["answer_relevancy"].ascore(user_input=claim, response=reasoning)
    prec = await scorers["context_precision"].ascore(user_input=claim, response=reasoning,
                                                     retrieved_contexts=passages)
    return {"faithfulness": float(faith.value), "answer_relevancy": float(relev.value),
            "context_precision": float(prec.value),
            "reasons": {"faithfulness": str(getattr(faith, "reason", "") or "")}}


async def grade(records: list[dict]) -> None:
    scorers = build_scorers()
    for rec in records:
        if "excluded" in rec or "judge_passes" in rec:
            continue
        rec["judge_passes"] = [await judge_once(scorers, rec) for _ in range(2)]
        a, b = rec["judge_passes"]
        rec["faithfulness_spread"] = abs(a["faithfulness"] - b["faithfulness"])
        rec["unstable"] = rec["faithfulness_spread"] > STABILITY_FLAG
        print(f"  graded {rec['id']} ({rec['kind']}): faithfulness {a['faithfulness']:.2f}/{b['faithfulness']:.2f} "
              f"relevancy {a['answer_relevancy']:.2f}/{b['answer_relevancy']:.2f} "
              f"precision {a['context_precision']:.2f}/{b['context_precision']:.2f}"
              + ("  UNSTABLE" if rec["unstable"] else ""))


# --- summary and outputs ------------------------------------------------------------

METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarise(records: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        scored = [r for r in rs if "judge_passes" in r]
        per = {m: [sum(p[m] for p in r["judge_passes"]) / 2 for r in scored] for m in METRICS}
        return {
            "claims": len(rs), "scored": len(scored), "excluded": sum("excluded" in r for r in rs),
            "verdict_matches_expected": sum(r["verdict"] == r["expected_verdict"] for r in rs),
            **{f"{m}_mean": mean(per[m]) for m in METRICS},
            "faithfulness_min": round(min(per["faithfulness"]), 3) if scored else None,
        }
    kinds = sorted({r["kind"] for r in records})
    return {
        "all": block(records),
        "by_kind": {k: block([r for r in records if r["kind"] == k]) for k in kinds},
        "excluded": {r["id"]: r["excluded"] for r in records if "excluded" in r},
        "unstable_claims": [r["id"] for r in records if r.get("unstable")],
        "bar": {"faithfulness_mean": 0.85, "faithfulness_min": 0.50,
                "applies_to": "all scored claims; set before any answer was collected"},
    }


def public_view(rec: dict) -> dict:
    out = {k: rec[k] for k in ("id", "kind", "origin", "expected_verdict", "http", "verdict",
                               "confidence", "passage_identities") if k in rec}
    out["reasoning_sha256"] = sha256(rec["reasoning"])
    out["reasoning_words"] = len(rec["reasoning"].split())
    if "excluded" in rec:
        out["excluded"] = rec["excluded"]
    if "judge_passes" in rec:
        out["judge_passes"] = [{m: round(p[m], 3) for m in METRICS} for p in rec["judge_passes"]]
        out["faithfulness_spread"] = round(rec["faithfulness_spread"], 3)
        out["unstable"] = rec["unstable"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation of FinVet's filing-text answers")
    sub = parser.add_subparsers(dest="phase", required=True)
    a = sub.add_parser("answer", help="phase 1: FinVet answers the 50 claims (VPN off)")
    a.add_argument("--api", default=os.environ.get("FINVET_API_URL", "http://127.0.0.1:8000"))
    a.add_argument("--extra-claims", required=True, help="the frozen 38-claim file (notes/)")
    a.add_argument("--record", required=True, help="private record to write (notes/)")
    g = sub.add_parser("grade", help="phase 2: RAGAS grades the saved answers (VPN on)")
    g.add_argument("--record", required=True, help="private record from the answer phase (notes/)")
    g.add_argument("--out", default=None, help="public artifact path")
    args = parser.parse_args()
    record_path = private_path(args.record)

    if args.phase == "answer":
        golden_dir = os.environ.get("FINVET_GOLDEN_DIR")
        if not golden_dir:
            raise SystemExit("FINVET_GOLDEN_DIR is unset")
        claims = load_claims(Path(golden_dir).expanduser(), Path(args.extra_claims))
        generator, records = answer(claims, args.api)
        record = {"kind": "rag_ragas_record", "framework": "ragas 0.4.3",
                  "answered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "generator": generator, "extra_claims_sha256": EXTRA_CLAIMS_SHA256,
                  "claims": records}
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=1))
        print(f"answered {len(records)} claims, excluded {sum('excluded' in r for r in records)}; "
              f"record: {record_path}")
        return 0

    record = json.loads(record_path.read_text())
    records = record["claims"]
    asyncio.run(grade(records))
    run_utc = datetime.now(timezone.utc)
    record.update({
        "graded_utc": run_utc.isoformat(timespec="seconds"),
        "judge": {"model": JUDGE_MODEL, "via": "GMU ORC LiteLLM gateway", "temperature": 0,
                  "extra_body": JUDGE_EXTRA_BODY, "passes": 2},
        "embedder": {"model": EMBED_MODEL, "via": "local Ollama"},
        "inputs": "user_input = claim; response = model reasoning with FinVet's appended "
                  "summary removed; retrieved_contexts = retrieved passages, delimiters removed",
        "summary": summarise(records),
    })
    record_path.write_text(json.dumps(record, indent=1))
    public = {k: v for k, v in record.items() if k != "claims"}
    public["withheld"] = ("claim text, reasoning text and judge reasons (they restate held-out "
                          "claims); see the private record")
    public["claims"] = [public_view(r) for r in records]
    out = Path(args.out) if args.out else Path("docs/eval/runs") / f"rag-ragas-{run_utc:%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(public, indent=1))
    print(json.dumps(record["summary"], indent=1))
    print(f"public artifact: {out}\nprivate record:  {record_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
