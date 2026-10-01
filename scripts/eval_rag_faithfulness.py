#!/usr/bin/env python
"""Judge-score the model's reading of filing text on the 12 qualitative claims.

FinVet decides numeric verdicts by a deterministic comparison, so the model's
prose is not the answer there. On the benchmark's 12 `sec/qualitative` claims
it is: the SEC agent retrieves passages and the model's explanation states
what they say. This runner sends those claims through the API, captures the
passages and the explanation from the response, and scores two things with
DeepEval, a second model acting as judge:

  faithfulness      every statement in the explanation is supported by the
                    retrieved passages
  answer relevancy  the explanation addresses the claim

The artifact withholds the claim text and the explanation text -- the claims
are held out, and an explanation restates its claim -- and records the scores,
the passages' hashes and a hash of the explanation. The judge's reasons
paraphrase the claim too, so they are printed for the operator, not stored.

The judge is DeepSeek through its OpenAI-compatible API, driven in JSON mode
because it does not support json_schema. These figures are judge-scored; the
retrieval figures in eval_rag_heldout.py are not.

Usage:
    set -a; source .env; set +a
    FINVET_GOLDEN_DIR=... PYTHONPATH=src .venv/bin/python scripts/eval_rag_faithfulness.py
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

CLAIM_IDS = list(range(37, 49))
DEFAULT_API = "http://127.0.0.1:8000"


def load_claims(golden_dir: Path) -> dict[int, dict]:
    for name in ("golden_u.jsonl", "golden_c.jsonl"):
        path = golden_dir / name
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            found = {r["id"]: r for r in rows if r.get("id") in CLAIM_IDS}
            if len(found) == len(CLAIM_IDS):
                return found
    raise SystemExit("FINVET_GOLDEN_DIR does not hold the 12 qualitative claims")


def judge_model():
    """DeepSeek as a DeepEval judge, in JSON mode."""
    from deepeval.models import DeepEvalBaseLLM
    from openai import OpenAI

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is unset; the judge needs it")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    class DeepSeekJudge(DeepEvalBaseLLM):
        def load_model(self):
            return client

        def get_model_name(self):
            return "deepseek-chat"

        def generate(self, prompt: str, schema=None):
            kwargs = {"model": "deepseek-chat", "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]}
            if schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["messages"][0]["content"] += (
                    "\n\nRespond with a single JSON object matching this schema:\n"
                    + json.dumps(schema.model_json_schema()))
            text = client.chat.completions.create(**kwargs).choices[0].message.content or ""
            if schema is None:
                return text
            return schema.model_validate_json(text)

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

    return DeepSeekJudge()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", default=os.environ.get("FINVET_API_URL", DEFAULT_API))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    golden_dir = os.environ.get("FINVET_GOLDEN_DIR")
    if not golden_dir:
        raise SystemExit("FINVET_GOLDEN_DIR is unset")
    claims = load_claims(Path(golden_dir).expanduser())

    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    judge = judge_model()
    records = []
    for cid in CLAIM_IDS:
        row = claims[cid]
        r = httpx.post(f"{args.api}/verify", json={"claim": row["claim"]}, timeout=300)
        body = r.json()
        explanation = body.get("explanation") or ""
        evidence = (((body.get("metadata") or {}).get("data_sources") or {}).get("rag") or {}).get("evidence") or []
        if not evidence:
            evidence = ((body.get("data_sources") or {}).get("rag") or {}).get("evidence") or []
        passages = [e.get("excerpt") or "" for e in evidence]
        rec = {
            "id": cid, "category": row.get("category"), "expected_verdict": (row.get("expected") or {}).get("verdict"),
            "http": r.status_code, "verdict": body.get("verdict"), "confidence": body.get("confidence"),
            "passages": [{"content_sha256": e.get("content_sha256"), "section": e.get("section"),
                          "filing_type": e.get("filing_type"), "period_end": e.get("period_end")} for e in evidence],
            "explanation_sha256": hashlib.sha256(explanation.encode()).hexdigest(),
            "explanation_words": len(explanation.split()),
        }
        if not passages or not explanation:
            rec["skipped"] = "no passages or no explanation in the response"
            records.append(rec)
            print(f"  id {cid}: skipped ({rec['skipped']})")
            continue
        case = LLMTestCase(input=row["claim"], actual_output=explanation, retrieval_context=passages)
        faith = FaithfulnessMetric(model=judge, threshold=0.5, include_reason=True)
        relev = AnswerRelevancyMetric(model=judge, threshold=0.5, include_reason=True)
        faith.measure(case)
        relev.measure(case)
        # The judge's reasons paraphrase the claim, so they stay out of the artifact.
        rec.update({"faithfulness": round(faith.score, 3), "answer_relevancy": round(relev.score, 3)})
        print(f"    reasons (not stored): {faith.reason[:160]}")
        records.append(rec)
        print(f"  id {cid}: verdict {body.get('verdict')}  faithfulness {faith.score:.2f}  relevancy {relev.score:.2f}  passages {len(passages)}")

    scored = [x for x in records if "faithfulness" in x]
    summary = {
        "claims": len(records), "scored": len(scored),
        "faithfulness_mean": round(sum(x["faithfulness"] for x in scored) / len(scored), 3) if scored else None,
        "faithfulness_min": min((x["faithfulness"] for x in scored), default=None),
        "answer_relevancy_mean": round(sum(x["answer_relevancy"] for x in scored) / len(scored), 3) if scored else None,
        "judge": "deepseek-chat via DeepEval FaithfulnessMetric / AnswerRelevancyMetric",
    }
    artifact = {"kind": "rag_faithfulness", "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "api": args.api, "summary": summary, "claims": records}
    out = Path(args.out) if args.out else Path("docs/eval/runs") / f"rag-faithfulness-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
