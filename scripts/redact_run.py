#!/usr/bin/env python
"""Publish a golden-run artifact with the held-out claim text withheld.

The first two redacted artifacts were made by hand, and the hand missed two
fields: the absolute dataset path (which carried a home directory) and the
per-role `base_url` (which carried the address of a self-hosted inference
gateway). Both reached git history. This script is the only sanctioned way
to produce a published artifact, and it refuses to write anything that still
contains a home path or an IPv4 literal.

What is withheld and what is kept:
  - `results[*].claim` is replaced for every row except the ones registered
    as burned for that dataset in `finvet.eval.exclusions` (rows already
    published in full in DATASET_CARD.md). Expected labels, verdicts,
    confidences, tool calls and timings are all kept, so every published
    metric recomputes from the redacted file.
  - `dataset` is reduced to its basename.
  - every `base_url` whose host is not a public provider is replaced with a
    placeholder. The model name, temperature and structured-output method
    stay, because a run that does not say what produced it cannot be
    compared with anything.

Reads the source through FINVET_GOLDEN_DIR (or --src), never writes into
that directory, and never overwrites an existing output.

Usage:
    FINVET_GOLDEN_DIR=... .venv/bin/python scripts/redact_run.py --label deepseek-c2
    .venv/bin/python scripts/redact_run.py --src path/to/run-...json --out-dir docs/eval
"""

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finvet.eval.exclusions import burned_ids  # noqa: E402

WITHHELD = "[withheld — see DATASET_CARD.md]"
GATEWAY_PLACEHOLDER = "<self-hosted LiteLLM gateway>"
PUBLIC_HOSTS = frozenset({"api.deepseek.com"})
REDACTION_NOTE = (
    "claim text withheld for all rows except the ones burned in "
    "DATASET_CARD.md; expected labels and all recorded behavior retained, so "
    "every published metric recomputes from this file. Dataset path reduced "
    "to its basename; non-public base_url values replaced. Produced by "
    "scripts/redact_run.py."
)

_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOME = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+")


def _host(url: str) -> str:
    """Just the hostname, so a gateway on a non-standard port still matches."""
    return re.sub(r"^[a-z]+://", "", url).split("/")[0].split(":")[0]


def redact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Return a redacted deep copy; the input is not modified."""
    out = copy.deepcopy(artifact)

    dataset_name = Path(out["dataset"]).name
    out["dataset"] = dataset_name
    keep = burned_ids(dataset_name)

    for row in out.get("results", []):
        if row.get("id") not in keep and "claim" in row:
            row["claim"] = WITHHELD

    for section in ("llm_config", "llm_config_client"):
        for role_cfg in (out.get(section) or {}).values():
            url = role_cfg.get("base_url")
            if url and _host(url) not in PUBLIC_HOSTS:
                role_cfg["base_url"] = GATEWAY_PLACEHOLDER

    out["redaction"] = REDACTION_NOTE
    return out


def leaks(text: str) -> List[str]:
    """Anything that must never appear in a published artifact."""
    found = []
    found += [m.group(0) for m in _HOME.finditer(text)]
    found += [m.group(0) for m in _IPV4.finditer(text)
              if not m.group(0).startswith("127.")]
    return found


def write_redacted(src: Path, out_dir: Path, force: bool = False) -> Path:
    """Redact one artifact and write it somewhere it is safe to publish.

    Three refusals, each cheaper than the mistake it prevents. It will not
    write into the directory it read from, because the source is a paid
    artifact that cannot be regenerated. It will not overwrite an existing
    redaction without being told to, because a published file is something
    a reader may already have cited. And it will not write at all if the
    result still contains a home directory or a routable address -- the
    rules above catch the two fields we know about, and `leaks` is the
    backstop for the field nobody thought of, which is exactly how the
    gateway address reached git history the first time.
    """
    src = src.resolve()
    out_dir = out_dir.resolve()
    if out_dir == src.parent:
        raise SystemExit(f"refusing to write into the source directory {out_dir}")
    dest = out_dir / f"{src.stem}-redacted.json"
    if dest.exists() and not force:
        raise SystemExit(f"{dest} exists; pass --force to replace it")

    artifact = json.loads(src.read_text())
    text = json.dumps(redact(artifact), indent=1)
    bad = leaks(text)
    if bad:
        raise SystemExit(f"refusing to write {dest.name}: still contains {sorted(set(bad))}")
    dest.write_text(text)
    return dest


def _find_source(directory: Path, label: str) -> Path:
    """The one complete run carrying this label, or an error naming the rest.

    Incomplete runs share a label with the run that replaced them -- an
    aborted attempt and its retry are both `deepseek-c2` -- so matching on
    the label alone would pick whichever sorted first. Only complete runs
    are candidates, and an ambiguous match refuses rather than guessing,
    because publishing the wrong artifact under a label that already appears
    in a benchmark write-up is not something a reader could detect.
    """
    candidates = []
    for path in sorted(directory.glob(f"run-*-{label}.json")):
        try:
            if json.loads(path.read_text()).get("complete"):
                candidates.append(path)
        except json.JSONDecodeError:
            continue
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one complete artifact labelled {label!r} in "
            f"{directory}, found {len(candidates)}: {[p.name for p in candidates]}. "
            f"Pass --src explicitly."
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="run label, e.g. deepseek-c2")
    parser.add_argument("--src", help="explicit artifact path (overrides --label)")
    parser.add_argument("--dir", default=os.environ.get("FINVET_GOLDEN_DIR"),
                        help="where to look for --label (default: $FINVET_GOLDEN_DIR)")
    parser.add_argument("--out-dir", default="docs/eval")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing redacted output")
    args = parser.parse_args()

    if args.src:
        src = Path(args.src)
    elif args.label and args.dir:
        src = _find_source(Path(args.dir).expanduser(), args.label)
    else:
        parser.error("give --src, or --label with --dir / FINVET_GOLDEN_DIR")

    dest = write_redacted(src, Path(args.out_dir), force=args.force)
    print(f"written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
