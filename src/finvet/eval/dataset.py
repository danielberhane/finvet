"""Locator for the external evaluation dataset (optional).

FinVet is benchmarked against a privately held evaluation dataset. Point
FINVET_EVAL_DATA_DIR at its data/clean directory to enable those harnesses
and tests; without it they skip cleanly. The dataset is not distributed with
FinVet.
"""

import os
from pathlib import Path
from typing import Optional


def eval_data_dir() -> Optional[Path]:
    """The external eval dataset's data/clean directory, or None."""
    p = os.environ.get("FINVET_EVAL_DATA_DIR")
    return Path(p).expanduser() if p else None


def eval_data_file(name: str) -> Optional[Path]:
    d = eval_data_dir()
    return d / name if d else None
