# Contributing

FinVet is a reference implementation maintained by one person, so the most
useful contributions are bug reports, reproductions, and small focused fixes.
Large feature PRs are unlikely to be merged without discussion first.

## Setup

```bash
uv sync                                        # dev tools come from the dev group
cp .env.example .env                           # fill in the required keys
.venv/bin/python -m pytest tests/unit -q       # ~1,565 tests, no services needed
```

Use `.venv/bin/python`, never bare `python`. Run tests from a clean shell: an
env file that overrides `LLM_*__MODEL` leaks into the test environment and
fails the tests that assert the default provider.

## What CI enforces

Your PR must pass what `.github/workflows/ci.yml` runs:

- `ruff check .`
- `mypy` on six trust-boundary modules (deliberately not the whole codebase)
- `python -m compileall` over `src ui tests scripts`
- a check that every `src/finvet/**.py` path named in the docs exists
- `pytest tests/unit` on Python 3.11, 3.12 and 3.13, with overall coverage at
  or above 75% and per-file floors on six modules
- a build of all three Docker images

Integration tests are excluded by default and each one skips itself when its
service is absent, so you do not need Postgres or the MCP server to contribute.

## The one rule about tests

**Tests must drive the producer.** A test covering a verdict, an escalation, or
a persistence path has to enter through a route callable, a graph node, or a
decorated tool, never a hand-built dictionary. Three defects once survived a
suite of hundreds of tests because their tests constructed their own inputs: a
fixture asserts the shape you remembered, not the shape the system emits.

## Things that will be declined

- **Midpoint comparison for range claims.** Comparing the midpoint of a stated
  band refutes true claims whose band exceeds the tolerance. Range claims
  decline on purpose (`docs/RELEASE_A_DECISIONS.md`, D18).
- **A free-form context field on the verify request.** It was removed as a
  prompt-injection channel and is not coming back; the API takes an identifier
  the server resolves itself.
- **Letting the model decide a numeric verdict.** The comparison is computed in
  Python from a structured observation, and a claim without one declines.

## Evaluation

The golden dataset is held out privately, so benchmark runs are not
reproducible from a fork. What is public is the method: redacted per-claim run
artifacts, layer summaries, and the dataset card in `docs/eval/`, from which
every published number recomputes. If you change anything the evaluation
measures, say so in the PR; the numbers in the README are tied to specific
recorded runs.

## Commits and PRs

Explain why in the commit message, not just what. Reference the incident or the
behaviour that motivated the change if there is one; most of the odd-looking
code here exists because something broke.
