---
name: benchmark-run
description: Run the golden 97 through a named model, freeze-safe and comparable across runs
disable-model-invocation: true
---
Run the full golden benchmark for: $ARGUMENTS
(expected form: `<env-name> <label>` — e.g. `minimax minimax-c2` or
`default deepseek-c2`. `default` means `.env`; anything else means `.env.<env-name>`.)

Every step below exists because skipping it once cost something real. Do not
skip steps, and do not reorder them.

## 1. Freeze check — refuse to run on uncommitted state
```
git status --short                          # code repo: must be empty
git -C "$FINVET_GOLDEN_DIR" status --short  # data repo: must be empty
git log -1 --format=%h
git -C "$FINVET_GOLDEN_DIR" log -1 --format=%h
```
Record both SHAs — they go in the final commit message. If either repo is
dirty: STOP and tell the user. Runs are only comparable when produced by
identical code and data; a dirty run is spend that buys nothing.

## 2. Serve the intended model — and verify from the serving process
Kill port 8000, start the API under the chosen env file, then:
```
curl -s localhost:8000/health
```
All three roles (parser, agent, verdict) must show the intended model.
Why: 56 rows were once labeled MiniMax while DeepSeek actually served them —
the client shell's belief about the model is not evidence; /health is.

## 3. Launch, in the background, with a monitor
From a shell that sourced ONLY the intended env file:
```
export FINVET_GOLDEN_DIR=/path/to/the/private/golden/repo
export LANGCHAIN_TRACING_V2=false
.venv/bin/python scripts/run_golden.py --label <label>
```
Run it as a background task and attach a Monitor grepping for:
`ERR |ESC |Timeout|503|429|401|Traceback|written to`
Why: a silent vendor hang once ate 97 minutes; the runner writes the artifact
after every row, so a killed run loses at most one claim — but only a monitor
tells you to kill it.

The runner itself refuses if the API serves a different model than the shell
expects. If it refuses: fix the mismatch. Never pass --allow-model-drift to
get past it during a benchmark.

## 4. Quarantine the shell afterward
Do NOT run pytest from any shell that sourced a non-default env.
Why: LLM_*__MODEL leaks into the test environment and produces ~8 false
failures against the DeepSeek defaults.

## 5. Score
```
.venv/bin/python scripts/eval_layers.py --label <label>
```
NEVER `--json <artifact path>` — --json is an OUTPUT path and once
overwrote 100 paid rows. (A hook now blocks this; the rule stands anyway.)

## 6. Commit — artifact and layer summary together
In finvet-golden: force-add the run artifact (run-*.json is gitignored by
default so scratch runs stay out) plus the layer summary. The commit message
states both SHAs from step 1 and the served model from step 2.

## 7. Comparison rules
- Compare only runs produced at the same two SHAs.
- Exclude `market/quote` (live data — one run met a Finnhub outage, the
  next didn't; that difference is the vendor, not the model).
- One run per model gives pass@1 only. Differences of ≤3 rows out of ~86
  are within the measured noise floor (DeepSeek pass^4 = 0.945 over c1–c4 on
  the frozen set, `docs/eval/layers-deepseek-c1-c4.json`); claiming a model
  difference needs ~3 runs per model.
