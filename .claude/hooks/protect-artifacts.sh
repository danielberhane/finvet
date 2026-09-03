#!/bin/bash
# PreToolUse guard: block any tool call that touches paid benchmark artifacts
# or the frozen golden dataset.
#
# Exists because of a real incident (2026-08-31): `eval_layers.py --json <path>`
# was run with an artifact path in the belief that --json selected an input.
# It is an output flag; 100 recorded rows of paid API spend were overwritten.
# The CLAUDE.md rule existed and was known — the action was misclassified.
# A pattern match has no beliefs to be wrong about.
#
# Contract: stdin = JSON {"tool_name": ..., "tool_input": {...}}
#           exit 0 = allow, exit 2 = block (stderr is shown to the model).

input=$(cat)

# Only inspect tools that can write. Read-only tools must never be blocked —
# an over-broad hook gets disabled, and then there is nothing.
tool=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)
case "$tool" in
  Bash|Write|Edit|NotebookEdit) ;;
  *) exit 0 ;;
esac

# Protected: run artifacts and the frozen dataset, in the golden repo.
# golden_100.jsonl.pre-* backups included. _archive/_backup copies included.
PATTERN='finvet-golden/[^"]*(run-[^"]*\.json|golden_c\.jsonl|golden_100[^"]*)'

if ! echo "$input" | grep -qE "$PATTERN"; then
  exit 0
fi

# For Bash: allow plainly read-only commands on those files. Everything else
# that mentions them is blocked — including redirects, cp onto them, rm, mv,
# sed -i, python scripts taking them as args (a script arg may be an output
# path; that is exactly the incident).
if [ "$tool" = "Bash" ]; then
  cmd=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)
  first_word=$(echo "$cmd" | awk '{print $1}')
  case "$first_word" in
    cat|head|tail|less|wc|grep|md5|shasum|ls|stat|diff|jq)
      # read-only verb AND no redirect anywhere in the command
      if ! echo "$cmd" | grep -q '[>]'; then
        exit 0
      fi
      ;;
  esac
fi

echo "BLOCKED by .claude/hooks/protect-artifacts.sh:" >&2
echo "this call touches a paid benchmark artifact or the frozen golden dataset." >&2
echo "These cost real API spend and cannot be regenerated identically." >&2
echo "If you genuinely need to modify one: copy it to the scratchpad and work" >&2
echo "there, or ask the user to run the command themselves." >&2
exit 2
