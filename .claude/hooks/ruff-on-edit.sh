#!/bin/bash
# PostToolUse: lint any edited .py file immediately.
#
# Twice in one week ruff caught real residue (a dead import, leftover dict
# keys) only because it happened to be run. This makes it non-optional.
# Non-blocking by design: PostToolUse cannot undo the edit, and a formatter
# war would be worse than a warning. Output goes to the model, which fixes it.

input=$(cat)
fp=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)
case "$fp" in
  *.py)
    cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
    [ -x .venv/bin/ruff ] || exit 0
    out=$(.venv/bin/ruff check "$fp" 2>&1)
    if [ -n "$out" ] && ! echo "$out" | grep -q "All checks passed"; then
      echo "ruff findings in $fp (fix before moving on):" >&2
      echo "$out" | head -20 >&2
      exit 2   # exit 2 on PostToolUse: shows stderr to the model as feedback
    fi
    ;;
esac
exit 0
