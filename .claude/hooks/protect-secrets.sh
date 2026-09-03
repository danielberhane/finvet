#!/bin/bash
# PreToolUse guard: secrets must be neither modified nor printed raw.
#
# Ported from finvet-v2.0.6 (write-guard) after it was lost in the directory
# migration, and extended to the read direction after a live incident: a key
# in .env.minimax with an unusual casing (LiteLLM_API_Key) slipped past a
# redaction pattern and landed in a session transcript. Sourcing env files
# for runs stays allowed; printing them raw does not.
#
# Contract: stdin JSON {"tool_name",...}; exit 0 allow, exit 2 block.

input=$(cat)
tool=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

SECRET_PATH='(^|/)\.env[^/"]*|credentials|secrets/|\.pem|\.key'

case "$tool" in
  Write|Edit|NotebookEdit)
    fp=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)
    if echo "$fp" | grep -qE "$SECRET_PATH"; then
      echo "BLOCKED by protect-secrets.sh: direct edits to secret files are" >&2
      echo "not allowed. Ask the user to change $fp themselves." >&2
      exit 2
    fi
    ;;
  Bash)
    cmd=$(echo "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)
    # Block only commands that PRINT a secret file raw. `source`/`.`/`set -a`
    # loading, grep -c, checking a var is set — all legitimate, all allowed.
    if echo "$cmd" | grep -qE '\b(cat|head|tail|less|more|bat)\b[^|;&]*\.env' \
       && ! echo "$cmd" | grep -qE '\|\s*(grep|wc|cut|sed|awk|python)'; then
      echo "BLOCKED by protect-secrets.sh: this would print an env file raw" >&2
      echo "into the transcript. Keys have leaked this way before. Use" >&2
      echo "grep for the specific NON-secret line you need, or pipe through" >&2
      echo "a redaction: sed 's/=.*/=REDACTED/' <file>." >&2
      exit 2
    fi
    # Writes/deletes aimed at secret files via shell.
    if echo "$cmd" | grep -qE '(>{1,2}[[:space:]]*[^ |;&]*\.env|(\brm\b|\bmv\b|\bcp\b|sed -i)[^|;&]*\.env)' ; then
      echo "BLOCKED by protect-secrets.sh: shell modification of an env file." >&2
      echo "Ask the user to make this change themselves." >&2
      exit 2
    fi
    ;;
esac
exit 0
