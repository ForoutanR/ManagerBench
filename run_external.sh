#!/usr/bin/env bash
# Launch the external-eval harnesses (T12 XSTest, T13 HarmBench) sequentially.
# Contains NO secret: OPENROUTER_API_KEY must already be in the environment
# (e.g. exported in the shell/profile on the machine that runs this).
set -euo pipefail
cd "$(dirname "$0")"
export HARD_CAP="${HARD_CAP:-27}"   # total-usage ceiling ($); key hard-caps at $30
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
mkdir -p results/external

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set in this environment. Export it, then re-run." >&2
  exit 3
fi

echo "=== T12 XSTest ==="
"$PY" run_xstest.py    2>&1 | tee results/external/xstest_run.log
echo "=== T13 HarmBench ==="
"$PY" run_harmbench.py 2>&1 | tee results/external/harmbench_run.log
echo ALL_DONE
