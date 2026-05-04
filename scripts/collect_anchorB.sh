#!/usr/bin/env bash
set -euo pipefail

POINT="${1:?Usage: collect_anchorB.sh POINT CONDITION [SECONDS]}"
COND="${2:?Usage: collect_anchorB.sh POINT CONDITION [SECONDS]}"
DURATION="${3:-60}"

ANCHOR="anchorB"
PORT="$("$HOME/.pyenv/versions/3.11.9/bin/python" - <<'PY2'
import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
matches = [p.device for p in ports if "2DE0:1337" in p.hwid or "Raspberry Pi Composite Gadget" in p.description]
if not matches:
    raise SystemExit("ERROR: Raspberry Pi UWB serial device not found")
print(matches[0])
PY2
)"
SESSION_ID="45"

BASE="$HOME/Desktop/qorvo_data_redo"
OUTDIR="$BASE/raw"
SDK="$HOME/Desktop/qm35-sdk/Samples/Python/UWB-Qorvo-Tools"
PREFIX="${POINT}_${COND}_${ANCHOR}"

mkdir -p "$OUTDIR"
cd "$OUTDIR"
source "$SDK/.venv/bin/activate"

RUN_FIRA_TWR="$HOME/.pyenv/versions/3.11.9/bin/run_fira_twr"
if [[ ! -x "$RUN_FIRA_TWR" ]]; then
  echo "ERROR: run_fira_twr not found at $RUN_FIRA_TWR"
  exit 1
fi

echo "============================================================"
echo "Collecting $ANCHOR / CONTROLEE"
echo "Point:      $POINT"
echo "Condition:  $COND"
echo "Duration:   $DURATION seconds"
echo "Port:       $PORT"
echo "Prefix:     $PREFIX"
echo "Start time: $(date --iso-8601=seconds)"
echo "============================================================"

"$RUN_FIRA_TWR" -p "$PORT" -s "$SESSION_ID" --controlee -t "$DURATION" \
  --en-rssi \
  --en-diag \
  --stats \
  --diag_dump \
  2>&1 | tee "${PREFIX}_stdout.txt"

echo "End time: $(date --iso-8601=seconds)" | tee -a "${PREFIX}_stdout.txt"

JSON="$(ls -t *.json 2>/dev/null | head -1 || true)"
if [[ -z "$JSON" ]]; then
  echo "ERROR: No diagnostic JSON file found."
  exit 1
fi

mv "$JSON" "${PREFIX}_diag.json"

echo "Saved:"
echo "  $OUTDIR/${PREFIX}_stdout.txt"
echo "  $OUTDIR/${PREFIX}_diag.json"

echo "Checking for distance lines:"
grep -n "distance:" "${PREFIX}_stdout.txt" | head || echo "WARNING: no distance lines found"
