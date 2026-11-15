#!/usr/bin/env bash
# Run one heavy command (a measurement or a training) in one of a few machine-wide slots.
#
#   scripts/heavy_slot.sh CMD [ARG ...]
#
# Several agents share one machine, and memory ran out once when too many pipelines ran at the same
# time. A slot is a directory under $HEAVY_SLOT_DIR (default /tmp/possession-slots), created
# atomically with mkdir and removed when the command ends, however it ends. With every slot taken
# the script waits. HEAVY_SLOTS sets the number of slots (default 2). A slot left by a killed shell
# is reclaimed when the process that held it is gone.
set -uo pipefail

DIR=${HEAVY_SLOT_DIR:-/tmp/possession-slots}
SLOTS=${HEAVY_SLOTS:-2}
mkdir -p "$DIR"
held=""
while [ -z "$held" ]; do
  for i in $(seq 1 "$SLOTS"); do
    slot="$DIR/slot$i"
    if mkdir "$slot" 2>/dev/null; then
      echo $$ > "$slot/pid"
      held=$slot
      break
    fi
    owner=$(cat "$slot/pid" 2>/dev/null || true)
    if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
      rm -rf "$slot"  # the holder is gone: reclaim on the next pass
    fi
  done
  [ -z "$held" ] && sleep 15
done
trap 'rm -rf "$held"' EXIT INT TERM
echo "[heavy_slot] $(basename "$held") taken by $$: $*" >&2
"$@"
