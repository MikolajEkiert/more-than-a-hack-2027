#!/usr/bin/env bash
# Regenerate a configuration's submission file in clean Linux containers (task 15).
#
#   scripts/repro_container.sh COMMIT CONFIG [PLATFORM ...]
#   scripts/repro_container.sh HEAD configs/c14_best.toml linux/arm64 linux/amd64
#
# What task 08 did by hand for version 1, as a script. A fresh clone of COMMIT is mounted read-only
# and copied inside the container, so the container's virtual environment and caches never touch
# the host; the data bundle is mounted read-only (MTAH_DATA_DIR); the cache starts empty, so every
# feature is computed again and the committed models are unpacked from models/. Inside, `uv sync
# --frozen` installs the locked environment and `possession submit` runs the pipeline twice from a
# fresh parse, requires the same sha256 and calls the organisers' validator. Prints the sha256 per
# platform (LightGBM needs the OpenMP runtime, libgomp1, which the slim image lacks). Default platforms: linux/arm64 and linux/amd64 (emulated on an arm64 host).
# Only the submission's sha256 and the validator's verdict are printed, never a row of the data.
set -euo pipefail

COMMIT=${1:?usage: repro_container.sh COMMIT CONFIG [PLATFORM ...]}
CONFIG=${2:?usage: repro_container.sh COMMIT CONFIG [PLATFORM ...]}
shift 2
PLATFORMS=("$@")
if [ ${#PLATFORMS[@]} -eq 0 ]; then
  PLATFORMS=(linux/arm64 linux/amd64)
fi

ROOT=$(git rev-parse --show-toplevel)
DATA=${MTAH_DATA_DIR:-$ROOT/data}
SHA=$(git -C "$ROOT" rev-parse "$COMMIT")
WORK=$(mktemp -d "${TMPDIR:-/tmp}/repro.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --no-hardlinks "$ROOT" "$WORK/src"
git -C "$WORK/src" checkout --quiet "$SHA"
IMAGE=${REPRO_IMAGE:-ghcr.io/astral-sh/uv:python3.11-bookworm-slim}

for platform in "${PLATFORMS[@]}"; do
  echo "[$platform] commit ${SHA:0:7}, $CONFIG, image $IMAGE"
  docker run --rm --platform "$platform" \
    -v "$WORK/src:/src:ro" -v "$DATA:/data:ro" -e MTAH_DATA_DIR=/data \
    "$IMAGE" bash -c "
      set -euo pipefail
      apt-get update -qq && apt-get install -y -qq --no-install-recommends libgomp1 >/dev/null
      cp -r /src /repo && cd /repo
      uv sync --frozen --quiet
      start=\$(date +%s)
      uv run possession submit '$CONFIG' --team culers 2>&1 \
        | grep -E 'SUBMISSION|^Valid|FAIL|differ|Error|Traceback' || true
      echo \"seconds \$(( \$(date +%s) - start ))\"
    " 2>&1 | sed "s|^|[$platform] |"
done
