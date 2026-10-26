#!/usr/bin/env bash
# Run on the simulator host after forwarding the NX service and copying its metadata.
set -euo pipefail
: "${EMBODIRUN:?Set EMBODIRUN to the downstream checkout containing activevln-navigation}"
: "${HABITAT_PYTHON:?Set HABITAT_PYTHON to the existing Habitat interpreter}"
: "${SCENES:?Set SCENES to the scene_datasets directory}"
if [[ $# != 3 ]]; then
    echo "Usage: bash navigate.sh RUN_ROOT ENDPOINT OUTPUT" >&2
    exit 2
fi
run_root=$1
endpoint=$2
output=$3
for name in config.json service-ready.json service-token selected-r2r-navigation-48.json; do
    test -f "$run_root/$name" || { echo "Missing $run_root/$name" >&2; exit 1; }
done
python - "$run_root/selected-r2r-navigation-48.json" <<'PY'
import hashlib
import sys
from pathlib import Path
expected = "4cee972c4be8b19657839ca6bfa62cab4019c38b6d5cffdc672814495bec85f4"
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise ValueError("Use the retained R2R 48-episode manifest for this comparison")
PY
export PYTHONPATH="$EMBODIRUN/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u "$EMBODIRUN/benchmarks/activevln-navigation/benchmark.py" \
    --root "$run_root" --config "$run_root/config.json" \
    --habitat-python "$HABITAT_PYTHON" --scenes "$SCENES" --output "$output" \
    --endpoint "$endpoint" --service-ready "$run_root/service-ready.json" \
    --token-file "$run_root/service-token" --datasets r2r --episode-count 48 \
    --batch-size 1 --simulator-gpu "${SIMULATOR_GPU:-0}"
