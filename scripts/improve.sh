#!/usr/bin/env bash
# One round of the improvement loop. The bot plays, its own crashes become training data, a
# candidate zone model is trained, and it replaces the live one only if it benchmarks better.
#
#   scripts/improve.sh base   <tag> [runs]   bench the live model, pull the pre-crash frames
#   (label data/crash_<tag> -> labels/crash_<tag>.jsonl: ask Claude Code, or `ssbot label`)
#   scripts/improve.sh train  <tag>          train a candidate zone CNN on every labelled set
#   scripts/improve.sh verify <tag> [runs]   bench the candidate; promote it if it wins
#
# Judge on the median over 20+ runs: single runs are noise.

set -euo pipefail
cd "$(dirname "$0")/.."

SSBOT=${SSBOT:-./target/release/ssbot}
PYTHON=${PYTHON:-../.venv/bin/python}
# Recordings whose frames don't line up with the zones (cyan border: the game didn't fill the
# window) would teach the classifier the wrong crops.
EXCLUDE=${EXCLUDE:-"human_round2"}
CANDIDATE_MODEL=zone_model.candidate.json   # a CNN, see sidecar/zone_cnn.py
CANDIDATE_CALIB=calibration.candidate.toml
# Never trained on: the fixed test set the eyes are scored on (`ssbot eval-zones data/$FROZEN`).
FROZEN=${FROZEN:-crash_r3}

stage=${1:?usage: improve.sh base|train|verify <tag> [runs]}
tag=${2:?need a round tag, e.g. r1}
runs=${3:-20}

labelled_dirs() {
  for f in labels/*.jsonl; do
    name=$(basename "$f" .jsonl)
    [[ -s "$f" && -d "data/$name" ]] || continue
    [[ " $EXCLUDE " == *" $name "* ]] && continue
    echo "data/$name"
  done
}

case "$stage" in
  base)
    "$SSBOT" bench --runs "$runs" --tag "$tag-base"
    "$SSBOT" crash-frames "runs/bench_$tag-base.json" --out "data/crash_$tag"
    echo
    echo "Next: label data/crash_$tag into labels/crash_$tag.jsonl, then: scripts/improve.sh train $tag"
    ;;
  train)
    [[ -s "labels/crash_$tag.jsonl" ]] || { echo "labels/crash_$tag.jsonl is missing or empty: label data/crash_$tag first" >&2; exit 1; }
    # shellcheck disable=SC2046
    # Full-resolution crops with 2x context around each zone: held-out missed hazards fell 36%.
    "$SSBOT" zone-crops $(labelled_dirs) --out data/zone_crops --native --crop 48 --ctx 2.0
    "$PYTHON" sidecar/zone_cnn.py --holdout "$FROZEN" --export "$CANDIDATE_MODEL"
    cp calibration.toml "$CANDIDATE_CALIB"
    sed -i.bak "s#^zone_model = .*#zone_model = \"$CANDIDATE_MODEL\"#" "$CANDIDATE_CALIB" && rm -f "$CANDIDATE_CALIB.bak"
    echo
    echo "Candidate saved as $CANDIDATE_MODEL. Next: scripts/improve.sh verify $tag"
    ;;
  verify)
    [[ -f "$CANDIDATE_MODEL" && -f "$CANDIDATE_CALIB" ]] || { echo "no candidate: run 'train' first" >&2; exit 1; }
    "$SSBOT" --calibration "$CANDIDATE_CALIB" bench --runs "$runs" --tag "$tag-candidate"
    echo
    "$SSBOT" bench-compare "runs/bench_$tag-base.json" "runs/bench_$tag-candidate.json"
    if python3 - "$tag" <<'PY'
import json, sys
tag = sys.argv[1]
agg = lambda t: json.load(open(f"runs/bench_{tag}-{t}.json"))["aggregate"]
base, cand = agg("base"), agg("candidate")
sys.exit(0 if cand["survival_median_s"] > base["survival_median_s"] else 1)
PY
    then
      cp zone_model.json "zone_model.before-$tag.json"
      cp "$CANDIDATE_MODEL" zone_model.json
      echo "Promoted: candidate beat the live model on median survival (old model kept as zone_model.before-$tag.json)."
    else
      echo "Not promoted: the candidate did not beat the live model on median survival."
    fi
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 1
    ;;
esac
