#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SYNTHETIC=(fac-loc g-cut g-cut-8 int-2-1 log log-det log-times-log-det log-times-sqrt sqrt)
ABLATION=(fac-loc g-cut int-2-1 log-det)
LAYER=(fac-loc g-cut log log-det sqrt)
SET_SIZES=(128 256 512 1024 2048)
CATEGORIES=(apparel bath bedding diaper feeding gear health media)
DEPTHS=(1 2 3 5 7 10)
TABLES=(synthetic synthetic-ablations synthetic-layers synthetic-set-size amazon amazon-layers dspn)
failed=0

run() {
    local experiment=$1 options=(--out "${OUT:-runs}")
    shift
    case $experiment in
        synthetic) options+=(${DEVICE:+--device "$DEVICE"} ${COMPILE:+--compile}) ;;
        amazon) options+=(${DEVICE:+--device "$DEVICE"}) ;;
        dspn) options+=(${DATA:+--data "$DATA"} ${FEATURES:+--features "$FEATURES"}) ;;
    esac
    python scripts/run.py "$experiment" "$@" "${options[@]}" ${DRY_RUN:+--dry-run}
}

spawn() {
    while (( $(jobs -rp | wc -l) >= ${JOBS:-1} )); do wait -n || failed=1; done
    "$@" &
}

finish() {
    while (( $(jobs -rp | wc -l) > 0 )); do wait -n || failed=1; done
}

synthetic_runs() {
    local -n datasets=$1
    local dataset model
    shift
    for dataset in "${datasets[@]}"; do
        for model in "$@"; do
            spawn run synthetic --model $model --dataset "$dataset"
        done
    done
}

amazon_category() {
    local category=$1 model
    shift
    for model in "$@"; do
        run amazon --model $model --dataset "$category" || return 1
    done
}

amazon_runs() {
    local category
    for category in "${CATEGORIES[@]}"; do
        spawn amazon_category "$category" "$@"
    done
}

depths=()
for layers in "${DEPTHS[@]}"; do
    depths+=("SubmodInt --layers $layers")
done
tables=("$@")
(( ${#tables[@]} )) || tables=("${TABLES[@]}")
for table in "${tables[@]}"; do
    [[ " ${TABLES[*]} " == *" $table "* ]] || { echo "unknown table $table (${TABLES[*]})" >&2; exit 2; }
done
for table in "${tables[@]}"; do
    case $table in
        synthetic) synthetic_runs SYNTHETIC SubmodInt FlexSubNet DSF SetTransformer EDSF DeepSet ;;
        synthetic-ablations) synthetic_runs ABLATION SubmodInt "SubmodInt --layers 1" "SubmodInt --fixed" DSF "DSF --layers 1" ;;
        synthetic-layers) synthetic_runs LAYER "${depths[@]}" ;;
        synthetic-set-size) for size in "${SET_SIZES[@]}"; do
                                for model in SubmodInt SetTransformer; do
                                    spawn run synthetic --model $model --dataset log-det --max-set-size "$size"
                                done
                            done ;;
        amazon) amazon_runs SubmodInt FlexSubNet DSF SetTransformer DeepSet ;;
        amazon-layers) amazon_runs "${depths[@]}" ;;
        dspn) for model in DSF SubmodInt "SubmodInt --fixed" Hybrid DeepSet SetTransformer; do
                  run dspn --model $model || failed=1
              done ;;
    esac
    finish
done
(( failed == 0 )) || { echo "some runs failed; see their output above" >&2; exit 1; }
