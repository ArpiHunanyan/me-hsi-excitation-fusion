#!/usr/bin/env bash
# Train and evaluate all excitation-fusion models (paper, Table 2):
#   2 backbones x 2 fusion stages x 7 operators = 28 models,
#   on 8 images x 5 click splits.
#
# Usage:
#   DATA_ROOT=./data GT_ROOT=./data/gt CUDA=0 ./run_fusion.sh
#   ./run_fusion.sh levit_logit_wavg mlp_feat_att      # only the listed models

set -u

DATA_ROOT="${DATA_ROOT:-./data}"
GT_ROOT="${GT_ROOT:-./data/gt}"
RESULTS_ROOT="${RESULTS_ROOT:-./results}"
CUDA="${CUDA:-0}"

# Image ids and spatial sizes (H x W)
IMAGES=(241127_1 241205_1 241205_2 241205_3 241205_4 241207_1 241207_2 241207_3)
SHAPES=(696x520  348x256  348x256  696x520  696x520  696x512  696x520  568x392)
SPLITS=(0 1 2 3 4)

# Training clicks: 5 clicks per class, 0.2% of the image each (2% in total)
CLICKS="gt_circle_1_percentage_rcount5"

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    for backbone in levit mlp; do
        for stage in feat logit; do
            for op in avg sum fc gate cnn wavg att; do
                MODELS+=("${backbone}_${stage}_${op}")
            done
        done
    done
fi

failed=()

for model in "${MODELS[@]}"; do
    for i in "${!IMAGES[@]}"; do
        image="${IMAGES[$i]}"
        shape="${SHAPES[$i]}"
        for split in "${SPLITS[@]}"; do
            dataset="${image}_norm_${shape}x31x4_4d"
            gt_train="${image}_${shape}_${CLICKS}_split${split}"

            echo "=== ${model} | ${dataset} | split ${split} ==="
            if ! python main.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --gt_train "${gt_train}" \
                    --data_root "${DATA_ROOT}" \
                    --gt_root "${GT_ROOT}" \
                    --results_root "${RESULTS_ROOT}" \
                    --cuda "${CUDA}"; then
                failed+=("${model} ${dataset} split${split}")
            fi
        done
    done
done

echo "Finished. ${#failed[@]} run(s) failed."
for run in "${failed[@]+"${failed[@]}"}"; do
    echo "  FAILED: ${run}"
done