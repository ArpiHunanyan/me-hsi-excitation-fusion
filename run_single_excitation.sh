#!/usr/bin/env bash
# Train and evaluate single-excitation models, then combine them with
# soft majority voting (Soft MV baseline in the paper):
#   2 backbones x 4 excitations x 8 images x 5 click splits.
#
# Excitation suffixes: 2 = 360 nm, 3 = 370 nm, 4 = 380 nm, 5 = 390 nm.
#
# Usage:
#   DATA_ROOT=./data GT_ROOT=./data/gt CUDA=0 ./run_single_excitation.sh

set -u

DATA_ROOT="${DATA_ROOT:-./data}"
GT_ROOT="${GT_ROOT:-./data/gt}"
RESULTS_ROOT="${RESULTS_ROOT:-./results}"
CUDA="${CUDA:-0}"

IMAGES=(241127_1 241205_1 241205_2 241205_3 241205_4 241207_1 241207_2 241207_3)
SHAPES=(696x520  348x256  348x256  696x520  696x520  696x512  696x520  568x392)
SPLITS=(0 1 2 3 4)
EXCITATIONS=(2 3 4 5)
MODELS=(levit_single mlp_single)

CLICKS="gt_circle_1_percentage_rcount5"

failed=()

for model in "${MODELS[@]}"; do
    for i in "${!IMAGES[@]}"; do
        image="${IMAGES[$i]}"
        shape="${SHAPES[$i]}"
        for exc in "${EXCITATIONS[@]}"; do
            for split in "${SPLITS[@]}"; do
                dataset="${image}_norm_${shape}x31_${exc}"
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
done

echo "Training finished. ${#failed[@]} run(s) failed."
for run in "${failed[@]+"${failed[@]}"}"; do
    echo "  FAILED: ${run}"
done

# Soft MV: average the logits of the four single-excitation models
python soft_majority_voting.py \
    --results_root "${RESULTS_ROOT}" \
    --gt_root "${GT_ROOT}" \
    --models "${MODELS[@]}" \
    --excitations "${EXCITATIONS[@]}"