# -*- coding: utf-8 -*-
"""
Soft majority voting (Soft MV) baseline.

E single-excitation models are trained independently (run_single_excitation.sh).
At inference, their output scores (logits, no softmax) are averaged pixel-wise
and the class with the highest averaged score is predicted:

    prediction = argmax_c  (1/E) * sum_e  logits_e[c]

There is no joint training and no normalization across excitations; this is
what distinguishes Soft MV from the logit-level AVG fusion model.

Inputs (written by main.py for each excitation e):
    <results_root>/<id>_norm_<HxW>x31_<e>/<model>/<gt_train>/logits.npy

Outputs (next to the fusion results of the same image):
    <results_root>/<id>_norm_<HxW>x31xE_4d/<backbone>_softmv/<gt_train>/
        logits.npy, metrics.txt, prediction.png

Example:
    python soft_majority_voting.py --results_root ./results --gt_root ./data/gt
"""

import argparse
import glob
import os

import numpy as np

from datasets import load_ground_truth, LABEL_VALUES, IGNORED_LABELS
from utils import compute_metrics, save_metrics, save_prediction_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Soft majority voting over single-excitation models"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="./results",
        help="Folder with single-excitation results (default: ./results)",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        required=True,
        help="Folder containing gt_circle/, gt_test/ and gt/",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["levit_single", "mlp_single"],
        help="Single-excitation models to combine",
    )
    parser.add_argument(
        "--excitations",
        nargs="+",
        default=["2", "3", "4", "5"],
        help="Excitation suffixes of the single-excitation datasets "
        "(default: 2 3 4 5 = 360, 370, 380, 390 nm)",
    )
    return parser.parse_args()


def soft_vote(logit_maps):
    """Average E logit maps of shape (H, W, C) and take the argmax."""
    mean_logits = np.mean(np.stack(logit_maps, axis=0), axis=0)
    return mean_logits, np.argmax(mean_logits, axis=-1)


def main():
    args = parse_args()
    first, n_exc = args.excitations[0], len(args.excitations)

    # One single-excitation result folder per image, found via the first excitation
    image_dirs = sorted(glob.glob(os.path.join(args.results_root, f"*_{first}")))
    if not image_dirs:
        raise FileNotFoundError(
            f"No single-excitation results found in {args.results_root}"
        )

    n_done = 0

    for image_dir in image_dirs:
        prefix = os.path.basename(image_dir)[: -len(f"_{first}")]  # <id>_norm_<HxW>x31
        fused_name = f"{prefix}x{n_exc}_4d"

        for model in args.models:
            backbone = model.replace("_single", "")
            split_dirs = sorted(glob.glob(os.path.join(image_dir, model, "*")))

            for split_dir in split_dirs:
                gt_train = os.path.basename(split_dir)
                paths = [
                    os.path.join(
                        args.results_root,
                        f"{prefix}_{e}",
                        model,
                        gt_train,
                        "logits.npy",
                    )
                    for e in args.excitations
                ]

                missing = [p for p in paths if not os.path.isfile(p)]
                if missing:
                    print(
                        f"Skipping {prefix} / {model} / {gt_train}: "
                        f"{len(missing)} of {n_exc} excitations missing"
                    )
                    continue

                mean_logits, prediction = soft_vote([np.load(p) for p in paths])

                _, test_gt, predict_on_gt = load_ground_truth(gt_train, args.gt_root)
                prediction[predict_on_gt == 0] = 0  # background is never predicted

                results = compute_metrics(
                    prediction,
                    test_gt,
                    ignored_labels=IGNORED_LABELS,
                    n_classes=len(LABEL_VALUES),
                )

                out_dir = os.path.join(
                    args.results_root, fused_name, f"{backbone}_softmv", gt_train
                )
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, "logits.npy"), mean_logits)
                save_metrics(results, os.path.join(out_dir, "metrics.txt"))
                save_prediction_map(prediction, os.path.join(out_dir, "prediction.png"))

                print(
                    f"{fused_name} / {backbone}_softmv / {gt_train}: "
                    f"OA={results['OA']:.4f}  AA={results['AA']:.4f}  Kappa={results['Kappa']:.4f}"
                )
                n_done += 1

    print(f"Soft MV computed for {n_done} image/split/model combinations")


if __name__ == "__main__":
    main()
