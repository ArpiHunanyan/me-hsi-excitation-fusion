# -*- coding: utf-8 -*-
"""
Train and evaluate one model on one image and one click split.

Example (feature-level CNN fusion with LeViT branches):

    python main.py --model levit_feat_cnn \
        --dataset 241127_1_norm_696x520x31x4_4d \
        --gt_train 241127_1_696x520_gt_circle_1_percentage_rcount5_split0 \
        --data_root ./data --gt_root ./data/gt --cuda 0

Outputs in <results_root>/<dataset>/<model>/<gt_train>/:
    logits.npy             (H, W, C) logits (input for soft_majority_voting.py)
    metrics.txt    OA, AA and kappa on the test pixels
    prediction.png      predicted label map
"""

import argparse
import os

import numpy as np
import torch
import torch.utils.data as data

from datasets import load_dataset, PatchDataset
from models import build_model, train, predict, AVAILABLE_MODELS
from utils import select_device, compute_metrics, save_metrics, save_prediction_map


def parse_args():
    parser = argparse.ArgumentParser(description="ME-HSI excitation-fusion experiments")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=AVAILABLE_MODELS,
        help="Model name, e.g. levit_logit_wavg or mlp_single",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Image name; names ending in '_4d' are multi-excitation",
    )
    parser.add_argument(
        "--gt_train",
        type=str,
        required=True,
        help="Training-GT name (simulated clicks of one split)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Folder containing 4D/ and Separate/",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        required=True,
        help="Folder containing gt_circle/, gt_test/ and gt/",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="./results",
        help="Where results are written (default: ./results)",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="CUDA device index; -1 for CPU (default: -1)",
    )

    group = parser.add_argument_group("Training")
    group.add_argument("--epoch", type=int, default=100)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--batch_size", type=int, default=100)
    group.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Spatial patch size (default: 15 for LeViT, 1 for MLP)",
    )
    group.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="Sliding-window stride at inference (default: 1)",
    )
    group.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="Also save the trained weights (checkpoint.pth)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.cuda)

    results_dir = os.path.join(
        args.results_root, args.dataset, args.model, args.gt_train
    )
    os.makedirs(results_dir, exist_ok=True)

    # ---------- Data ----------
    img, train_gt, test_gt, predict_on_gt, label_values, ignored_labels = load_dataset(
        args.dataset, args.gt_train, args.data_root, args.gt_root
    )

    is_single = args.model.endswith("_single")
    if is_single and img.ndim != 3:
        raise ValueError(
            f"{args.model} needs a single-excitation image, got {args.dataset}"
        )
    if not is_single and img.ndim != 4:
        raise ValueError(
            f"{args.model} needs a multi-excitation (_4d) image, got {args.dataset}"
        )

    n_bands = img.shape[2]
    n_excitations = img.shape[3] if img.ndim == 4 else 1

    print(f"Image {args.dataset}: shape {img.shape}")
    print(
        f"{np.count_nonzero(train_gt)} training pixels, {np.count_nonzero(test_gt)} test pixels"
    )

    # ---------- Model ----------
    model, optimizer, criterion, scheduler, hyperparams = build_model(
        args.model,
        n_bands=n_bands,
        n_classes=len(label_values),
        n_excitations=n_excitations,
        ignored_labels=ignored_labels,
        device=device,
        epoch=args.epoch,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
    )
    hyperparams["test_stride"] = args.test_stride

    # ---------- Training ----------
    train_dataset = PatchDataset(img, train_gt, **hyperparams)
    train_loader = data.DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )

    print(f"Training {args.model} on {len(train_dataset)} samples")
    train(
        model,
        optimizer,
        criterion,
        train_loader,
        hyperparams["epoch"],
        scheduler=scheduler,
        device=device,
    )

    if args.save_checkpoint:
        torch.save(model.state_dict(), os.path.join(results_dir, "checkpoint.pth"))

    # ---------- Inference and evaluation ----------
    logits = predict(model, img, hyperparams, predict_on_gt)
    prediction = np.argmax(logits, axis=-1)

    np.save(os.path.join(results_dir, "logits.npy"), logits)

    results = compute_metrics(
        prediction, test_gt, ignored_labels=ignored_labels, n_classes=len(label_values)
    )
    save_metrics(results, os.path.join(results_dir, "metrics.txt"))
    save_prediction_map(prediction, os.path.join(results_dir, "prediction.png"))

    print(
        f"OA={results['OA']:.4f}  AA={results['AA']:.4f}  Kappa={results['Kappa']:.4f}"
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
