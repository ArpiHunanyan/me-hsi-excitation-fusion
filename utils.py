# -*- coding: utf-8 -*-
"""Helpers for inference, evaluation and saving results."""

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Prediction-map colors: 0 background (black), 1 ablated (red), 2 unablated (green)
PALETTE = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}


def select_device(ordinal):
    """CPU if ordinal < 0, otherwise the requested CUDA device when available."""
    if ordinal < 0:
        print("Computation on CPU")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print(f"Computation on CUDA GPU device {ordinal}")
        return torch.device(f"cuda:{ordinal}")
    print("CUDA was requested but is not available; computation will run on CPU.")
    return torch.device("cpu")


# ============================================================
# Sliding-window inference
# ============================================================


def iter_windows(image, step=1, window_size=(15, 15)):
    """
    Slide a window over an image.

    Yields:
        (window, x, y, w, h): the window data and its top-left corner and size
    """
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step

    for x in range(0, W - w + offset_w + 1, step):
        x = min(x, W - w)
        for y in range(0, H - h + offset_h + 1, step):
            y = min(y, H - h)
            yield image[x : x + w, y : y + h], x, y, w, h


def count_windows(image, step=1, window_size=(15, 15)):
    """Number of windows produced by iter_windows."""
    w, h = window_size
    W, H = image.shape[:2]
    n_x = len(range(0, W - w + (W - w) % step + 1, step))
    n_y = len(range(0, H - h + (H - h) % step + 1, step))
    return n_x * n_y


def batched(n, iterable):
    """Yield successive chunks of n elements from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# ============================================================
# Evaluation
# ============================================================


def compute_metrics(prediction, target, ignored_labels, n_classes):
    """
    Overall accuracy (OA), average accuracy (AA) and Cohen's kappa.

    Pixels whose target is in ignored_labels are excluded. AA averages the
    per-class accuracy over the evaluated classes (labels 1 .. n_classes - 1).
    """
    keep = ~np.isin(target, ignored_labels)
    target = target[keep].astype(int)
    prediction = prediction[keep].astype(int)

    cm = np.bincount(
        n_classes * target + prediction, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)

    total = cm.sum()
    oa = np.trace(cm) / total

    classes = [c for c in range(n_classes) if c not in ignored_labels]
    aa = np.mean([cm[c, c] / cm[c].sum() for c in classes])

    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / total**2
    kappa = (oa - pe) / (1 - pe)

    return {"OA": oa, "AA": aa, "Kappa": kappa, "Confusion matrix": cm}


# ============================================================
# Saving
# ============================================================


def save_metrics(results, path):
    """Write OA, AA and kappa to a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"OA={results['OA']:.6f}\n")
        f.write(f"AA={results['AA']:.6f}\n")
        f.write(f"Kappa={results['Kappa']:.6f}\n")


def save_prediction_map(labels, path, palette=PALETTE):
    """Save a 2D label map as a color PNG (one image pixel per data pixel)."""
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for label, color in palette.items():
        rgb[labels == label] = color
    plt.imsave(path, rgb)
