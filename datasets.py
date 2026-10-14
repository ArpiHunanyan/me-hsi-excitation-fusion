# -*- coding: utf-8 -*-
"""
Data loading for multi-excitation hyperspectral imaging (ME-HSI).

Two image types are supported:
    multi-excitation (4D):   (H, W, S, E)   e.g. 241127_1_norm_696x520x31x4_4d
    single excitation (3D):  (H, W, S)      e.g. 241127_1_norm_696x520x31_2

Expected layout (see README):
    <data_root>/4D/<dataset>.mat           multi-excitation cubes
    <data_root>/Separate/<dataset>.mat     single-excitation cubes
    <gt_root>/gt_circle/<gt_train>.mat     training clicks (one file per split)
    <gt_root>/gt_test/<id>_gt_test.mat     test labels
    <gt_root>/gt/<id>_gt.mat               tissue mask (pixels to predict on)

Labels:
    0 = background (ignored)
    1 = ablated
    2 = unablated
    3 = unsure (only in the tissue mask; excluded from training and testing,
        but predicted for qualitative inspection)
"""

import os

import numpy as np
import torch
import torch.utils.data
from scipy.io import loadmat

LABEL_VALUES = ["background", "ablated", "unablated"]
IGNORED_LABELS = [0]


# ============================================================
# Loading
# ============================================================


def _load_array(path, key=None):
    """Load one array from a .mat file (by key, or the first non-metadata variable)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    mat = loadmat(path)
    if key is not None:
        return mat[key]

    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise KeyError(f"No data variable found in {path}")
    return mat[keys[0]]


def _parse_gt_name(gt_train):
    """
    Split a training-GT name into its image id and GT folder.

        241127_1_696x520_gt_circle_1_percentage_rcount5_split0
        -> base_id = '241127_1_696x520', folder = 'gt_circle'
    """
    parts = gt_train.split("_")
    if "gt" not in parts:
        raise ValueError(f"Cannot parse GT name '{gt_train}': missing '_gt_' token")

    gt_idx = parts.index("gt")
    base_id = "_".join(parts[:gt_idx])
    folder = f"{parts[gt_idx]}_{parts[gt_idx + 1]}"
    return base_id, folder


def load_ground_truth(gt_train, gt_root):
    """
    Load the ground-truth maps of one click split.

    Returns:
        train_gt:       (H, W) training labels (simulated clicks)
        test_gt:        (H, W) test labels, training pixels removed
        predict_on_gt:  (H, W) tissue mask; inference is run where != 0
    """
    base_id, train_folder = _parse_gt_name(gt_train)

    train_gt = _load_array(
        os.path.join(gt_root, train_folder, f"{gt_train}.mat"), key=gt_train
    )

    test_name = f"{base_id}_gt_test"
    test_gt = _load_array(
        os.path.join(gt_root, "gt_test", f"{test_name}.mat"), key=test_name
    )
    test_gt = np.where(train_gt > 0, 0, test_gt)  # training pixels are never tested

    mask_name = f"{base_id}_gt"
    predict_on_gt = _load_array(
        os.path.join(gt_root, "gt", f"{mask_name}.mat"), key=mask_name
    )

    return train_gt, test_gt, predict_on_gt


def load_dataset(dataset_name, gt_train, data_root, gt_root):
    """
    Load an ME-HSI image and its ground-truth maps.

    Args:
        dataset_name: image name; names ending in '_4d' are multi-excitation
        gt_train:     training-GT name (simulated clicks for one split)
        data_root:    folder containing '4D/' and 'Separate/'
        gt_root:      folder containing 'gt_circle/', 'gt_test/' and 'gt/'

    Returns:
        img:            (H, W, S, E) or (H, W, S) float32, min-max scaled to [0, 1]
        train_gt:       (H, W) training labels
        test_gt:        (H, W) test labels, training pixels removed
        predict_on_gt:  (H, W) tissue mask; inference is run where != 0
        label_values:   class names
        ignored_labels: labels excluded from training and evaluation
    """
    is_4d = dataset_name.endswith("_4d")
    subfolder = "4D" if is_4d else "Separate"
    img = _load_array(os.path.join(data_root, subfolder, f"{dataset_name}.mat"))

    expected_ndim = 4 if is_4d else 3
    if img.ndim != expected_ndim:
        raise ValueError(
            f"{dataset_name}: expected a {expected_ndim}D image, got shape {img.shape}"
        )

    train_gt, test_gt, predict_on_gt = load_ground_truth(gt_train, gt_root)

    # Global min-max scaling over the whole image (applied on top of the
    # pixel-wise normalization already present in the *_norm_* files)
    img = np.asarray(img, dtype="float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return (
        img,
        train_gt,
        test_gt,
        predict_on_gt,
        list(LABEL_VALUES),
        list(IGNORED_LABELS),
    )


# ============================================================
# PyTorch dataset
# ============================================================


class PatchDataset(torch.utils.data.Dataset):
    """
    Patch dataset over the labelled pixels of one hyperspectral image.

    Sample shapes (before batching):
        4D image, patch > 1:   (S, E, H, W)
        4D image, patch = 1:   (S, E)
        3D image, patch > 1:   (1, S, H, W)
        3D image, patch = 1:   (S,)

    Each sample is labelled by its center pixel. Pixels closer than
    patch_size // 2 to the image border are skipped.
    """

    def __init__(self, data, gt, **hyperparams):
        super().__init__()
        self.data = data
        self.label = gt
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.is_4d = data.ndim == 4

        mask = np.ones_like(gt)
        for label in self.ignored_labels:
            mask[gt == label] = 0

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if p < x < data.shape[0] - p and p < y < data.shape[1] - p
            ]
        )
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.is_4d:
            # (H, W, S, E) -> (S, E, H, W)
            data = np.asarray(data.transpose((2, 3, 0, 1)), dtype="float32")
        else:
            # (H, W, S) -> (S, H, W)
            data = np.asarray(data.transpose((2, 0, 1)), dtype="float32")

        data = torch.from_numpy(np.copy(data))
        label = torch.from_numpy(np.asarray(label, dtype="int64"))

        if self.patch_size == 1:
            # Single spectrum: drop spatial dimensions
            data = data[..., 0, 0]
            label = label[0, 0]
        else:
            label = label[
                self.patch_size // 2, self.patch_size // 2
            ]  # center-pixel label
            if not self.is_4d:
                data = data.unsqueeze(0)  # (1, S, H, W)

        return data, label
