# -*- coding: utf-8 -*-
"""
Models for multi-excitation hyperspectral imaging (ME-HSI) fusion.

Contents
    1. Single-excitation backbones      MLP, LeViT_FEATURE
    2. Excitation-wise extractors       ExMLP, ExMLPFeature, ExLeViT, ExLeViTFeature
    3. Fusion operators                 AVG, SUM, FC, Gate, CNN, WAVG, ATT
    4. Fusion wrappers                  LateFusion, FeatureLevelFusion
    5. Model registry                   build_model
    6. Training / inference             train, test

Tensor notation
    B: batch size     S: spectral bands (31)     E: excitations (4)
    H, W: patch size  D: fused dimension         C: number of classes

Fusion stages (see paper, Table 2)
    feature level:  E x Phi -> Operator -> Psi       (FeatureLevelFusion)
    logit level:    E x B   -> Operator              (LateFusion)
"""

from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from tqdm import tqdm

from levit import LeViT, Transformer, cast_tuple, Rearrange
from utils import batched, iter_windows, count_windows

# ============================================================
# 1. SINGLE-EXCITATION BACKBONES
# ============================================================


class MLP(nn.Module):
    """
    Single-excitation MLP classifier.

    Input:  (B, S)
    Output: (B, C)
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super().__init__()

        self.dropout = nn.Dropout(p=0.5) if dropout else nn.Identity()

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class LeViT_FEATURE(nn.Module):
    """
    LeViT backbone returning pooled features only (no classification head).

    Input:  (B, S, H, W) or (B, 1, S, H, W)
    Output: (B, dim_last)
    """

    def __init__(
        self,
        *,
        image_size,
        channels,
        dim,
        depth,
        heads,
        mlp_mult,
        stages=3,
        dim_key=32,
        dim_value=64,
        dropout=0.0,
    ):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)
        self.channels = channels

        assert all(
            map(lambda t: len(t) == stages, (dims, depths, layer_heads))
        ), "dimensions, depths, and heads must match number of stages"

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(self.channels, 32, 4, stride=1, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, dims[0], 3, stride=1, padding=1),
        )

        fmap_size = image_size // (2**4)
        layers = []

        for ind, (d, dep, h) in enumerate(zip(dims, depths, layer_heads)):
            is_last = ind == (stages - 1)
            layers.append(
                Transformer(d, fmap_size, dep, h, dim_key, dim_value, mlp_mult, dropout)
            )

            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(
                    Transformer(
                        d,
                        fmap_size,
                        1,
                        h * 2,
                        dim_key,
                        dim_value,
                        mlp_mult=mlp_mult,
                        dropout=dropout,
                        dim_out=next_dim,
                        downsample=True,
                    )
                )
                fmap_size = ceil(fmap_size / 2)

        self.backbone = nn.Sequential(*layers)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), Rearrange("... () () -> ...")
        )

        # Feature dimension after pooling = last stage dim
        self.feature_dim = dims[-1]

    def forward(self, img):
        if img.dim() == 5:
            if img.size(1) != 1:
                raise ValueError(f"Expected shape (B,1,S,H,W), got {img.shape}")
            img = img.squeeze(1)

        elif img.dim() != 4:
            raise ValueError(
                f"Expected shape (B,S,H,W) or (B,1,S,H,W), got {img.shape}"
            )

        x = self.conv_embedding(img)
        x = self.backbone(x)
        x = self.pool(x)

        return x


# ============================================================
# 2. EXCITATION-WISE EXTRACTORS
# ============================================================
# E independent branches, no parameter sharing across excitations.
#     logit level:    ExMLP, ExLeViT                ->  (B, E, C)
#     feature level:  ExMLPFeature, ExLeViTFeature  ->  (B, E, D)
# ============================================================


class ExMLP(nn.Module):
    """
    Excitation-wise MLP classifiers.

    Input:  (B, S, E), (B, 1, S, E) or (B, S, E, H, W)  [center pixel used]
    Output: (B, E, C)
    """

    def __init__(
        self,
        input_channels=31,
        n_classes=2,
        n_excitations=4,
        dropout=False,
        base_model=MLP,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = n_classes
        self.n_excitations = n_excitations

        self.models = nn.ModuleList(
            [
                base_model(
                    input_channels=input_channels,
                    n_classes=n_classes,
                    dropout=dropout,
                )
                for _ in range(n_excitations)
            ]
        )

    def forward(self, x):
        if x.dim() == 5:
            B, S, E, H, W = x.shape
            h = H // 2
            w = W // 2
            x = x[:, :, :, h, w]  # (B, S, E)

        elif x.dim() == 4:
            B, one, S, E = x.shape
            if one != 1:
                raise ValueError(f"Expected shape (B,1,S,E), got {x.shape}")
            x = x.squeeze(1)  # (B, S, E)

        elif x.dim() == 3:
            B, S, E = x.shape

        else:
            raise ValueError(
                f"Expected input shape (B,S,E), (B,1,S,E), or (B,S,E,H,W), got {x.shape}"
            )

        if S != self.input_channels:
            raise ValueError(f"Expected S={self.input_channels}, got S={S}")

        if E != self.n_excitations:
            raise ValueError(f"Expected E={self.n_excitations}, got E={E}")

        outputs = []

        for e in range(E):
            x_e = x[:, :, e]  # (B, S)
            out_e = self.models[e](x_e)  # (B, C)
            outputs.append(out_e.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, E, C)


class ExMLPFeature(nn.Module):
    """
    Excitation-wise MLP feature extractors.

    Input:  (B, S, E), (B, 1, S, E) or (B, S, E, H, W)  [center pixel used]
    Output: (B, E, 2048)
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def __init__(
        self,
        input_channels=31,
        n_excitations=4,
        dropout=False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.n_excitations = n_excitations
        self.feature_dim = 2048

        def make_branch():
            return nn.Sequential(
                nn.Linear(input_channels, 2048),
                nn.ReLU(),
                nn.Dropout(p=0.5) if dropout else nn.Identity(),
                nn.Linear(2048, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5) if dropout else nn.Identity(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Dropout(p=0.5) if dropout else nn.Identity(),
            )

        self.models = nn.ModuleList([make_branch() for _ in range(n_excitations)])

        self.apply(self.weight_init)

    def forward(self, x):
        if x.dim() == 5:
            B, S, E, H, W = x.shape
            h = H // 2
            w = W // 2
            x = x[:, :, :, h, w]  # (B, S, E)

        elif x.dim() == 4:
            B, one, S, E = x.shape
            if one != 1:
                raise ValueError(f"Expected shape (B,1,S,E), got {x.shape}")
            x = x.squeeze(1)  # (B, S, E)

        elif x.dim() == 3:
            B, S, E = x.shape

        else:
            raise ValueError(
                f"Expected (B,S,E), (B,1,S,E), or (B,S,E,H,W), got {x.shape}"
            )

        if S != self.input_channels:
            raise ValueError(f"Expected S={self.input_channels}, got {S}")

        if E != self.n_excitations:
            raise ValueError(f"Expected E={self.n_excitations}, got {E}")

        outputs = []

        for e in range(E):
            x_e = x[:, :, e]  # (B, S)
            feat_e = self.models[e](x_e)  # (B, 2048)
            outputs.append(feat_e.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, E, 2048)


class ExLeViTFeature(nn.Module):
    """
    Excitation-wise LeViT feature extractors.

    Input:  (B, S, E, H, W)
    Output: (B, E, 512)
    """

    def __init__(
        self,
        input_channels=31,
        n_excitations=4,
        image_size=224,
        dim=(256, 384, 512),
        stages=3,
        depth=4,
        heads=(4, 6, 8),
        mlp_mult=2,
        dropout=0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.n_excitations = n_excitations
        self.feature_dim = dim[-1]  # 512

        self.models = nn.ModuleList(
            [
                LeViT_FEATURE(
                    image_size=image_size,
                    channels=input_channels,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    mlp_mult=mlp_mult,
                    stages=stages,
                    dropout=dropout,
                )
                for _ in range(n_excitations)
            ]
        )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected shape (B,S,E,H,W), got {x.shape}")

        B, S, E, H, W = x.shape

        if S != self.input_channels:
            raise ValueError(f"Expected S={self.input_channels}, got {S}")

        if E != self.n_excitations:
            raise ValueError(f"Expected E={self.n_excitations}, got {E}")

        outputs = []

        for e in range(E):
            x_e = x[:, :, e, :, :]  # (B, S, H, W)
            feat_e = self.models[e](x_e)  # (B, 512)
            outputs.append(feat_e.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, E, 512)


class ExLeViT(nn.Module):
    """
    Excitation-wise LeViT classifiers.

    Input:  (B, S, E, H, W)
    Output: (B, E, C)
    """

    def __init__(
        self,
        input_channels=31,
        n_classes=2,
        n_excitations=4,
        image_size=224,
        dim=(256, 384, 512),
        stages=3,
        depth=4,
        heads=(4, 6, 8),
        mlp_mult=2,
        dropout=0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = n_classes
        self.n_excitations = n_excitations

        self.models = nn.ModuleList(
            [
                LeViT(
                    dim=dim,
                    image_size=image_size,
                    stages=stages,
                    depth=depth,
                    heads=heads,
                    mlp_mult=mlp_mult,
                    dropout=dropout,
                    num_classes=n_classes,
                    channels=input_channels,
                )
                for _ in range(n_excitations)
            ]
        )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected shape (B,S,E,H,W), got {x.shape}")

        B, S, E, H, W = x.shape

        if S != self.input_channels:
            raise ValueError(f"Expected S={self.input_channels}, got {S}")

        if E != self.n_excitations:
            raise ValueError(f"Expected E={self.n_excitations}, got {E}")

        outputs = []

        for e in range(E):
            x_e = x[:, :, e, :, :]  # (B, S, H, W)
            out_e = self.models[e](x_e)  # (B, C)

            if isinstance(out_e, tuple):
                out_e = out_e[0]

            outputs.append(out_e.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, E, C)


# ============================================================
# 3. FUSION OPERATORS
# ============================================================
# Every operator maps (B, E, D) -> (B, D).
# Each excitation representation is normalized with LayerNorm before fusion.
#     logit level:    D = C
#     feature level:  D = 2048 (MLP) or 512 (LeViT)
# ============================================================


class AvgFusion(nn.Module):
    """AVG: fixed uniform average over excitations."""

    def __init__(self, use_norm=True, feature_dim=2):
        super().__init__()

        self.norm = nn.LayerNorm(feature_dim) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        return x.mean(dim=1)


class SumFusion(nn.Module):
    """SUM: fixed summation over excitations."""

    def __init__(self, use_norm=True, feature_dim=2):
        super().__init__()

        self.norm = nn.LayerNorm(feature_dim) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        return x.sum(dim=1)


class FCFusion(nn.Module):
    """FC: learned joint linear mixing of all excitation representations."""

    def __init__(self, n_excitations=4, feature_dim=2, use_norm=True):
        super().__init__()
        self.n_excitations = n_excitations
        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim) if use_norm else nn.Identity()
        self.fc = nn.Linear(n_excitations * feature_dim, feature_dim)

        init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected (B,E,D), got {x.shape}")

        B, E, D = x.shape

        if E != self.n_excitations:
            raise ValueError(f"Expected E={self.n_excitations}, got {E}")

        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")

        x = self.norm(x)  # (B,E,D)
        x = x.reshape(B, E * D)  # (B,E*D)
        return self.fc(x)  # (B,D)


class GateFusion(nn.Module):
    """Gate: sample-dependent softmax weights from a shared linear scorer."""

    def __init__(self, feature_dim=2, use_norm=True):
        super().__init__()

        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim) if use_norm else nn.Identity()

        # One scalar gate score per excitation token
        self.gate = nn.Linear(feature_dim, 1)

        init.kaiming_normal_(self.gate.weight)
        if self.gate.bias is not None:
            init.zeros_(self.gate.bias)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected shape (B,E,D), got {x.shape}")

        B, E, D = x.shape

        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")

        x = self.norm(x)  # (B,E,D)
        scores = self.gate(x).squeeze(-1)  # (B,E)
        weights = torch.softmax(scores, dim=1)  # (B,E)

        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B,D)


class CNNFusion(nn.Module):
    """
    CNN: sample-dependent weights from a Conv1D along the ordered excitation axis.

    x -> LayerNorm -> Conv1D(D -> 1, kernel=3) -> softmax over E -> weighted sum
    """

    def __init__(
        self,
        n_excitations=4,
        feature_dim=2,
        kernel_size=3,
        use_norm=True,
    ):
        super().__init__()

        self.n_excitations = n_excitations
        self.feature_dim = feature_dim

        self.norm = nn.LayerNorm(feature_dim) if use_norm else nn.Identity()

        padding = kernel_size // 2

        self.score_conv = nn.Conv1d(
            in_channels=feature_dim,  # D
            out_channels=1,  # one importance score per excitation
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,E,D), got {x.shape}")

        B, E, D = x.shape

        if E != self.n_excitations:
            raise ValueError(f"Expected {self.n_excitations} excitations, got {E}")

        if D != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {D}")

        x = self.norm(x)  # (B,E,D)
        z = x.transpose(1, 2)  # (B,D,E): channels=D, length=E
        scores = self.score_conv(z).squeeze(1)  # (B,E)
        weights = torch.softmax(scores, dim=1)  # (B,E)

        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B,D)


class WeightedAvgFusion(nn.Module):
    """WAVG: one globally learned softmax weight per excitation position."""

    def __init__(
        self,
        n_excitations=4,
        feature_dim=2,
        use_norm=True,
    ):
        super().__init__()

        self.n_excitations = n_excitations
        self.feature_dim = feature_dim

        self.norm = nn.LayerNorm(feature_dim) if use_norm else nn.Identity()

        # softmax(0, ..., 0) = uniform average at initialization
        self.exc_logits = nn.Parameter(torch.zeros(n_excitations))

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B,E,D), got {x.shape}")

        B, E, D = x.shape

        if E != self.n_excitations:
            raise ValueError(f"Expected E={self.n_excitations}, got {E}")

        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, got {D}")

        x = self.norm(x)

        weights = torch.softmax(self.exc_logits, dim=0)  # (E,)

        return (x * weights.view(1, E, 1)).sum(dim=1)  # (B, D)


class AttentionFusion(nn.Module):
    """ATT: a learned query attends over the excitation tokens (multi-head attention)."""

    def __init__(
        self,
        n_excitations=4,
        feature_dim=2,
        num_heads=1,
        use_norm=True,
    ):
        super().__init__()

        self.n_excitations = n_excitations
        self.feature_dim = feature_dim

        self.norm = nn.LayerNorm(feature_dim) if use_norm else nn.Identity()

        self.fusion_query = nn.Parameter(torch.zeros(1, 1, feature_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,E,D), got {x.shape}")

        B, E, D = x.shape

        if E != self.n_excitations:
            raise ValueError(f"Expected {self.n_excitations} excitations, got {E}")

        if D != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {D}")

        x = self.norm(x)  # (B,E,D)
        q = self.fusion_query.expand(B, -1, -1)  # (B,1,D)

        fused, _ = self.attn(query=q, key=x, value=x)

        return fused.squeeze(1)  # (B,D)


# ============================================================
# 4. FUSION WRAPPERS
# ============================================================


class LateFusion(nn.Module):
    """
    Logit-level fusion:  E x B -> Operator

    x -> extractor -> (B, E, C) -> fuser -> (B, C)
    """

    def __init__(self, extractor, fuser):
        super().__init__()

        self.extractor = extractor
        self.fuser = fuser

    def forward(self, x):
        return self.fuser(self.extractor(x))


class FeatureLevelFusion(nn.Module):
    """
    Feature-level fusion:  E x Phi -> Operator -> Psi

    x -> extractor -> (B, E, D) -> fuser -> (B, D) -> classifier -> (B, C)
    """

    def __init__(self, extractor, fuser, classifier):
        super().__init__()
        self.extractor = extractor
        self.fuser = fuser
        self.classifier = classifier

    def forward(self, x):
        x = self.extractor(x)  # (B,E,D)
        x = self.fuser(x)  # (B,D)
        return self.classifier(x)  # (B,C)


# ============================================================
# 5. MODEL REGISTRY
# ============================================================
# Fusion models:       {levit|mlp}_{feat|logit}_{avg|sum|fc|gate|cnn|wavg|att}
# Single-excitation:   levit_single, mlp_single   (inputs to Soft MV)
# ============================================================

BACKBONES = ("levit", "mlp")
STAGES = ("feat", "logit")
OPERATORS = ("avg", "sum", "fc", "gate", "cnn", "wavg", "att")

FUSION_MODELS = [f"{b}_{s}_{o}" for b in BACKBONES for s in STAGES for o in OPERATORS]
SINGLE_MODELS = ["levit_single", "mlp_single"]
AVAILABLE_MODELS = FUSION_MODELS + SINGLE_MODELS

# LeViT configuration shared by all LeViT-based models
LEVIT_KWARGS = dict(
    image_size=224,  # sets fmap_size = 14, matching 15x15 input patches
    dim=(256, 384, 512),
    stages=3,
    depth=4,
    heads=(4, 6, 8),
    mlp_mult=2,
    dropout=0.1,
)

# Default spatial patch size per backbone
PATCH_SIZE = {"levit": 15, "mlp": 1}

# Attention heads of the ATT operator per fusion stage
ATT_HEADS = {"feat": 4, "logit": 1}


def build_fuser(operator, n_excitations, feature_dim, num_heads=1):
    """Return the fusion operator mapping (B, E, D) -> (B, D)."""
    if operator == "avg":
        return AvgFusion(feature_dim=feature_dim, use_norm=True)
    if operator == "sum":
        return SumFusion(feature_dim=feature_dim, use_norm=True)
    if operator == "fc":
        return FCFusion(
            n_excitations=n_excitations, feature_dim=feature_dim, use_norm=True
        )
    if operator == "gate":
        return GateFusion(feature_dim=feature_dim, use_norm=True)
    if operator == "cnn":
        return CNNFusion(
            n_excitations=n_excitations,
            feature_dim=feature_dim,
            kernel_size=3,
            use_norm=True,
        )
    if operator == "wavg":
        return WeightedAvgFusion(
            n_excitations=n_excitations, feature_dim=feature_dim, use_norm=True
        )
    if operator == "att":
        return AttentionFusion(
            n_excitations=n_excitations,
            feature_dim=feature_dim,
            num_heads=num_heads,
            use_norm=True,
        )
    raise KeyError(f"Unknown fusion operator: {operator}")


def build_fusion_model(
    backbone, stage, operator, n_bands, n_classes, n_excitations, dropout=False
):
    """Build a feature-level or logit-level excitation-fusion model."""
    if stage == "feat":
        if backbone == "mlp":
            extractor = ExMLPFeature(
                input_channels=n_bands, n_excitations=n_excitations, dropout=dropout
            )
        else:
            extractor = ExLeViTFeature(
                input_channels=n_bands, n_excitations=n_excitations, **LEVIT_KWARGS
            )

        feature_dim = extractor.feature_dim
        fuser = build_fuser(
            operator, n_excitations, feature_dim, num_heads=ATT_HEADS["feat"]
        )
        classifier = nn.Linear(feature_dim, n_classes)
        return FeatureLevelFusion(
            extractor=extractor, fuser=fuser, classifier=classifier
        )

    if stage == "logit":
        if backbone == "mlp":
            extractor = ExMLP(
                input_channels=n_bands,
                n_classes=n_classes,
                n_excitations=n_excitations,
                dropout=dropout,
                base_model=MLP,
            )
        else:
            extractor = ExLeViT(
                input_channels=n_bands,
                n_classes=n_classes,
                n_excitations=n_excitations,
                **LEVIT_KWARGS,
            )

        fuser = build_fuser(
            operator, n_excitations, n_classes, num_heads=ATT_HEADS["logit"]
        )
        return LateFusion(extractor=extractor, fuser=fuser)

    raise KeyError(f"Unknown fusion stage: {stage}")


def build_model(
    name,
    n_bands,
    n_classes,
    n_excitations,
    ignored_labels,
    device,
    epoch=100,
    learning_rate=1e-4,
    batch_size=100,
    patch_size=None,
):
    """
    Instantiate a model with its optimizer, loss, scheduler and hyperparameters.

    Args:
        name:           one of AVAILABLE_MODELS
        n_bands:        spectral bands per excitation (31)
        n_classes:      number of labels, including the ignored label 0
        n_excitations:  number of excitations (4); unused for *_single models
        ignored_labels: labels excluded from the loss
        device:         torch device
        epoch, learning_rate, batch_size: training settings
        patch_size:     spatial patch size (default: 15 for LeViT, 1 for MLP)

    Returns:
        model, optimizer, criterion, scheduler, hyperparams
    """
    if name in FUSION_MODELS:
        backbone, stage, operator = name.split("_")
        model = build_fusion_model(
            backbone, stage, operator, n_bands, n_classes, n_excitations
        )

    elif name == "mlp_single":
        backbone = "mlp"
        model = MLP(n_bands, n_classes)

    elif name == "levit_single":
        backbone = "levit"
        model = LeViT(num_classes=n_classes, channels=n_bands, **LEVIT_KWARGS)

    else:
        raise KeyError(f"{name} model is unknown. Available: {AVAILABLE_MODELS}")

    model = model.to(device)

    # Ignored labels receive (almost) zero weight in the loss
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(ignored_labels)] = 1.0e-30
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=epoch // 4
    )

    hyperparams = dict(
        patch_size=patch_size if patch_size is not None else PATCH_SIZE[backbone],
        epoch=epoch,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_classes=n_classes,
        ignored_labels=list(ignored_labels),
        device=device,
    )

    return model, optimizer, criterion, scheduler, hyperparams


# ============================================================
# 6. TRAINING / INFERENCE
# ============================================================


def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    device=torch.device("cpu"),
    display_iter=100,
):
    """
    Train a network for a number of epochs.
    The learning-rate scheduler (ReduceLROnPlateau) monitors the mean training loss.
    """
    net.to(device)

    recent_losses = []
    iter_ = 1

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        net.train()
        avg_loss = 0.0

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            loss = criterion(net(data), target)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            recent_losses = (recent_losses + [loss.item()])[-100:]

            if display_iter and iter_ % display_iter == 0:
                tqdm.write(
                    "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        e,
                        epoch,
                        batch_idx * len(data),
                        len(data) * len(data_loader),
                        100.0 * batch_idx / len(data_loader),
                        np.mean(recent_losses),
                    )
                )
            iter_ += 1

        avg_loss /= len(data_loader)

        if scheduler is not None:
            scheduler.step(avg_loss)


def predict(net, img, hyperparams, gt):
    """
    Sliding-window inference on a full image; each window predicts its center pixel.

    Args:
        net:         trained model
        img:         (H, W, S, E) multi-excitation or (H, W, S) single-excitation image
        hyperparams: dict with patch_size, batch_size, device, n_classes, test_stride
        gt:          (H, W) tissue mask; only pixels with gt != 0 are predicted

    Returns:
        (H, W, C) raw network outputs (logits); zeros where nothing was predicted
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    logits = np.zeros(img.shape[:2] + (hyperparams["n_classes"],))
    iterations = count_windows(img, **kwargs) // batch_size

    with torch.no_grad():
        for batch in tqdm(
            batched(batch_size, iter_windows(img, **kwargs)),
            total=iterations,
            desc="Inference on the image",
        ):
            data, centers = [], []

            for window, x, y, w, h in batch:
                cx, cy = x + w // 2, y + h // 2
                if gt[cx, cy] != 0:
                    data.append(window)
                    centers.append((cx, cy))

            if len(data) == 0:
                continue

            data = np.stack(data, axis=0)  # (N, h, w, S[, E])

            if patch_size == 1:
                data = data[:, 0, 0]  # (N, S) or (N, S, E)
            elif data.ndim == 4:
                data = data.transpose(0, 3, 1, 2)[:, None]  # (N, 1, S, h, w)
            else:
                data = data.transpose(0, 3, 4, 1, 2)  # (N, S, E, h, w)

            output = net(
                torch.from_numpy(np.ascontiguousarray(data)).float().to(device)
            )
            output = output.cpu().numpy()

            for (cx, cy), out in zip(centers, output):
                logits[cx, cy] += out

    return logits
