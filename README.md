# Where Should Excitations Meet? Deep Fusion Strategies for Multi-Excitation Hyperspectral Imaging of Ablated Atrial Tissue

Code and experimental configurations for the paper submitted to the **16th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS 2026)**.

> **Status:** The implementation will be released here upon acceptance.

## Overview

This work studies excitation fusion in multi-excitation hyperspectral imaging (ME-HSI) as a controlled architectural choice, applied to the classification of radiofrequency-ablated and unablated bovine atrial tissue.

Two questions are varied independently:

- **Where** should excitation-specific information be combined — at feature level (`E × Φ → fusion → Ψ`) or at logit level (`E × B → fusion`)?
- **How** should it be aggregated — using fixed, globally learned, or sample-dependent operators?

Seven fusion operators (`AVG`, `SUM`, `FC`, `Gate`, `CNN`, `WAVG`, `ATT`) are benchmarked across two backbones (MLP, LeViT) against a decision-level soft majority-voting baseline.

## Planned contents

- Excitation-branch model definitions for both backbones
- Implementations of all seven fusion operators at both fusion stages
- Training and evaluation configurations
- Scripts to reproduce the reported OA, AA, and κ results

## Data

Bovine left atrial tissue imaged under four excitation wavelengths (360, 370, 380, 390 nm) with 31 emission bands (420–720 nm, 10 nm step). Ground truth was derived from TTC histological staining.

## Citation

Citation details will be added upon acceptance.

```bibtex
@inproceedings{TBD,
  title     = {Where Should Excitations Meet? Deep Fusion Strategies for
               Multi-Excitation Hyperspectral Imaging of Ablated Atrial Tissue},
  booktitle = {Workshop on Hyperspectral Imaging and Signal Processing:
               Evolution in Remote Sensing (WHISPERS)},
  year      = {2026}
}
```

## Contact

Questions about the code or data: *(add email once review is complete)*
