# Fabric Distortion Index (FDI)

A quantitative metric for evaluating stripe texture fidelity in image-based virtual try-on (VTON) systems. FDI measures the structural degradation of periodic stripe patterns in garments synthesized by neural VTON architectures.

**Paper submitted to Q3 SCI journal.** This repository contains the complete experimental code, test dataset, and analysis results accompanying the manuscript.

## Overview

FDI integrates three complementary computational modules, preceded by a Spatial Defense preprocessing step:

**Preprocessing: Spatial Defense** — Mask-guided ROI extraction with skin suppression and texture density validation

1. **C-GOV** — Circular Gradient Orientation Variance via phase-doubling circular statistics
2. **CURV** — Menger Curvature for microscopic contour jaggedness
3. **Periodicity (2D-FFT)** — Spectral regularity assessment as a structural discount factor

**Ablation Framework** — Separate component-wise contribution analysis (α/β weight grid search) and Spatial Defense validation

**Formula**: `FDI = ψ · (α · C-GOV + β · CURV) · (2.0 − P)`
**Calibrated scale**: `FDI < 15` = Excellent | `15–30` = Acceptable | `FDI ≥ 30` = Structural Violation

## Repository Structure

```
FDI_repo/
├── stripe_distortion_analyzer.py     # Core FDI algorithm (sRGB pipeline)
├── batch_analysis_final.py           # Batch analysis over test_images/
├── grid_search_alpha.py              # α/β weight grid search (0.05–0.95)
├── requirements.txt                  # Python dependencies
├── test_images/                      # 53 test images
│   ├── synth_*.png                   # Synthetic stripe patterns
│   ├── pexels_*.jpg / unsplash_*.jpg # Real garment photos
│   ├── vton_*.png                    # Simulated VTON distortions (10 types)
│   └── ref_*.png                     # Reference images for full-reference metrics
├── output/                           # Results & visualizations
│   ├── FDI_alpha_grid_search.png     # Optimal α/β = 0.65/0.35 heatmap
│   ├── fdi_ablation_chart.png        # Ablation study results
│   ├── fdi_vs_ssim_psnr_comparison.png # Comparison with traditional IQAs
│   └── vton_fdi_results.json         # FDI on 9 simulated VTON distortions
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python batch_analysis_final.py       # Analyze all test images
python grid_search_alpha.py          # Reproduce α/β grid search
```

### Example Output (VTON Simulation)

```
Image                      FDI   C-GOV    CURV    P      Verdict
───────────────────────────────────────────────────────────────
Perfect Vertical           0.0   0.0000   0.0000  1.000  Excellent
Torso Drape (VTON)        15.7   0.0200   0.6800  1.000  Good
Extreme AI Degradation    39.3   0.4828   0.8288  1.000  Severe
Texture Blur (VTON)        0.0   0.0000   0.0000  1.000  Excellent*
```

*Note: Texture Blur and Ghost Artifact yield FDI = 0.0 due to edge-gradient blind spot; see paper for discussion.*

## Dataset

| Category | Count | Source | Purpose |
|----------|-------|--------|---------|
| Synthetic stripes | 15 | Programmatic generation | FDI calibration & noise/JPEG sensitivity |
| Real garment photos | 10 | Unsplash / Pexels (CC0) | Real-world validation |
| VTON simulation | 10 | Distortion + component analysis | Failure-mode coverage |

**Total test images**: 54 (includes references, ablation, and grid search images)

## Results Summary

| Finding | Detail |
|---------|--------|
| 1 | GCC and RMSE are unreliable for stripe fidelity evaluation (R² < 0.3 with FDI) |
| 2 | α = 0.65, β = 0.35 yields maximal ranking discrimination |
| 3 | FDI achieves clear separation across all distortion severity levels |

## Citation

```bibtex
@article{teng2025fdi,
  author    = {Teng, Peilin},
  title     = {Fabric Distortion Index: A Perceptual Metric for Evaluating Stripe Texture Fidelity in Image-Based Virtual Try-On},
  journal   = {Submitted to SCI Q3 Journal},
  year      = {2025},
  institution = {Suzhou University of Technology}
}
```

---
*Correspondence: tengpeilin@stu.szit.edu.cn*
