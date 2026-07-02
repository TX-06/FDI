# Fabric Distortion Index (FDI)

A stripe-specific structural metric for evaluating texture fidelity in image-based virtual try-on (VTON) outputs. FDI measures geometric degradation in periodic stripe garments, including orientation disorder, local contour roughness, and loss of global periodic structure.

**Paper-ready Q3 SCI package.** This repository contains the core FDI implementation, reproducible test images, analysis scripts, key output artifacts, and the latest reviewed manuscript draft.

## Overview

FDI integrates three complementary computational modules, preceded by a Spatial Defense preprocessing step:

**Preprocessing: Spatial Defense** — Mask-guided ROI extraction with skin suppression and texture density validation

1. **C-GOV** — ROI-level Circular Gradient Orientation Variance used in the FDI formula
2. **CURV** — Menger Curvature for microscopic contour jaggedness
3. **Periodicity (2D-FFT)** — Spectral regularity assessment as a structural discount factor

**Ablation Framework** — Separate component-wise contribution analysis (α/β weight grid search) and Spatial Defense validation

**Formula**: `FDI = ψ · (α · C-GOV + β · CURV) · (1 − γ · P)`

Here, `C-GOV` denotes the ROI-level aggregate GOV value used by the scoring function. The console also prints a global circular-orientation diagnostic, but that diagnostic is not substituted into the final FDI formula.

Default parameters: `α = 0.65`, `β = 0.35`, `γ = 0.35`, `ψ = 100`.

**Calibrated scale**: `FDI < 15` = Excellent | `15–30` = Acceptable | `FDI ≥ 30` = Structural Violation

## Repository Structure

```
FDI_repo/
├── stripe_distortion_analyzer.py     # Core FDI algorithm (sRGB pipeline)
├── batch_analysis_final.py           # Batch analysis over test_images/
├── grid_search_alpha.py              # α/β weight grid search (0.05–0.95)
├── real_vton_stableviton_analysis.py # Real StableVITON stripe-output evaluation
├── requirements.txt                  # Python dependencies
├── docs/
│   └── FDI_SCI_Paper_Draft_Q3_best_reviewed.docx
├── test_images/
│   ├── synth_*.png                   # Synthetic stripe patterns
│   ├── pexels_*.jpg / unsplash_*.jpg # Real garment photos
│   ├── vton_*.png                    # Simulated VTON distortions
│   ├── real_vton_stableviton/        # Official StableVITON stripe examples
│   └── ref_*.png                     # Reference images for full-reference metrics
├── output/                           # Results & visualizations
│   ├── FDI_alpha_grid_search.png     # Optimal α/β = 0.65/0.35 heatmap
│   ├── fdi_ablation_chart.png        # Ablation study results
│   ├── fdi_vs_ssim_psnr_comparison.png # Comparison with traditional IQAs
│   ├── stableviton_real_vton_stripe_panel.png
│   ├── stableviton_real_vton_results.json
│   └── vton_fdi_results.json         # FDI on 9 simulated VTON distortions
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python batch_analysis_final.py       # Analyze all test images
python grid_search_alpha.py          # Reproduce α/β grid search
python real_vton_stableviton_analysis.py
```

### Example Output (VTON Simulation)

```
Image                      FDI   C-GOV    CURV    P      Verdict
───────────────────────────────────────────────────────────────
Perfect Vertical           0.0   0.0000   0.0000  1.000  Excellent
Torso Drape (VTON)        15.7   0.0200   0.6800  1.000  Acceptable
Extreme AI Degradation    39.3   0.4828   0.8288  1.000  Violation
Texture Blur (VTON)        0.0   0.0000   0.0000  1.000  Known blind spot*
```

*Texture Blur and Ghost Artifact yield FDI = 0.0 because FDI is a stripe-geometry metric, not a blur or translucency detector. They are intentionally marked as known blind spots in the manuscript.*

### Real StableVITON Outputs

The reviewed manuscript adds a real-output check using publicly released examples from the official StableVITON project page. A single black-and-white horizontal stripe top is evaluated across five target people:

| Sample | FDI | Category |
|--------|----:|----------|
| Runway female | 24.1 | Acceptable |
| Stage male | 25.6 | Acceptable |
| Street male | 26.0 | Acceptable |
| Event female | 27.2 | Acceptable |
| Red-carpet male | 28.1 | Acceptable |
| Mean ± SD | 26.2 ± 1.37 | Acceptable |

## Dataset

| Category | Count | Source | Purpose |
|----------|-------|--------|---------|
| Synthetic stripes | 15 | Programmatic generation | FDI calibration & noise/JPEG sensitivity |
| Real garment photos | 5 | Unsplash / Pexels | Real-world validation |
| Simulated VTON distortion | 9 | Distortion + component analysis | Failure-mode coverage |
| Real StableVITON outputs | 5 | Official StableVITON project page | Public real-output validation |

The manuscript reports 30 evaluated configurations. The repository also keeps auxiliary reference and stress-test images used during calibration and figure generation.

## Results Summary

| Finding | Detail |
|---------|--------|
| 1 | Generic PSNR/SSIM trends do not reliably reflect stripe-structure integrity |
| 2 | α = 0.65, β = 0.35, γ = 0.35 provides a stable operating point |
| 3 | Official StableVITON stripe outputs fall in the acceptable band (24.1–28.1) |
| 4 | Blur and ghosting are known blind spots and should be handled with complementary IQA metrics |

## Citation

```bibtex
@article{teng2026fdi,
  author    = {Teng, Peilin},
  title     = {Fabric Distortion Index: A Stripe-Specific Structural Metric for Evaluating Texture Fidelity in Image-Based Virtual Try-On},
  journal   = {Submitted to SCI Q3 Journal},
  year      = {2026},
  institution = {Suzhou University of Technology}
}
```

---
*Correspondence: tengpeilin@stu.szit.edu.cn*
