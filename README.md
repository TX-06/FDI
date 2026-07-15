# Fabric Distortion Index (FDI)

This repository contains the public code and non-restricted data records for the Fabric Distortion Index (FDI), a textile-structure descriptor for local stripe-geometry consistency in digital garment try-on outputs.

FDI is designed for stripe-dominant garments. It is not a universal virtual try-on quality score, and it is not intended to measure face quality, body realism, color accuracy, blur, transparency, or background quality.

## Method Summary

FDI combines three interpretable cues:

- `C-GOV`: circular gradient-orientation variance inside the garment region.
- `CURV`: stride-based Menger curvature for local stripe-contour instability.
- `P`: normalized FFT periodicity evidence used as a regularization factor, not as an independent distortion penalty.

The manuscript uses:

```text
FDI = min{100, psi * (alpha * C-GOV + beta * CURV) * (1 - gamma * P)}
```

Default parameters:

```text
alpha = 0.65
beta  = 0.35
gamma = 0.35
psi   = 220
```

The provisional reporting bands are:

```text
LR: FDI < 15
IS: 15 <= FDI < 30
VI: 30 <= FDI < 45
HR: FDI >= 45
```

`SRI = 100 * P` may be reported as a complementary stripe-repeat descriptor.

## Repository Contents

```text
FDI_repo/
├── stripe_distortion_analyzer.py      # Core FDI implementation
├── batch_analysis_final.py            # Batch analysis for local test images
├── spectral_periodicity_analysis.py   # FFT periodicity sensitivity analysis
├── rotation_robustness_analysis.py    # Regular-stripe rotation sanity check
├── wavy_sensitivity_analysis.py       # Wavelength/amplitude sensitivity analysis
├── real_vton_public_stripe_analysis.py# Public VTON numeric FDI analysis
├── iqa_comparison_analysis.py         # General IQA comparison scripts
├── data/                              # Public numeric records and source inventory
├── figures/                           # Public derived plots without third-party image crops
├── test_images/                       # Programmatically generated test images
├── requirements.txt
└── requirements-iqa.txt
```

## Public Data Policy

This repository does not redistribute third-party public VTON image crops or human-participant data.

The `data/public_image_source_inventory.*` files list the public image sources used for manuscript stress testing, including source URLs and permission status notes. The image crops themselves are not included because their reuse depends on the original source licences and journal permission requirements.

The `data/public_vton_fdi_results.*` files provide the numeric FDI outputs for the public VTON stress-test set. Local crop paths have been removed and replaced with `not_released_due_to_third_party_permissions`.

No observer ratings, human-subject records, or perception-study data are included in this repository.

## Key Data Files

- `data/public_vton_fdi_results.csv`: numeric FDI records for the public VTON stress-test samples.
- `data/public_image_source_inventory.csv`: source and permission-tracking inventory for public VTON images.
- `data/rotation_robustness_results.csv`: regular-stripe rotation sanity-check results.
- `data/wavy_sensitivity_results.csv`: sinusoidal waviness sensitivity results.
- `data/spectral_periodicity_results.json`: FFT periodicity branch results.
- `data/vton_simulated_distortion_results.json`: simulated VTON failure-mode results.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python batch_analysis_final.py
python spectral_periodicity_analysis.py
python rotation_robustness_analysis.py
python wavy_sensitivity_analysis.py
```

Some optional IQA comparisons require additional packages listed in `requirements-iqa.txt`.

## Citation

```bibtex
@article{teng2026fdi,
  author  = {Teng Peilin and Wu Shigang},
  title   = {Fabric Distortion Index: A Textile-Structure Descriptor for Local Stripe-Geometry Analysis in Digital Garment Try-On},
  journal = {Submitted},
  year    = {2026}
}
```
