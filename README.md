# Fabric Distortion Index (FDI)

A quantitative metric for evaluating stripe texture fidelity in image-based virtual try-on (VTON) systems. FDI measures the structural degradation of periodic stripe patterns in garments synthesized by neural VTON architectures.

## Overview

FDI integrates four computational modules:
1. **Spatial Defense**: Mask-guided ROI extraction with skin suppression
2. **C-GOV**: Circular Gradient Orientation Variance via phase-doubling circular statistics
3. **CURV**: Menger Curvature for microscopic contour jaggedness
4. **2D-FFT**: Spectral regularity assessment

**Formula**: `FDI = psi * (alpha * C-GOV + beta * CURV) * (2.0 - P)` where alpha=0.65, beta=0.35

## Quick Start

```bash
pip install -r requirements.txt
python batch_analysis_final.py
```

## Citation

Teng, P. "Fabric Distortion Index: A Perceptual Metric for Evaluating Stripe Texture Fidelity in Image-Based Virtual Try-On." Suzhou University of Technology, 2025.
