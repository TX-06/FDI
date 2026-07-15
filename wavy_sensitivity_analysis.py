#!/usr/bin/env python3
"""连续波浪条纹敏感性实验。

该实验回应“固定少量幅度样本不满足单调性”的审稿风险。它在多个波长下
扫描波浪幅度，报告 FDI 与幅度的 Spearman 相关，并保留真实曲线形态。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from stripe_distortion_analyzer import (
    build_garment_mask,
    compute_circ_gov_global,
    compute_curvature_distortion,
    compute_fdi,
    compute_gov,
    compute_periodicity_fft,
    estimate_dominant_orientation,
    load_and_preprocess,
)


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "test_images" / "wavy_sensitivity"
OUTDIR = ROOT / "output"
CSV_PATH = OUTDIR / "wavy_sensitivity_results.csv"
JSON_PATH = OUTDIR / "wavy_sensitivity_summary.json"
FIG_PATH = OUTDIR / "wavy_sensitivity_curves.png"


def make_vertical_stripes(h: int = 512, w: int = 512, period: int = 20, stripe_width: int = 8) -> np.ndarray:
    image = np.ones((h, w), dtype=np.uint8) * 245
    for x in range(0, w, period):
        image[:, x:x + stripe_width] = 60
    return image


def make_wavy_stripes(amplitude: int, wavelength: int) -> np.ndarray:
    base = make_vertical_stripes()
    out = np.empty_like(base)
    h = base.shape[0]
    for y in range(h):
        shift = int(round(amplitude * np.sin(2.0 * np.pi * y / wavelength)))
        out[y] = np.roll(base[y], shift)
    return out


def save_gray(gray: np.ndarray, path: Path) -> None:
    rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), rgb)


def analyze_fast(path: Path) -> tuple[float, float, float, float]:
    gray, rgb = load_and_preprocess(str(path), blur_ksize=3)
    roi = build_garment_mask(
        rgb,
        gray,
        use_skin_suppression=False,
        use_texture_filter=False,
    )
    gov_map, mag, ang = compute_gov(gray, window_size=7, mag_threshold=15.0, roi_mask=roi)
    dom, _ = estimate_dominant_orientation(ang, mag, mag_threshold=15.0, roi_mask=roi)
    curv_map = compute_curvature_distortion(gray, 15.0, 50.0, 20, roi)
    periodicity, _ = compute_periodicity_fft(gray, roi, dom)
    c_gov, _ = compute_circ_gov_global(ang, mag, roi_mask=roi)
    fdi, _, _, curv_mean, periodicity = compute_fdi(
        gov_map,
        curv_map,
        periodicity,
        w_gov=0.65,
        w_curv=0.35,
        w_period=0.35,
        roi_mask=roi,
    )
    return fdi, c_gov, curv_mean, periodicity


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(exist_ok=True)

    amplitudes = list(range(0, 31, 2))
    wavelengths = [48, 96, 160]
    rows: list[dict[str, float | int | str]] = []

    for wavelength in wavelengths:
        for amplitude in amplitudes:
            image = make_wavy_stripes(amplitude, wavelength)
            path = IMAGE_DIR / f"wavy_amp{amplitude:02d}_lambda{wavelength}.png"
            save_gray(image, path)
            fdi, c_gov, curv, p = analyze_fast(path)
            rows.append(
                {
                    "amplitude_px": amplitude,
                    "wavelength_px": wavelength,
                    "fdi": round(fdi, 4),
                    "c_gov": round(c_gov, 4),
                    "curv": round(curv, 4),
                    "p": round(p, 4),
                    "image": str(path.relative_to(ROOT)),
                }
            )
            print(f"lambda={wavelength}, amp={amplitude}: FDI={fdi:.2f}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"amplitudes": amplitudes, "wavelengths": wavelengths, "groups": {}}
    for wavelength in wavelengths:
        group = [row for row in rows if row["wavelength_px"] == wavelength]
        amp = [float(row["amplitude_px"]) for row in group]
        fdi = [float(row["fdi"]) for row in group]
        rho = float(spearmanr(amp, fdi).statistic)
        summary["groups"][str(wavelength)] = {
            "spearman_amplitude_fdi": round(rho, 4),
            "minimum_fdi": round(min(fdi), 4),
            "maximum_fdi": round(max(fdi), 4),
        }

    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for wavelength in wavelengths:
        group = [row for row in rows if row["wavelength_px"] == wavelength]
        ax.plot(
            [row["amplitude_px"] for row in group],
            [row["fdi"] for row in group],
            marker="o",
            linewidth=2,
            label=f"wavelength {wavelength}px",
        )
    ax.set_xlabel("Sinusoidal amplitude (px)")
    ax.set_ylabel("FDI")
    ax.set_title("Wavy-stripe sensitivity across amplitude and wavelength")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {JSON_PATH}")
    print(f"Saved: {FIG_PATH}")


if __name__ == "__main__":
    main()
