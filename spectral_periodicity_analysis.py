#!/usr/bin/env python3
"""
2D-FFT 周期性正则敏感性实验。

目的：
验证 FDI 中的 P 项不是固定常数。实验构造从稳定周期条纹到周期崩塌的
多类样本，比较含 FFT 正则与不含 FFT 正则的 FDI 差异。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stripe_distortion_analyzer import analyze_stripe_distortion


ROOT = Path(__file__).resolve().parent
IMGDIR = ROOT / "test_images" / "spectral_periodicity"
OUTDIR = ROOT / "output"
IMGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(exist_ok=True)


def save_gray(gray: np.ndarray, name: str) -> Path:
    path = IMGDIR / name
    rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), rgb)
    return path


def make_vertical_stripes(h: int = 512, w: int = 512, period: int = 20, stripe_width: int = 8) -> np.ndarray:
    img = np.ones((h, w), dtype=np.uint8) * 245
    for x in range(0, w, period):
        img[:, x:x + stripe_width] = 60
    return img


def generate_samples() -> list[tuple[str, str, Path]]:
    h = w = 512
    rng = np.random.default_rng(2026)
    samples: list[tuple[str, str, Path]] = []

    perfect = make_vertical_stripes(h, w)
    samples.append(("Perfect periodic stripes", "稳定周期条纹", save_gray(perfect, "spectral_perfect_periodic.png")))

    wavy = perfect.copy()
    for y in range(0, h, 8):
        shift = int(14 * np.sin(y / 42.0))
        wavy[y:y + 8] = np.roll(wavy[y:y + 8], shift, axis=1)
    samples.append(("Wavy but periodic stripes", "自然波浪但周期仍稳定", save_gray(wavy, "spectral_wavy_periodic.png")))

    nonuniform = np.ones((h, w), dtype=np.uint8) * 245
    x = 0
    while x < w:
        stripe_width = int(rng.integers(4, 12))
        gap = int(rng.integers(8, 26))
        nonuniform[:, x:min(w, x + stripe_width)] = 60
        x += stripe_width + gap
    samples.append(("Non-uniform stripe spacing", "非均匀条纹间距", save_gray(nonuniform, "spectral_nonuniform_spacing.png")))

    wavy_nonuniform = nonuniform.copy()
    for y in range(0, h, 8):
        shift = int(16 * np.sin(y / 36.0))
        wavy_nonuniform[y:y + 8] = np.roll(wavy_nonuniform[y:y + 8], shift, axis=1)
    samples.append(("Wavy non-uniform stripes", "非均匀间距叠加波浪形变", save_gray(wavy_nonuniform, "spectral_wavy_nonuniform.png")))

    drift = np.ones((h, w), dtype=np.uint8) * 245
    for y in range(h):
        phase = (y // 22) * 7
        row = np.ones(w, dtype=np.uint8) * 245
        for x in range(0, w, 37):
            start = (x + phase) % w
            row[start:min(w, start + 9)] = 60
        drift[y] = row
    samples.append(("Phase-drift stripe field", "相位漂移条纹场", save_gray(drift, "spectral_phase_drift.png")))

    random_tex = rng.integers(0, 256, (h, w), dtype=np.uint8)
    samples.append(("Random non-periodic texture", "随机非周期纹理", save_gray(random_tex, "spectral_random_texture.png")))

    return samples


def main() -> None:
    samples = generate_samples()
    results = []

    for english_name, chinese_name, path in samples:
        fdi, c_gov, curv, periodicity = analyze_stripe_distortion(
            image_path=str(path),
            output_dir=str(OUTDIR),
            blur_ksize=3,
            gov_window_size=7,
            mag_threshold=15.0,
            canny_low=15.0,
            canny_high=50.0,
            min_contour_len=20,
            w_gov=0.65,
            w_curv=0.35,
            use_skin_suppression=False,
            use_texture_filter=False,
        )
        geometric_evidence = 100.0 * (0.65 * c_gov + 0.35 * curv)
        results.append(
            {
                "sample": english_name,
                "sample_cn": chinese_name,
                "image": str(path.relative_to(ROOT)),
                "p": round(periodicity, 3),
                "c_gov": round(c_gov, 4),
                "curv": round(curv, 4),
                "fdi_with_fft": round(fdi, 1),
                "fdi_without_fft": round(geometric_evidence, 1),
                "fft_discount": round(1.0 - 0.35 * periodicity, 3),
            }
        )

    json_path = OUTDIR / "spectral_periodicity_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    names = [r["sample"] for r in results]
    p_vals = [r["p"] for r in results]
    fdi_fft = [r["fdi_with_fft"] for r in results]
    fdi_no = [r["fdi_without_fft"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(names))
    axes[0].bar(x, p_vals, color="#2E74B5")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Periodicity P")
    axes[0].set_title("2D-FFT Regularization Sensitivity")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.36
    axes[1].bar(x - width / 2, fdi_no, width, label="FDI without FFT", color="#888888")
    axes[1].bar(x + width / 2, fdi_fft, width, label="FDI with FFT", color="#D35400")
    axes[1].set_ylabel("FDI")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    chart_path = OUTDIR / "spectral_periodicity_analysis.png"
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
