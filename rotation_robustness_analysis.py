#!/usr/bin/env python3
"""规则单方向条纹的旋转鲁棒性实验。

该脚本专门检查审稿人可能质疑的情况：完全规则的斜向条纹是否会
因为像素栅格化、Canny 轮廓连接或 FFT 主频估计而得到偏高 FDI。
结果用于论文中的限制说明和补充鲁棒性分析。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stripe_distortion_analyzer import (
    build_garment_mask,
    compute_curvature_distortion,
    compute_fdi,
    compute_gov,
    compute_periodicity_fft,
    estimate_dominant_orientation,
    load_and_preprocess,
)


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "test_images" / "rotation_robustness"
OUT_DIR = ROOT / "output"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def make_stripes(angle_deg: float, size: int = 512, period: int = 32, duty: float = 0.48) -> np.ndarray:
    """生成抗锯齿的规则单方向条纹图像。"""
    scale = 4
    big = size * scale
    period_big = period * scale
    yy, xx = np.mgrid[0:big, 0:big]
    cx = xx - big / 2
    cy = yy - big / 2
    theta = np.deg2rad(angle_deg)
    coord = cx * np.cos(theta) + cy * np.sin(theta)
    phase = np.mod(coord, period_big) / period_big
    stripe = (phase < duty).astype(np.uint8) * 255
    rgb_big = np.dstack([stripe, stripe, stripe])
    rgb = cv2.resize(rgb_big, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def analyze_image(path: Path) -> dict[str, float]:
    """只计算数值，不保存诊断大图，避免批量实验产生过多中间图。"""
    gray, rgb = load_and_preprocess(str(path), blur_ksize=3)
    roi = build_garment_mask(
        rgb,
        gray,
        use_skin_suppression=False,
        use_texture_filter=False,
    )
    gov_map, mag, angles = compute_gov(gray, window_size=7, mag_threshold=15.0, roi_mask=roi)
    dom, consistency = estimate_dominant_orientation(angles, mag, mag_threshold=15.0, roi_mask=roi)
    periodicity, _ = compute_periodicity_fft(gray, roi_mask=roi, dominant_deg=dom)
    curv_map = compute_curvature_distortion(
        gray,
        canny_low=15.0,
        canny_high=50.0,
        min_contour_len=20,
        roi_mask=roi,
    )
    fdi, _, c_gov, curv, p = compute_fdi(
        gov_map,
        curv_map,
        periodicity=periodicity,
        w_gov=0.65,
        w_curv=0.35,
        w_period=0.35,
        roi_mask=roi,
    )
    return {
        "fdi": round(float(fdi), 3),
        "c_gov": round(float(c_gov), 4),
        "curv": round(float(curv), 4),
        "periodicity": round(float(p), 3),
        "dominant_deg": round(float(dom), 2),
        "consistency": round(float(consistency), 4),
    }


def main() -> None:
    angles = [0, 15, 30, 45, 60, 75, 90]
    rows: list[dict[str, str | float]] = []
    for angle in angles:
        img = make_stripes(angle)
        path = IMG_DIR / f"regular_stripe_angle_{angle:02d}.png"
        Image.fromarray(img).save(path)
        metrics = analyze_image(path)
        rows.append({"angle_deg": angle, "image_path": str(path), **metrics})

    csv_path = OUT_DIR / "rotation_robustness_results.csv"
    json_path = OUT_DIR / "rotation_robustness_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(json.dumps(rows, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
