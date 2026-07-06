#!/usr/bin/env python3
"""
真实 StableVITON 条纹输出复现实验。

该脚本分析 test_images/real_vton_stableviton/ 中的公开示例裁剪图，
并将论文中的 Table 7 结果写入 output/stableviton_real_vton_results.json。
"""
from __future__ import annotations

import json
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from stripe_distortion_analyzer import analyze_stripe_distortion


ROOT = Path(__file__).resolve().parent
IMGDIR = ROOT / "test_images" / "real_vton_stableviton"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)

SAMPLES = [
    ("subject1_runway_female.png", "Runway female"),
    ("subject2_stage_male.png", "Stage male"),
    ("subject3_street_male.png", "Street male"),
    ("subject4_event_female.png", "Event female"),
    ("subject5_redcarpet_male.png", "Red-carpet male"),
]

PANEL_IMAGES = [("target_cloth.png", "Target stripe cloth", None), *[
    (filename, description, None) for filename, description in SAMPLES
]]


def categorize_fdi(fdi: float) -> str:
    """根据 FDI 分数返回论文表格中的简短类别。"""
    if fdi < 15:
        return "Excellent"
    if fdi < 30:
        return "Acceptable"
    if fdi < 45:
        return "Moderate"
    if fdi < 70:
        return "Severe"
    if fdi < 85:
        return "Bad"
    return "Critical"


def format_one_decimal(value: float) -> str:
    """按论文表格习惯使用 half-up 规则显示一位小数。"""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def save_summary_panel(results: list[dict]) -> Path:
    """生成论文图 5 使用的 StableVITON 真实输出总览图。"""
    score_by_name = {item["name"]: item["fdi"] for item in results}
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (filename, title, _) in zip(axes, PANEL_IMAGES):
        image = Image.open(IMGDIR / filename).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        if filename == "target_cloth.png":
            label = title
        else:
            label = f"{title}\nFDI={format_one_decimal(score_by_name[filename])}"
        ax.text(0.0, -0.04, label, transform=ax.transAxes, fontsize=8, va="top")

    fig.tight_layout()
    panel_path = OUTDIR / "stableviton_real_vton_stripe_panel.png"
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def main() -> None:
    results = []
    for filename, description in SAMPLES:
        image_path = IMGDIR / filename
        fdi, roi_gov, curv, periodicity = analyze_stripe_distortion(
            image_path=str(image_path),
            output_dir=str(OUTDIR),
            blur_ksize=3,
            gov_window_size=7,
            mag_threshold=15.0,
            canny_low=15.0,
            canny_high=50.0,
            min_contour_len=20,
            w_gov=0.65,
            w_curv=0.35,
            use_skin_suppression=True,
            use_texture_filter=True,
        )
        results.append(
            {
                "name": filename,
                "description": description,
                "source_model": "StableVITON",
                "target_garment": "Horizontal stripe top",
                "fdi": round(fdi, 2),
                "c_gov": round(roi_gov, 4),
                "curv": round(curv, 4),
                "periodicity": round(periodicity, 3),
                "category": categorize_fdi(fdi),
            }
        )

    values = [r["fdi"] for r in results]
    summary = {
        "mean": round(statistics.mean(values), 2),
        "std": round(statistics.pstdev(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }

    payload = {"samples": results, "summary": summary}
    json_path = OUTDIR / "stableviton_real_vton_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    panel_path = save_summary_panel(results)

    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {panel_path}")


if __name__ == "__main__":
    main()
