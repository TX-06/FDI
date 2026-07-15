#!/usr/bin/env python3
"""FDI 与通用图像质量指标的可复现对比实验。

实验分为两部分：
1. 对 15 张具有像素对齐理想参考图的合成条纹图计算全参考指标。
2. 对 30 张无理想参考图的公开 VTON 输出计算无参考指标。

运行前先执行 ``batch_analysis_final.py`` 和
``real_vton_public_stripe_analysis.py``，确保 FDI 结果与样本清单为最新版本。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pyiqa
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "test_images"
OUTPUT_DIR = ROOT / "output"
FULL_REFERENCE_CSV = OUTPUT_DIR / "iqa_full_reference_results.csv"
NO_REFERENCE_CSV = OUTPUT_DIR / "iqa_no_reference_vton_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "iqa_comparison_summary.json"

REFERENCE_IMAGE = IMAGE_DIR / "ref_perfect_vertical_20px.png"

FULL_REFERENCE_CASES = [
    ("synth_wavy_amp4.png", "Wavy stripe (amplitude 4 px)", "wavy", 1),
    ("synth_wavy_amp8.png", "Wavy stripe (amplitude 8 px)", "wavy", 2),
    ("synth_wavy_amp12.png", "Wavy stripe (amplitude 12 px)", "wavy", 3),
    ("synth_wavy_amp16.png", "Wavy stripe (amplitude 16 px)", "wavy", 4),
    ("synth_wavy_amp20.png", "Wavy stripe (amplitude 20 px)", "wavy", 5),
    ("synth_noise_std10.png", "Gaussian noise (sigma 10)", "noise", 1),
    ("synth_noise_std20.png", "Gaussian noise (sigma 20)", "noise", 2),
    ("synth_noise_std30.png", "Gaussian noise (sigma 30)", "noise", 3),
    ("synth_noise_std40.png", "Gaussian noise (sigma 40)", "noise", 4),
    ("synth_jpeg_q85.png", "JPEG compression (quality 85)", "jpeg", 1),
    ("synth_jpeg_q70.png", "JPEG compression (quality 70)", "jpeg", 2),
    ("synth_jpeg_q50.png", "JPEG compression (quality 50)", "jpeg", 3),
    ("synth_jpeg_q30.png", "JPEG compression (quality 30)", "jpeg", 4),
    ("synth_jpeg_q15.png", "JPEG compression (quality 15)", "jpeg", 5),
    ("synth_cross_hatch.png", "Cross-hatch pattern (out of scope)", "scope", 1),
]

FULL_REFERENCE_METRICS = [
    "psnr",
    "ssim",
    "ms_ssim",
    "gmsd",
    "lpips",
    "dists",
]

NO_REFERENCE_METRICS = ["brisque", "niqe", "maniqa"]


def safe_spearman(x: list[float], y: list[float]) -> float | None:
    """计算 Spearman 相关；常量序列返回 None，避免结果 JSON 出现 NaN。"""
    rho = spearmanr(x, y).statistic
    rho_float = float(rho)
    if not math.isfinite(rho_float):
        return None
    return round(rho_float, 4)


def load_json(path: Path) -> Any:
    """读取 JSON，并在缺少上游结果时给出明确提示。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少上游结果文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少上游结果文件：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def create_metrics(names: list[str]) -> dict[str, Any]:
    """在 CPU 上加载指标模型，避免逐图重复初始化。"""
    metrics: dict[str, Any] = {}
    for name in names:
        print(f"Loading metric: {name}", flush=True)
        metrics[name] = pyiqa.create_metric(name, device="cpu")
    return metrics


def score_metric(metric: Any, image: Path, reference: Path | None = None) -> float:
    """计算单个指标并返回普通浮点数。"""
    with torch.no_grad():
        value = metric(str(image)) if reference is None else metric(str(image), str(reference))
    return float(value.detach().cpu().item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按首行字段顺序写入结果。"""
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_full_reference(metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """计算具有理想参考图的合成条纹实验。"""
    if not REFERENCE_IMAGE.exists():
        raise FileNotFoundError(f"缺少理想条纹参考图：{REFERENCE_IMAGE}")

    batch_rows = load_json(OUTPUT_DIR / "batch_results.json")
    fdi_by_name = {row["name"]: float(row["fdi"]) for row in batch_rows}
    rows: list[dict[str, Any]] = []

    for file_name, description, family, severity in FULL_REFERENCE_CASES:
        image_path = IMAGE_DIR / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"缺少合成测试图：{image_path}")
        if file_name not in fdi_by_name:
            raise KeyError(f"batch_results.json 中缺少 {file_name} 的 FDI")

        row: dict[str, Any] = {
            "sample_id": image_path.stem,
            "description": description,
            "family": family,
            "severity_level": severity,
            "fdi": round(fdi_by_name[file_name], 4),
        }
        for metric_name in FULL_REFERENCE_METRICS:
            score = score_metric(metrics[metric_name], image_path, REFERENCE_IMAGE)
            row[metric_name] = round(score, 6)
        rows.append(row)
        print(f"FR {image_path.stem}: complete", flush=True)

    # 所有指标统一转换为“数值越大表示失真越强”的方向后计算单调性。
    high_quality_metrics = {"psnr", "ssim", "ms_ssim"}
    trend_summary: dict[str, Any] = {}
    for family in ("wavy", "noise", "jpeg"):
        family_rows = [row for row in rows if row["family"] == family]
        severity = [row["severity_level"] for row in family_rows]
        family_summary: dict[str, float] = {}
        for metric_name in ["fdi", *FULL_REFERENCE_METRICS]:
            values = [row[metric_name] for row in family_rows]
            if metric_name in high_quality_metrics:
                values = [-value for value in values]
            family_summary[metric_name] = safe_spearman(severity, values)
        trend_summary[family] = family_summary

    return rows, trend_summary


def run_no_reference(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """计算 30 张公开 VTON 输出的无参考整体质量指标。"""
    source_rows = load_csv(OUTPUT_DIR / "public_real_vton_stripe_results.csv")
    if len(source_rows) != 30:
        raise ValueError(f"真实 VTON 样本应为 30 张，当前为 {len(source_rows)} 张")

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        image_path = Path(source["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"缺少真实 VTON 样本：{image_path}")

        row: dict[str, Any] = {
            "sample_id": source["sample_id"],
            "method": source["method"],
            "stripe_type": source["stripe_type"],
            "fdi": round(float(source["fdi"]), 4),
        }
        for metric_name in NO_REFERENCE_METRICS:
            # pyiqa 0.1.13 的 BRISQUE 对连续调用存在内部状态问题；
            # 每张图重新初始化一次，可保证 30 个样本稳定复现。
            if metric_name == "brisque":
                metric = pyiqa.create_metric("brisque", device="cpu")
            else:
                metric = metrics[metric_name]
            score = score_metric(metric, image_path)
            row[metric_name] = round(score, 6)
        rows.append(row)
        print(f"NR {source['sample_id']}: complete", flush=True)

    return rows


def summarize_no_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总无参考指标，并报告它们与 FDI 的秩相关而不冒充主观评价。"""
    summary: dict[str, Any] = {"sample_count": len(rows), "metrics": {}}
    fdi = [row["fdi"] for row in rows]
    lower_is_better = {"brisque", "niqe"}

    for metric_name in NO_REFERENCE_METRICS:
        values = [row[metric_name] for row in rows]
        distortion_values = values if metric_name in lower_is_better else [-value for value in values]
        summary["metrics"][metric_name] = {
            "mean": round(sum(values) / len(values), 6),
            "minimum": round(min(values), 6),
            "maximum": round(max(values), 6),
            "spearman_with_fdi_distortion_direction": safe_spearman(fdi, distortion_values),
        }
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    full_metrics = create_metrics(FULL_REFERENCE_METRICS)
    no_reference_metrics = create_metrics([name for name in NO_REFERENCE_METRICS if name != "brisque"])

    full_rows, trend_summary = run_full_reference(full_metrics)
    write_csv(FULL_REFERENCE_CSV, full_rows)

    no_reference_rows = run_no_reference(no_reference_metrics)
    write_csv(NO_REFERENCE_CSV, no_reference_rows)

    summary = {
        "implementation": {
            "library": "pyiqa",
            "version": pyiqa.__version__,
            "device": "cpu",
            "full_reference": FULL_REFERENCE_METRICS,
            "no_reference": NO_REFERENCE_METRICS,
        },
        "full_reference_monotonicity_spearman": trend_summary,
        "no_reference_vton": summarize_no_reference(no_reference_rows),
        "interpretation_boundary": (
            "Full-reference metrics use pixel-aligned synthetic images. "
            "No-reference metrics use the complete cropped VTON outputs and do not isolate stripe geometry."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {FULL_REFERENCE_CSV}")
    print(f"Saved: {NO_REFERENCE_CSV}")
    print(f"Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
