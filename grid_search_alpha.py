#!/usr/bin/env python3
"""
α/β 权重 grid search — 从 0.05 到 0.95 步进 0.05 扫描
输出：
  1. grid_search_results.json — 详细数据
  2. FDI_alpha_grid_search.png — 对比曲线图
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMGDIR = Path("/Users/tengpeilin/sci/test_images")
OUTDIR = Path("/Users/tengpeilin/sci/output")
IMGDIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

from stripe_distortion_analyzer import analyze_stripe_distortion

# ── 测试图片 ──
test_set = [
    ("ref_perfect_vertical_20px.png", "Perfect Vertical"),
    ("synth_wavy_amp8.png",       "Wavy (Amp=8)"),
    ("synth_noise_std20.png",     "Noise (σ=20)"),
    ("synth_jpeg_q15.png",        "JPEG (Q=15)"),
    ("realistic_torso_stripe.png", "Real Torso"),
    ("pexels_stripe_1.jpg",       "Real Stripe 1"),
]

# ── α 扫描范围 ──
alpha_values = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
results = []

print("=" * 70)
print("α/β 权重 Grid Search")
print("=" * 70)
print(f"α 范围: {alpha_values}")
print()

for img_name, img_desc in test_set:
    img_path = str(IMGDIR / img_name)
    if not os.path.exists(img_path):
        print(f"  ✗ 文件不存在: {img_path}")
        continue

    for alpha in alpha_values:
        beta = round(1.0 - alpha, 2)
        try:
            fdi = analyze_stripe_distortion(
                image_path=img_path,
                output_dir="output",
                blur_ksize=3,
                gov_window_size=7,
                mag_threshold=15.0,
                canny_low=15.0,
                canny_high=50.0,
                min_contour_len=20,
                w_gov=alpha,
                w_curv=beta,
                use_skin_suppression=True,
                use_texture_filter=True,
            )
        except Exception as e:
            print(f"  ✗ {img_desc} α={alpha}: {e}")
            fdi = None
        
        results.append({
            "image": img_desc,
            "alpha": alpha,
            "beta": beta,
            "fdi": fdi,
        })
        print(f"  {img_desc:25s} α={alpha:.2f} β={beta:.2f} → FDI={fdi}")

# ── 保存数据 ──
with open(OUTDIR / "grid_search_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n数据已保存: {OUTDIR / 'grid_search_results.json'}")

# ── 绘制曲线 ──
fig, ax = plt.subplots(figsize=(12, 7))

image_names = sorted(set(r["image"] for r in results))
colors = plt.cm.tab10(np.linspace(0, 1, len(image_names)))

for img_name, color in zip(image_names, colors):
    data = [r for r in results if r["image"] == img_name]
    data.sort(key=lambda x: x["alpha"])
    alphas = [d["alpha"] for d in data]
    fdis = [d["fdi"] if d["fdi"] is not None else 0 for d in data]
    ax.plot(alphas, fdis, 'o-', color=color, label=img_name, linewidth=2, markersize=5)

ax.axvline(x=0.65, color='red', linestyle='--', alpha=0.7, label='Selected α=0.65')
ax.set_xlabel("α Weight (C-GOV coefficient)", fontsize=12)
ax.set_ylabel("FDI Score", fontsize=12)
ax.set_title("Grid Search: Effect of α/β Weighting on FDI Scores", fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)

plt.tight_layout()
chart_path = OUTDIR / "FDI_alpha_grid_search.png"
plt.savefig(str(chart_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {chart_path}")
