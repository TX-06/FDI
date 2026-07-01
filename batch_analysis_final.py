#!/usr/bin/env python3
"""
最终批次分析脚本：
1. 生成更多合成测试条纹图像（含已知参考图像，用于SSIM/PSNR对比）
2. 对每张图像计算 FDI + SSIM + PSNR
3. 输出对比表格数据
4. 生成对比图表
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path
import json

IMGDIR = Path("/Users/tengpeilin/sci/test_images")
OUTDIR = Path("/Users/tengpeilin/sci/output")
ABLATION_OUT = Path("/Users/tengpeilin/sci/output_ablation")
IMGDIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

# ── 导入 FDI 分析器中的关键函数 ──
from stripe_distortion_analyzer import (
    analyze_stripe_distortion,
    compute_circ_gov_global,
    compute_periodicity_fft,
)

# ── SSIM / PSNR ──
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_ssim_psnr(test_path, ref_path):
    """计算两张图片之间的 SSIM 和 PSNR"""
    img_test = cv2.imread(str(test_path))
    img_ref = cv2.imread(str(ref_path))
    if img_test is None or img_ref is None:
        return None, None
    # 确保尺寸一致
    h, w = img_test.shape[:2]
    img_ref = cv2.resize(img_ref, (w, h))
    # 转为灰度
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    
    s = ssim(gray_test, gray_ref, data_range=255)
    p = psnr(gray_test, gray_ref, data_range=255)
    return round(s, 4), round(p, 2)


# ═══════════════════════════════════════════════════════════
# 1. 生成更多合成测试图像（含参考图像）
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  1. 生成合成测试条纹图像")
print("=" * 70)

H, W = 512, 512

def save_synth(img, name):
    path = str(IMGDIR / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"     ✓ {name}")
    return path

# ── 参考图像：完美垂直条纹（全分辨率、不同周期）──
ref_20px = np.ones((H, W, 3), dtype=np.uint8) * 255
for x in range(0, W, 20):
    cv2.line(ref_20px, (x, 0), (x, H), (80, 80, 80), 8)
save_synth(ref_20px, "ref_perfect_vertical_20px.png")

ref_12px = np.ones((H, W, 3), dtype=np.uint8) * 255
for x in range(0, W, 12):
    cv2.line(ref_12px, (x, 0), (x, H), (80, 80, 80), 5)
save_synth(ref_12px, "ref_perfect_vertical_12px.png")

# ── 受控失真测试集 ──
test_cases = []

# (A) 轻度波浪失真（模拟自然褶皱）
for amp in [4, 8, 12, 16, 20]:
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    for i in range(0, W, 20):
        pts = []
        for y in range(H):
            offset = int(amp * np.sin(y * 2 * np.pi / 150))
            x = i + offset
            if 0 <= x < W:
                pts.append([x, y])
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (80, 80, 80), 8)
    fname = f"synth_wavy_amp{amp}.png"
    save_synth(img, fname)
    test_cases.append((fname, "ref_perfect_vertical_20px.png", f"Wavy Stripe (Amp={amp})"))

# (B) 频率噪声失真
for noise_std in [10, 20, 30, 40]:
    ref = ref_20px.copy()
    noise = np.random.normal(0, noise_std, ref.shape).astype(np.int16)
    noisy = np.clip(ref.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    fname = f"synth_noise_std{noise_std}.png"
    save_synth(noisy, fname)
    test_cases.append((fname, "ref_perfect_vertical_20px.png", f"Gaussian Noise (σ={noise_std})"))

# (C) JPEG 压缩失真（不同质量）
ref_path = str(IMGDIR / "ref_perfect_vertical_20px.png")
ref_bgr = cv2.imread(ref_path)
for quality in [85, 70, 50, 30, 15]:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', ref_bgr, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec_rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    fname = f"synth_jpeg_q{quality}.png"
    save_synth(dec_rgb, fname)
    test_cases.append((fname, "ref_perfect_vertical_20px.png", f"JPEG Compression (Q={quality})"))

# (D) 多方向条纹混合
img = np.ones((H, W, 3), dtype=np.uint8) * 255
for x in range(0, W, 20):
    cv2.line(img, (x, 0), (x, H), (80, 80, 80), 8)
for y in range(0, H, 20):
    cv2.line(img, (0, y), (W, y), (60, 60, 60), 4)
fname = "synth_cross_hatch.png"
save_synth(img, fname)
test_cases.append((fname, "ref_perfect_vertical_20px.png", "Cross-Hatch Pattern"))

# ═══════════════════════════════════════════════════════════
# 2. 批次分析：FDI + SSIM + PSNR
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2. 批次分析：FDI + SSIM + PSNR")
print("=" * 70)

results = []

for fname, ref_fname, desc in test_cases:
    img_path = str(IMGDIR / fname)
    ref_path = str(IMGDIR / ref_fname)
    
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
            w_gov=0.65,
            w_curv=0.35,
            use_skin_suppression=True,
            use_texture_filter=True,
        )
    except Exception as e:
        print(f"  ✗ {fname}: FDI failed - {e}")
        fdi = None
    
    s, p = compute_ssim_psnr(img_path, ref_path)
    
    results.append({
        "name": fname,
        "description": desc,
        "category": "Synthetic",
        "fdi": fdi,
        "ssim": s,
        "psnr": p,
    })
    print(f"  {desc:40s} FDI={fdi:5.1f}  SSIM={s}  PSNR={p}")

# ── 保存结果 ──
with open(Path.cwd() / "batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n结果已保存到: {Path.cwd() / 'batch_results.json'}")

# ═══════════════════════════════════════════════════════════
# 3. 生成对比图表
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  3. 生成对比图表")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 提取数据
names = [r["description"] for r in results]
fdi_vals = [r["fdi"] for r in results]
ssim_vals = [r["ssim"] for r in results]
psnr_vals = [r["psnr"] for r in results]

# 筛选有效数据
valid = [(i, n) for i, n in enumerate(names) if fdi_vals[i] is not None and ssim_vals[i] is not None]
v_idx = [i for i, _ in valid]
v_names = [n for _, n in valid]
v_fdi = [fdi_vals[i] for i in v_idx]
v_ssim = [ssim_vals[i] for i in v_idx]
v_psnr = [psnr_vals[i] for i in v_idx]

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

x = np.arange(len(v_names))
width = 0.3

# 上：FDI vs SSIM
ax = axes[0]
bars1 = ax.bar(x - width/2, v_fdi, width, label='FDI', color='darkred', alpha=0.85)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, v_ssim, width, label='SSIM', color='steelblue', alpha=0.7)
ax.set_xlabel('Test Image', fontsize=10)
ax.set_ylabel('FDI (0-100)', color='darkred', fontsize=11)
ax2.set_ylabel('SSIM (0-1)', color='steelblue', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(v_names, rotation=45, ha='right', fontsize=9)
ax.set_title('FDI vs SSIM: Synthetic Stripe Distortion Analysis', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# 下：FDI vs PSNR
ax = axes[1]
bars1 = ax.bar(x - width/2, v_fdi, width, label='FDI', color='darkred', alpha=0.85)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, v_psnr, width, label='PSNR (dB)', color='forestgreen', alpha=0.7)
ax.set_xlabel('Test Image', fontsize=10)
ax.set_ylabel('FDI (0-100)', color='darkred', fontsize=11)
ax2.set_ylabel('PSNR (dB)', color='forestgreen', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(v_names, rotation=45, ha='right', fontsize=9)
ax.set_title('FDI vs PSNR: Synthetic Stripe Distortion Analysis', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = OUTDIR / "fdi_vs_ssim_psnr_comparison.png"
plt.savefig(str(chart_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"  对比图: {chart_path}")

# ═══════════════════════════════════════════════════════════
# 4. 打印LaTeX表格
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  LaTeX 表格（可复制到论文中）")
print("=" * 70)
print()
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Comparison of FDI with SSIM and PSNR on synthetic stripe patterns.}")
print(r"\label{tab:comparison}")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"Test Image & FDI & C-GOV & SSIM & PSNR (dB) \\")
print(r"\midrule")
for r in results:
    fdi_str = f"{r['fdi']:.1f}" if r['fdi'] is not None else "N/A"
    ssim_str = f"{r['ssim']:.4f}" if r['ssim'] is not None else "N/A"
    psnr_str = f"{r['psnr']:.2f}" if r['psnr'] is not None else "N/A"
    print(f"{r['description']:40s} & {fdi_str} & {ssim_str} & {psnr_str} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n" + "=" * 70)
print("  完成！")
print("=" * 70)
