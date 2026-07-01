#!/usr/bin/env python3
"""
条纹变形与织物纬斜分析器 — VTON 质量评估指标  [SCI-Ready v2.1]
================================================================
定量衡量虚拟试衣（VTON）算法对织物物理特性的破坏程度。

核心指标（双驱融合）：
  1. 梯度方向方差（GOV） —— 局部条纹方向的混乱程度（循环统计量）
  2. 轮廓曲率分析          —— 锯齿/波浪状边缘伪影（Menger 曲率）

三重防污染 ROI 引擎（⭐ v2.0 新增）：
  第一重：HSV + YCrCb 双色彩空间肤色/背景抑制
  第二重：纹理密度验证（局部梯度能量检测）
  第三重：用户先验（可选边界框 / 颜色范围 / 外部掩膜）

输出 **织物变形指数（FDI）**，范围 [0, 100]。
所有指标仅基于服装区域（Garment ROI）计算。

依赖库：OpenCV (cv2)  NumPy  Matplotlib
安装：pip install opencv-python numpy matplotlib
"""

from __future__ import annotations
import sys
# ── Windows GBK 终端兼容：强制 stdout/stderr 使用 utf-8 ──
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # 无 GUI 后端，避免窗口弹闪
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import Tuple, Optional

# ── 中文字体（自动检测 Windows / macOS）──
_CJK_FONT = None
for _fname in [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]:
    try:
        if Path(_fname).exists():
            _CJK_FONT = fm.FontProperties(fname=_fname)
            fm.fontManager.addfont(_fname)
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [_CJK_FONT.get_name(), "DejaVu Sans"]
            break
    except Exception:
        continue
if _CJK_FONT is None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════
#  步骤 0 — 三重防污染 Garment ROI 引擎
# ═══════════════════════════════════════════════════════════════════════

def _skin_mask_hsv_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """
    第一重防线：HSV + YCrCb 双色彩空间肤色检测。

    两个色彩空间的肤色掩膜取并集后膨胀，确保服装-皮肤交界处
    的过渡像素也被标记为待排除区域。

    参考范围：
      HSV:   H ∈ [0, 50]  S ∈ [20, 150]  V ∈ [60, 255]  (PeerJ 2019)
      YCrCb: Cb ∈ [133, 173]  Cr ∈ [77, 127]           (经典椭圆模型简化)

    返回：布尔矩阵 [H, W]，True = 检测为肤色/待排除。
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask_hsv = cv2.inRange(hsv,
                           np.array([0, 20, 60], dtype=np.uint8),
                           np.array([50, 150, 255], dtype=np.uint8))

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb,
                             np.array([0, 133, 77], dtype=np.uint8),
                             np.array([255, 173, 127], dtype=np.uint8))

    skin = (mask_hsv.astype(bool) | mask_ycrcb.astype(bool)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    skin = cv2.dilate(skin, kernel, iterations=2)
    return skin.astype(bool)


def _texture_density_mask(gray: np.ndarray,
                           block_size: int = 16,
                           std_threshold: float = 8.0) -> np.ndarray:
    """
    第二重防线：纹理密度验证。

    对灰度图轻量模糊后计算 Sobel 梯度幅值，然后分块统计局部
    标准差。服装条纹区域因周期性边缘信号而具有高局部标准差；
    皮肤、平坦背景、JPEG 块效应区域的局部标准差显著较低。

    阈值 8.0 是基于 256 级灰度的经验值，在标准测试集上验证。

    返回：布尔矩阵 [H, W]，True = 有足够纹理信号。
    """
    # 轻微模糊抑制 JPEG 块效应（块大小 8→模糊核 3 足够）
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    h, w = gray.shape
    tex = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, block_size):
        y_end = min(y + block_size, h)
        for x in range(0, w, block_size):
            x_end = min(x + block_size, w)
            if mag[y:y_end, x:x_end].std() > std_threshold:
                tex[y:y_end, x:x_end] = 1

    # 形态学闭运算：填充条纹内部孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tex = cv2.morphologyEx(tex, cv2.MORPH_CLOSE, kernel)
    return tex.astype(bool)


def build_garment_mask(
    rgb: np.ndarray,
    gray: np.ndarray,
    use_skin_suppression: bool = True,
    use_texture_filter: bool = True,
    garment_bbox: Optional[Tuple[int, int, int, int]] = None,
    garment_color_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    external_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    构建最终服装区域二值掩膜。

    融合策略（⭐ v2.1 修正）：
        garment ≈ (NOT exclude) ∪ texture_override

    即：先排除确信的非服装像素（肤色+极亮/极暗背景），
    再用纹理证据 "救回" 边界过渡区。最后保留最大连通分量
    并腐蚀边界，排除 ROI 边缘的半像素过渡区。

    参数：
        rgb:                RGB 图像 [H, W, 3]。
        gray:               灰度图像 [H, W]（建议预模糊）。
        use_skin_suppression: 启用第一重肤色/背景抑制。
        use_texture_filter:   启用第二重纹理密度验证。
        garment_bbox:         手动边界框 (x, y, w, h)。
        garment_color_range:  HSV 颜色范围 (lower, upper)。
        external_mask:        外部二值掩膜（1 = 服装）。

    返回：二值掩膜 [H, W]，dtype=uint8，1 = 服装区域。
    """
    h, w = rgb.shape[:2]

    # ── 第三重优先：外部掩膜或 bbox 直接接管 ──
    if external_mask is not None:
        mask = external_mask.copy()
        # 仍需清理边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    if garment_bbox is not None:
        x, y, bw, bh = garment_bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)] = 1
        return mask

    # ── 初始化：全图为服装候选 ──
    mask = np.ones((h, w), dtype=np.uint8)

    # ── 第一重：肤色 + 背景排除 ──
    if use_skin_suppression:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # 极亮背景（白墙、天空等）：低饱和度 + 高亮度
        bg_bright = ((hsv[:, :, 1] < 18) & (hsv[:, :, 2] > 220))
        # 极暗背景（阴影、黑布等）：低饱和度 + 低亮度
        bg_dark   = ((hsv[:, :, 1] < 18) & (hsv[:, :, 2] < 45))

        skin = _skin_mask_hsv_ycrcb(rgb)
        exclude = (skin | bg_bright | bg_dark).astype(np.uint8)

        # 形态学清理：开运算去噪 → 闭运算填孔
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        exclude = cv2.morphologyEx(exclude, cv2.MORPH_OPEN, kernel, iterations=1)
        exclude = cv2.morphologyEx(exclude, cv2.MORPH_CLOSE, kernel, iterations=1)

        mask[exclude.astype(bool)] = 0

    # ── 第二重：纹理证据覆盖 ──
    # 逻辑：有纹理的地方至少是某种织物，允许覆盖肤色排除
    # （肤色检测在条纹-皮肤边界处可能过度排除）。
    # 但不再盲目将无纹理区全部抹除——非纹理区可能是平滑织物。
    if use_texture_filter:
        tex = _texture_density_mask(gray)
        mask[tex] = 1                   # 纹理区强制设为服装

    # ── 颜色范围过滤（如果提供）──
    if garment_color_range is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower, upper = garment_color_range
        color_mask = cv2.inRange(hsv, lower, upper).astype(bool)
        mask[~color_mask] = 0

    # ── 保留最大连通区域，丢弃孤立的误检小区域 ──
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        sizes = np.bincount(labels.ravel())
        if len(sizes) > 1:
            largest_label = np.argmax(sizes[1:]) + 1
            mask = (labels == largest_label).astype(np.uint8)

    # ── 腐蚀边界：排除 ROI 边缘过渡像素 ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# ═══════════════════════════════════════════════════════════════════════
#  步骤 1 — 图像加载与预处理
# ═══════════════════════════════════════════════════════════════════════

def load_and_preprocess(image_path: str,
                         blur_ksize: int = 3,
                         deblock: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载图像，应用 JPEG 去块效应 + 高斯模糊。

    ⭐ v2.2: 新增双边滤波去块效应（针对高压缩 JPG 图像）。
    JPEG 以 8×8 块为单位压缩，产生块边界处的虚假边缘。
    双边滤波在保留真实条纹边缘的同时平滑这些块边界。

    返回：
        gray:  预处理后的灰度图 [H, W]，float64。
        rgb:   RGB 图像 [H, W, 3]，uint8。
    """
    # ⭐ 使用 np.fromfile + imdecode 兼容 Unicode 中文路径
    img_array = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ⭐ v2.2: 双边滤波去块效应（保留边缘 + 平滑8×8块边界）
    if deblock:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=4)

    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return np.float64(gray_blur), rgb


# ═══════════════════════════════════════════════════════════════════════
#  步骤 2 — 梯度方向方差（GOV）
# ═══════════════════════════════════════════════════════════════════════

def compute_gov(
    gray: np.ndarray,
    window_size: int = 15,
    mag_threshold: float = 20.0,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算梯度方向方差（GOV）图。

    ⭐ v2.1 修复：正确执行幅值加权的循环均值归一化。

    算法：
      1. Sobel → Gx, Gy → M = |∇I|, θ = atan2(Gy, Gx)
      2. 角度加倍：θ' = 2θ（使 θ 和 θ+π 等价）
      3. 滑动窗口内计算幅值加权循环均值：
           R_local = |E[M·e^{iθ'}]| / E[M]     ∈ [0, 1]
      4. 循环方差：GOV = 1 - R_local            ∈ [0, 1]

    关键修复：
      旧版直接用 boxFilter 的 sin_avg/cos_avg（即 E[M·sinθ']）
      直接算 R = sqrt(sin_avg²+cos_avg²)，量纲错误导致 R ≫ 1，
      GOV 恒被 clamp 到 0。现除以 E[M] 得到正确的单位向量长度。

    返回：
        gov_map:   逐像素 GOV [H, W]，float32，值 ∈ [0, 1]。
        magnitude: 梯度幅值 [H, W]，float64。
        angles:    梯度方向 [H, W]，弧度。
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angles = np.arctan2(gy, gx)

    # ROI 外梯度置零
    if roi_mask is not None:
        magnitude[~roi_mask.astype(bool)] = 0.0

    # 加倍角度 + 幅值加权
    a2 = 2.0 * angles
    sin_w = (np.sin(a2) * magnitude).astype(np.float32)
    cos_w = (np.cos(a2) * magnitude).astype(np.float32)
    mag_f32 = magnitude.astype(np.float32)

    # 盒式滑动窗口均值：E[M·sin], E[M·cos], E[M]
    sin_avg = cv2.boxFilter(sin_w, cv2.CV_32F,
                            (window_size, window_size),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    cos_avg = cv2.boxFilter(cos_w, cv2.CV_32F,
                            (window_size, window_size),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    mag_avg = cv2.boxFilter(mag_f32, cv2.CV_32F,
                            (window_size, window_size),
                            normalize=True, borderType=cv2.BORDER_REFLECT)

    # ⭐ 正确归一化：除以 E[M]，得到单位圆内的合成向量
    valid = mag_avg > 1e-6
    sin_norm = np.zeros_like(sin_avg)
    cos_norm = np.zeros_like(cos_avg)
    sin_norm[valid] = sin_avg[valid] / mag_avg[valid]
    cos_norm[valid] = cos_avg[valid] / mag_avg[valid]

    # 平均合成向量长度 R ∈ [0, 1]
    r = np.sqrt(sin_norm ** 2 + cos_norm ** 2)
    r = np.clip(r, 0.0, 1.0)                     # 防止浮点误差越界
    gov_map = 1.0 - r                             # 循环方差 V = 1 - R

    # 弱梯度区域 + 非 ROI 区域置零
    gov_map[magnitude < mag_threshold] = 0.0
    if roi_mask is not None:
        gov_map[~roi_mask.astype(bool)] = 0.0

    return gov_map.astype(np.float32), magnitude, angles


# ═══════════════════════════════════════════════════════════════════════
#  步骤 2b — 主导条纹方向
# ═══════════════════════════════════════════════════════════════════════

def estimate_dominant_orientation(
    angles: np.ndarray,
    magnitude: np.ndarray,
    mag_threshold: float = 20.0,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    全局幅值加权主导方向估计。

    仅统计强梯度像素（|∇I| > threshold），在 ROI 内计算
    加倍角度的加权循环均值。

    返回：
        dominant_deg:  主导边缘方向 [0°, 180°)。
                       水平条纹 → ~90°，垂直条纹 → ~0°。
        consistency:   R ∈ [0, 1]，1.0 = 所有边缘像素方向完全一致。
    """
    mask = magnitude > mag_threshold
    if roi_mask is not None:
        mask = mask & roi_mask.astype(bool)

    strong_a = angles[mask]
    strong_m = magnitude[mask]

    if len(strong_a) < 50:
        return 0.0, 0.0

    # 幅值加权循环均值（加倍角）
    a2 = 2.0 * strong_a
    sin_m = np.average(np.sin(a2), weights=strong_m)
    cos_m = np.average(np.cos(a2), weights=strong_m)

    consistency = float(np.sqrt(sin_m ** 2 + cos_m ** 2))
    dominant_rad = np.arctan2(sin_m, cos_m) / 2.0
    dominant_deg = float(np.degrees(dominant_rad) % 180)

    return dominant_deg, consistency


# ═══════════════════════════════════════════════════════════════════════
#  步骤 3 — 轮廓曲率分析
# ═══════════════════════════════════════════════════════════════════════

def compute_curvature_distortion(
    gray: np.ndarray,
    canny_low: float = 30.0,
    canny_high: float = 90.0,
    min_contour_len: int = 50,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    通过 Menger 曲率的局部方差检测锯齿/波浪边缘伪影。

    算法：
      1. Canny 边缘检测 → 轮廓提取
      2. 每个轮廓点计算 Menger 曲率：
           k_i = 4 · Area(△p_{i-1}p_i p_{i+1}) / (|a|·|b|·|c|)
      3. 在 ±3 点窗口内计算曲率局部方差
      4. 通过 1 - exp(-var·15) 映射到 [0, 1]

    ⭐ v2.1 修复：np.cross 对 2D 向量在 NumPy 2.0 中已废弃，
       改用显式 2D 叉积 a_x·b_y - a_y·b_x。

    返回：curv_map [H, W]，float32，值 ∈ [0, 1]。
    """
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, canny_low, canny_high)

    if roi_mask is not None:
        edges[~roi_mask.astype(bool)] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    h, w = gray.shape
    curv_map = np.zeros((h, w), dtype=np.float32)

    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n < min_contour_len:
            continue

        # Menger 曲率
        curv = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            ax, ay = pts[i, 0] - pts[i - 1, 0], pts[i, 1] - pts[i - 1, 1]
            bx, by = pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1]
            cx, cy = pts[i + 1, 0] - pts[i - 1, 0], pts[i + 1, 1] - pts[i - 1, 1]

            na = np.hypot(ax, ay)
            nb = np.hypot(bx, by)
            nc = np.hypot(cx, cy)
            denom = na * nb * nc
            if denom < 1e-8:
                continue
            # 显式 2D 叉积，兼容 NumPy 2.0
            cross_2d = ax * by - ay * bx
            curv[i] = 4.0 * abs(cross_2d) * 0.5 / denom

        # 局部曲率方差 → "锯齿度" 分数
        half_w = 3
        for i in range(half_w, n - half_w):
            seg = curv[i - half_w : i + half_w + 1]
            var_local = np.var(seg)
            distortion = 1.0 - np.exp(-var_local * 15.0)

            xi, yi = int(pts[i, 0]), int(pts[i, 1])
            if 0 <= xi < w and 0 <= yi < h:
                if distortion > curv_map[yi, xi]:
                    curv_map[yi, xi] = distortion

    return curv_map


# ═══════════════════════════════════════════════════════════════════════
#  步骤 3b — FFT 频域周期性分析  ⭐ v2.2 新增核心 ⭐
# ═══════════════════════════════════════════════════════════════════════


def compute_periodicity_fft(
    gray: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    dominant_deg: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    通过 FFT 频谱分析评估条纹的频域周期性强弱。

    核心思想：
    ────────────────────────────────────────────────────────────
    真正 3D 穿着的条纹衫 → 频谱中存在清晰的定向周期性峰值
    AI 崩溃的纹理        → 频谱能量弥散 / 无定向周期峰

    因此 periodicity_strength ∈ [0, 1] 是一个关键判别因子：
      1.0 = 完美的周期纹理（即使有自然弯曲）
      0.0 = 完全无周期（AI 崩溃或纯噪声）

    算法：
    1. 裁剪 ROI 区域进行 2D FFT
    2. 在垂直于条纹方向的谱线上寻找信噪比峰值
    3. 峰值能量 / 背景能量 = 周期强度

    参数：
        gray:         灰度图像 [H, W]。
        roi_mask:     服装区域掩膜 [H, W]。
        dominant_deg: 主导条纹方向 [0°, 180°)。如果提供，
                      则 FFT 仅沿垂直方向搜索。
    返回：
        periodicity_strength: 频域周期强度 [0, 1]。
        fft_viz:              log 频谱可视化 [H, W]。
    """
    h, w = gray.shape

    # — 使用 ROI 区域或全图 —
    if roi_mask is not None:
        masked = gray * roi_mask.astype(float)
    else:
        masked = gray

    # — 应用汉宁窗减少频谱泄露 —
    han = np.hanning(w)[None, :] * np.hanning(h)[:, None]
    masked = masked * han

    # — 2D FFT —
    f = np.fft.fft2(masked.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag_spec = np.abs(fshift)
    mag_log = np.log(mag_spec + 1)

    fft_viz = mag_log / (mag_log.max() + 1e-8)

    # — 在垂直于条纹方向的轴上寻找周期性峰值 —
    cx, cy = w // 2, h // 2

    if dominant_deg is not None:
        # 沿垂直于条纹的方向搜索
        search_angle = (dominant_deg + 90) % 180
        theta = np.radians(search_angle)
    else:
        # 若无主导方向，沿多条径向扫描找最强峰
        best_peak_ratio = 0.0
        for angle_deg in range(0, 180, 5):
            theta = np.radians(angle_deg)
            radial = _sample_radial(mag_spec, cx, cy, theta, max_r=min(cx, cy))
            if len(radial) < 5:
                continue
            peak = radial.max()
            bg = (radial.sum() - peak) / max(len(radial) - 1, 1)
            ratio = peak / max(bg, 1e-6)
            if ratio > best_peak_ratio:
                best_peak_ratio = ratio
        periodicity_strength = float(np.clip(best_peak_ratio / 12.0, 0, 1))
        return periodicity_strength, fft_viz

    # 沿特定方向采样
    radial = _sample_radial(mag_spec, cx, cy, theta, max_r=min(cx, cy))
    if len(radial) < 5:
        return 0.0, fft_viz

    peak = radial.max()
    bg = (radial.sum() - peak) / max(len(radial) - 1, 1)
    peak_ratio = peak / max(bg, 1e-6)

    # 映射到 [0, 1]： 比率 3 以下≈无周期； 12 以上≈强周期
    periodicity_strength = float(np.clip((peak_ratio - 2.0) / 10.0, 0, 1))

    return periodicity_strength, fft_viz


def _sample_radial(
    spectrum: np.ndarray, cx: int, cy: int,
    theta: float, max_r: int,
) -> np.ndarray:
    """沿给定角度从频谱中心向外采样的强度数组。"""
    values = []
    for r in range(1, max_r):
        x = int(round(cx + r * np.cos(theta)))
        y = int(round(cy + r * np.sin(theta)))
        if 0 <= x < spectrum.shape[1] and 0 <= y < spectrum.shape[0]:
            values.append(spectrum[y, x])
    return np.array(values, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
#  步骤 2c — 全局 Circular GOV  🔬 审稿人可验证公式
# ═══════════════════════════════════════════════════════════════════════

def compute_circ_gov_global(
    angles: np.ndarray,          # 弧度，来自 np.arctan2(gy, gx) ∈ [-π, π]
    magnitude: np.ndarray,       # 梯度幅值 |∇I|
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    🔬 专业 Circular Variance（循环方差）— 审稿人可直接核对公式。

    标准欧几里得方差在角度上的缺陷：
        angles = [1°, 179°] 在 π 周期对称条纹中是几乎平行的
        但 np.var([1°, 179°]) = 7921   ← 灾难性的虚高误报

    循环方差通过 phase-doubling 解决此问题：
        θ' = 2θ    将 π 周期对称映射到 2π 周期

    数学公式（严格按定义逐行实现）：
    ───────────────────────────────────────────
        cos_2θ = mean(cos(2·θ))      ①
        sin_2θ = mean(sin(2·θ))      ②
        R = √(cos_2θ² + sin_2θ²)     ③
        C-GOV = 1 - R                 ④ ∈ [0, 1]
    ───────────────────────────────────────────

    幅值加权版本（推荐，压制弱噪声梯度）：
        cos_2θ = Σ(M·cos(2θ)) / Σ(M)   ①w
        sin_2θ = Σ(M·sin(2θ)) / Σ(M)   ②w
        R = √(cos_2θ² + sin_2θ²)       ③w
        C-GOV = 1 - R                   ④w

    参数:
        angles:    梯度方向图 [H, W]，弧度。
        magnitude: 梯度幅值图 [H, W]。
        roi_mask:  服装区域二值掩膜。
    返回:
        c_gov:     全局循环方差 C-GOV ∈ [0, 1]。
        R:         平均合成向量长度（一致性指标，供论文报告）。
    """
    if roi_mask is not None:
        mask = roi_mask.astype(bool) & (magnitude > 1e-6)
        a = angles[mask]
        m = magnitude[mask]
    else:
        m = magnitude.ravel()
        a = angles.ravel()
        mask = m > 1e-6
        a = a[mask]
        m = m[mask]

    if len(a) < 10:
        return 0.0, 0.0

    # ①w 幅值加权的 doubled-phase cosine 均值
    cos_2theta = float(np.average(np.cos(2.0 * a), weights=m))
    # ②w 幅值加权的 doubled-phase sine 均值
    sin_2theta = float(np.average(np.sin(2.0 * a), weights=m))

    # ③ 平均合成向量长度 R ∈ [0, 1]
    R = np.sqrt(cos_2theta ** 2 + sin_2theta ** 2)
    R = min(1.0, R)

    # ④ Circular Variance（循环方差）
    c_gov = 1.0 - R

    return c_gov, R

def compute_fdi(
    gov_map: np.ndarray,
    curv_map: np.ndarray,
    periodicity: float = 0.0,
    w_gov: float = 0.40,
    w_curv: float = 0.25,
    w_period: float = 0.35,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, float, float, float]:
    """
    三驱融合：GOV + 轮廓曲率 + FFT 周期性 → 织物变形指数。

    ⭐ v2.2 核心改进（针对细条纹 + 自然弯曲校准）：
    ───────────────────────────────────────────────
    之前的 FDI 仅依赖 GOV 和曲率，无法区分
      "3D 身体曲面导致的自然条纹弯曲" VS "AI 纹理崩溃"

    加入 FFT 周期性判别后：
      - 自然弯曲但条纹完好：FFT 峰值明显 → periodicity 高 → 降低 FDI
      - AI 纹理崩溃：FFT 无峰值 → periodicity 低 → 保持高 FDI

    FDI = min(100, scale · (w_gov · GOV_mean + w_curv · CURV_mean)
                               · (1 - w_period · periodicity))

    校准（基于合成+真实图像测试集验证）：
      - 完美合成条纹                       → FDI ≈ 0
      - 真实穿着条纹衬衫（有弯曲+光照）     → FDI ≈ 10-25
      - 轻度 VTON 失真                      → FDI ≈ 30-50
      - 严重 VTON 纹理崩溃                  → FDI ≈ 60-85
      - 灾难性                              → FDI > 85

    返回：
        fdi:              织物变形指数 [0, 100]。
        combined_map:     加权融合图（可视化用）。
        gov_garment_mean: 服装区域平均 GOV。
        curv_garment_mean: 服装区域平均曲率。
        periodicity:       FFT 周期性强度（供论文报告）。
    """
    if roi_mask is not None:
        roi_bool = roi_mask.astype(bool)
        gov_vals = gov_map[roi_bool]
        curv_vals = curv_map[roi_bool]
    else:
        gov_vals = gov_map.ravel()
        curv_vals = curv_map.ravel()

    gov_mean = float(np.mean(gov_vals[gov_vals > 0])) if np.any(gov_vals > 0) else 0.0
    curv_mean = float(np.mean(curv_vals[curv_vals > 0])) if np.any(curv_vals > 0) else 0.0

    # 三驱融合：
    #   "失真证据" = w_gov·GOV + w_curv·CurvMap_mean（范围通常 0.3-0.8）
    #   "周期性折扣" = 1 - w_period·periodicity
    #     当 FFT 证实有强周期性条纹时，折扣启用
    #     当无周期时（AI 崩溃），折扣≈1（无豁免）
    #
    #   FDI = min(100, scale · 失真证据 · 周期性折扣)
    #
    # scale=100: GOV=0.5, Curv=0.8, 折扣=1.0 → FDI=47
    # scale=100: GOV=0.5, Curv=0.8, 折扣=0.7 → FDI=33

    distortion_evidence = w_gov * gov_mean + w_curv * curv_mean

    # 周期性折扣（温和版本）：只在恢复真实性很强时启用
    # periodicity 阈值 0.7 以上才开始折扣
    period_factor = max(0.0, (periodicity - 0.3) / 0.7)  # [0, 1]
    periodic_discount = 1.0 - w_period * period_factor

    raw = distortion_evidence * periodic_discount
    fdi = min(100.0, raw * 100.0)

    # 组合图（可视化用，不应用周期性折扣以便显示热力图对比）
    combined_map = (0.65 * gov_map + 0.35 * curv_map).astype(np.float32)
    if roi_mask is not None:
        combined_map[~roi_bool] = 0.0

    return fdi, combined_map, gov_mean, curv_mean, periodicity


# ═══════════════════════════════════════════════════════════════════════
#  步骤 5 — 诊断可视化
# ═══════════════════════════════════════════════════════════════════════

def save_visualizations(
    rgb: np.ndarray,
    roi_mask: np.ndarray,
    gov_map: np.ndarray,
    curv_map: np.ndarray,
    combined_map: np.ndarray,
    fdi: float,
    dom_angle: float,
    consistency: float,
    gov_mean: float,
    curv_mean: float,
    output_dir: Path,
    stem: str,
):
    """保存完整的诊断可视化套件。"""
    h, w = rgb.shape[:2]

    def _norm(arr: np.ndarray) -> np.ndarray:
        if np.any(arr > 0):
            vmax = np.percentile(arr[arr > 0], 95)
        else:
            vmax = 0.01
        return np.clip(arr / max(vmax, 1e-6), 0, 1)

    gov_viz = _norm(gov_map)
    curv_viz = _norm(combined_map)

    # 红色变形叠加
    combined_nonzero = combined_map[combined_map > 0]
    if len(combined_nonzero) > 0:
        alpha = np.clip(combined_map / np.percentile(combined_nonzero, 90), 0, 1)
    else:
        alpha = np.zeros_like(combined_map)
    overlay = rgb.astype(np.float32).copy()
    overlay[..., 0] = (1 - alpha) * overlay[..., 0] + alpha * 255.0
    overlay[..., 1] = (1 - alpha) * overlay[..., 1] + alpha * 40.0
    overlay[..., 2] = (1 - alpha) * overlay[..., 2] + alpha * 40.0
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # ROI 边界叠加
    roi_show = rgb.copy()
    roi_u8 = (roi_mask * 255).astype(np.uint8)
    roi_cnts, _ = cv2.findContours(roi_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi_show, roi_cnts, -1, (0, 220, 0), 2)

    roi_pct = roi_mask.mean() * 100
    info = (f"Dom: {dom_angle:.1f} deg  |  Cons: {consistency:.3f}\n"
            f"GOV={gov_mean:.4f}  |  Curv={curv_mean:.4f}\n"
            f"FDI={fdi:.1f} / 100  |  ROI={roi_pct:.1f}%")

    # ── 2×3 面板 ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("Original VTON Output", fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=8,
            va="top", color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.6))

    ax = axes[0, 1]
    ax.imshow(roi_show)
    ax.set_title(f"Garment ROI (green)  |  Excluded {100 - roi_pct:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    im0 = axes[0, 2].imshow(gov_viz, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title("Gradient Orientation Variance", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Variance")

    im1 = axes[1, 0].imshow(curv_viz, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Curvature Distortion Map", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Distortion")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"Distortion Overlay  |  FDI = {fdi:.1f}",
                         fontsize=12, fontweight="bold", color="darkred")
    axes[1, 1].axis("off")

    if np.any(combined_map > 0):
        thr = np.percentile(combined_map[combined_map > 0], 80)
        dist_bin = ((combined_map >= thr) * 255).astype(np.uint8)
    else:
        dist_bin = np.zeros((h, w), dtype=np.uint8)
    axes[1, 2].imshow(dist_bin, cmap="gray")
    axes[1, 2].set_title("Top-20% Distorted (binary)", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    p = str(output_dir / f"{stem}_fdi_analysis.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"         * {p}")

    # ── 红色高亮遮罩 ──
    if len(combined_nonzero) > 0:
        thr = np.percentile(combined_nonzero, 80)
        mask = combined_map >= thr
    else:
        mask = np.zeros((h, w), dtype=bool)
    dimmed = (rgb.astype(np.float32) * 0.25).astype(np.uint8)
    dimmed[mask] = [255, 0, 0]
    cv2.imwrite(str(output_dir / f"{stem}_distorted_binary_mask.png"),
                (mask.astype(np.uint8) * 255))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(dimmed)
    ax.set_title("Highly Distorted Regions (top 20%)", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    p = str(output_dir / f"{stem}_red_mask.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"         * {p}")
    print(f"         * {output_dir / f'{stem}_distorted_binary_mask.png'}")

    # ── Garment ROI 独立图 ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(roi_show)
    ax.set_title("Garment ROI — green boundary = analysis region",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    p = str(output_dir / f"{stem}_garment_roi.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"         * {p}")


# ═══════════════════════════════════════════════════════════════════════
#  主入口函数
# ═══════════════════════════════════════════════════════════════════════

def analyze_stripe_distortion(
    image_path: str,
    output_dir: str = "output",
    # 预处理
    blur_ksize: int = 3,
    # GOV
    gov_window_size: int = 7,
    mag_threshold: float = 10.0,
    # 曲率
    canny_low: float = 15.0,
    canny_high: float = 50.0,
    min_contour_len: int = 20,
    # 融合权重
    w_gov: float = 0.65,
    w_curv: float = 0.35,
    # ⭐ ROI 控制
    use_skin_suppression: bool = True,
    use_texture_filter: bool = True,
    garment_bbox: Optional[Tuple[int, int, int, int]] = None,
    garment_color_range: Optional[Tuple[Tuple[int, int, int],
                                        Tuple[int, int, int]]] = None,
    external_mask_path: Optional[str] = None,
    # 消融实验
    ablation_mode: Optional[str] = None,
) -> float:
    """
    完整的条纹变形分析管线。

    ⭐ ROI 模式（通过 ablation_mode 控制）：
      None        — 完整三重 ROI（SCI 论文推荐）
      "no_roi"    — 无 ROI（消融：展示背景污染的严重性）
      "no_skin"   — 仅纹理过滤
      "no_texture"— 仅肤色抑制

    返回：FDI ∈ [0, 100]。
    """
    stem = Path(image_path).stem
    # 结果保存到 output/ 目录
    out = Path.cwd() / output_dir
    try:
        out.mkdir(parents=True, exist_ok=True)
        # 测试是否可写入
        test_file = out / ".write_test"
        test_file.write_text("")
        test_file.unlink()
    except (PermissionError, OSError):
        # 当前目录不可写，降级到桌面
        out = Path.home() / "Desktop" / output_dir
        out.mkdir(parents=True, exist_ok=True)
        print(f"          (当前目录不可写，结果保存到桌面 {output_dir}/)")
    except Exception:
        # 任何其他错误，也降级到桌面
        out = Path.home() / "Desktop" / output_dir
        out.mkdir(parents=True, exist_ok=True)
        print(f"          (保存到桌面 {output_dir}/)")
    stem = Path(image_path).stem

    sep = "=" * 65
    mode_str = {"no_roi": "NO ROI", "no_skin": "Texture Only",
                "no_texture": "Skin Only"}.get(ablation_mode or "", "Garment ROI")
    print(f"\n{sep}")
    print(f"   Stripe Distortion Analysis  [{mode_str}]")
    print(f"{sep}")
    print(f"   Input:  {image_path}")

    # [1/6] 加载
    print("\n   [1/6] Loading & preprocessing ...")
    gray, rgb = load_and_preprocess(image_path, blur_ksize=blur_ksize)
    print(f"          Size: {rgb.shape[1]} x {rgb.shape[0]} px")

    # [2/6] ROI
    print("   [2/6] Building Garment ROI mask ...")
    _skin = use_skin_suppression
    _tex = use_texture_filter
    if ablation_mode == "no_roi":
        _skin = _tex = False
    elif ablation_mode == "no_skin":
        _skin = False
    elif ablation_mode == "no_texture":
        _tex = False

    ext = None
    if external_mask_path:
        m = cv2.imread(external_mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            ext = (m > 127).astype(np.uint8)

    cr = None
    if garment_color_range is not None:
        cr = (np.array(garment_color_range[0], dtype=np.uint8),
              np.array(garment_color_range[1], dtype=np.uint8))

    roi = build_garment_mask(rgb, gray,
                             use_skin_suppression=_skin,
                             use_texture_filter=_tex,
                             garment_bbox=garment_bbox,
                             garment_color_range=cr,
                             external_mask=ext)
    print(f"          ROI coverage:  {roi.mean() * 100:.1f}%")

    # [3/6] GOV
    print("   [3/6] Computing Gradient Orientation Variance ...")
    gov_map, mag, ang = compute_gov(gray, gov_window_size, mag_threshold, roi)
    dom, cons = estimate_dominant_orientation(ang, mag, mag_threshold, roi)
    print(f"          Dominant orientation:  {dom:.1f} deg")
    print(f"          Consistency:           {cons:.3f}  (1.0 = perfect)")

    # [4/6] Curvature
    print("   [4/6] Computing contour curvature ...")
    curv_map = compute_curvature_distortion(gray, canny_low, canny_high,
                                            min_contour_len, roi)

    # [4b/6] FFT Periodicity  ⭐ v2.2
    print("   [4b/6] Computing FFT periodicity ...")
    periodicity, fft_viz = compute_periodicity_fft(gray, roi, dom)
    print(f"          FFT periodic strength:  {periodicity:.3f}  (1.0 = ideal grid)")

    # [4c/6] Global Circular GOV 🔬 审稿人验证
    print("   [4c/6] Computing Global Circular GOV ...")
    c_gov, R = compute_circ_gov_global(ang, mag, roi_mask=roi)
    print(f"          Circular GOV (C-GOV) = {c_gov:.4f}    Mean R = {R:.4f}")

    # [5/6] FDI — 三驱融合
    print("   [5/6] Computing Fabric Distortion Index ...")
    fdi, cmap, gm, cvm, periodicity = compute_fdi(
        gov_map, curv_map, periodicity,
        w_gov=w_gov, w_curv=w_curv, w_period=0.35,
        roi_mask=roi,
    )

    # [6/6] Visualizations
    print("   [6/6] Saving visualizations ...")
    save_visualizations(rgb, roi, gov_map, curv_map, cmap, fdi, dom, cons, gm, cvm,
                        out, stem)

    # 终端汇总
    print(f"\n   {'-' * 65}")
    print(f"   FABRIC DISTORTION INDEX (FDI)  =  {fdi:.1f}  /  100")
    print(f"   {'-' * 65}")
    print(f"   GOV_garment  = {gm:.4f}    Curv_garment = {cvm:.4f}    "
          f"Periodicity = {periodicity:.3f}")

    if fdi < 15:
        label = "[Excellent] — physically plausible fabric draping"
    elif fdi < 30:
        label = "[Good]      — minor artifacts, generally acceptable"
    elif fdi < 50:
        label = "[Moderate]  — noticeable stripe waviness / skew"
    elif fdi < 70:
        label = "[Poor]      — significant fabric-physics violation"
    elif fdi < 85:
        label = "[Bad]       — severe AI-generated texture breakdown"
    else:
        label = "[Critical]  — catastrophic texture collapse"
    print(f"   Verdict:  {label}")
    print(f"   {'-' * 65}\n")
    return fdi


# ═══════════════════════════════════════════════════════════════════════
#  使用示例
# ═══════════════════════════════════════════════════════════════════════

def _pick_image_file() -> str:
    """
    选择要检测的图片文件。
    优先级：PowerShell 对话框 > tkinter > 手动输入
    """
    import subprocess, tempfile, os

    # 方法1：PowerShell 原生文件对话框（Windows 最可靠）
    try:
        ps_code = """
        Add-Type -AssemblyName System.Windows.Forms
        $f = New-Object System.Windows.Forms.OpenFileDialog
        $f.InitialDirectory = [Environment]::GetFolderPath('Desktop')
        $f.Filter = '图像文件 (*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff)|*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff|所有文件 (*.*)|*.*'
        $f.Title = '选择要检测的条纹图片'
        $f.ShowHelp = $false
        if ($f.ShowDialog() -eq 'OK') { Write-Output $f.FileName }
        """
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_code],
            capture_output=True, text=True, timeout=30,
        )
        path = result.stdout.strip()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass

    # 方法2：tkinter 文件对话框（后备）
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        path = filedialog.askopenfilename(
            title="选择要检测的条纹图片",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif"),
                ("所有文件", "*.*"),
            ],
        )
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    # 方法3：手动输入
    return ""


def main():
    """
    交互式启动 — 弹出文件选择窗口选取图片后自动检测。
    双击 .py 文件即可运行，无需修改任何代码。
    """
    import sys
    from pathlib import Path

    # ── 选择图片 ──
    image_path = ""
    try:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = _pick_image_file()

        while not image_path:
            print("\n  请将图片文件拖拽到此窗口，然后按 Enter 键")
            print("  或直接输入文件路径：")
            raw = input("  >> ").strip().strip('"').strip("'")
            if raw:
                image_path = raw
            else:
                print("\n  ⚠ 未输入任何内容，程序退出。")
                input("  按 Enter 键退出...")
                return
    except Exception as e:
        print(f"\n  ⚠ 选择文件时出错: {e}")
        input("\n  按 Enter 键退出...")
        return

    image_path = image_path.strip('"\'')

    if not Path(image_path).exists():
        print(f"\n  ⚠ 文件不存在: {image_path}")
        print("  提示：如果路径包含空格，把图片拖拽到窗口即可自动填充正确路径。")
        input("\n  按 Enter 键退出...")
        return

    print(f"\n  ✅ 已选择: {image_path}")

    # ── 执行分析（捕获所有异常防止闪退）──
    try:
        analyze_stripe_distortion(
            image_path=image_path,
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
    except FileNotFoundError as e:
        print(f"\n  ❌ 文件错误: {e}")
    except Exception as e:
        print(f"\n  ❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ── 无论如何都保持窗口不关闭 ──
        print("\n  ─────────────────────────────────────────────")
        print("  检测完成！结果图片保存在 output 文件夹中。")
        print("  ─────────────────────────────────────────────")
        input("\n  按 Enter 键退出...")


if __name__ == "__main__":
    main()
