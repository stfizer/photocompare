# -*- coding: utf-8 -*-
"""
图表数据生成：RGB 直方图 / 波形图(Waveform) / 矢量示波器(Vectorscope) / CIE xy 色域散点。
所有函数输入 float32 [0,1] RGB (H,W,3)，输出可直接渲染的 2D 密度/统计数组。
"""
from __future__ import annotations
import numpy as np
import cv2

# ---------------- RGB 直方图 ----------------

def rgb_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """返回 (3, bins) 各通道计数（归一化为最大值=1）"""
    out = np.zeros((3, bins), dtype=np.float64)
    for c in range(3):
        h = np.histogram(img[..., c], bins=bins, range=(0.0, 1.0))[0].astype(np.float64)
        out[c] = h
    mx = out.max()
    if mx > 0:
        out /= mx
    return out


# ---------------- 波形图 ----------------

def waveform_rgb(img: np.ndarray, width: int = 480, height: int = 256) -> np.ndarray:
    """
    RGB 波形图：横轴=图像列，纵轴=像素值 0..255，颜色密度累积。
    返回 (height, width, 3) float [0,1]，log 密度。专业示波器样式。
    """
    h, w = img.shape[:2]
    scale = width / w
    small = cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    cols = np.broadcast_to(np.arange(sw), (sh, sw))
    out = np.zeros((height, sw, 3), dtype=np.float64)
    for c in range(3):
        rows = np.clip((small[..., c] * (height - 1)).astype(np.int64), 0, height - 1)
        flat_idx = (rows * sw + cols).ravel()
        cnt = np.bincount(flat_idx, minlength=height * sw).reshape(height, sw)
        out[..., c] = cnt
    out = np.log1p(out)
    mx = out.max()
    if mx > 0:
        out /= mx
    return out.astype(np.float32)


# ---------------- 矢量示波器 ----------------

def rgb_to_yuv(img: np.ndarray):
    """BT.601 YUV；U,V 范围约 [-0.5,0.5]"""
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 0.492 * (B - Y)
    V = 0.877 * (R - Y)
    return Y, U, V


def vectorscope(img: np.ndarray, size: int = 512, scale: float = 0.6) -> np.ndarray:
    """
    矢量示波器：像素按其自身颜色累积到 (U,V) 平面。
    返回 (size,size,3) float [0,1]，密度做 log 映射。中心=无色，半径=饱和度，角度=色相。
    scale: (U,V) 显示半径（0.6 覆盖 75% 饱和色靶）
    """
    Y, U, V = rgb_to_yuv(img)
    iu = np.clip(((U / scale + 1.0) * 0.5 * (size - 1)).astype(np.int64), 0, size - 1)
    iv = np.clip(((1.0 - (V / scale + 1.0) * 0.5) * (size - 1)).astype(np.int64), 0, size - 1)
    flat = iv * size + iu
    cnt = np.bincount(flat.ravel(), minlength=size * size).astype(np.float64)
    acc = np.zeros((size * size, 3), dtype=np.float64)
    for c in range(3):
        acc[:, c] = np.bincount(flat.ravel(), weights=img[..., c].ravel(), minlength=size * size)
    valid = cnt > 0
    color = np.zeros((size * size, 3), dtype=np.float64)
    color[valid] = acc[valid] / cnt[valid, None]
    dens = np.zeros(size * size, dtype=np.float64)
    dens[valid] = np.log1p(cnt[valid])
    mx = dens.max()
    if mx > 0:
        dens /= mx
    out = color * dens[:, None]
    out = np.clip(out * 2.2, 0.0, 1.0)  # 增益让暗密度可见
    return out.reshape(size, size, 3).astype(np.float32)


# 75% 饱和色靶 (BT.601)，用于矢量示波器刻度
TARGETS_UV = {
    "R":  (0.75 * 0.492 * (0 - 0.299), 0.75 * 0.877 * (1 - 0.299)),
    "Mg": (0.75 * 0.492 * (1 - 0.413), 0.75 * 0.877 * (1 - 0.413)),
    "B":  (0.75 * 0.492 * (1 - 0.114), 0.75 * 0.877 * (0 - 0.114)),
    "Cy": (0.75 * 0.492 * (1 - 0.701), 0.75 * 0.877 * (0 - 0.701)),
    "G":  (0.75 * 0.492 * (0 - 0.587), 0.75 * 0.877 * (0 - 0.587)),
    "Yl": (0.75 * 0.492 * (0 - 0.886), 0.75 * 0.877 * (1 - 0.886)),
}


# ---------------- CIE xy 色域 ----------------

_M_SRGB_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# CIE 1931 2° 光谱轨迹 (10nm, 380-700nm)
SPECTRUM_XY = [
    (380, 0.1741, 0.0050), (390, 0.1738, 0.0049), (400, 0.1733, 0.0048),
    (410, 0.1726, 0.0048), (420, 0.1714, 0.0051), (430, 0.1689, 0.0069),
    (440, 0.1644, 0.0109), (450, 0.1566, 0.0177), (460, 0.1440, 0.0297),
    (470, 0.1241, 0.0578), (480, 0.0913, 0.1327), (490, 0.0454, 0.2950),
    (500, 0.0082, 0.5384), (510, 0.0139, 0.7502), (520, 0.0743, 0.8338),
    (530, 0.1547, 0.8059), (540, 0.2296, 0.7543), (550, 0.3016, 0.6923),
    (560, 0.3731, 0.6245), (570, 0.4441, 0.5547), (580, 0.5125, 0.4866),
    (590, 0.5752, 0.4242), (600, 0.6270, 0.3725), (610, 0.6658, 0.3340),
    (620, 0.6915, 0.3083), (630, 0.7079, 0.2920), (640, 0.7190, 0.2809),
    (650, 0.7270, 0.2720), (660, 0.7300, 0.2700), (670, 0.7320, 0.2680),
    (680, 0.7334, 0.2666), (690, 0.7344, 0.2656), (700, 0.7347, 0.2653),
]
D65_XY = (0.3127, 0.3290)
SRGB_PRIMARIES = {"R": (0.6400, 0.3300), "G": (0.3000, 0.6000), "B": (0.1500, 0.0600)}

# 绘图范围
GAMUT_X_RANGE = (0.0, 0.8)
GAMUT_Y_RANGE = (0.0, 0.9)


def rgb_to_xy(img: np.ndarray):
    """[0,1] RGB -> CIE xy（逐像素），Y<阈值 的点剔除"""
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_SRGB_XYZ.T
    sums = xyz.sum(axis=-1)
    safe = sums > 1e-6
    x = np.zeros_like(sums)
    y = np.zeros_like(sums)
    x[safe] = xyz[..., 0][safe] / sums[safe]
    y[safe] = xyz[..., 1][safe] / sums[safe]
    return x, y, safe


def gamut_scatter(img: np.ndarray, size: int = 512) -> np.ndarray:
    """
    像素 xy 色度散点（密度+原色着色）。
    返回 (size,size,3) float [0,1]。
    """
    x, y, valid = rgb_to_xy(img)
    x0, x1 = GAMUT_X_RANGE
    y0, y1 = GAMUT_Y_RANGE
    iu = np.clip(((x - x0) / (x1 - x0) * (size - 1)).astype(np.int64), 0, size - 1)
    iv = np.clip(((1.0 - (y - y0) / (y1 - y0)) * (size - 1)).astype(np.int64), 0, size - 1)
    iv = np.where(valid, iv, -1)
    keep = iv >= 0
    flat = (iv[keep] * size + iu[keep])
    cnt = np.bincount(flat, minlength=size * size).astype(np.float64)
    acc = np.zeros((size * size, 3), dtype=np.float64)
    for c in range(3):
        acc[:, c] = np.bincount(flat, weights=img[..., c].reshape(-1)[keep.reshape(-1)], minlength=size * size)
    m = cnt > 0
    color = np.zeros((size * size, 3), dtype=np.float64)
    color[m] = acc[m] / cnt[m, None]
    dens = np.zeros(size * size, dtype=np.float64)
    dens[m] = np.log1p(cnt[m])
    mx = dens.max()
    if mx > 0:
        dens /= mx
    out = color * dens[:, None]
    return np.clip(out * 2.2, 0.0, 1.0).reshape(size, size, 3).astype(np.float32)


def wavelength_to_rgb(nm: float):
    """近似可见光波长 -> RGB（用于光谱轨迹着色）"""
    if nm < 380 or nm > 700:
        return (0.0, 0.0, 0.0)
    if nm < 440:
        r, g, b = -(nm - 440) / 60.0, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50.0, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / 20.0
    elif nm < 580:
        r, g, b = (nm - 510) / 70.0, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, -(nm - 645) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # 边缘亮度衰减
    if nm < 420:
        f = 0.3 + 0.7 * (nm - 380) / 40.0
    elif nm > 660:
        f = 0.3 + 0.7 * (700 - nm) / 40.0
    else:
        f = 1.0
    gamma = 0.8
    return tuple(max(0.0, min(1.0, (v * f) ** gamma)) for v in (r, g, b))
