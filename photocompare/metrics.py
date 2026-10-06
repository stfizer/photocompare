# -*- coding: utf-8 -*-
"""
量化指标：ΔE00 (CIEDE2000)、SSIM、MSE/RMSE、PSNR。
全部基于 numpy + cv2 向量化实现，无额外重依赖。
"""
from __future__ import annotations
import numpy as np
import cv2

# ---------------- sRGB -> Lab (D65) ----------------

_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    """img: (H,W,3) float [0,1] -> Lab (H,W,3)"""
    c = np.clip(img.astype(np.float64), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T / _D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ---------------- CIEDE2000 ----------------

def ciede2000_map(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """逐像素 CIEDE2000 色差, 返回 (H,W)"""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2.0
    Cb7 = Cb ** 7
    G = 0.5 * (1.0 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p))
    h1p = np.where(h1p < 0, h1p + 360.0, h1p)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, h1p)
    h2p = np.degrees(np.arctan2(b2, a2p))
    h2p = np.where(h2p < 0, h2p + 360.0, h2p)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) <= 180.0, hsum / 2.0,
                   np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0))
    hbp = np.where((C1p * C2p) == 0, hsum, hbp)

    T = (1.0
         - 0.17 * np.cos(np.radians(hbp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbp))
         + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))

    dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp ** 7
    RC = 2.0 * np.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    SL = 1.0 + 0.015 * (Lbp - 50.0) ** 2 / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC

    tL = dLp / SL
    tC = dCp / SC
    tH = dHp / SH
    return np.sqrt(np.maximum(tL * tL + tC * tC + tH * tH + RT * tC * tH, 0.0))


def delta_e_stats(img_a: np.ndarray, img_b: np.ndarray, mask=None) -> dict:
    """两张 [0,1] RGB 图的 ΔE00 统计；mask 给定时只统计重叠区"""
    lab1 = srgb_to_lab(img_a)
    lab2 = srgb_to_lab(img_b)
    de = ciede2000_map(lab1, lab2)
    if mask is not None and de[mask].size >= 100:
        vals = de[mask]
    else:
        vals = de.ravel()
    return {
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "p95": float(np.percentile(vals, 95)),
    }


def delta_e_heatmap_rgb(img_a: np.ndarray, img_b: np.ndarray, scale: float = 6.0) -> np.ndarray:
    """ΔE 热力图 -> uint8 RGB (H,W,3)。scale: ΔE 达到多少算满刻度"""
    lab1 = srgb_to_lab(img_a)
    lab2 = srgb_to_lab(img_b)
    de = ciede2000_map(lab1, lab2)
    de8 = np.clip(de / scale * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(de8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------- SSIM ----------------

def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """a,b: (H,W) float [0,1]。11x11 高斯窗 σ=1.5（标准配置）"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu1, mu2 = g(a), g(b)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    s1 = g(a * a) - mu1_sq
    s2 = g(b * b) - mu2_sq
    cov = g(a * b) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float((num / den).mean())


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """对亮度通道计算 SSIM（兼顾亮度/对比度/结构）"""
    la = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
    lb = b[..., 0] * 0.2126 + b[..., 1] * 0.7152 + b[..., 2] * 0.0722
    return ssim_gray(la, lb)


# ---------------- 简单指标 ----------------

def mse_rmse_psnr(a: np.ndarray, b: np.ndarray) -> dict:
    d = (a.astype(np.float64) - b.astype(np.float64))
    mse = float((d * d).mean())
    rmse = float(np.sqrt(mse))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 1e-12 else 99.0
    return {"mse": mse, "rmse": rmse, "psnr": psnr}


def channel_mean_shift(a: np.ndarray, b: np.ndarray) -> dict:
    """R/G/B 通道均值变化（帮助判断色温/色调偏移方向）"""
    out = {}
    for i, name in enumerate(("R", "G", "B")):
        out[name] = float(b[..., i].mean() - a[..., i].mean())
    return out


def luma_mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    la = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
    lb = b[..., 0] * 0.2126 + b[..., 1] * 0.7152 + b[..., 2] * 0.0722
    return float(lb.mean() - la.mean())
