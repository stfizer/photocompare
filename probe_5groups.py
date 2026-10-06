# -*- coding: utf-8 -*-
"""5 组真实样图基线 + 差异结构诊断（一轮输出，供优化决策）"""
import sys
import numpy as np
import cv2

sys.path.insert(0, r"F:\workbuddy空间任务\修图对比器")
from photocompare import analyzer, metrics

PAIRS = [
    ("组1 凯奇超人",
     r"C:\Users\stfizer\Desktop\尼古拉斯凯奇超人.jpg",
     r"C:\Users\stfizer\Desktop\尼古拉斯凯奇超人-p过.jpg"),
    ("组2 女演员",
     r"C:\Users\stfizer\Desktop\女超人演员.jpg",
     r"C:\Users\stfizer\Desktop\女超人演员-p过.jpg"),
    ("组3 COS(108)",
     r"C:\Users\stfizer\.workbuddy\clipboard-images\clipboard-2026-10-05T04-39-48-363Z-2d4f07e8.jpg",
     r"C:\Users\stfizer\.workbuddy\clipboard-images\clipboard-2026-10-05T04-39-48-365Z-0e0d772d.jpg"),
    ("组4 合照(110)",
     r"C:\Users\stfizer\Desktop\微信图片_20261004001109_110_3.jpg",
     r"C:\Users\stfizer\Desktop\微信图片_20261004001109_110_3-p过.jpg"),
    ("组5 train报点",
     r"C:\Users\stfizer\Desktop\train报点.jpg",
     r"C:\Users\stfizer\Desktop\train报点-p过.jpg"),
]


def luma(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def de_mean(a, b):
    return float(metrics.ciede2000_map(metrics.srgb_to_lab(a), metrics.srgb_to_lab(b)).mean())


def quantile_lut_rgb(A, B, n=257):
    """A→B 每通道分位数 LUT（全局曲线上限）"""
    out = np.empty_like(A)
    qs = np.linspace(0, 1, n)
    for c in range(3):
        sa = np.quantile(A[..., c], qs)
        sb = np.quantile(B[..., c], qs)
        lut = np.interp(np.linspace(0, 1, 1025), sa, sb)
        out[..., c] = lut[np.clip(A[..., c] * 1024, 0, 1024).astype(np.int32)]
    return out


def tone_slope(la, lb):
    m = (la > 0.03) & (la < 0.97)
    if m.sum() < 500:
        m = np.ones_like(la, bool)
    k, b = np.polyfit(la[m], lb[m], 1)
    return float(k), float(b)


def jpeg_block(img):
    g = (luma(img) * 255).astype(np.float32)
    gx = np.abs(np.diff(g, axis=1)); gy = np.abs(np.diff(g, axis=0))
    bx = np.zeros(gx.shape[1], bool); bx[7::8] = True; bx[-1] = False
    by = np.zeros(gy.shape[0], bool); by[7::8] = True; by[-1] = False
    return (gx[:, bx].mean() / (gx[:, ~bx].mean() + 1e-9)
            + gy[by, :].mean() / (gy[~by, :].mean() + 1e-9)) / 2


for name, pa, pb in PAIRS:
    print(f"\n===== {name} =====")
    try:
        _, a_prev, ia = analyzer.load_image(pa)
        _, b_prev, ib = analyzer.load_image(pb)
    except Exception as e:
        print("读取失败:", e)
        continue
    H = min(a_prev.shape[0], b_prev.shape[0])
    W = min(a_prev.shape[1], b_prev.shape[1])
    A = cv2.resize(a_prev, (W, H), interpolation=cv2.INTER_AREA)
    B = cv2.resize(b_prev, (W, H), interpolation=cv2.INTER_AREA)
    la, lb = luma(A), luma(B)
    ch = [f"{n}{B[...,c].mean()-A[...,c].mean():+.3f}" for c, n in enumerate("RGB")]
    ka, ba_ = tone_slope(la, lb)
    kb, bb_ = tone_slope(lb, la)   # 反向斜率（可逆性参考）
    print(f"[{ia} -> {ib}] dE={de_mean(A,B):.2f} luma {la.mean():.4f}->{lb.mean():.4f} "
          f"std {la.std():.4f}->{lb.std():.4f}")
    print(f"  通道均值差 {' '.join(ch)} | 正向斜率 k={ka:.3f} b={ba_:+.3f} "
          f"反向 k={kb:.3f} | jpeg块 A={jpeg_block(A):.3f} B={jpeg_block(B):.3f}")

    # 全局 LUT 上限
    ref = quantile_lut_rgb(A, B)
    print(f"  RGB-LUT上限 dE={de_mean(ref, B):.2f} (基线 {de_mean(A,B):.2f})")

    # 现有算法
    params, fitted, mse, _, lut = analyzer.analyze(A, B, work_size=256, rounds=3)
    fr = fitted if fitted.shape[:2] == (H, W) else cv2.resize(
        fitted, (W, H), interpolation=cv2.INTER_AREA)
    fr = np.clip(fr, 0, 1)
    nz = {k: round(v, 3) for k, v in params.items() if abs(v) > 0.02}
    fit = de_mean(fr, B)
    base = de_mean(A, B)
    print(f"  检出: {nz}")
    print(f"  拟合: dE {base:.2f}->{fit:.2f} (降消 {(1-fit/base)*100:.0f}%)")
