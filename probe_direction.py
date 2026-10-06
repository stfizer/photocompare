# -*- coding: utf-8 -*-
"""方向性验证：A→B 与 B→A 双向拟合对比"""
import sys
import numpy as np
import cv2

sys.path.insert(0, r"F:\workbuddy空间任务\修图对比器")
from photocompare import analyzer, metrics


def luma(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def de_mean(a, b):
    return float(metrics.ciede2000_map(metrics.srgb_to_lab(a), metrics.srgb_to_lab(b)).mean())


PAIRS = [
    ("组1(人像)", r"C:\Users\stfizer\.workbuddy\clipboard-images\clipboard-2026-10-03T16-28-11-152Z-2d4f07e8.jpg",
     r"C:\Users\stfizer\.workbuddy\clipboard-images\clipboard-2026-10-03T16-28-11-154Z-4e0274d1.jpg"),
    ("组2(合照)", r"C:\Users\stfizer\Desktop\微信图片_20261004001109_110_3.jpg",
     r"C:\Users\stfizer\Desktop\微信图片_20261004001110_111_3.jpg"),
]

for name, p1, p2 in PAIRS:
    print(f"\n===== {name} =====")
    _, a_prev, _ = analyzer.load_image(p1)
    _, b_prev, _ = analyzer.load_image(p2)
    H = min(a_prev.shape[0], b_prev.shape[0])
    W = min(a_prev.shape[1], b_prev.shape[1])
    A = cv2.resize(a_prev, (W, H), interpolation=cv2.INTER_AREA)
    B = cv2.resize(b_prev, (W, H), interpolation=cv2.INTER_AREA)

    for tag, src, dst in (("正问  A→B", A, B), ("反问  B→A", B, A)):
        params, fitted, mse, _ = analyzer.analyze(src, dst, work_size=256, rounds=3)
        fr = fitted if fitted.shape[:2] == (H, W) else cv2.resize(
            fitted, (W, H), interpolation=cv2.INTER_AREA)
        fr = np.clip(fr, 0, 1)
        nz = {k: round(v, 3) for k, v in params.items() if abs(v) > 0.02}
        base = de_mean(src, dst)
        fit = de_mean(fr, dst)
        ls, ld = luma(src), luma(dst)
        print(f"{tag}: 基线dE={base:.2f} 拟合dE={fit:.2f} (降消 {(1-fit/base)*100:.0f}%) "
              f"luma {ls.mean():+.4f}->{ld.mean():+.4f}")
        print(f"      检出: {nz}")
