# -*- coding: utf-8 -*-
"""真实样图基线分析：差异结构 + 现有算法拟合效果"""
import sys
import numpy as np
sys.path.insert(0, r"F:\workbuddy空间任务\修图对比器")
from photocompare import analyzer, pipeline, metrics, scopes
from photocompare.pipeline import DEFAULT_PARAMS

PAIRS = [
    ("组1(剪贴板)", r"C:\Users\stfizer\.workbuddy\clipboard-images\clipboard-2026-10-03T16-28-11-152Z-2d4f07e8.jpg",
     r"C:\Users\stfizer\.workbuddy\clipboard-images\clipboard-2026-10-03T16-28-11-154Z-4e0274d1.jpg"),
    ("组2(桌面110/111)", r"C:\Users\stfizer\Desktop\微信图片_20261004001109_110_3.jpg",
     r"C:\Users\stfizer\Desktop\微信图片_20261004001110_111_3.jpg"),
]

for name, pa, pb in PAIRS:
    print(f"\n===== {name} =====")
    try:
        a_full, a_prev, ia = analyzer.load_image(pa)
        b_full, b_prev, ib = analyzer.load_image(pb)
    except Exception as e:
        print("读取失败:", e)
        continue
    print(f"A: {ia}  B: {ib}")

    # 统一到共同尺寸
    H = min(a_prev.shape[0], b_prev.shape[0])
    W = min(a_prev.shape[1], b_prev.shape[1])
    import cv2
    A = cv2.resize(a_prev, (W, H), interpolation=cv2.INTER_AREA)
    B = cv2.resize(b_prev, (W, H), interpolation=cv2.INTER_AREA)

    # 真实差异结构
    de = metrics.delta_e_stats(A, B)
    ssim = metrics.ssim_rgb(A, B)
    ch = metrics.channel_mean_shift(A, B)
    la = A[...,0]*0.2126 + A[...,1]*0.7152 + A[...,2]*0.0722
    lb = B[...,0]*0.2126 + B[...,1]*0.7152 + B[...,2]*0.0722
    print(f"真实差异: dE={de['mean']:.2f} (p95={de['p95']:.1f}) SSIM={ssim:.4f} RMSE={metrics.mse_rmse_psnr(A,B)['rmse']*255:.1f}/255")
    print(f"  通道均值差 R{ch['R']:+.4f} G{ch['G']:+.4f} B{ch['B']:+.4f} | luma {lb.mean()-la.mean():+.4f} | std {la.std():.3f}->{lb.std():.3f}")

    # 现有算法拟合
    params, fitted, err, used = analyzer.analyze(A, B, work_size=256, rounds=3)
    nz = {k: round(v, 3) for k, v in params.items() if abs(v) > 0.02}
    print(f"现有算法检出: {nz}")
    # 拟合质量（在样图分辨率上评估）
    fitted_r = cv2.resize(fitted, (W, H), interpolation=cv2.INTER_AREA) if fitted.shape[:2] != (H, W) else fitted
    de_fit = metrics.delta_e_stats(fitted_r, B)
    print(f"拟合质量: dE={de_fit['mean']:.2f} (基线 dE={de['mean']:.2f}, 降消 {(1-de_fit['mean']/max(de['mean'],1e-6))*100:.0f}%)")
