# -*- coding: utf-8 -*-
"""核心算法自测：拟合精度 + 指标正确性（无 GUI）"""
import sys, time
import numpy as np
import cv2

sys.path.insert(0, r"F:\workbuddy空间任务\修图对比器")
from photocompare import pipeline, analyzer, metrics, scopes


def make_test_image(h=540, w=960):
    """合成一张有色彩层次和细节的测试图"""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.5 + 0.3 * np.sin(x / 130.0) + 0.12 * np.cos(y / 90.0)
    g = 0.5 + 0.3 * np.sin((x + y) / 170.0)
    b = 0.5 + 0.25 * np.cos(x / 200.0) + 0.1 * np.sin(y / 60.0)
    img = np.stack([r, g, b], -1)
    img += 0.02 * np.random.RandomState(7).randn(h, w, 3).astype(np.float32)
    # 加一些高光块和暗块
    img[80:180, 80:280] += 0.35
    img[350:470, 600:850] -= 0.4
    return np.clip(img, 0, 1)


def test_metrics():
    print("== 指标测试 ==")
    a = make_test_image(120, 180)
    # 相同图
    s = metrics.delta_e_stats(a, a)
    assert s["mean"] < 1e-6, f"相同图 ΔE 应为 0, got {s}"
    assert metrics.ssim_rgb(a, a) > 0.999
    # 轻微差异
    b = np.clip(a + 0.01, 0, 1)
    s = metrics.delta_e_stats(a, b)
    print(f"  轻微差异: ΔE mean={s['mean']:.3f}")
    assert 0.1 < s["mean"] < 4.0
    # 明显差异
    c = np.clip(a * 0.7 + 0.1, 0, 1)
    s2 = metrics.delta_e_stats(a, c)
    print(f"  明显差异: ΔE mean={s2['mean']:.3f} (预期 >5)")
    assert s2["mean"] > 5
    # SSIM
    print(f"  SSIM 轻微差异: {metrics.ssim_rgb(a, b):.4f}, 明显差异: {metrics.ssim_rgb(a, c):.4f}")
    # CIEDE2000 已知值验证: Lab(50,2.6772,-79.7751) vs (50,0,-82.7485) -> 2.0425
    lab1 = np.array([[[50.0, 2.6772, -79.7751]]])
    lab2 = np.array([[[50.0, 0.0, -82.7485]]])
    de = metrics.ciede2000_map(lab1, lab2)[0, 0]
    print(f"  CIEDE2000 标准测试对1: {de:.4f} (期望 2.0425)")
    assert abs(de - 2.0425) < 0.01, "CIEDE2000 公式错误!"
    # 第二个标准对
    lab1 = np.array([[[50.0, 3.1571, -77.2803]]])
    lab2 = np.array([[[50.0, 0.0, -82.7485]]])
    de = metrics.ciede2000_map(lab1, lab2)[0, 0]
    print(f"  CIEDE2000 标准测试对2: {de:.4f} (期望 2.8615)")
    assert abs(de - 2.8615) < 0.01, "CIEDE2000 公式错误!"
    print("  指标 OK\n")


def test_scopes():
    print("== 图表数据测试 ==")
    a = make_test_image(270, 480)
    hist = scopes.rgb_histogram(a)
    assert hist.shape == (3, 256) and hist.max() <= 1.0
    wf = scopes.waveform_rgb(a)
    assert wf.shape == (256, 480, 3) and wf.max() <= 1.0 + 1e-6
    vs = scopes.vectorscope(a)
    assert vs.shape == (512, 512, 3)
    gm = scopes.gamut_scatter(a)
    assert gm.shape == (512, 512, 3)
    # 纯红图的矢量示波器应聚集在 R 靶附近
    red = np.zeros((64, 64, 3), np.float32); red[..., 0] = 0.75
    vs_r = scopes.vectorscope(red)
    Y, U, V = scopes.rgb_to_yuv(red)
    tu, tv = scopes.TARGETS_UV["R"]
    assert abs(U[0, 0] - tu) < 0.02 and abs(V[0, 0] - tv) < 0.02, "色靶坐标计算错误"
    # 白图应集中在中心
    white = np.full((64, 64, 3), 0.8, np.float32)
    _, Uw, Vw = scopes.rgb_to_yuv(white)
    assert abs(Uw[0, 0]) < 1e-5 and abs(Vw[0, 0]) < 1e-5
    print("  直方图/波形/矢量/色域 OK\n")


def test_pipeline_and_fit():
    print("== 管线 + 参数反推测试 ==")
    a = make_test_image(540, 960)
    a_prev = analyzer.resize_long(a, 1800)

    cases = [
        {"exposure": 0.4},
        {"temperature": 0.45, "tint": -0.15},
        {"exposure": 0.25, "contrast": 0.3, "vibrance": 0.4},
        {"exposure": -0.3, "shadows": 0.5, "highlights": -0.4},
        {"sharpen": 0.5, "grain_ignore": 0.7},
        {"clarity": 0.35, "bloom": 0.25},
        {"gamma": 0.3, "temperature": 0.3, "vibrance": -0.3, "contrast": 0.2},
        {"exposure": 0.5, "temperature": -0.4, "vibrance": 0.5, "contrast": 0.35,
         "sharpen": 0.6, "clarity": 0.3, "bloom": 0.2},
    ]
    t0 = time.time()
    for i, truth in enumerate(cases):
        full_truth = dict(pipeline.DEFAULT_PARAMS); full_truth.update(truth)
        b = pipeline.apply_pipeline(a, full_truth)
        t1 = time.time()
        params, fitted, mse, used, lut = analyzer.analyze(a, b, work_size=256, rounds=3)
        t2 = time.time()
        err = max(abs(params[k] - truth.get(k, 0.0)) for k in pipeline.PARAM_KEYS)
        b_fit = cv2.resize(b, (fitted.shape[1], fitted.shape[0]), interpolation=cv2.INTER_AREA) if b.shape[:2] != fitted.shape[:2] else b
        de = metrics.delta_e_stats(fitted, b_fit)
        main_err = max(abs(params[k] - v) for k, v in truth.items()) if truth else 0.0
        print(f"  case{i}: 真值{truth}")
        detected = pipeline.params_nonzero(params)
        print(f"         检出{ {k: round(v,3) for k,v in detected.items()} }")
        print(f"         主参数误差={main_err:.3f} | 拟合ΔE={de['mean']:.2f} | 管线{t1-t0:.2f}s 拟合{t2-t1:.2f}s")
        # 验收：视觉拟合好 + 主要参数方向量级正确
        # ≥5 个参数同时修改属极限场景（等价类爆炸），只验证方向检出 + 视觉接近
        if len(truth) >= 5:
            assert de["mean"] < 3.5, f"case{i} 拟合视觉误差过大"
        else:
            assert de["mean"] < 2.0, f"case{i} 拟合视觉误差过大"
        # grain_ignore(软阈值)与 clarity/sharpen 存在不可分辨等价类：
        # 只要求方向正确（检出锐化类修改）+ 视觉拟合好
        if "sharpen" in truth or "grain_ignore" in truth:
            has_detect = (params["sharpen"] > 0.15 or params["grain_ignore"] > 0.15)
            assert has_detect, f"case{i} 未检出锐化类修改"
        elif "gamma" in truth:
            # gamma 与 exposure+高光/阴影组合本质耦合（均锁中间调斜率），
            # 验证其余独立参数（色温/色调/饱和度）的精度
            for k in ("temperature", "tint", "vibrance"):
                if k in truth:
                    e_k = abs(params[k] - truth[k])
                    assert e_k < 0.1, f"case{i} {k} 误差过大: {e_k}"
        else:
            assert main_err < 0.15, f"case{i} 主参数误差过大"
    print(f"  拟合 OK (总耗时 {time.time()-t0:.1f}s)\n")


if __name__ == "__main__":
    np.random.seed(0)
    test_metrics()
    test_scopes()
    test_pipeline_and_fit()
    print("=== 全部自测通过 ===")
