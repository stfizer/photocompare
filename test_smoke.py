# -*- coding: utf-8 -*-
"""无头冒烟测试：实例化 UI + 走通核心业务流（不显示窗口）"""
import os, sys, traceback
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"F:\workbuddy空间任务\修图对比器")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from photocompare import pipeline, analyzer

def make_image(h=540, w=960, seed=1):
    rs = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.5 + 0.3 * np.sin(x / 130.0) + 0.12 * np.cos(y / 90.0)
    g = 0.5 + 0.3 * np.sin((x + y) / 170.0)
    b = 0.5 + 0.25 * np.cos(x / 200.0)
    img = np.stack([r, g, b], -1) + 0.02 * rs.randn(h, w, 3).astype(np.float32)
    img[80:180, 80:280] += 0.3
    img[350:470, 600:850] -= 0.35
    return np.clip(img, 0, 1)

def main():
    app = QApplication(sys.argv)
    print("[1] import main_window")
    from photocompare.main_window import MainWindow
    win = MainWindow()
    print("[2] MainWindow created")

    # 模拟加载数据（绕过文件对话框）
    a = make_image(540, 960, seed=1)
    truth = dict(pipeline.DEFAULT_PARAMS)
    truth.update({"exposure": 0.35, "temperature": 0.4, "vibrance": 0.3, "contrast": 0.2})
    b = pipeline.apply_pipeline(a, truth)

    win.img_a_full, win.prev_a = a, analyzer.resize_long(a, 1800)
    win.img_b_full, win.prev_b = b, analyzer.resize_long(b, 1800)
    win.viewer.set_image("A", (np.clip(win.prev_a,0,1)*255).astype(np.uint8))
    win.viewer.set_image("B", (np.clip(win.prev_b,0,1)*255).astype(np.uint8))
    win._refresh_charts()
    win._refresh_metrics()
    win._update_actions()
    print("[3] images loaded, charts/metrics refreshed")

    # 模式切换
    for m in ("split", "wipe", "blink", "single", "wipe"):
        win._set_mode(m)
    win.viewer.set_grayscale(True); win.viewer.set_grayscale(False)
    print("[4] modes cycled")

    # 模拟分析（同步跑 analyze）
    params, fitted, err, used, lut = analyzer.analyze(win.prev_a, win.prev_b, work_size=256, rounds=3)
    win._on_analyze_done(params, fitted, err, lut, None, None, "")
    print("[5] analyze done:", {k: round(v,2) for k,v in params.items() if abs(v)>0.02})

    # 模拟滑块变化 -> 预览管线（同步调度）
    win.proc_params = dict(params)
    win._schedule_preview(hi=False)
    win._proc_worker.wait(60000)
    app.processEvents()
    assert "P" in win.viewer.images, "P 未生成"
    print("[6] preview pipeline OK, P generated")

    # 图表源切换
    for i in range(3):
        win.combo_chart_src.setCurrentIndex(i)
    print("[7] chart sources cycled")

    # 指标双模式
    win.metrics_panel.combo.setCurrentIndex(0)
    win._refresh_metrics()
    win.metrics_panel.combo.setCurrentIndex(1)
    win._refresh_metrics()
    print("[8] metrics both modes OK, dE=", round(win.metrics_panel.de_label.text() and float(win.metrics_panel.de_label.text()), 3))

    # 灰度图性能：Wipe 绘制
    from PySide6.QtGui import QPixmap
    win.viewer.grab()
    print("[9] viewer rendered (grab) OK")

    print("=== 冒烟测试全部通过 ===")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
