# -*- coding: utf-8 -*-
"""后台线程：自动对齐 / 参数分析 / 全分辨率处理，避免卡 UI"""
from __future__ import annotations
import traceback
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import analyzer, pipeline


class AnalyzeWorker(QThread):
    """自动对齐 + 参数反推分析（后台）"""
    progress = Signal(float, str)          # 0~1, 消息
    # params, fitted_preview, final_mse, tone_lut, b_aligned, align_mask, align_info
    finished_ok = Signal(dict, object, float, object, object, object, str)
    failed = Signal(str)

    def __init__(self, img_a, img_b, parent=None):
        super().__init__(parent)
        self.img_a = img_a
        self.img_b = img_b

    def run(self):
        try:
            def cb(frac, msg):
                self.progress.emit(frac * 0.9, msg)

            # 自动对齐（尺寸不一致或存在位移时；失败则按原图对比）
            b_aligned, amask, ainfo = None, None, ""
            try:
                b_aligned, amask, ainfo = analyzer.align_pair(self.img_a, self.img_b)
            except Exception:
                ainfo = ""
                b_aligned, amask = None, None
            img_b = b_aligned if b_aligned is not None else self.img_b

            params, fitted, mse, _, lut = analyzer.analyze(
                self.img_a, img_b, progress_cb=cb, work_size=256, rounds=3,
                region_mask=amask)
            self.finished_ok.emit(params, fitted, mse, lut, b_aligned, amask, ainfo)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ProcessWorker(QThread):
    """全分辨率应用参数（预览/固化/导出用），可选叠加色调曲线 LUT"""
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, img, params, parent=None, lut=None):
        super().__init__(parent)
        self.img = img
        self.params = params
        self.lut = lut

    def run(self):
        try:
            out = pipeline.apply_pipeline(self.img, self.params)
            if self.lut:
                out = pipeline.apply_tone_lut(out, self.lut)
            self.finished_ok.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())
