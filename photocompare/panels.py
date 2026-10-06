# -*- coding: utf-8 -*-
"""右侧面板：修图栏 / 分析结果 / 量化指标"""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QSlider, QDoubleSpinBox, QGroupBox,
                               QScrollArea, QPushButton, QTreeWidget,
                               QTreeWidgetItem, QComboBox, QFileDialog)

from .pipeline import PARAM_SPECS, PARAM_NAMES, PARAM_RANGE, PARAM_DESC, PARAM_FMT
from . import metrics

# 分组定义：组名 -> [keys]
GROUPS = [
    ("光线", ["exposure", "contrast", "highlights", "shadows", "gamma"]),
    ("颜色", ["temperature", "tint", "vibrance"]),
    ("细节", ["clarity", "sharpen", "grain_ignore", "bloom"]),
]


class _ParamRow(QWidget):
    """单参数行：名称 + 滑块 + 数值框"""
    changed = Signal(str, float)
    released = Signal()

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        key, name, lo, hi, dflt, fmt, desc = spec
        self.key = key
        self.lo, self.hi = lo, hi
        self.fmt = fmt

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 6, 1)
        lay.setSpacing(6)

        self.label = QLabel(name)
        self.label.setFixedWidth(76)
        self.label.setToolTip(desc)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setToolTip(desc)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(2)
        self.spin.setSingleStep((hi - lo) / 200.0)
        self.spin.setFixedWidth(72)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)

        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.slider.sliderReleased.connect(self.released)

        self._guard = False
        self.set_value(dflt)

    def _to_slider(self, v):
        return int(round((v - self.lo) / (self.hi - self.lo) * 1000))

    def _from_slider(self, s):
        return self.lo + (self.hi - self.lo) * s / 1000.0

    def set_value(self, v: float):
        self._guard = True
        v = float(np.clip(v, self.lo, self.hi))
        self.slider.setValue(self._to_slider(v))
        self.spin.setValue(v)
        self._guard = False

    def value(self) -> float:
        return float(self.spin.value())

    def _on_slider(self, s):
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(round(self._from_slider(s), 3))
        self._guard = False
        self.changed.emit(self.key, self.value())

    def _on_spin(self, v):
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(self._to_slider(v))
        self._guard = False
        self.changed.emit(self.key, float(v))


class AdjustPanel(QWidget):
    """修图栏：12 参数实时调节"""
    paramsChanged = Signal(dict)
    previewReleased = Signal()          # 滑块松开 → 高精度重算
    resetRequested = Signal()           # 用户点"全部归零"（主窗口需同时清色调曲线）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: dict[str, _ParamRow] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        self._hint = QLabel("拖动滑块实时修图；「分析修改」后自动填入检测参数")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#889; font-size:11px;")
        v.addWidget(self._hint)

        for gname, keys in GROUPS:
            box = QGroupBox(gname)
            bv = QVBoxLayout(box)
            bv.setContentsMargins(4, 4, 4, 4)
            bv.setSpacing(2)
            for spec in PARAM_SPECS:
                if spec[0] in keys:
                    row = _ParamRow(spec)
                    row.changed.connect(self._on_changed)
                    row.released.connect(self.previewReleased)
                    self.rows[spec[0]] = row
                    bv.addWidget(row)
            v.addWidget(box)
        v.addStretch(1)

        reset_btn = QPushButton("全部归零")
        reset_btn.clicked.connect(self.reset_all)
        v.addWidget(reset_btn)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _on_changed(self, key, val):
        self.paramsChanged.emit(self.values())

    def values(self) -> dict:
        return {k: r.value() for k, r in self.rows.items()}

    def set_params(self, params: dict):
        for k, v in params.items():
            if k in self.rows:
                self.rows[k].set_value(float(v))

    def reset_all(self):
        for r in self.rows.values():
            r.set_value(0.0)
        self.paramsChanged.emit(self.values())
        self.resetRequested.emit()


class AnalysisPanel(QWidget):
    """分析结果：自然语言结论 + 参数表 + 应用按钮"""
    applyRequested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)

        self.conclusion = QLabel("尚未分析。打开两张图后点「分析修改」。")
        self.conclusion.setWordWrap(True)
        self.conclusion.setStyleSheet("font-size:13px; font-weight:600; color:#e8c66a;")
        v.addWidget(self.conclusion)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["参数", "估计值", "说明"])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 110)
        self.tree.setColumnWidth(1, 110)
        v.addWidget(self.tree, 1)

        self.quality = QLabel("")
        self.quality.setStyleSheet("color:#8a8; font-size:12px;")
        v.addWidget(self.quality)

        hb = QHBoxLayout()
        self.apply_btn = QPushButton("应用到修图栏")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        hb.addWidget(self.apply_btn)
        hb.addStretch(1)
        v.addLayout(hb)

        self._params = {}

    def _apply(self):
        if self._params:
            self.applyRequested.emit(dict(self._params))

    def show_result(self, params: dict, fitted_de: float | None, fitted_ssim: float | None,
                    lut_desc: str | None = None):
        self._params = dict(params)
        # 结论
        from .analyzer import describe_params
        text = "检测到修改：" + describe_params(params)
        if lut_desc:
            text += f"另有{lut_desc}（已并入拟合，见处理后效果）。"
        self.conclusion.setText(text)
        # 表
        self.tree.clear()
        nonzero = {k: v for k, v in params.items() if abs(v) > 0.02}
        if not nonzero:
            item = QTreeWidgetItem(["—", "0", "未检测到明显修改"])
            self.tree.addTopLevelItem(item)
        for k, v in nonzero.items():
            val_txt = PARAM_FMT.get(k, "{:+.2f}").format(v)
            gamma_note = ""
            if k == "gamma" and abs(v) > 0.02:
                gamma_note = f"（γ={2**(-v):.2f}）"
            item = QTreeWidgetItem([PARAM_NAMES.get(k, k), val_txt,
                                    PARAM_DESC.get(k, "") + gamma_note])
            f = self.tree.font()
            f.setBold(True)
            item.setFont(1, f)
            self.tree.addTopLevelItem(item)
        for k in ("exposure", "contrast", "temperature", "vibrance", "sharpen", "clarity"):
            if k in nonzero:
                continue
            self.tree.addTopLevelItem(QTreeWidgetItem([PARAM_NAMES.get(k, k), "≈0", "无明显变化"]))
        if fitted_de is not None:
            q = "优" if fitted_de < 1 else ("良" if fitted_de < 2.5 else "一般")
            self.quality.setText(
                f"拟合质量：反推参数回放到低分辨率图 ΔE={fitted_de:.2f}（{q}）"
                + (f" · SSIM={fitted_ssim:.3f}" if fitted_ssim is not None else "")
                + "。参数为估计值，可用修图栏微调。")
        self.apply_btn.setEnabled(bool(self._params))


class MetricsPanel(QWidget):
    """量化指标：ΔE00 / SSIM / RMSE / PSNR + ΔE 热力图"""
    compareChanged = Signal(str)     # "AB" / "PB"

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        top.addWidget(QLabel("对比对象："))
        self.combo = QComboBox()
        self.combo.addItem("原图 A vs 修改图 B", "AB")
        self.combo.addItem("处理后 vs 修改图 B（验证拟合）", "PB")
        self.combo.currentIndexChanged.connect(
            lambda _: self.compareChanged.emit(self.combo.currentData()))
        top.addWidget(self.combo, 1)
        v.addLayout(top)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.de_label = self._big_label("—")
        self.de_eval = QLabel("")
        self.ssim_label = self._big_label("—")
        self.ssim_eval = QLabel("")
        self.rmse_label = self._big_label("—")
        self.psnr_label = self._big_label("—")
        grid.addWidget(self._title("ΔE00 平均色差"), 0, 0)
        grid.addWidget(self.de_label, 1, 0)
        grid.addWidget(self.de_eval, 2, 0)
        grid.addWidget(self._title("SSIM 结构相似度"), 0, 1)
        grid.addWidget(self.ssim_label, 1, 1)
        grid.addWidget(self.ssim_eval, 2, 1)
        grid.addWidget(self._title("RMSE"), 3, 0)
        grid.addWidget(self.rmse_label, 4, 0)
        grid.addWidget(self._title("PSNR (dB)"), 3, 1)
        grid.addWidget(self.psnr_label, 4, 1)
        v.addLayout(grid)

        v.addWidget(QLabel("ΔE 热力图（亮=差异大）："))
        self.heatmap = QLabel("—")
        self.heatmap.setMinimumHeight(150)
        self.heatmap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap.setStyleSheet("background:#0d0d11; border:1px solid #26262e;")
        v.addWidget(self.heatmap, 1)
        v.addStretch(0)

    def _title(self, t):
        lbl = QLabel(t)
        lbl.setStyleSheet("color:#99a; font-size:11px;")
        return lbl

    def _big_label(self, t):
        lbl = QLabel(t)
        f = QFont(self.font())
        f.setPointSizeF(f.pointSizeF() * 1.9)
        f.setBold(True)
        lbl.setFont(f)
        lbl.setStyleSheet("color:#fff;")
        return lbl

    @staticmethod
    def _de_review(de: float) -> str:
        if de < 1:
            return "＜1：人眼几乎看不出差异"
        if de < 3:
            return "1~3：仔细观察才能发现"
        if de < 6:
            return "3~6：明显不同"
        return "＞6：显著不同（滤镜级调色）"

    def compute(self, a: np.ndarray, b: np.ndarray, mask=None):
        """a,b: float [0,1] RGB（预览分辨率）；mask: 重叠区（对齐模式）"""
        if a is None or b is None:
            return
        if a.shape != b.shape:
            import cv2
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        if mask is not None and mask.shape != a.shape[:2]:
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (a.shape[1], a.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        de = metrics.delta_e_stats(a, b, mask=mask)
        ssim = metrics.ssim_rgb(a, b)
        m = metrics.mse_rmse_psnr(a, b)
        ch = metrics.channel_mean_shift(a, b)
        lum = metrics.luma_mean_shift(a, b)

        self.de_label.setText(f"{de['mean']:.2f}")
        self.de_eval.setText(self._de_review(de["mean"]) + f"（最大 {de['max']:.1f}）")
        self.ssim_label.setText(f"{ssim:.4f}")
        s = ssim
        se = ("几乎相同" if s > 0.99 else ("轻微差异" if s > 0.95 else
              ("明显差异" if s > 0.85 else "结构差异大")))
        self.ssim_eval.setText(f"{se}")
        self.rmse_label.setText(f"{m['rmse']*255:.1f}/255")
        self.psnr_label.setText(f"{m['psnr']:.1f}")

        hm = metrics.delta_e_heatmap_rgb(a, b)
        if mask is not None:
            # 重叠区外涂灰（未参与对比）
            hm[~mask] = (58, 58, 64)
        h, w = hm.shape[:2]
        qi = QImage(hm.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.heatmap.setPixmap(QPixmap.fromImage(qi).scaled(
            self.heatmap.width(), self.heatmap.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        # 通道均值偏移提示
        tips = []
        if abs(ch["R"] - ch["B"]) > 0.01:
            tips.append("R-B 通道偏移 → 色温" + ("偏暖" if ch["R"] > ch["B"] else "偏冷"))
        if abs(ch["G"]) > 0.015:
            tips.append("G 通道 " + ("+" if ch["G"] > 0 else "-") + f"{abs(ch['G'])*255:.0f}")
        if abs(lum) > 0.01:
            tips.append("亮度 " + ("↑" if lum > 0 else "↓") + f"{abs(lum)*100:.0f}%")
        if tips:
            self.de_eval.setToolTip("；".join(tips))
