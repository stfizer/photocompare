# -*- coding: utf-8 -*-
"""主窗口：工具栏 / 对比查看器 / 修图栏 / 分析 / 图表 / 指标 的组装与业务流"""
from __future__ import annotations
import os
import traceback
import numpy as np
import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QImage, QPixmap, QIcon
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                               QSplitter, QTabWidget, QToolBar, QFileDialog,
                               QLabel, QComboBox, QStatusBar, QMessageBox,
                               QToolButton, QButtonGroup, QStyle)

from . import analyzer, pipeline, metrics, scopes
from .viewer import CompareViewer, SLOT_NAMES
from .charts import HistogramChart, WaveformChart, VectorscopeChart, GamutChart
from .panels import AdjustPanel, AnalysisPanel, MetricsPanel
from .worker import AnalyzeWorker, ProcessWorker

IMG_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;所有文件 (*.*)"
PREVIEW_LONG = 1800     # 对比查看用
CHART_LONG = 720        # 图表/指标用
PROC_PREVIEW_LONG = 1200  # 滑块实时预览用


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("修图参数对比器")
        self.resize(1560, 920)

        # 数据
        self.img_a_full: np.ndarray | None = None   # float [0,1] 全分辨率
        self.img_b_full: np.ndarray | None = None
        self.prev_a: np.ndarray | None = None       # 长边 PREVIEW_LONG
        self.prev_b: np.ndarray | None = None
        self.proc_params: dict = dict(pipeline.DEFAULT_PARAMS)
        self.proc_lut: dict | None = None      # 分析得到的残差色调曲线
        self.align_mask = None                 # 自动对齐的重叠区（预览尺寸）
        self._analyze_worker: AnalyzeWorker | None = None
        self._proc_worker: ProcessWorker | None = None
        self._proc_pending = False
        self._proc_hi = False
        self._last_dir = os.path.expanduser("~")
        self._chart_imgs: dict[str, np.ndarray] = {}

        self._build_ui()
        self._connect()
        self._update_actions()

    # ---------------- UI 构建 ----------------

    def _build_ui(self):
        # 工具栏
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize() * 1.1)
        self.addToolBar(tb)

        st = self.style()
        self.act_open_a = QAction(st.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
                                  "打开原图 A", self)
        self.act_open_a.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open_b = QAction(st.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
                                  "打开修改图 B", self)
        self.act_open_b.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_analyze = QAction(st.standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
                                   "分析修改", self)
        self.act_analyze.setShortcut(QKeySequence("F5"))
        self.act_bake = QAction(st.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton),
                                "参数固化到 A", self)
        self.act_export = QAction(st.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
                                  "导出处理图", self)
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        for a in (self.act_open_a, self.act_open_b, self.act_analyze,
                  self.act_bake, self.act_export):
            tb.addAction(a)
            a.triggered.connect(self._dispatch(a))

        # 视图工具条
        vt = QToolBar("视图")
        vt.setMovable(False)
        self.addToolBarBreak()
        self.addToolBar(vt)

        self.viewer = CompareViewer()
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for text, mode in (("并排", "split"), ("滑块擦除", "wipe"),
                           ("闪烁", "blink"), ("单图", "single")):
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self._mode_group.addButton(btn)
            btn.clicked.connect(lambda _, m=mode: self._set_mode(m))
            vt.addWidget(btn)
            if mode == "wipe":
                btn.setChecked(True)
        self._mode_buttons = {b.text(): b for b in self._mode_group.buttons()}

        vt.addSeparator()
        vt.addWidget(QLabel(" 左: "))
        self.combo_left = QComboBox()
        self.combo_right = QComboBox()
        for c in (self.combo_left, self.combo_right):
            c.addItems(["原图 A", "处理后", "修改图 B"])
            c.currentIndexChanged.connect(self._sources_changed)
        vt.addWidget(self.combo_left)
        vt.addWidget(QLabel(" 右: "))
        vt.addWidget(self.combo_right)

        vt.addSeparator()
        self.btn_gray = QToolButton()
        self.btn_gray.setText("灰度对比")
        self.btn_gray.setCheckable(True)
        self.btn_gray.setToolTip("灰度分离：剥离色彩，只看影调/明暗/对比度")
        self.btn_gray.clicked.connect(self._toggle_gray)
        vt.addWidget(self.btn_gray)

        self.btn_fit = QToolButton()
        self.btn_fit.setText("适配窗口")
        self.btn_fit.clicked.connect(self.viewer.fit)
        vt.addWidget(self.btn_fit)

        # 中央布局
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Orientation.Horizontal)
        lay.addWidget(split)

        split.addWidget(self.viewer)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(360)

        self.adjust_panel = AdjustPanel()
        self.analysis_panel = AnalysisPanel()
        self.metrics_panel = MetricsPanel()

        # 图表
        chart_w = QWidget()
        cv = QVBoxLayout(chart_w)
        cv.setContentsMargins(2, 2, 2, 2)
        top = QHBoxLayout()
        top.addWidget(QLabel("波形/矢量/色域 显示:"))
        self.combo_chart_src = QComboBox()
        self.combo_chart_src.addItems(["原图 A", "修改图 B", "处理后"])
        self.combo_chart_src.setCurrentIndex(1)
        self.combo_chart_src.currentIndexChanged.connect(self._chart_src_changed)
        top.addWidget(self.combo_chart_src)
        top.addStretch(1)
        cv.addLayout(top)
        self.chart_hist = HistogramChart()
        self.chart_wave = WaveformChart()
        self.chart_vec = VectorscopeChart()
        self.chart_gamut = GamutChart()
        grid_like = QVBoxLayout()
        grid_like.addWidget(self.chart_hist, 1)
        grid_like.addWidget(self.chart_wave, 1)
        row2 = QHBoxLayout()
        row2.addWidget(self.chart_vec, 1)
        row2.addWidget(self.chart_gamut, 1)
        grid_like.addLayout(row2, 2)
        cv.addLayout(grid_like)

        self.tabs.addTab(self.adjust_panel, "修图栏")
        self.tabs.addTab(self.analysis_panel, "分析结果")
        self.tabs.addTab(chart_w, "图表")
        self.tabs.addTab(self.metrics_panel, "量化指标")
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 7)
        split.setStretchFactor(1, 3)
        split.setSizes([1050, 460])

        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_status = QLabel("就绪")
        self.status.addWidget(self.lbl_status, 1)

        self._progress_timer = QTimer(self)
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(130)
        self._preview_debounce.timeout.connect(lambda: self._schedule_preview(hi=False))

    def _dispatch(self, action):
        def handler():
            if action is self.act_open_a:
                self.open_image("A")
            elif action is self.act_open_b:
                self.open_image("B")
            elif action is self.act_analyze:
                self.start_analyze()
            elif action is self.act_bake:
                self.bake_to_a()
            elif action is self.act_export:
                self.export_processed()
        return handler

    # ---------------- 事件连接 ----------------

    def _connect(self):
        self.adjust_panel.paramsChanged.connect(self._on_params_changed)
        self.adjust_panel.previewReleased.connect(lambda: self._schedule_preview(hi=True))
        self.adjust_panel.resetRequested.connect(lambda: setattr(self, "proc_lut", None))
        self.analysis_panel.applyRequested.connect(self._apply_analysis_params)
        self.metrics_panel.compareChanged.connect(lambda _: self._refresh_metrics())
        self.viewer.set_status_callback(lambda s: self.lbl_status.setText(s))

    # ---------------- 图片加载 ----------------

    def open_image(self, slot: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"打开{'原图 A' if slot == 'A' else '修改图 B'}", self._last_dir, IMG_FILTER)
        if not path:
            return
        try:
            full, prev, info = analyzer.load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法读取图片：\n{e}")
            return
        self._last_dir = os.path.dirname(path)
        if slot == "A":
            self.img_a_full, self.prev_a = full, prev
            self.viewer.set_image("A", to_u8(prev))
        else:
            self.img_b_full, self.prev_b = full, prev
            self.viewer.set_image("B", to_u8(prev))
            self.align_mask = None     # 新 B 尚未对齐
            # 默认进入滑块对比
            self._set_mode("wipe")
            self.combo_left.setCurrentIndex(0)
            self.combo_right.setCurrentIndex(2)
        self.viewer.fit()
        self._refresh_charts()
        self._refresh_metrics()
        self._update_actions()
        name = "A" if slot == "A" else "B"
        self.lbl_status.setText(f"已加载{SLOT_NAMES.get(name)}：{os.path.basename(path)} · {info}")

    # ---------------- 参数/预览管线 ----------------

    def _on_params_changed(self, params: dict):
        self.proc_params = dict(params)
        self._preview_debounce.start()

    def _schedule_preview(self, hi: bool):
        if self.prev_a is None:
            return
        if self._proc_worker is not None and self._proc_worker.isRunning():
            self._proc_pending = True
            self._proc_hi = self._proc_hi or hi
            return
        src = self.prev_a if hi else analyzer.resize_long(self.prev_a, PROC_PREVIEW_LONG)
        self._proc_hi = hi
        self._proc_worker = ProcessWorker(src, self.proc_params, self, lut=self.proc_lut)
        self._proc_worker.finished_ok.connect(self._on_preview_done)
        self._proc_worker.failed.connect(self._on_worker_fail)
        self.lbl_status.setText("正在应用参数…")
        self._proc_worker.start()

    def _on_preview_done(self, out: np.ndarray):
        self.viewer.set_image("P", to_u8(out))
        self._chart_imgs["P"] = analyzer.resize_long(out, CHART_LONG)
        self._refresh_chart_values()
        self._refresh_metrics()
        self.lbl_status.setText("预览已更新（P=处理后）")
        self._proc_worker = None
        if self._proc_pending:
            self._proc_pending = False
            self._schedule_preview(hi=self._proc_hi)

    def _on_worker_fail(self, tb_text: str):
        self._proc_worker = None
        self.lbl_status.setText("处理失败")
        QMessageBox.critical(self, "错误", "处理失败：\n" + tb_text[-800:])

    def _apply_analysis_params(self, params: dict, lut: dict | None = None):
        self.proc_params = dict(params)
        self.proc_lut = lut
        self.adjust_panel.set_params(params)
        self._schedule_preview(hi=True)

    # ---------------- 模式/源 ----------------

    def _set_mode(self, mode: str):
        self.viewer.set_mode(mode)
        self.lbl_status.setText({
            "split": "并排对比：左右各自显示",
            "wipe": "滑块擦除对比：拖动白色手柄",
            "blink": "闪烁对比：两图定时交替",
            "single": "单图查看（源=左源下拉）",
        }.get(mode, ""))

    def _sources_changed(self):
        li = self.combo_left.currentIndex()
        ri = self.combo_right.currentIndex()
        slots = ["A", "P", "B"]
        self.viewer.set_sources(slots[li], slots[ri])
        if self.viewer.mode == "single":
            self.viewer.set_single_slot(slots[li])

    def _toggle_gray(self):
        self.viewer.set_grayscale(self.btn_gray.isChecked())

    # ---------------- 分析 ----------------

    def start_analyze(self):
        if self.prev_a is None or self.prev_b is None:
            QMessageBox.information(self, "提示", "请先打开原图 A 和修改图 B 两张图片。")
            return
        if self._analyze_worker is not None and self._analyze_worker.isRunning():
            return
        self.act_analyze.setEnabled(False)
        self.lbl_status.setText("分析中…（低分辨率拟合 → 高分辨率精化）")
        self._analyze_worker = AnalyzeWorker(self.prev_a, self.prev_b, self)
        self._analyze_worker.progress.connect(
            lambda f, m: self.lbl_status.setText(f"分析中 {f*100:.0f}% · {m}"))
        self._analyze_worker.finished_ok.connect(self._on_analyze_done)
        self._analyze_worker.failed.connect(self._on_worker_fail)
        self._analyze_worker.start()

    def _on_analyze_done(self, params: dict, fitted: np.ndarray, final_mse: float,
                         lut: dict | None, b_aligned, amask, align_info: str):
        self.act_analyze.setEnabled(True)
        # 自动对齐：用对齐后的 B 替换显示（P vs B 同坐标系可直接对比）
        if b_aligned is not None:
            b_clipped = np.clip(b_aligned, 0, 1)
            self.img_b_full = b_clipped
            self.prev_b = analyzer.resize_long(b_clipped, PREVIEW_LONG)
            self.viewer.set_image("B", to_u8(self.prev_b))
            self.viewer.fit()
            self._refresh_charts()
            self.lbl_status.setText(align_info)
        self.align_mask = amask     # 重叠区（预览尺寸），指标面板使用
        try:
            # 拟合质量：fitted(768) vs B(768)
            b768 = analyzer.resize_long(self.prev_b, 768)
            if fitted.shape[:2] != b768.shape[:2]:
                fitted_r = cv2.resize(fitted, (b768.shape[1], b768.shape[0]),
                                      interpolation=cv2.INTER_AREA)
            else:
                fitted_r = fitted
            de = metrics.delta_e_stats(fitted_r, b768)["mean"]
            ssim = metrics.ssim_rgb(fitted_r, b768)
        except Exception:
            de = ssim = None

        # 填入修图栏（触发预览，预览/固化会叠加色调曲线）
        self._apply_analysis_params(params, lut)
        lut_desc = pipeline.describe_tone_lut(lut) if lut else None
        self.analysis_panel.show_result(params, de, ssim, lut_desc=lut_desc)
        self.tabs.setCurrentWidget(self.analysis_panel)
        # 对比源切到 处理后 vs 修改图（验证拟合效果）
        self.combo_left.setCurrentIndex(1)   # P
        self.combo_right.setCurrentIndex(2)  # B
        self._set_mode("wipe")
        self.metrics_panel.combo.setCurrentIndex(1)
        self.lbl_status.setText(
            f"分析完成：ΔE拟合={de:.2f}" if de is not None else "分析完成")
        self._update_actions()

    # ---------------- 固化 / 导出 ----------------

    def bake_to_a(self):
        if self.img_a_full is None:
            return
        if not QMessageBox.question(
                self, "确认", "把当前修图参数永久应用到原图 A？\n（A 将被替换为处理结果，参数清零）"):
            return
        self.lbl_status.setText("正在全分辨率应用参数…")
        w = ProcessWorker(self.img_a_full, self.proc_params, self, lut=self.proc_lut)
        w.finished_ok.connect(lambda out: self._on_bake_done(out))
        w.failed.connect(self._on_worker_fail)
        self._bake_worker = w
        w.start()

    def _on_bake_done(self, out: np.ndarray):
        self.img_a_full = out
        self.prev_a = analyzer.resize_long(out, PREVIEW_LONG)
        self.viewer.set_image("A", to_u8(self.prev_a))
        self.proc_lut = None
        self.adjust_panel.reset_all()
        self._refresh_charts()
        self._refresh_metrics()
        self.lbl_status.setText("已固化：A = 处理结果，参数已归零")

    def export_processed(self):
        if self.img_a_full is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出处理图", os.path.join(self._last_dir, "processed.png"),
            "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        self.lbl_status.setText("正在导出（全分辨率）…")
        w = ProcessWorker(self.img_a_full, self.proc_params, self, lut=self.proc_lut)
        w.finished_ok.connect(lambda out, p=path: self._on_export_done(out, p))
        w.failed.connect(self._on_worker_fail)
        self._export_worker = w
        w.start()

    def _on_export_done(self, out: np.ndarray, path: str):
        try:
            from PIL import Image
            Image.fromarray(to_u8(out)).save(path)
            self.lbl_status.setText(f"已导出：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ---------------- 图表 / 指标 ----------------

    def _refresh_charts(self):
        self._chart_imgs.pop("A", None)
        self._chart_imgs.pop("B", None)
        if self.prev_a is not None:
            self._chart_imgs["A"] = analyzer.resize_long(self.prev_a, CHART_LONG)
        if self.prev_b is not None:
            self._chart_imgs["B"] = analyzer.resize_long(self.prev_b, CHART_LONG)
        self._refresh_chart_values()

    def _refresh_chart_values(self):
        imgs = self._chart_imgs
        # 直方图（A/B 叠加）
        ha = scopes.rgb_histogram(imgs["A"]) if "A" in imgs else None
        hb = scopes.rgb_histogram(imgs["B"]) if "B" in imgs else None
        self.chart_hist.set_data(ha, hb)
        # 波形/矢量/色域（按源）
        wf, vs, gm = {}, {}, {}
        for k in ("A", "B", "P"):
            if k in imgs:
                wf[k] = scopes.waveform_rgb(imgs[k])
                vs[k] = scopes.vectorscope(imgs[k])
                gm[k] = scopes.gamut_scatter(imgs[k])
        self.chart_wave.set_data(wf)
        self.chart_vec.set_data(vs)
        self.chart_gamut.set_data(gm)
        self._chart_src_changed()

    def _chart_src_changed(self):
        idx = self.combo_chart_src.currentIndex()
        slot = ("A", "B", "P")[idx]
        self.chart_wave.current = slot
        self.chart_vec.current = slot
        self.chart_gamut.current = slot
        self.chart_wave.update()
        self.chart_vec.update()
        self.chart_gamut.update()

    def _refresh_metrics(self):
        mode = self.metrics_panel.combo.currentData()
        a = self._chart_imgs.get("A" if mode == "AB" else "P")
        b = self._chart_imgs.get("B")
        if mode == "PB" and "P" not in self._chart_imgs:
            self.metrics_panel.de_label.setText("—")
            return
        if a is None or b is None:
            return
        mask = None
        if self.align_mask is not None:
            mask = cv2.resize(self.align_mask.astype(np.uint8),
                              (b.shape[1], b.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        self.metrics_panel.compute(a, b, mask=mask)

    # ---------------- 杂项 ----------------

    def _update_actions(self):
        has_a = self.img_a_full is not None
        has_b = self.img_b_full is not None
        self.act_analyze.setEnabled(has_a and has_b)
        self.act_bake.setEnabled(has_a)
        self.act_export.setEnabled(has_a)

    def closeEvent(self, ev):
        if self._analyze_worker is not None and self._analyze_worker.isRunning():
            self._analyze_worker.terminate()
        if self._proc_worker is not None and self._proc_worker.isRunning():
            self._proc_worker.terminate()
        super().closeEvent(ev)
