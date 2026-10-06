# -*- coding: utf-8 -*-
"""
专业图表：RGB 直方图 / 波形图 / 矢量示波器 / CIE xy 色域图。
数据来自 scopes 模块（密度数组），用 QPainter 渲染 + 专业刻度叠加。
"""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (QPainter, QImage, QColor, QPen, QFont, QBrush,
                           QPainterPath, QPixmap)
from PySide6.QtWidgets import QWidget

from .scopes import (SPECTRUM_XY, D65_XY, SRGB_PRIMARIES,
                     GAMUT_X_RANGE, GAMUT_Y_RANGE, TARGETS_UV, wavelength_to_rgb)


def _density_to_qimage(arr: np.ndarray) -> QImage:
    """float [0,1] (H,W,3) -> QImage"""
    a8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    h, w = a8.shape[:2]
    return QImage(a8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class _ChartBase(QWidget):
    """深底图表基类"""

    BG = QColor("#0d0d11")
    GRID = QColor("#26262e")
    TEXT = QColor("#9a9aa4")

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.setMinimumHeight(170)

    def _frame(self, p: QPainter):
        p.fillRect(self.rect(), self.BG)
        p.setPen(QPen(self.GRID, 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.setPen(self.TEXT)
        f = QFont(self.font()); f.setPointSizeF(8.5)
        p.setFont(f)
        p.drawText(QRectF(8, 4, self.width() - 16, 16), Qt.AlignmentFlag.AlignLeft, self.title)

    def _plot_rect(self) -> QRectF:
        return QRectF(44, 24, self.width() - 56, self.height() - 44)


class HistogramChart(_ChartBase):
    """RGB 直方图：A 填充 + B 描边 叠加"""

    def __init__(self, parent=None):
        super().__init__("RGB 直方图", parent)
        self.hist_a: np.ndarray | None = None
        self.hist_b: np.ndarray | None = None
        self.show_b = True

    def set_data(self, hist_a=None, hist_b=None):
        self.hist_a = hist_a
        self.hist_b = hist_b
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._frame(p)
        r = self._plot_rect()
        p.setPen(QPen(self.GRID, 1))
        for i in range(1, 4):
            y = r.top() + r.height() * i / 4
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        def draw_hist(h, fill: bool, alpha: int):
            if h is None:
                return
            n = h.shape[1]
            for c, col in enumerate(((230, 70, 70), (80, 200, 100), (90, 130, 250))):
                path = QPainterPath()
                path.moveTo(r.left(), r.bottom())
                for i in range(n):
                    x = r.left() + r.width() * i / (n - 1)
                    y = r.bottom() - r.height() * float(np.clip(h[c, i], 0, 1))
                    path.lineTo(x, y)
                path.lineTo(r.right(), r.bottom())
                color = QColor(*col, alpha)
                if fill:
                    p.fillPath(path, QBrush(color))
                else:
                    p.setPen(QPen(color, 1.4))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPath(path)

        draw_hist(self.hist_a, fill=True, alpha=110)
        if self.show_b:
            draw_hist(self.hist_b, fill=False, alpha=230)
        # 图例
        p.setPen(QColor("#ddd"))
        p.drawText(QPointF(r.right() - 90, r.top() + 12), "■ A原图  ─ B修改")
        p.end()


class WaveformChart(_ChartBase):
    """RGB 波形图（选择显示 A/B/P）"""

    def __init__(self, parent=None):
        super().__init__("RGB 波形图（横轴=画面从左到右，纵轴=亮度 0-100%）", parent)
        self.data: dict[str, np.ndarray] = {}
        self.current = "A"

    def set_data(self, data: dict):
        self.data = data
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._frame(p)
        r = self._plot_rect()
        arr = self.data.get(self.current)
        if arr is not None:
            qi = _density_to_qimage(arr)
            p.drawImage(r, qi)
        # 刻度
        p.setPen(QPen(self.GRID, 1))
        for f in (0.25, 0.5, 0.75):
            y = r.bottom() - r.height() * f
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(self.TEXT)
            p.drawText(QPointF(6, y + 4), f"{f*100:.0f}")
            p.setPen(QPen(self.GRID, 1))
        p.setPen(self.TEXT)
        for f, t in ((0.0, "0"), (1.0, "100")):
            y = r.bottom() - r.height() * f
            p.drawText(QPointF(6, y + 4), t)
        p.end()


class VectorscopeChart(_ChartBase):
    """矢量示波器（选择显示 A/B/P）+ 六色靶刻度"""

    def __init__(self, parent=None):
        super().__init__("矢量示波器（角度=色相，离圆心=饱和度）", parent)
        self.data: dict[str, np.ndarray] = {}
        self.current = "A"

    def set_data(self, data: dict):
        self.data = data
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._frame(p)
        size = min(self.width() - 60, self.height() - 40)
        r = QRectF(48, 26, size, size)
        cx, cy = r.center().x(), r.center().y()
        radius = size / 2.0

        arr = self.data.get(self.current)
        if arr is not None:
            qi = _density_to_qimage(arr)
            p.drawImage(r, qi)

        # 刻度
        p.setPen(QPen(self.GRID, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(r)
        p.drawEllipse(QRectF(cx - radius * 0.5, cy - radius * 0.5, radius, radius))
        p.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))
        p.drawLine(QPointF(cx, cy - radius), QPointF(cx, cy + radius))

        # 六色靶
        f = QFont(self.font()); f.setPointSizeF(8)
        p.setFont(f)
        for name, (u, v) in TARGETS_UV.items():
            # U 横轴（右正），V 纵轴（上正）→ 屏幕 y 向下
            x = cx + (u / 0.6) * radius
            y = cy - (v / 0.6) * radius
            col = {"R": (255, 60, 60), "G": (60, 230, 90), "B": (90, 120, 255),
                   "Cy": (60, 220, 220), "Mg": (240, 80, 240), "Yl": (230, 220, 60)}[name]
            p.setPen(QPen(QColor(*col, 170), 1))
            p.drawLine(QPointF(cx, cy), QPointF(x, y))
            p.setBrush(QColor(*col, 200))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, y), 3, 3)
            p.setPen(QColor(*col))
            p.drawText(QPointF(x + 4, y - 4), name)

        p.setPen(self.TEXT)
        p.drawText(QPointF(cx - 8, cy + size / 2 + 14), "U")
        p.drawText(QPointF(cx + size / 2 + 6, cy + 4), "V")
        p.end()


class GamutChart(_ChartBase):
    """CIE xy 色域图：散点密度 + 光谱轨迹 + sRGB 三角形"""

    def __init__(self, parent=None):
        super().__init__("CIE xy 色域图（sRGB 三角内=显示器可显示）", parent)
        self.data: dict[str, np.ndarray] = {}
        self.current = "A"
        self._trail_img: QImage | None = None
        self._trail_size = 0

    def set_data(self, data: dict):
        self.data = data
        self.update()

    def _build_trail(self, size: int) -> QImage:
        """光谱轨迹底图（带内部淡填充）"""
        img = np.zeros((size, size, 3), np.float32)
        x0, x1 = GAMUT_X_RANGE
        y0, y1 = GAMUT_Y_RANGE

        def to_px(x, y):
            px = int((x - x0) / (x1 - x0) * (size - 1))
            py = int((1.0 - (y - y0) / (y1 - y0)) * (size - 1))
            return px, py

        # 光谱轨迹 + 底边（紫线）围成的区域淡填充
        pts = [(x, y) for _, x, y in SPECTRUM_XY]
        purple = (pts[-1][0] * 0.08 + 0.0, pts[0][1] * 0.05)  # 底边近似
        mask = np.zeros((size, size), bool)
        poly = [to_px(x, y) for x, y in pts]
        rr, cc = self._polygon_mask(poly, size)
        mask[rr, cc] = True
        img[mask] = (0.10, 0.10, 0.14)

        # 轨迹着色线
        for i in range(len(SPECTRUM_XY) - 1):
            nm1, x1_, y1_ = SPECTRUM_XY[i]
            nm2, x2_, y2_ = SPECTRUM_XY[i + 1]
            col = np.array(wavelength_to_rgb((nm1 + nm2) / 2), np.float32)
            for t in np.linspace(0, 1, 40):
                x = x1_ + (x2_ - x1_) * t
                y = y1_ + (y2_ - y1_) * t
                px, py = to_px(x, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = py + dy, px + dx
                        if 0 <= yy < size and 0 <= xx < size:
                            img[yy, xx] = col
        return _density_to_qimage(img)

    @staticmethod
    def _polygon_mask(poly, size):
        """简单多边形填充掩码（scanline）"""
        rr, cc = [], []
        # 用 matplotlib-free 的点在多边形内测试（射线法），仅对小图使用
        ys, xs = np.mgrid[0:size, 0:size]
        pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        n = len(poly)
        inside = np.zeros(len(pts), bool)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            cond = ((yi > pts[:, 1]) != (yj > pts[:, 1]))
            denom = (yj - yi)
            if abs(denom) < 1e-9:
                j = i
                continue
            xint = (xj - xi) * (pts[:, 1] - yi) / denom + xi
            inside ^= (cond & (pts[:, 0] < xint))
            j = i
        idx = np.where(inside)[0]
        return idx // size, idx % size

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._frame(p)
        size = min(self.width() - 60, self.height() - 40)
        r = QRectF(48, 26, size, size)

        if self._trail_img is None or self._trail_size != int(size):
            self._trail_img = self._build_trail(int(size))
            self._trail_size = int(size)
        p.drawImage(r, self._trail_img)

        arr = self.data.get(self.current)
        if arr is not None:
            qi = _density_to_qimage(arr)
            p.drawImage(r, qi)

        # sRGB 三角形 + D65
        x0, x1 = GAMUT_X_RANGE
        y0, y1 = GAMUT_Y_RANGE

        def to_widget(x, y):
            return QPointF(r.left() + (x - x0) / (x1 - x0) * r.width(),
                           r.top() + (1.0 - (y - y0) / (y1 - y0)) * r.height())

        R = to_widget(*SRGB_PRIMARIES["R"])
        G = to_widget(*SRGB_PRIMARIES["G"])
        B = to_widget(*SRGB_PRIMARIES["B"])
        pen = QPen(QColor(255, 255, 255, 200), 1.6)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(R)
        path.lineTo(G); path.lineTo(B); path.lineTo(R)
        p.drawPath(path)
        for pt, name, col in ((R, "R", (255, 80, 80)), (G, "G", (80, 230, 100)), (B, "B", (90, 120, 255))):
            p.setBrush(QColor(*col))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(pt, 3, 3)
        d65 = to_widget(*D65_XY)
        p.setPen(QPen(QColor("#ffffff"), 1.4))
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(d65, 2.6, 2.6)
        p.setPen(QColor("#ddd"))
        f = QFont(self.font()); f.setPointSizeF(8)
        p.setFont(f)
        p.drawText(QPointF(d65.x() + 5, d65.y() + 3), "D65")
        p.end()
