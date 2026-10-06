# -*- coding: utf-8 -*-
"""
对比查看器：单图 / 并排(Split) / 滑块擦除(Wipe) / 闪烁(Blink) 四种模式，
支持缩放(滚轮)、平移(拖动)、灰度切换。
图像槽位：A=原图, P=处理后, B=修改图。
"""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QImage, QColor, QPen, QFont, QTransform, QPixmap
from PySide6.QtWidgets import QWidget

SLOT_NAMES = {"A": "原图 A", "P": "处理后", "B": "修改图 B"}


def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """uint8 RGB (H,W,3) -> QImage（保持数据引用）"""
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()  # 拷贝脱离 numpy 生命周期，安全


class CompareViewer(QWidget):
    MODES = ("split", "wipe", "blink", "single")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.images: dict[str, np.ndarray] = {}       # slot -> uint8 RGB
        self._qimg: dict[str, QImage] = {}            # slot -> 彩色 QImage
        self._qimg_gray: dict[str, QImage] = {}       # slot -> 灰度 QImage

        self.mode = "wipe"
        self.sources = ("A", "B")      # 左/右（或闪烁的两源）
        self.single_slot = "A"
        self.grayscale = False

        self.zoom = 1.0
        self.offset = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_start = QPointF()
        self._dragging_wipe = False

        self.wipe_pos = 0.5
        self._blink_left = True
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._apply_blink_timer()

        self._status_cb = None

    # ---------- 公共接口 ----------

    def set_status_callback(self, cb):
        self._status_cb = cb

    def set_image(self, slot: str, arr: np.ndarray | None):
        """arr: uint8 RGB (H,W,3)；None 清除"""
        if arr is None:
            self.images.pop(slot, None)
            self._qimg.pop(slot, None)
            self._qimg_gray.pop(slot, None)
        else:
            self.images[slot] = arr
            self._qimg[slot] = ndarray_to_qimage(arr)
            g = (arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722)
            g8 = np.clip(g, 0, 255).astype(np.uint8)
            g3 = np.repeat(g8[..., None], 3, axis=2)
            self._qimg_gray[slot] = ndarray_to_qimage(g3)
        self.update()

    def set_mode(self, mode: str):
        if mode in self.MODES:
            self.mode = mode
            if mode == "blink":
                self._blink_timer.start()
            else:
                self._blink_timer.stop()
            self.update()

    def set_sources(self, left: str, right: str):
        self.sources = (left, right)
        self.update()

    def set_single_slot(self, slot: str):
        self.single_slot = slot
        self.update()

    def set_grayscale(self, on: bool):
        self.grayscale = on
        self.update()

    def set_blink_interval(self, ms: int):
        self._blink_ms = ms
        self._apply_blink_timer()

    def fit(self):
        self.zoom = 1.0
        self.offset = QPointF(0.0, 0.0)
        self.update()

    def _apply_blink_timer(self):
        self._blink_ms = getattr(self, "_blink_ms", 500)
        self._blink_timer.setInterval(self._blink_ms)

    def _toggle_blink(self):
        self._blink_left = not self._blink_left
        self.update()

    # ---------- 绘制 ----------

    def _pick_qimg(self, slot: str) -> QImage | None:
        if self.grayscale:
            return self._qimg_gray.get(slot)
        return self._qimg.get(slot)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101014"))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.mode == "split":
            self._paint_split(p)
        elif self.mode == "wipe":
            self._paint_wipe(p)
        elif self.mode == "blink":
            self._paint_blink(p)
        else:
            self._paint_single(p)
        p.end()

    def _virtual_rect(self, slot: str, area: QRectF) -> QRectF:
        """把图像 keep-aspect fit 到 area，返回目标矩形（应用 zoom/pan）"""
        qi = self._pick_qimg(slot)
        if qi is None or qi.width() < 1:
            return QRectF(area)
        # fit
        iw, ih = float(qi.width()), float(qi.height())
        scale = min(area.width() / iw, area.height() / ih) * 0.98
        w, h = iw * scale, ih * scale
        x = area.center().x() - w / 2.0
        y = area.center().y() - h / 2.0
        rect = QRectF(x, y, w, h)
        if self.zoom != 1.0 or self.offset != QPointF(0, 0):
            c = rect.center()
            t = QTransform()
            t.translate(c.x() + self.offset.x(), c.y() + self.offset.y())
            t.scale(self.zoom, self.zoom)
            t.translate(-c.x(), -c.y())
            rect = t.mapRect(rect)
        return rect

    def _draw_label(self, p: QPainter, text: str, x: float, y: float):
        f = QFont(self.font())
        f.setPointSizeF(max(9.0, self.font().pointSizeF() * 1.05))
        p.setFont(f)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 14
        h = fm.height() + 6
        p.fillRect(QRectF(x, y, w, h), QColor(20, 20, 24, 190))
        p.setPen(QColor("#e8e8ec"))
        p.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)

    def _paint_single_area(self, p: QPainter, slot: str, area: QRectF):
        qi = self._pick_qimg(slot)
        if qi is None:
            return
        rect = self._virtual_rect(slot, area)
        p.drawImage(rect, qi)

    def _paint_single(self, p: QPainter):
        slot = self.single_slot if self.single_slot in self.images else next(iter(self.images), None)
        if slot is None:
            self._paint_placeholder(p)
            return
        self._paint_single_area(p, slot, QRectF(self.rect()))
        self._draw_label(p, SLOT_NAMES.get(slot, slot), 10, 10)

    def _paint_placeholder(self, p: QPainter):
        p.setPen(QColor("#666670"))
        f = QFont(self.font()); f.setPointSize(13)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                   "打开两张图片开始对比\n工具栏：打开原图 A → 打开修改图 B")

    def _paint_split(self, p: QPainter):
        la, rb = self.sources
        w, h = self.width(), self.height()
        left_area = QRectF(0, 0, w / 2 - 1, h)
        right_area = QRectF(w / 2 + 1, 0, w / 2 - 1, h)
        p.setPen(QPen(QColor("#333338"), 1))
        p.drawLine(QPointF(w / 2, 0), QPointF(w / 2, h))
        if la in self.images:
            self._paint_single_area(p, la, left_area)
            self._draw_label(p, SLOT_NAMES.get(la, la), 10, 10)
        if rb in self.images:
            self._paint_single_area(p, rb, right_area)
            self._draw_label(p, SLOT_NAMES.get(rb, rb), w / 2 + 12, 10)

    def _paint_wipe(self, p: QPainter):
        la, rb = self.sources
        if la not in self.images and rb not in self.images:
            self._paint_placeholder(p)
            return
        area = QRectF(self.rect())
        w = self.width()
        wx = self.wipe_pos * w

        if la in self.images:
            qi = self._pick_qimg(la)
            rect = self._virtual_rect(la, area)
            p.save()
            p.setClipRect(QRectF(0, 0, wx, self.height()))
            p.drawImage(rect, qi)
            p.restore()
        if rb in self.images:
            qi = self._pick_qimg(rb)
            rect = self._virtual_rect(rb, area)
            p.save()
            p.setClipRect(QRectF(wx, 0, w - wx, self.height()))
            p.drawImage(rect, qi)
            p.restore()

        # 分割线 + 手柄
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(QPointF(wx, 0), QPointF(wx, self.height()))
        handle_y = self.height() / 2
        p.setBrush(QColor("#ffffff"))
        p.setPen(QPen(QColor("#888890"), 2))
        p.drawEllipse(QPointF(wx, handle_y), 11, 11)
        p.setPen(QPen(QColor("#333"), 2))
        p.drawLine(QPointF(wx - 5, handle_y), QPointF(wx + 5, handle_y))
        # 两侧标签
        if la in self.images:
            self._draw_label(p, SLOT_NAMES.get(la, la), 10, 10)
        if rb in self.images:
            self._draw_label(p, SLOT_NAMES.get(rb, rb), max(10, wx - 90), self.height() - 34)

    def _paint_blink(self, p: QPainter):
        slot = self.sources[0] if self._blink_left else self.sources[1]
        if slot not in self.images:
            slot = next(iter(self.images), None)
        if slot is None:
            self._paint_placeholder(p)
            return
        self._paint_single_area(p, slot, QRectF(self.rect()))
        # 闪烁角标
        p.setPen(QPen(QColor("#ffcc44"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(self.rect().adjusted(1, 1, -2, -2))
        self._draw_label(p, SLOT_NAMES.get(slot, slot), 10, 10)

    # ---------- 交互 ----------

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = float(np.clip(self.zoom * factor, 0.2, 12.0))
        # 锚定鼠标点（近似：以视口中心为锚）
        self.zoom = new_zoom
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self.mode == "wipe":
            wx = self.wipe_pos * self.width()
            if abs(ev.position().x() - wx) < 16:
                self._dragging_wipe = True
                return
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_start = ev.position() - self.offset

    def mouseMoveEvent(self, ev):
        if self._dragging_wipe:
            self.wipe_pos = float(np.clip(ev.position().x() / max(self.width(), 1), 0.0, 1.0))
            self.update()
            return
        if self._panning:
            self.offset = ev.position() - self._pan_start
            self.update()

    def mouseReleaseEvent(self, ev):
        self._dragging_wipe = False
        self._panning = False

    def mouseDoubleClickEvent(self, ev):
        self.fit()
