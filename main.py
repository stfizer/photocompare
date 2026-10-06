# -*- coding: utf-8 -*-
"""修图参数对比器 - 程序入口"""
import os
import sys


def _fix_stream(name):
    """PyInstaller --windowed 下 sys.stdout/stderr 为 None，库内部写日志会崩溃"""
    if getattr(sys, name) is None:
        try:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
        except Exception:
            pass


_fix_stream("stdout")
_fix_stream("stderr")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QIcon, QFont
from PySide6.QtWidgets import QApplication

DARK_QSS = """
QMainWindow, QWidget { background: #17171c; color: #d8d8de; font-size: 12px; }
QToolBar { background: #1d1d24; border-bottom: 1px solid #2a2a33; spacing: 4px; padding: 3px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 4px;
              padding: 5px 9px; color: #cfcfd6; }
QToolButton:hover { background: #2a2a35; }
QToolButton:pressed, QToolButton:checked { background: #33406b; border-color: #46538a; color: #fff; }
QComboBox, QDoubleSpinBox { background: #22222b; border: 1px solid #33333f; border-radius: 4px;
                            padding: 3px 6px; }
QComboBox QAbstractItemView { background: #22222b; selection-background-color: #33406b; }
QTabWidget::pane { border: 1px solid #2a2a33; border-radius: 4px; }
QTabBar::tab { background: #1d1d24; padding: 7px 14px; margin-right: 2px;
               border-top-left-radius: 4px; border-top-right-radius: 4px; color: #9a9aa4; }
QTabBar::tab:selected { background: #262633; color: #ffffff; border-bottom: 2px solid #5a78d6; }
QGroupBox { border: 1px solid #2c2c36; border-radius: 6px; margin-top: 10px; padding-top: 4px;
            font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8fa4d9; }
QSlider::groove:horizontal { height: 5px; background: #2b2b36; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -6px 0; border-radius: 7px;
                             background: #5a78d6; }
QSlider::handle:horizontal:hover { background: #7a95e8; }
QSlider::sub-page:horizontal { background: #3d4f8a; border-radius: 2px; }
QPushButton { background: #2c2c3a; border: 1px solid #3c3c4c; border-radius: 5px;
              padding: 6px 14px; color: #d8d8de; }
QPushButton:hover { background: #36364a; }
QPushButton:pressed { background: #2a3355; }
QPushButton:disabled { color: #666; background: #222228; }
QTreeWidget { background: #141419; border: 1px solid #2a2a33; border-radius: 4px; }
QTreeWidget::item:selected { background: #33406b; }
QHeaderView::section { background: #1d1d24; border: none; padding: 4px; color: #9a9aa4; }
QStatusBar { background: #1d1d24; border-top: 1px solid #2a2a33; color: #9a9aa4; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #1a1a20; width: 10px; }
QScrollBar::handle:vertical { background: #3a3a48; border-radius: 5px; min-height: 30px; }
QScrollBar:horizontal { background: #1a1a20; height: 10px; }
QScrollBar::handle:horizontal { background: #3a3a48; border-radius: 5px; min-width: 30px; }
QLabel { background: transparent; }
QSplitter::handle { background: #26262e; width: 3px; }
"""


def _write_crash_log(text: str):
    """诊断/崩溃信息写到 exe 旁边（追加模式）"""
    try:
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        with open(os.path.join(base, "boot.log"), "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def main():
    # 全局异常钩子：windowed 模式下 Python 异常无处可去，写日志
    def _hook(tp, val, tb):
        import traceback
        _write_crash_log("".join(traceback.format_exception(tp, val, tb)))
        sys.__excepthook__(tp, val, tb)
    sys.excepthook = _hook

    try:
        app = QApplication(sys.argv)
    except Exception:
        import traceback
        _write_crash_log(traceback.format_exc())
        raise

    app.setStyle("Fusion")
    app.setApplicationName("修图参数对比器")

    # 深色调色板（QSS 之外的基础色）
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#17171c"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#d8d8de"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#141419"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#d8d8de"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#2c2c3a"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#d8d8de"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#33406b"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(DARK_QSS)

    # 图标（可选）
    try:
        import os
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "app.ico")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass

    from photocompare.main_window import MainWindow
    try:
        win = MainWindow()
        win.show()
        _write_crash_log("window shown OK")
    except Exception:
        import traceback
        _write_crash_log(traceback.format_exc())
        raise
    # 事件循环心跳：若 1.5s 后能写入，说明 exec 正常运行
    from PySide6.QtCore import QTimer
    QTimer.singleShot(1500, lambda: _write_crash_log("event loop alive"))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
