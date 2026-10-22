"""Shared status, theme and vector figure viewer for desktop control pages."""
from decimal import Context, ROUND_HALF_UP
from pathlib import Path
import json
import re

from PySide6.QtCore import QByteArray, QObject, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QTransform
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import (QGraphicsScene, QGraphicsView, QGridLayout, QHBoxLayout, QFrame, QProgressBar, QSizePolicy,
                               QLabel, QPushButton, QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget)


def _svg_for_qt(data):
    """Represent Matplotlib's empty space glyphs as non-painting SVG groups.

    Qt rejects a path without geometry and then cannot resolve its <use>
    references. Keep the ID and transforms without rewriting XML namespaces,
    touching real outlines, or changing the saved SVG used by the Web page.
    """
    def replace_space(match):
        tag = match.group()
        if not re.search(rb'\sid\s*=\s*(["\'])[^"\']+-20\1', tag):
            return tag
        geometry = re.search(rb'\sd\s*=\s*(["\'])(.*?)\1', tag, re.DOTALL)
        if geometry is not None:
            if geometry.group(2).strip():
                return tag
            tag = tag[:geometry.start()] + tag[geometry.end():]
        return tag.replace(b"<path", b"<g", 1)

    return re.sub(rb"<path\b[^<>]*/>", replace_space, data)


class SvgPlotCanvas(QGraphicsView):
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.svg_renderer = QSvgRenderer(self)
        self._item = QGraphicsSvgItem()
        self._item.setCacheMode(QGraphicsSvgItem.NoCache)
        self._item.setSharedRenderer(self.svg_renderer)
        self._scene.addItem(self._item)
        self._loaded = False
        self._placeholder = "运行后显示本次计算结果"
        self._mode = "fit"
        self._zoom = 1.0
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background: white; border: 1px solid #7999b9; border-radius: 5px;")
        self.setAccessibleName("可缩放图表")
        self.setToolTip("滚轮缩放；按住左键拖动；双击适应窗口")

    def _fit_scale(self):
        bounds = self._item.boundingRect()
        return max(.001, min(max(1, self.viewport().width()-16) / max(1, bounds.width()),
                             max(1, self.viewport().height()-16) / max(1, bounds.height())))

    def load(self, filename):
        # Close the file before rendering so live preview replacement is not held up.
        self.load_svg(Path(filename).read_bytes())

    def load_svg(self, data):
        center = self.mapToScene(self.viewport().rect().center())
        first = not self._loaded
        if not self.svg_renderer.load(QByteArray(_svg_for_qt(data))):
            raise ValueError("无法加载 SVG 图表。")
        self._item.setSharedRenderer(self.svg_renderer)
        bounds = self._item.boundingRect()
        self.setSceneRect(bounds.adjusted(-bounds.width(), -bounds.height(), bounds.width(), bounds.height()))
        self._loaded = True
        self._item.setVisible(True)
        if first or self._mode != "manual":
            self.fit(self._mode if not first else "fit")
        else:
            self.centerOn(center)
        self.viewport().update()

    def clear(self):
        self._loaded = False
        self._mode, self._zoom = "fit", 1.0
        self._item.setVisible(False)
        self.resetTransform()
        self.zoom_changed.emit(1.0)
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._loaded:
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#334155"))
            painter.drawText(self.viewport().rect().adjusted(16, 16, -16, -16),
                             Qt.AlignCenter | Qt.TextWordWrap, self._placeholder)
            painter.end()

    def fit(self, mode="fit"):
        self._mode = mode
        if not self._loaded:
            return
        bounds = self._item.boundingRect()
        base = self._fit_scale()
        self._zoom = min(8., max(.5, (self.viewport().width()-16) / bounds.width() / base)) if mode == "width" else 1.
        self.setTransform(QTransform.fromScale(base*self._zoom, base*self._zoom))
        self.centerOn(bounds.center())
        self.zoom_changed.emit(self._zoom)

    def zoom_by(self, factor, position=None):
        if not self._loaded:
            return
        point = position if position is not None else self.viewport().rect().center()
        anchor = self.mapToScene(point)
        zoom = min(8., max(.5, self._zoom * factor))
        self.scale(zoom/self._zoom, zoom/self._zoom)
        self._mode, self._zoom = "manual", zoom
        after = self.mapToScene(point)
        self.centerOn(self.mapToScene(self.viewport().rect().center()) + anchor - after)
        self.zoom_changed.emit(zoom)

    def wheelEvent(self, event):
        if self._loaded:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            self.zoom_by(1.2 ** max(-5., min(5., delta/120)), event.position().toPoint())
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if self._loaded and event.button() == Qt.LeftButton:
            self._mode = "manual"
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.fit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        center = self.mapToScene(self.viewport().rect().center())
        super().resizeEvent(event)
        if self._loaded:
            if self._mode == "manual":
                scale = self._fit_scale() * self._zoom
                self.setTransform(QTransform.fromScale(scale, scale))
                self.centerOn(center)
            else:
                self.fit(self._mode)


class DnnMpcPlotView(QWidget):
    def __init__(self, parent=None, *, aspect_ratio=None):
        super().__init__(parent)
        self._aspect_ratio = aspect_ratio
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        toolbar = QHBoxLayout()
        self.canvas = SvgPlotCanvas(self)
        self._tool_buttons = []
        for label, action in (("缩小", lambda: self.canvas.zoom_by(1/1.2)),
                              ("放大", lambda: self.canvas.zoom_by(1.2)),
                              ("适应窗口", lambda: self.canvas.fit()),
                              ("适应宽度", lambda: self.canvas.fit("width"))):
            button = QPushButton(label)
            button.setAccessibleName("图表" + label)
            button.setProperty("controlPlotTool", True)
            button.setEnabled(False)
            button.clicked.connect(action)
            toolbar.addWidget(button)
            self._tool_buttons.append(button)
        self._zoom_label = QLabel("100%")
        toolbar.addWidget(self._zoom_label)
        toolbar.addStretch()
        self.canvas.zoom_changed.connect(lambda value: self._zoom_label.setText(f"{value:.0%}"))
        self.canvas.zoom_changed.connect(self._update_tools)
        layout.addLayout(toolbar)
        hint = QLabel("滚轮缩放 · 按住左键拖动 · 双击适应窗口")
        hint.setProperty("controlHint", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, 1)
        self.setMinimumSize(300, 280)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._aspect_ratio:
            self.canvas.setFixedHeight(max(260, min(650, round(self.width() / self._aspect_ratio))))



    def load(self, filename):
        self.canvas.load(filename)

    def load_svg(self, data):
        self.canvas.load_svg(data)

    def clear(self):
        self.canvas.clear()

    def set_placeholder(self, text):
        self.canvas._placeholder = text
        self.canvas.viewport().update()

    def _update_tools(self, zoom):
        for index, button in enumerate(self._tool_buttons):
            button.setEnabled(self.canvas._loaded and
                              (zoom > .5 if index == 0 else zoom < 8 if index == 1 else True))

    def renderer(self):
        return self.canvas.svg_renderer


class ControlPanel(QFrame):
    """The same in-card headings and padding as the Web control panels."""
    def __init__(self, title, *, secondary=False, parent=None):
        super().__init__(parent)
        self.setProperty("controlPanel", True)
        self.setProperty("controlInset", secondary)
        self.body = QVBoxLayout(self)
        margin = 12 if secondary else 20
        self.body.setContentsMargins(margin, margin, margin, margin)
        self.body.setSpacing(12)
        self.heading = QLabel(title)
        self.heading.setProperty("controlSubheading" if secondary else "controlHeading", True)
        self.body.addWidget(self.heading)


class ControlScrollArea(QScrollArea):
    """Keep settings, result captions and logs reachable on smaller windows."""
    def __init__(self, content, *, width=None, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)
        self.setWidget(content)
        if width is not None:
            self.setFixedWidth(width)
        else:
            self.setMinimumWidth(360)


class ControlRunLog(QWidget):
    """Collapsible, selectable worker logs shared by the four desktop pages."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.toggle = QPushButton("▸ 运行日志")
        self.toggle.setAccessibleName("运行日志")
        self.toggle.setProperty("controlLogToggle", True)
        self.toggle.setCheckable(True)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(2000)
        self.text.setFixedHeight(120)
        self.text.setVisible(False)
        self.toggle.toggled.connect(self._toggle)
        layout.addWidget(self.toggle, 0, Qt.AlignLeft)
        layout.addWidget(self.text)
        self.hide()

    def _toggle(self, expanded):
        self.toggle.setText(("▾" if expanded else "▸") + " 运行日志")
        self.text.setVisible(expanded)

    def append(self, message):
        self.text.appendPlainText(str(message))
        self.show()

    def clear(self):
        self.text.clear()
        self.toggle.setChecked(False)
        self.hide()


def control_page_style(root_name):
    """Keep this theme inside the requesting page, away from other modules."""
    return """
        QWidget#ROOT { background: #0d1f3b; color: #d9ecff; }
        QWidget#ROOT QWidget[controlSurface="true"] { background: #0d1f3b; }
        QWidget#ROOT QGroupBox, QWidget#ROOT QFrame[controlPanel="true"] {
            background: #102640; border: 1px solid #365477; border-radius: 8px;
        }
        QWidget#ROOT QFrame[controlInset="true"] { background: #0d2035; }
        QWidget#ROOT QGroupBox { margin-top: 12px; padding-top: 12px; color: #d9ecff; font-size: 15px; font-weight: 600; }
        QWidget#ROOT QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; }
        QWidget#ROOT QGroupBox[controlHeading="true"] { font-size: 18px; }
        QWidget#ROOT QScrollArea { border: none; background: transparent; }
        QWidget#ROOT QScrollBar:vertical { background: #0d2035; width: 12px; margin: 0; }
        QWidget#ROOT QScrollBar::handle:vertical { background: #365477; min-height: 30px; border-radius: 5px; }
        QWidget#ROOT QScrollBar::handle:vertical:hover { background: #4c83ac; }
        QWidget#ROOT QScrollBar::add-line:vertical, QWidget#ROOT QScrollBar::sub-line:vertical { height: 0; }
        QWidget#ROOT QScrollBar::add-page:vertical, QWidget#ROOT QScrollBar::sub-page:vertical { background: transparent; }
        QWidget#ROOT QLabel { color: #d9ecff; font-size: 14px; border: none; }
        QWidget#ROOT QLabel[controlHeading="true"] { font-size: 18px; font-weight: 600; padding: 4px 0; }
        QWidget#ROOT QLabel[controlSubheading="true"] { font-size: 15px; font-weight: 600; }
        QWidget#ROOT QLabel[controlPlotHeading="true"] { font-size: 16px; font-weight: 600; }
        QWidget#ROOT QLabel#parameterRangeHint, QWidget#ROOT QLabel[controlHint="true"] { font-size: 13px; color: #a3bdd5; }
        QWidget#ROOT QLabel#resultPathLabel { background: #0d2035; border: 1px solid #365477; border-radius: 6px; padding: 7px 9px; font-size: 13px; color: #a3bdd5; }
        QWidget#ROOT QLineEdit, QWidget#ROOT QSpinBox, QWidget#ROOT QDoubleSpinBox {
            background: #0d2035; color: #d9ecff; border: 1px solid #365477;
            border-radius: 6px; min-height: 28px; padding: 2px 8px; font-size: 14px;
        }
        QWidget#ROOT QLineEdit:disabled, QWidget#ROOT QSpinBox:disabled, QWidget#ROOT QDoubleSpinBox:disabled { color: #87a2bd; }
        QWidget#ROOT QPushButton { background: #214c70; color: #e2efff; border: 1px solid #4c83ac; border-radius: 8px; min-height: 20px; padding: 7px 12px; font-size: 14px; font-weight: 600; }
        QWidget#ROOT QPushButton[controlPrimary="true"] { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #438fc3, stop:1 #2f689b); border-color: #85cef6; }
        QWidget#ROOT QPushButton[controlPlotTool="true"] { min-height: 20px; padding: 4px 10px; font-size: 13px; }
        QWidget#ROOT QPushButton:hover { background: #31678f; }
        QWidget#ROOT QPushButton:focus { border-color: #a8ddff; }
        QWidget#ROOT QPushButton:disabled, QWidget#ROOT QPushButton[controlPrimary="true"]:disabled { background: #21374d; border-color: #365477; color: #87a2bd; }
        QWidget#ROOT QPushButton[controlLogToggle="true"] { background: transparent; border: none; padding: 0; color: #9bc8ff; font-size: 13px; font-weight: 400; text-align: left; }
        QWidget#ROOT QPlainTextEdit { background: #0d2035; color: #d9ecff; border: 1px solid #365477; border-radius: 6px; font-size: 13px; }
        QWidget#ROOT QFrame[controlDivider="true"] { background: #365477; border: none; max-height: 1px; }
        QWidget#ROOT QSplitter::handle { background: #365477; }
        QWidget#ROOT QTabWidget::pane { border: none; }
        QWidget#ROOT QTabBar::tab { background: #214c70; color: #e2efff; border: 1px solid #4c83ac; border-radius: 8px; padding: 10px 18px; margin-right: 8px; font-size: 14px; font-weight: 600; }
        QWidget#ROOT QTabBar::tab:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #438fc3, stop:1 #2f689b); border-color: #85cef6; }
    """.replace("#ROOT", "#" + root_name)


class ControlPlotRefresh(QObject):
    """Poll published preview revisions without resetting a viewer's zoom/pan."""
    def __init__(self, views, parent=None):
        super().__init__(parent)
        self.views = views
        self.directory = None
        self.revision = 0
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)

    def start(self, directory):
        self.directory = Path(directory)
        self.revision = 0
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def refresh(self):
        if self.directory is None:
            return
        try:
            progress = json.loads((self.directory / "progress.json").read_text(encoding="utf-8"))
            revision = progress.get("image_revision", 0)
            if not revision or revision == self.revision:
                return
            for name, view in self.views.items():
                if name in progress.get("outputs", {}):
                    view.load(str(self.directory / f"{name}.svg"))
            self.revision = revision
        except (OSError, ValueError):
            # A temporarily busy/missing file is retried by the next timer tick.
            pass


def format_rmse(value):
    """Match Web Number.toPrecision(4), including trailing zeros and exponents."""
    # Round the actual binary float once, using JavaScript's tie rule. Python's
    # .4g drops zeros and switches to scientific notation at a different limit.
    rounded = Context(prec=4, rounding=ROUND_HALF_UP).create_decimal_from_float(float(value))
    if not rounded.is_finite():
        return str(rounded)
    if not rounded:
        return "0.000"
    exponent = rounded.adjusted()
    if -6 <= exponent < 4:
        return format(rounded, f".{3 - exponent}f")
    return f"{rounded.scaleb(-exponent):.3f}e{exponent:+d}"


class ControlRmseMetrics(QFrame):
    """Sidebar RMSE labels and prominent values shared by FDI and replay."""
    FIELDS = (("tracking_rmse", "跟踪误差 RMSE"), ("disturbance_rmse", "扰动估计 RMSE"))

    def __init__(self, columns=1, parent=None):
        super().__init__(parent)
        self.setObjectName("controlRmseMetrics")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        # Together with the sidebar's 12 px spacing, this matches Web's 20 px gap.
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)
        divider = QFrame()
        divider.setObjectName("rmseDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(18)
        self.values = {}
        for index, (key, label) in enumerate(self.FIELDS):
            item = QVBoxLayout()
            item.setSpacing(4)
            title = QLabel(label)
            title.setWordWrap(True)
            item.addWidget(title)
            value = QLabel()
            value.setObjectName("rmseValue")
            value.setAccessibleName(label)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            item.addWidget(value)
            self.values[key] = value
            grid.addLayout(item, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)
        self.setStyleSheet("""
            QFrame#controlRmseMetrics { background: transparent; border: none; }
            QFrame#controlRmseMetrics QFrame#rmseDivider { background: #365477; border: none; }
            QFrame#controlRmseMetrics QLabel { color: #d9ecff; font-size: 14px; font-weight: 400; border: none; padding: 0; }
            QFrame#controlRmseMetrics QLabel#rmseValue { color: #86d4ff; font-size: 18px; font-weight: 600; }
        """)
        self.clear()

    def set_values(self, result):
        for key, value in self.values.items():
            value.setText(format_rmse(result[key]))
        self.show()

    def clear(self):
        for value in self.values.values():
            value.clear()
        self.hide()


class ControlTaskStatus(QFrame):
    """Status badge, wrapping message, percentage, and a full-width progress bar."""
    STATES = {
        "idle": ("待执行", "#8eaac4"), "running": ("运行中", "#79c9f8"),
        "succeeded": ("已完成", "#79d9b5"), "failed": ("失败", "#ffaaaa"),
        "cancelled": ("已取消", "#e9c27d"),
    }

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setObjectName("controlTaskStatus")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        summary = QHBoxLayout()
        summary.setSpacing(10)
        self.badge = QLabel()
        self.badge.setObjectName("taskBadge")
        self.message = QLabel()
        self.message.setObjectName("taskMessage")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.message.setTextFormat(Qt.PlainText)
        self.message.setAccessibleName(label + "状态")
        self.message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.percentage = QLabel()
        self.percentage.setObjectName("taskPercent")
        self.percentage.setMinimumWidth(44)
        self.percentage.setAlignment(Qt.AlignRight | Qt.AlignTop)
        summary.addWidget(self.badge, 0, Qt.AlignTop)
        summary.addWidget(self.message, 1)
        summary.addWidget(self.percentage, 0, Qt.AlignTop)
        layout.addLayout(summary)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setAccessibleName(label + "进度")
        layout.addWidget(self.progress)
        self.set_status("idle", "设置参数后开始运行", 0)

    def set_status(self, state, message, percent=None):
        label, accent = self.STATES[state]
        value = self.progress.value() if percent is None else round(percent)
        value = 100 if state == "succeeded" else 0 if state == "idle" else max(0, min(99, value))
        if self.property("state") != state:
            self.setProperty("state", state)
            self.setStyleSheet("""
                QFrame#controlTaskStatus { background: #0d2035; border: 1px solid #365477; border-radius: 8px; }
                QFrame#controlTaskStatus QLabel { border: none; background: transparent; font-size: 14px; font-weight: 400; color: #d9ecff; padding: 0; }
                QFrame#controlTaskStatus QLabel#taskBadge { color: ACCENT; border: 1px solid ACCENT; border-radius: 5px; padding: 2px 8px; font-size: 12px; font-weight: 600; }
                QFrame#controlTaskStatus QLabel#taskPercent { color: ACCENT; padding-top: 2px; }
                QFrame#controlTaskStatus QLabel#taskMessage { padding-top: 2px; }
                QFrame#controlTaskStatus QProgressBar { background: #233b53; border: none; border-radius: 4px; }
                QFrame#controlTaskStatus QProgressBar::chunk { background: ACCENT; border-radius: 4px; }
            """.replace("ACCENT", accent))
        self.badge.setText(label)
        self.message.setText(message)
        self.percentage.setText(f"{value}%")
        self.progress.setValue(value)
