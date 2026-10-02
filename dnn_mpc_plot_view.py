"""Interactive vector figure viewer for the two desktop DNN-MPC pages."""
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtGui import QPainter, QTransform
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QGraphicsSvgItem
from PySide6.QtWidgets import (QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout, QWidget)


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
        center = self.mapToScene(self.viewport().rect().center())
        first = not self._loaded
        # Close the file before rendering so live preview replacement is not held up.
        if not self.svg_renderer.load(QByteArray(Path(filename).read_bytes())):
            raise ValueError("无法加载 SVG 图表。")
        self._item.setSharedRenderer(self.svg_renderer)
        bounds = self._item.boundingRect()
        self.setSceneRect(bounds.adjusted(-bounds.width(), -bounds.height(), bounds.width(), bounds.height()))
        self._loaded = True
        if first or self._mode != "manual":
            self.fit(self._mode if not first else "fit")
        else:
            self.centerOn(center)
        self.viewport().update()

    def clear(self):
        self._loaded = False
        self._mode, self._zoom = "fit", 1.0

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
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        toolbar = QHBoxLayout()
        self.canvas = SvgPlotCanvas(self)
        for label, action in (("缩小", lambda: self.canvas.zoom_by(1/1.2)),
                              ("放大", lambda: self.canvas.zoom_by(1.2)),
                              ("适应窗口", lambda: self.canvas.fit()),
                              ("适应宽度", lambda: self.canvas.fit("width"))):
            button = QPushButton(label)
            button.setAccessibleName("图表" + label)
            button.clicked.connect(action)
            toolbar.addWidget(button)
        self._zoom_label = QLabel("100%")
        toolbar.addWidget(self._zoom_label)
        toolbar.addStretch()
        self.canvas.zoom_changed.connect(lambda value: self._zoom_label.setText(f"{value:.0%}"))
        layout.addLayout(toolbar)
        hint = QLabel("滚轮缩放 · 按住左键拖动 · 双击适应窗口")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, 1)
        self.setMinimumSize(300, 280)

    def load(self, filename):
        self.canvas.load(filename)

    def clear(self):
        self.canvas.clear()

    def renderer(self):
        return self.canvas.svg_renderer
