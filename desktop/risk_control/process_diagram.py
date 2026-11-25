"""Live CSTR schematic with moving stripes on the SVG's fluid pipes."""
from time import monotonic
from xml.etree import ElementTree

from PySide6.QtCore import QByteArray, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QSizePolicy, QWidget

from platform_core.risk_control import presentation


class ProcessDiagram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("夹套式 CSTR 实时工艺图")
        self._renderer = QSvgRenderer(self)
        svg = ElementTree.fromstring(presentation.template())
        self._size = QSize(int(svg.attrib["width"]), int(svg.attrib["height"]))
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMaximumHeight(285)
        self._background = QColor(svg.find(".//{*}rect").attrib["fill"])
        polylines = {
            line.attrib["id"]: QPolygonF([QPointF(*map(float, point.split(",")))
                                         for point in line.attrib["points"].split()])
            for line in svg.findall(".//{*}polyline[@id]")
        }
        layer = svg.find(".//*[@id='pipe-motion']")
        self._reference_flow = float(layer.attrib["data-reference-flow"])
        pattern = [float(n) for n in layer.attrib["stroke-dasharray"].split()]
        self._pitch = sum(pattern)
        width = float(layer.attrib["stroke-width"])
        self._streams = {}
        for group in layer:
            pen = QPen(QColor(group.attrib["stroke"]), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            # QPen expresses dash lengths and offsets in multiples of pen width.
            pen.setDashPattern([n/width for n in pattern])
            paths = [polylines[use.attrib["{http://www.w3.org/1999/xlink}href"][1:]] for use in group]
            self._streams[group.attrib["data-stream"]] = pen, paths

        self._phase = dict(material=0., coolant=0.)
        self._coolant_rate = 0.
        self._sample_key = None
        self._running = False
        self._last_frame = 0.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def sizeHint(self):
        return self._size

    def heightForWidth(self, width):
        return min(285, round(width * self._size.height() / self._size.width()))

    def set_sample(self, sample, run, running):
        if self._timer.isActive():
            self._tick()
        key = (run, sample["seq"], sample.get("signal_marker"))
        if self._sample_key != key:
            if self._sample_key is None or self._sample_key[0] != run:
                self._phase = dict(material=0., coolant=0.)
            self._renderer.load(QByteArray(presentation.render(sample).encode("utf-8")))
            marker = (sample.get("signal_marker") or {}).get("label", "")
            self.setAccessibleDescription((f"釜内温度 {sample['T']:.2f} K，夹套出口温度 {sample['Tc']:.2f} K，"
                                           f"实际冷却液流量 {sample['Qc']:.2f} L/min　{marker}").rstrip())
            self._sample_key = key
        self._coolant_rate = max(0., sample["Qc"])/self._reference_flow
        self.set_running(running)
        self.update()

    def set_running(self, running):
        self._running = running
        if running and self.isVisible():
            if not self._timer.isActive():
                self._last_frame = monotonic()
                self._timer.start()
        else:
            self._timer.stop()

    def _tick(self):
        now = monotonic()
        distance = (now-self._last_frame)*self._pitch
        self._last_frame = now
        self._phase["material"] = (self._phase["material"]+distance) % self._pitch
        self._phase["coolant"] = (self._phase["coolant"]+distance*self._coolant_rate) % self._pitch
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.set_running(self._running)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        if not self._renderer.isValid():
            return
        view = self._renderer.viewBoxF()
        scale = min(self.width()/view.width(), self.height()/view.height())
        if scale <= 0:
            return
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            outline = QPainterPath()
            outline.addRoundedRect(QRectF(self.rect()), 6, 6)
            painter.setClipPath(outline)
            painter.fillRect(self.rect(), self._background)
            painter.translate(self.width()/2, self.height()/2)
            painter.scale(scale, scale)
            painter.translate(-view.center())
            self._renderer.render(painter, view)
            painter.setBrush(Qt.NoBrush)
            for stream, (pen, paths) in self._streams.items():
                pen.setDashOffset(-self._phase[stream]/pen.widthF())
                painter.setPen(pen)
                for path in paths:
                    painter.drawPolyline(path)

