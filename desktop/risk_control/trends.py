"""Qt trend charts shared by the CSTR control pages."""
import math
from bisect import bisect_left
from operator import itemgetter

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QCoreApplication, QEvent, QMargins, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QSizePolicy

from .layout import label


class Trend(QChartView):
    pan_to = Signal(float)
    restore_live = Signal()
    scale_changed = Signal()

    def __init__(self, title, names, x_title, unit=""):
        chart = QChart()
        super().__init__(chart)
        self.pan_origin = None
        self.sample_unit = unit
        self.sample_digits = None
        self.sample_time = None
        self.sample_context = None
        self.sample_rows = self.sample_groups = ()
        self.sample_tip = QLabel(self.viewport())
        self.sample_tip.setTextFormat(Qt.PlainText)
        self.sample_tip.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.sample_tip.setStyleSheet("background: #10243a; color: #e6f2ff; border: 1px solid #476986;"
                                     "border-radius: 5px; padding: 4px 8px; font-size: 12px;")
        self.sample_tip.hide()
        self.setMouseTracking(True)
        self.domain = None
        self.y_zoom_level = 0
        self.manual_y_range = None
        self.auto_y_range = (0, 1)
        self.wheel_remainder = 0.
        chart.setAnimationOptions(QChart.NoAnimation)
        self.title = title
        self.heading = label(title, "subheading")
        self.setAccessibleName(title)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QBrush(QColor("#0e213c")))
        chart.setBackgroundVisible(False)
        chart.legend().setLabelColor(QColor("#d7e7f8"))
        chart.legend().setAlignment(Qt.AlignTop)
        chart.layout().setContentsMargins(0, 0, 0, 0)
        chart.setBackgroundRoundness(0)
        font = QFont(self.font())
        font.setPixelSize(12)
        chart.legend().setFont(font)
        chart.setMargins(QMargins(4, 0, 4, 0))
        self.axes = [QValueAxis(), QValueAxis()]
        for axis, alignment in zip(self.axes, (Qt.AlignBottom, Qt.AlignLeft)):
            axis.setLabelsColor(QColor("#b9cee2"))
            axis.setGridLineColor(QColor("#30465d"))
            axis.setTitleBrush(QBrush(QColor("#bdd0de")))
            axis.setLabelsFont(font)
            chart.addAxis(axis, alignment)
        self.axes[0].setGridLineVisible(False)
        self.axes[0].setTickCount(7)
        self.axes[1].setTickCount(5)
        self.axes[0].setTitleText(x_title)
        self.lines = []
        self.comparison_lines = []
        self.pulse_bands = []
        for i, name in enumerate(names):
            series = QLineSeries()
            series.setName(name)
            color = ("#69baff", "#f4bd73")[i % 2]
            series.setPen(QPen(QColor(color), 2, Qt.SolidLine if i == 0 else Qt.DashLine))
            chart.addSeries(series)
            for axis in self.axes:
                series.attachAxis(axis)
            self.lines.append(series)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumHeight(135)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear_sample(self):
        if self.sample_time is not None:
            self.sample_time = None
            self.sample_tip.hide()
            self.viewport().update()

    def viewportEvent(self, event):
        if event.type() in (QEvent.Leave, QEvent.Hide) and hasattr(self, "sample_tip"):
            self.clear_sample()
        return super().viewportEvent(event)

    def inspect_sample(self, position):
        chart = self.chart()
        plot = chart.plotArea()
        point = chart.mapFromScene(self.mapToScene(position.toPoint()))
        rows = self.sample_rows
        if not rows or plot.isEmpty() or not plot.contains(point):
            return self.clear_sample()
        axis = self.axes[0]
        step = (axis.max()-axis.min()) / plot.width()
        at = axis.min() + (point.x()-plot.left()) * step
        if not rows[0][0]-6*step <= at <= rows[-1][0]+6*step:
            return self.clear_sample()
        i = min(bisect_left(rows, at, key=itemgetter(0)), len(rows)-1)
        if i and at-rows[i-1][0] <= rows[i][0]-at:
            i -= 1
        row = rows[i]
        self.sample_time = row[0]
        values = [(line.name(), value) for line, value in zip(self.lines, row[1:])]
        for group in self.sample_groups:
            j = bisect_left(group, row[0]-1e-9, key=itemgetter(0))
            if j < len(group) and abs(group[j][0]-row[0]) < 1e-9:
                values.append(("恒流对照", group[j][1]))
                break
        title = f"轮次 {row[0]:g}" if axis.titleText()=="轮次" else f"仿真时间 {row[0]:.1f} min"
        fmt = ".7g" if self.sample_digits is None else f".{self.sample_digits}f"
        unit = " "+self.sample_unit if self.sample_unit else ""
        self.sample_tip.setText("\n".join([title, *(f"{name}　{value:{fmt}}{unit}" for name, value in values if value is not None and math.isfinite(value))]))
        self.sample_tip.adjustSize()
        top = self.mapFromScene(chart.mapToScene(plot.topLeft())).y()
        side = 8 if point.x() > plot.center().x() else max(0, self.viewport().width()-self.sample_tip.width()-8)
        self.sample_tip.move(side, top)
        self.sample_tip.show()
        self.sample_tip.raise_()
        self.viewport().update()

    def mousePressEvent(self, event):
        self.clear_sample()
        if self.domain and event.button() == Qt.LeftButton:
            chart = self.chart()
            plot = chart.plotArea()
            if plot.contains(chart.mapFromScene(self.mapToScene(event.position().toPoint()))):
                self.pan_origin = event.position(), self.domain, self.manual_y_range or self.auto_y_range, plot.size()
                self.drag_x_active = self.drag_y_active = False
                self.viewport().setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan_origin is not None:
            origin, (left, right), (low, high), plot = self.pan_origin
            delta = event.position() - origin
            if self.drag_y_active or abs(delta.y()) >= 4:
                self.drag_y_active = True
                offset = delta.y() * (high-low) / plot.height()
                self.manual_y_range = low+offset, high+offset
                self._apply_y_range()
            if self.drag_x_active or abs(delta.x()) >= 4:
                self.drag_x_active = True
                self.pan_to.emit(left - delta.x() * (right-left) / plot.width())
            event.accept()
            return
        if not event.buttons():
            self.inspect_sample(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.pan_origin is not None:
            self.pan_origin = None
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        chart = self.chart()
        point = chart.mapFromScene(self.mapToScene(event.position().toPoint()))
        if self.domain and event.button() == Qt.LeftButton and chart.plotArea().contains(point):
            self.restore_live.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if self.domain is None:
            super().wheelEvent(event)
            return
        if event.modifiers() == Qt.NoModifier:
            parent = self.parentWidget()
            while parent is not None and not isinstance(parent, QScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                QCoreApplication.sendEvent(parent.viewport(), event)
                return
        if event.modifiers() != Qt.ShiftModifier:
            event.ignore()
            return
        chart = self.chart()
        point = chart.mapFromScene(self.mapToScene(event.position().toPoint()))
        plot = chart.plotArea()
        if not plot.contains(point):
            event.ignore()
            return
        delta = event.angleDelta()
        if delta.isNull():
            delta = event.pixelDelta()
            unit = 60
        else:
            unit = 120
        move = delta.y() or delta.x()
        if not move:
            event.ignore()
            return
        total = self.wheel_remainder + move / unit
        steps = int(total)
        self.wheel_remainder = total - steps
        if steps:
            self._zoom_y(steps, (plot.bottom()-point.y())/plot.height())
        event.accept()

    def _zoom_y(self, steps, fraction):
        level = max(0, min(10, self.y_zoom_level + steps))
        if level == self.y_zoom_level:
            return
        low, high = self.manual_y_range or self.auto_y_range
        span = (high - low) * 2**(self.y_zoom_level - level)
        anchor = low + fraction * (high - low)
        self.manual_y_range = (anchor - fraction*span, anchor + (1-fraction)*span) if level else None
        self.y_zoom_level = level
        self._apply_y_range()

    def reset_y_scale(self):
        self.clear_sample()
        self.pan_origin = None
        if self.domain:
            self.viewport().setCursor(Qt.OpenHandCursor)
        self.manual_y_range = None
        self.y_zoom_level = 0
        self.wheel_remainder = 0.
        self._apply_y_range()

    def _apply_y_range(self):
        low, high = self.manual_y_range or self.auto_y_range
        axis = self.axes[1]
        step = (high-low) / (axis.tickCount()-1)
        decimals = max(0, min(8, 1-math.floor(math.log10(step))))
        axis.setLabelFormat(f"%.{decimals}f")
        axis.setRange(low, high)
        suffix = f"纵轴 ×{2**self.y_zoom_level}" if self.y_zoom_level else "纵轴已平移"
        caption = self.title + ("　" + suffix if self.manual_y_range is not None else "")
        if self.heading.text() != caption:
            self.heading.setText(caption)
            self.scale_changed.emit()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        at = getattr(self, "sample_time", None)
        if not getattr(self, "pulse_bands", None) and at is None:
            return
        area = self.chart().mapRectToScene(self.chart().plotArea())
        left, right = self.axes[0].min(), self.axes[0].max()
        if area.isEmpty() or right <= left:
            return
        scale = area.width() / (right-left)
        painter.save()
        painter.setClipRect(area, Qt.IntersectClip)
        for start, end in self.pulse_bands:
            band = QRectF(area.left() + (start-left)*scale, area.top(), (end-start)*scale, area.height())
            painter.fillRect(band, QColor(237, 185, 103, 34))
        if at is not None and left <= at <= right:
            x = area.left() + (at-left)*scale
            painter.setPen(QPen(QColor("#c2d5e6"), 1, Qt.DashLine))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
        painter.restore()

    def draw(self, rows, domain=None, groups=(), bands=(), *, context=None):
        if context != self.sample_context or not rows:
            self.clear_sample()
        self.sample_context = context
        self.sample_rows, self.sample_groups = rows, groups
        if domain and self.domain is None:
            self.viewport().setCursor(Qt.OpenHandCursor)
        self.domain = domain
        self.pulse_bands = bands
        while len(self.comparison_lines) > len(groups):
            line = self.comparison_lines.pop()
            self.chart().removeSeries(line)
            line.deleteLater()
        for i, points in enumerate(groups):
            if i == len(self.comparison_lines):
                line = QLineSeries()
                line.setName("恒流对照")
                line.setPen(QPen(QColor("#c6a6ff"), 2, Qt.DotLine))
                self.chart().addSeries(line)
                for axis in self.axes:
                    line.attachAxis(axis)
                for marker in self.chart().legend().markers(line):
                    marker.setVisible(i == 0)
                self.comparison_lines.append(line)
            self.comparison_lines[i].replace([QPointF(t, v) for t, v in points])
        self.viewport().update()
        if not rows:
            return
        x = [float(row[0]) for row in rows]
        values = [float(v) for row in rows for v in row[1:] if v is not None]
        values.extend(v for points in groups for _, v in points)
        for i, series in enumerate(self.lines, 1):
            series.setPointsVisible(len(rows) == 1)
            series.replace([QPointF(t, float(row[i])) for t, row in zip(x, rows) if row[i] is not None])
        self.axes[0].setRange(*(domain or (x[0], max(x[-1], x[0]+0.1))))
        if not values:
            return
        delta = max(0.001, (max(values)-min(values))*0.1)
        self.auto_y_range = min(values)-delta, max(values)+delta
        if domain:
            self._apply_y_range()
        else:
            self.axes[1].setRange(*self.auto_y_range)
