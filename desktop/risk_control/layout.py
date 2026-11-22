"""Native cards and wrapping form rows for the risk-control workspace."""
from html import escape

from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)


def label(text, role="hint"):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    widget.setObjectName(role)
    widget.setWordWrap(True)
    return widget


class RunningValue(QLabel):
    """Show the effective session value underneath a locally edited parameter."""
    def __init__(self, form, box):
        super().__init__()
        self.box = box
        self.actual = None
        self.setObjectName("hint")
        self.setWordWrap(True)
        form.fields[box].layout().addWidget(self)
        self.hide()
        box.valueChanged.connect(self._refresh)

    def sync(self, value, *, restore=False):
        if self.actual == value and not restore:
            return
        previous = self.actual
        self.actual = value
        if restore or previous is None or self.box.textFromValue(previous) == self.box.textFromValue(self.box.value()):
            self.box.setValue(value)
        self._refresh()

    def _refresh(self):
        if self.actual is None:
            return
        actual = self.box.textFromValue(self.actual)
        pending = self.box.textFromValue(self.box.value()) != actual
        self.setText(f"当前运行值 {actual}，修改待应用" if pending else "")
        self.setVisible(pending)


class StatusStrip(QLabel):
    def __init__(self):
        super().__init__()
        self.setObjectName("statusStrip")
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)

    def show_state(self, state, title, *details):
        color = {"running": "#68bfff", "succeeded": "#61d09d", "failed": "#ff746d"}.get(state, "#7f9ab1")
        fields = [f'<span style="color:{color}">●</span>&nbsp; <b>{escape(title)}</b>']
        fields.extend(escape(str(text)) for text in details if text)
        self.setText("&nbsp;&nbsp;&nbsp; ".join(fields))


class FlowRow(QLayout):
    """Lay out native controls in reading order and wrap when the panel narrows."""
    def __init__(self, *widgets, gap=10, grow=None):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.items = []
        self.gap = gap
        self.grow = grow
        for widget in widgets:
            self.addWidget(widget)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < self.count() else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < self.count() else None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize(0, 0)
        for item in self.items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        return size

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, True)

    def _layout(self, rect, place):
        rows, row, used = [], [], 0
        for item in self.items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            width = min(rect.width(), size.width())
            if row and used + self.gap + width > rect.width():
                rows.append(row)
                row, used = [], 0
            used += (self.gap if row else 0) + width
            row.append((item, width, size.height()))
        if row:
            rows.append(row)
        y = rect.y()
        for row in rows:
            height = max(h for _, _, h in row)
            if place:
                spare = rect.width() - sum(w for _, w, _ in row) - self.gap * (len(row)-1)
                x = rect.x()
                for item, w, h in row:
                    if self.grow is not None and item.widget() is self.grow:
                        w += spare
                    item.setGeometry(QRect(x, y + (height-h)//2, w, h))
                    x += w + self.gap
            y += height + self.gap
        return y - rect.y() - (self.gap if rows else 0)


class FieldGrid(QWidget):
    def __init__(self, pane):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(11)
        self.entries = []
        self.fields = {}
        self.columns = 1
        pane.addWidget(self)

    def add_item(self, widget, wide=False):
        widget.setParent(self)
        widget.show()
        self.entries.append((widget, wide))
        self.arrange()

    def add_field(self, title, control, wide=False, beside=None):
        cell = QWidget()
        column = QVBoxLayout(cell)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(5)
        name = label(title, "fieldLabel")
        name.setBuddy(control)
        control.setAccessibleName(title)
        control.setMinimumWidth(0)
        control.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        column.addWidget(name)
        if beside is None:
            column.addWidget(control)
        else:
            row = QHBoxLayout()
            row.setSpacing(self.grid.horizontalSpacing())
            row.addWidget(control, 1, Qt.AlignVCenter)
            row.addWidget(beside, 1, Qt.AlignVCenter)
            column.addLayout(row)
        self.fields[control] = cell
        cell.installEventFilter(self)
        self.add_item(cell, wide or beside is not None)

    def eventFilter(self, cell, event):
        if event.type() in (QEvent.LayoutRequest, QEvent.Resize):
            # Keep wrapped hints and newly visible running values inside the field.
            height = cell.heightForWidth(cell.width())
            if height >= 0 and height != cell.minimumHeight():
                cell.setMinimumHeight(height)
        return super().eventFilter(cell, event)

    def add_hint(self, control, text):
        hint = label(text)
        control.setAccessibleDescription(text)
        self.fields[control].layout().addWidget(hint)
        return hint

    def show_field(self, control, show):
        self.fields[control].setVisible(show)
        self.arrange()

    def arrange(self):
        while self.grid.count():
            self.grid.takeAt(0)
        for i in range(self.grid.columnCount()):
            self.grid.setColumnStretch(i, int(i < self.columns))
        row = col = 0
        for widget, wide in self.entries:
            if widget.isHidden():
                continue
            if wide and col:
                row, col = row+1, 0
            span = self.columns if wide else 1
            self.grid.addWidget(widget, row, col, 1, span, Qt.AlignTop)
            col += span
            if col == self.columns:
                row, col = row+1, 0
        for i in range(self.columns):
            self.grid.setColumnStretch(i, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = max(1, (event.size().width()+11)//186)
        if columns != self.columns:
            self.columns = columns
            self.arrange()


class StagePage(QScrollArea):
    def __init__(self):
        super().__init__()
        self.simulation_bar = None
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        self.grid = QGridLayout(body)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(14)
        self.grid.setAlignment(Qt.AlignTop)
        self.panels = []
        for role in ("cstrParameters", "cstrDisplay"):
            panel = QFrame()
            panel.setObjectName(role)
            panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
            pane = QVBoxLayout(panel)
            pane.setContentsMargins(16, 16, 16, 16)
            pane.setSpacing(12)
            self.panels.append(panel)
        self.stacked = None
        self.set_columns(False)
        self.setWidget(body)

    @property
    def parameters(self):
        return self.panels[0].layout()

    @property
    def display(self):
        return self.panels[1].layout()

    def add_simulation_bar(self, command):
        self.simulation_bar = QFrame(self)
        row = QHBoxLayout(self.simulation_bar)
        row.setContentsMargins(4, 0, 4, 6)
        self.simulation_clock = label("", "guide")
        self.simulation_button = QPushButton("暂停仿真")
        self.simulation_button.setToolTip("暂停或继续当前仿真（Pause 键）")
        self.simulation_button.setEnabled(False)
        self.simulation_button.clicked.connect(lambda: command(self.simulation_action))
        self.pause_shortcut = QShortcut(QKeySequence(Qt.Key_Pause), self)
        self.pause_shortcut.setContext(Qt.WindowShortcut)
        self.pause_shortcut.setAutoRepeat(False)
        self.pause_shortcut.activated.connect(self.simulation_button.click)
        row.addWidget(self.simulation_clock, 1)
        row.addWidget(self.simulation_button)
        height = self.simulation_bar.sizeHint().height()
        self.setViewportMargins(0, height, 0, 0)
        self.simulation_bar.setGeometry(0, 0, self.width(), height)

    def show_simulation_state(self, state, time_min):
        self.simulation_action = "resume" if state in ("paused", "ready") else "pause"
        self.simulation_button.setText("继续仿真" if self.simulation_action == "resume" else "暂停仿真")
        self.simulation_button.setEnabled(state in ("running", "paused", "ready"))
        title = {"paused": "已暂停", "running": "运行中", "ended": "已结束"}.get(state, "就绪")
        self.simulation_clock.setText(f"{title}　仿真时间 {time_min:.1f} min")

    def set_columns(self, stacked):
        if stacked == self.stacked:
            return
        self.stacked = stacked
        for panel in self.panels:
            self.grid.removeWidget(panel)
        self.grid.addWidget(self.panels[0], 0, 0, Qt.AlignTop)
        self.grid.addWidget(self.panels[1], int(stacked), int(not stacked), Qt.AlignTop)
        self.grid.setColumnMinimumWidth(0, 0 if stacked else 270)
        self.grid.setColumnStretch(0, 14)
        self.grid.setColumnStretch(1, 0 if stacked else 33)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.simulation_bar is not None:
            self.simulation_bar.resize(self.width(), self.viewportMargins().top())
        self.set_columns(self.width() <= 728)

