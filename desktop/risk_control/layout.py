"""Native cards and wrapping form rows for the risk-control workspace."""
from html import escape

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QLabel, QLayout, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)


def label(text, role="hint"):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    widget.setObjectName(role)
    widget.setWordWrap(True)
    return widget


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

    def add_field(self, title, control, wide=False):
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
        column.addWidget(control)
        self.fields[control] = cell
        self.add_item(cell, wide)

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
        self.set_columns(self.width() <= 728)

