from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QValidator
from PySide6.QtWidgets import (QAbstractSpinBox, QComboBox, QDoubleSpinBox, QMessageBox, QSpinBox,
                               QStyle, QStyleOptionComboBox, QStyleOptionSpinBox, QTabBar, QTabWidget)


class _ScrollParameters:
    def wheelEvent(self, event):
        # Leave scrolling to the containing panel, even while focused.
        event.ignore()


class _ControlSpinButtons(_ScrollParameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_warning = None
        self.setKeyboardTracking(False)
        self.setCorrectionMode(QAbstractSpinBox.CorrectToPreviousValue)

    def validate(self, text, pos):
        state, text, pos = super().validate(text, pos)
        # Keep the whole draft. Rejecting a final digit would commit a valid prefix instead.
        if state == QValidator.Invalid:
            state = QValidator.Intermediate
        return state, text, pos

    def preview_value(self):
        return self.valueFromText(self.lineEdit().text()) if self.hasAcceptableInput() else self.value()

    def _invalid_message(self):
        if self._input_warning is not None or self.hasAcceptableInput():
            return None
        draft = self.cleanText().strip()
        reason = f"输入值 {draft} 无效" if draft else "输入不能为空"
        digits = self.decimals() if isinstance(self, QDoubleSpinBox) else 0
        kind = f"数值，最多 {digits} 位小数" if digits else "整数"
        return (f"{self.accessibleName() or '参数'}\n{reason}。允许范围为 {self.textFromValue(self.minimum())}"
                f" 至 {self.textFromValue(self.maximum())}，请填写{kind}。\n"
                f"已恢复为上一次确认的值 {self.textFromValue(self.value())}。")

    def _show_input_warning(self, message):
        dialog = QMessageBox(QMessageBox.Warning, "参数输入无效", message, QMessageBox.Ok, self)
        self._input_warning = dialog
        dialog.setTextFormat(Qt.PlainText)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.button(QMessageBox.Ok).setText("确定")
        dialog.finished.connect(self._dismiss_input_warning)
        dialog.open()

    def _dismiss_input_warning(self):
        self._input_warning = None
        if self.isVisible():
            self.window().activateWindow()
            self.setFocus(Qt.OtherFocusReason)
            self.selectAll()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            message = self._invalid_message()
            if message:
                self.interpretText()
                self._show_input_warning(message)
                event.accept()
                return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        # Qt replaces an intermediate draft during focusOutEvent. Read it before correction.
        message = self._invalid_message() if self.isVisible() and event.reason() != Qt.ActiveWindowFocusReason else None
        super().focusOutEvent(event)
        if message:
            self._show_input_warning(message)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.buttonSymbols() == QAbstractSpinBox.NoButtons:
            return
        # Qt's native arrows disappear once the dark stylesheet paints the
        # spin-box border. Draw only the glyphs; Qt retains native hit testing,
        # stepping, held-button repeat, keyboard input, and boundary handling.
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        for control, flag, direction in (
            (QStyle.SC_SpinBoxUp, QAbstractSpinBox.StepUpEnabled, -1),
            (QStyle.SC_SpinBoxDown, QAbstractSpinBox.StepDownEnabled, 1),
        ):
            rect = self.style().subControlRect(QStyle.CC_SpinBox, option, control, self)
            if not rect.isValid():
                continue
            x, y = rect.center().x(), rect.center().y()
            painter.setBrush(QColor("#e2efff" if self.isEnabled()
                             and option.stepEnabled & flag else "#60788f"))
            painter.drawPolygon(QPolygonF([
                QPointF(x - 4, y - direction * 2),
                QPointF(x + 4, y - direction * 2),
                QPointF(x, y + direction * 3),
            ]))
        painter.end()


class ControlSpinBox(_ControlSpinButtons, QSpinBox):
    pass


class ControlDoubleSpinBox(_ControlSpinButtons, QDoubleSpinBox):
    pass


class ControlComboBox(_ScrollParameters, QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        arrow = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxArrow, self)
        center = QPointF(arrow.center())
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor("#d8e7ff" if self.isEnabled() else "#8092a9"), 1.6))
            painter.drawPolyline(QPolygonF([center + QPointF(-4, -2), center + QPointF(0, 2),
                                            center + QPointF(4, -2)]))


class ControlTabBar(_ScrollParameters, QTabBar):
    pass


class ControlTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabBar(ControlTabBar(self))
        self.setDocumentMode(True)
        self.tabBar().setExpanding(False)
        self.tabBar().setDrawBase(False)
        self.setAccessibleName("深度神经网络优化控制工作流程")


def control_page_style(root):
    return """
    QWidget#ROOT, QWidget#ROOT QTabWidget::pane, QWidget#ROOT QStackedWidget, QWidget#ROOT QScrollArea, QWidget#ROOT QScrollArea > QWidget > QWidget { background: #0e203a; border: 0; }
    QWidget#ROOT QTabWidget::pane { padding-top: 13px; }
    QWidget#ROOT QTabBar::tab { color: #d8e7ff; background: #203958; border: 1px solid #385e85; border-radius: 6px; margin-right: 8px; min-height: 36px; padding: 0 13px; font-size: 15px; font-weight: bold; }
    QWidget#ROOT QTabBar::tab:selected { color: white; background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #4493ce,stop:1 #276399); border-color: #73b9eb; }
    QWidget#ROOT QTabBar::tab:hover:!selected, QWidget#ROOT QPushButton:hover { background: #2c4f73; }
    QWidget#ROOT QFrame#cstrParameters, QWidget#ROOT QFrame#cstrDisplay { background: #0e213c; border: 1px solid #304c70; border-radius: 8px; }
    QWidget#ROOT QLabel, QWidget#ROOT QCheckBox { background: transparent; color: #d8e7f8; font-size: 14px; }
    QWidget#ROOT QLabel#heading { color: #d9edff; font-size: 17px; font-weight: bold; }
    QWidget#ROOT QLabel#subheading { color: #b9d8f2; font-size: 15px; font-weight: bold; }
    QWidget#ROOT QLabel#fieldLabel, QWidget#ROOT QLabel#guide { color: #bbd4e9; font-size: 13px; }
    QWidget#ROOT QLabel#hint { color: #adc5d8; font-size: 12px; }
    QWidget#ROOT QLabel#statusStrip { border: 1px solid #3b536d; border-radius: 6px; background: #192637; color: #cfe3f5; padding: 7px 10px; font-size: 13px; }
    QWidget#ROOT QLabel#empty, QWidget#ROOT QFrame#deferred { border: 1px dashed #456783; border-radius: 8px; background: #0d1e32; padding: 12px; color: #9db9d1; }
    QWidget#ROOT QFrame#divider { background: #345470; border: 0; max-height: 1px; }
    QWidget#ROOT QAbstractSpinBox, QWidget#ROOT QComboBox { color: #e6f1ff; background: #0e203b; border: 1px solid #345172; border-radius: 4px; min-height: 30px; padding: 0 29px 0 8px; font-size: 14px; }
    QWidget#ROOT QComboBox::drop-down { border: 0; width: 23px; }
    QWidget#ROOT QComboBox::down-arrow { image: none; width: 0; height: 0; }
    QWidget#ROOT QFrame#fileField { background: #0e203b; border: 1px solid #345172; border-radius: 4px; }
    QWidget#ROOT QScrollBar::add-page:vertical, QWidget#ROOT QScrollBar::sub-page:vertical { background: transparent; }
    QWidget#ROOT QComboBox QAbstractItemView { color: #e6f1ff; background: #122740; selection-background-color: #31577c; }
    QWidget#ROOT QAbstractSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:26px; height:15px; border: 0; }
    QWidget#ROOT QAbstractSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:26px; height:15px; border: 0; }
    QWidget#ROOT QPushButton { color: #d8e7ff; background: #203958; border: 1px solid #385e85; border-radius: 6px; padding: 6px 12px; font-size: 14px; font-weight: bold; }
    QWidget#ROOT QPushButton[role="secondary"] { background: #1a334e; }
    QWidget#ROOT QPushButton[role="small"] { background: #1a334e; padding: 5px 9px; font-size: 12px; }
    QWidget#ROOT QPushButton[role="link"] { color: #8bcaff; border: 0; background: transparent; padding: 2px 0; text-decoration: underline; font-size: 13px; font-weight: normal; }
    QWidget#ROOT QPushButton:disabled { color: #8092a9; background: #172d46; border-color: #2c455f; }
    QWidget#ROOT QPushButton:focus, QWidget#ROOT QAbstractSpinBox:focus, QWidget#ROOT QComboBox:focus { border-color: #79baff; }
    QWidget#ROOT QProgressBar { color: #d7e7f8; background: #203043; border: 1px solid #456078; border-radius: 4px; max-height: 8px; }
    QWidget#ROOT QProgressBar::chunk { background: #68bfff; border-radius: 3px; }
    QWidget#ROOT QScrollBar:vertical { background: #10223a; width: 9px; }
    QWidget#ROOT QScrollBar::handle:vertical { background: #395570; min-height: 28px; border-radius: 4px; }
    QWidget#ROOT QScrollBar::add-line:vertical, QWidget#ROOT QScrollBar::sub-line:vertical { height: 0; }
    """.replace("ROOT", root)
