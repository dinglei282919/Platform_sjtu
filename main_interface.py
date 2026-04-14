import sys

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from anomaly_detection import MultiScenarioAnomalyDetectionWidget
from correlation_analysis import CorrelationAnalysisWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工业安全智能决策平台")
        self.resize(1700, 960)
        self._dropdown = None
        self._dropdown_layout = None
        self._nav_menu_map = {}
        self._nav_title_map = {}
        self._active_dropdown_button = None
        self._content_title_label = None
        self._anomaly_content_widget = None
        self._correlation_content_widget = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 12, 16, 14)
        root_layout.setSpacing(10)

        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_function_bar())
        root_layout.addWidget(self._build_content(), 1)

        self._apply_styles()
        self._on_submodule_clicked("异构数据治理", "关联分析")

    def _build_header(self):
        bar = QFrame()
        bar.setObjectName("headerBar")
        bar.setFixedHeight(72)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 0, 16, 0)
        layout.setSpacing(12)

        logo = QLabel("◈")
        logo.setObjectName("logoBlock")
        title = QLabel("工业安全智能决策平台")
        title.setObjectName("pageTitle")

        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addStretch()

        for icon in ("👤", "⚙", "⏻"):
            btn = QPushButton(icon)
            btn.setObjectName("headerIcon")
            btn.setCursor(Qt.PointingHandCursor)
            layout.addWidget(btn)

        return bar

    def _make_nav_button(self, icon: str, text: str, checked=False):
        btn = QPushButton(f"{icon}  {text}  ⌄")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setObjectName("navButton")
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _build_function_bar(self):
        bar = QFrame()
        bar.setObjectName("functionBar")
        bar.setFixedHeight(66)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        items = [
            ("📊", "异构数据治理", True, ["关联分析"]),
            ("🏭", "异常行为检测", False, ["多工况分层级异常检测"]),
        ]

        for icon, text, checked, menu_items in items:
            button = self._make_nav_button(icon, text, checked=checked)
            layout.addWidget(button)
            button.clicked.connect(lambda _checked=False, b=button: self._toggle_dropdown(b))
            self._nav_menu_map[button] = menu_items
            self._nav_title_map[button] = text
            if checked:
                self._active_dropdown_button = button

        layout.addStretch()

        self._dropdown = self._build_dropdown()
        self._dropdown.hide()
        return bar

    def _build_dropdown(self):
        panel = QFrame(self, Qt.Popup | Qt.FramelessWindowHint)
        panel.setObjectName("dropdownPanel")
        panel.setFixedSize(420, 260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        self._dropdown_layout = layout
        return panel

    def _build_content(self):
        container = QFrame()
        container.setObjectName("contentRoot")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        title_bar = QFrame()
        title_bar.setObjectName("contentTitleBar")
        title_bar.setFixedHeight(64)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(18, 0, 18, 0)
        title = QLabel("异构数据治理 - 关联分析")
        title.setObjectName("contentTitle")
        self._content_title_label = title
        title_layout.addWidget(title)
        title_layout.addStretch()

        body = QFrame()
        body.setObjectName("contentBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        anomaly_content = QFrame()
        anomaly_layout = QVBoxLayout(anomaly_content)
        anomaly_layout.setContentsMargins(0, 0, 0, 0)
        anomaly_layout.setSpacing(0)
        anomaly_layout.addWidget(MultiScenarioAnomalyDetectionWidget())
        self._anomaly_content_widget = anomaly_content
        self._anomaly_content_widget.hide()

        correlation_content = QFrame()
        correlation_layout = QVBoxLayout(correlation_content)
        correlation_layout.setContentsMargins(0, 0, 0, 0)
        correlation_layout.setSpacing(0)
        correlation_layout.addWidget(CorrelationAnalysisWidget())
        self._correlation_content_widget = correlation_content
        self._correlation_content_widget.hide()

        body_layout.addWidget(self._anomaly_content_widget, 1)
        body_layout.addWidget(self._correlation_content_widget, 1)

        vbox.addWidget(title_bar)
        vbox.addWidget(body, 1)
        return container

    def _set_dropdown_items(self, nav_title, items):
        while self._dropdown_layout.count():
            item = self._dropdown_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        item_height = 52
        font = QFont("Microsoft YaHei UI")
        font.setPixelSize(22)
        metrics = QFontMetrics(font)
        max_text_width = max((metrics.horizontalAdvance(name) for name in items), default=0)

        for i, name in enumerate(items):
            btn = QPushButton(name)
            btn.setObjectName("dropItem")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(item_height)
            btn.setFont(font)
            if i == 0:
                btn.setProperty("active", True)
            btn.clicked.connect(lambda _checked=False, t=nav_title, n=name: self._on_submodule_clicked(t, n))
            self._dropdown_layout.addWidget(btn)
        self._dropdown_layout.addStretch()

        panel_width = max(360, min(max_text_width + 120, 980))
        panel_height = 24 + item_height * len(items)
        self._dropdown.setFixedSize(panel_width, min(max(panel_height, 140), 460))

    def _toggle_dropdown(self, button):
        if self._dropdown.isVisible() and self._active_dropdown_button is button:
            self._dropdown.hide()
            return
        self._active_dropdown_button = button
        for nav_btn in self._nav_menu_map:
            nav_btn.setChecked(nav_btn is button)
        self._set_dropdown_items(self._nav_title_map.get(button, ""), self._nav_menu_map.get(button, []))
        self._position_dropdown()
        self._dropdown.show()
        self._dropdown.raise_()

    def _on_submodule_clicked(self, nav_title, submodule_title):
        self._dropdown.hide()
        if self._content_title_label is not None:
            self._content_title_label.setText(f"{nav_title} - {submodule_title}")

        if submodule_title == "多工况分层级异常检测":
            self._anomaly_content_widget.show()
            self._correlation_content_widget.hide()
        else:
            self._anomaly_content_widget.hide()
            self._correlation_content_widget.show()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._position_dropdown()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._position_dropdown)

    def _position_dropdown(self):
        if not self._dropdown or not self._active_dropdown_button:
            return
        anchor = self._active_dropdown_button.mapToGlobal(QPoint(0, 0))
        x = anchor.x() + (self._active_dropdown_button.width() - self._dropdown.width()) // 2
        y = anchor.y() + self._active_dropdown_button.height() + 8
        screen = QApplication.screenAt(anchor) or QApplication.primaryScreen()
        if screen:
            bounds = screen.availableGeometry()
            x = max(bounds.left() + 8, min(x, bounds.right() - self._dropdown.width() - 8))
        self._dropdown.move(x, y)

    def _apply_styles(self):
        self.setStyleSheet(
            """
            * {
                font-family: "Microsoft YaHei UI", "Segoe UI";
                color: #d8e7ff;
            }
            QMainWindow, QWidget#root {
                background: qradialgradient(cx:0.2, cy:0.1, radius:1.2,
                    fx:0.2, fy:0.1,
                    stop:0 #193457, stop:0.45 #0d1f3b, stop:1 #09162b);
            }
            QFrame#headerBar {
                border: 1px solid rgba(123, 176, 247, 0.35);
                border-radius: 14px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #102a4b, stop:0.55 #0f2f59, stop:1 #0b284b);
            }
            QLabel#logoBlock {
                min-width: 40px;
                max-width: 40px;
                min-height: 40px;
                max-height: 40px;
                border-radius: 10px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #72d4ff, stop:1 #2b72ca);
                color: #e9f7ff;
                font-size: 19px;
                font-weight: 700;
                qproperty-alignment: AlignCenter;
            }
            QLabel#pageTitle {
                font-size: 50px;
                font-weight: 700;
                color: #f2f6ff;
                letter-spacing: 1px;
            }
            QPushButton#headerIcon {
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                border-radius: 18px;
                background: rgba(62, 109, 165, 0.2);
                border: 1px solid rgba(133, 181, 245, 0.28);
                font-size: 16px;
            }
            QPushButton#headerIcon:hover {
                background: rgba(99, 157, 230, 0.3);
            }
            QFrame#functionBar {
                border: 1px solid rgba(123, 176, 247, 0.3);
                border-radius: 14px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #0f2948, stop:1 #102847);
            }
            QPushButton#navButton {
                border-radius: 10px;
                border: 1px solid transparent;
                padding: 8px 14px;
                background: transparent;
                font-size: 24px;
                color: #c3d8f6;
                text-align: left;
            }
            QPushButton#navButton:hover {
                background: rgba(80, 140, 210, 0.22);
            }
            QPushButton#navButton:checked {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2b5f93, stop:1 #3f77af);
                border: 1px solid rgba(128, 181, 245, 0.55);
                color: #c6ecff;
            }
            QFrame#dropdownPanel {
                border: 1px solid rgba(126, 176, 236, 0.55);
                border-radius: 14px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #3b5472, stop:1 #2a4567);
            }
            QPushButton#dropItem {
                text-align: center;
                padding: 8px 12px;
                border-radius: 10px;
                background: transparent;
                border: none;
                font-size: 22px;
                color: #d7e7fb;
            }
            QPushButton#dropItem[active="true"] {
                background: rgba(171, 207, 255, 0.16);
                border: 1px solid rgba(171, 207, 255, 0.3);
            }
            QPushButton#dropItem:hover {
                background: rgba(163, 206, 255, 0.18);
            }
            QFrame#contentTitleBar {
                border: 1px solid rgba(106, 154, 222, 0.28);
                border-radius: 14px 14px 0 0;
                background: rgba(17, 42, 73, 0.9);
            }
            QLabel#contentTitle {
                font-size: 34px;
                font-weight: 700;
                color: #eef6ff;
            }
            QFrame#contentBody {
                border: 1px solid rgba(106, 154, 222, 0.28);
                border-top: none;
                border-radius: 0 0 14px 14px;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 10px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(110, 130, 150, 0.4);
                border-radius: 5px;
            }
            QScrollBar:horizontal {
                height: 10px;
                background: transparent;
            }
            QScrollBar::handle:horizontal {
                background: rgba(110, 130, 150, 0.45);
                border-radius: 5px;
            }
            """
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
