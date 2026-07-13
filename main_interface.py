# -*- coding: utf-8 -*-
import sys
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ---- 原有模块导入（若缺失则用 try-except 保护） ----
try:
    from anomaly_detection import MultiScenarioAnomalyDetectionWidget
except ImportError:
    MultiScenarioAnomalyDetectionWidget = None
try:
    from correlation_analysis import CorrelationAnalysisWidget
except ImportError:
    CorrelationAnalysisWidget = None
try:
    from error_classification import ErrorClassificationWidget
except ImportError:
    ErrorClassificationWidget = None
try:
    from auto_score import AutoScoreWidget
except ImportError:
    AutoScoreWidget = None
try:
    from cdq_risk_matching import CDQMatchingWidget
except ImportError:
    CDQMatchingWidget = None
try:
    from process_control_dnn_mpc import ProcessControlDnnMpcWidget
except ImportError:
    ProcessControlDnnMpcWidget = None
try:
    from second_order_dynamic_system import SecondOrderDynamicSystemWidget
except ImportError:
    SecondOrderDynamicSystemWidget = None

# ---- 新增导入 ----
from sdg_hazop import SDG_HazopWidget
from sil_validation import SILValidationWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("流程行业动态风险管控工具集平台")
        self.setMinimumSize(1180, 760)
        self.resize(1500, 860)
        self._dropdown = None
        self._dropdown_layout = None
        self._nav_menu_map = {}
        self._nav_title_map = {}
        self._active_dropdown_button = None
        self._content_title_label = None

        # 所有页面容器
        self._anomaly_content_widget = None
        self._correlation_content_widget = None
        self._second_order_content_widget = None
        self._error_class_content_widget = None
        self._auto_score_content_widget = None
        self._cdq_matching_content_widget = None
        self._process_training_content_widget = None
        self._process_mpc_content_widget = None
        self._sis_detection_content_widget = None
        self._sil_validation_content_widget = None

        # 预创建两个新模块的实例（避免重复加载）
        self._sis_widget = SDG_HazopWidget()
        self._sil_widget = SILValidationWidget()

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
        # 默认进入 SIS自主化检测 -> SDG-HAZOP
        self._on_submodule_clicked("异构数据治理", "关联分析")

    def _build_header(self):
        bar = QFrame()
        bar.setObjectName("headerBar")
        bar.setFixedHeight(64)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 0, 16, 0)
        layout.setSpacing(12)

        logo = QLabel("◈")
        logo.setObjectName("logoBlock")
        title = QLabel("流程行业动态风险管控工具集平台")
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

    def _make_nav_button(self, icon: str, text: str, checked=False, has_dropdown=True):
        suffix = "  ⌄" if has_dropdown else ""
        btn = QPushButton(f"{icon}  {text}{suffix}")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setObjectName("navButton")
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _build_function_bar(self):
        bar = QFrame()
        bar.setObjectName("functionBar")
        bar.setFixedHeight(58)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 菜单项：为 SIS自主化检测 和 在线SIL验证 增加子菜单
        items = [
            ("📊", "异构数据治理", True, ["关联分析", "二阶非线性动态系统"]),
            ("🏭", "异常行为检测", False, ["基于移动目标防御的异常检测"]),
            ("📈", "风险动态分析", False,
             ["潜在安全威胁识别与自动分类",
              "多评估准则融合的风险学习分析",
              "风险场景动态匹配与适配方案生成算法"]),
            ("🎛", "风险管控优化决策", False, ["控制模型训练评估", "优化控制仿真验证"]),
            ("🛡", "SIS自主化检测", False, ["SDG-HAZOP"]),
            ("✅", "在线SIL验证", False, ["基于GSPN-MC模型的动态化SIL验证方法"]),
        ]

        for icon, text, checked, menu_items in items:
            button = self._make_nav_button(icon, text, checked=checked, has_dropdown=bool(menu_items))
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
        title_bar.setFixedHeight(56)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(18, 0, 18, 0)
        self._content_title_label = QLabel("SIS自主化检测 - SDG-HAZOP")
        self._content_title_label.setObjectName("contentTitle")
        title_layout.addWidget(self._content_title_label)
        title_layout.addStretch()

        body = QFrame()
        body.setObjectName("contentBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # ---- 异常行为检测 ----
        if MultiScenarioAnomalyDetectionWidget is not None:
            anomaly_content = QFrame()
            anomaly_layout = QVBoxLayout(anomaly_content)
            anomaly_layout.setContentsMargins(0, 0, 0, 0)
            anomaly_layout.setSpacing(0)
            anomaly_layout.addWidget(MultiScenarioAnomalyDetectionWidget())
            self._anomaly_content_widget = anomaly_content
            self._anomaly_content_widget.hide()
            body_layout.addWidget(self._anomaly_content_widget, 1)

        # ---- 关联分析（默认） ----
        if CorrelationAnalysisWidget is not None:
            correlation_content = QFrame()
            correlation_layout = QVBoxLayout(correlation_content)
            correlation_layout.setContentsMargins(0, 0, 0, 0)
            correlation_layout.setSpacing(0)
            correlation_layout.addWidget(CorrelationAnalysisWidget())
            self._correlation_content_widget = correlation_content
            self._correlation_content_widget.hide()
            body_layout.addWidget(self._correlation_content_widget, 1)

        # ---- 二阶非线性动态系统 ----
        if SecondOrderDynamicSystemWidget is not None:
            second_order_content = QFrame()
            second_order_layout = QVBoxLayout(second_order_content)
            second_order_layout.setContentsMargins(0, 0, 0, 0)
            second_order_layout.setSpacing(0)
            second_order_layout.addWidget(SecondOrderDynamicSystemWidget())
            self._second_order_content_widget = second_order_content
            self._second_order_content_widget.hide()
            body_layout.addWidget(self._second_order_content_widget, 1)

        # ---- 风险管控优化决策 ----
        if ProcessControlDnnMpcWidget is not None:
            process_training_content = QFrame()
            process_training_layout = QVBoxLayout(process_training_content)
            process_training_layout.setContentsMargins(0, 0, 0, 0)
            process_training_layout.setSpacing(0)
            process_training_layout.addWidget(ProcessControlDnnMpcWidget(page_mode="training"))
            self._process_training_content_widget = process_training_content
            self._process_training_content_widget.hide()
            body_layout.addWidget(self._process_training_content_widget, 1)

            process_mpc_content = QFrame()
            process_mpc_layout = QVBoxLayout(process_mpc_content)
            process_mpc_layout.setContentsMargins(0, 0, 0, 0)
            process_mpc_layout.setSpacing(0)
            process_mpc_layout.addWidget(ProcessControlDnnMpcWidget(page_mode="mpc"))
            self._process_mpc_content_widget = process_mpc_content
            self._process_mpc_content_widget.hide()
            body_layout.addWidget(self._process_mpc_content_widget, 1)

        # ---- 风险动态分析子模块 ----
        if ErrorClassificationWidget is not None:
            error_class_content = QFrame()
            error_class_layout = QVBoxLayout(error_class_content)
            error_class_layout.setContentsMargins(0, 0, 0, 0)
            error_class_layout.setSpacing(0)
            error_class_layout.addWidget(ErrorClassificationWidget())
            self._error_class_content_widget = error_class_content
            self._error_class_content_widget.hide()
            body_layout.addWidget(self._error_class_content_widget, 1)

        if AutoScoreWidget is not None:
            auto_score_content = QFrame()
            auto_score_layout = QVBoxLayout(auto_score_content)
            auto_score_layout.setContentsMargins(0, 0, 0, 0)
            auto_score_layout.setSpacing(0)
            auto_score_layout.addWidget(AutoScoreWidget())
            self._auto_score_content_widget = auto_score_content
            self._auto_score_content_widget.hide()
            body_layout.addWidget(self._auto_score_content_widget, 1)

        if CDQMatchingWidget is not None:
            cdq_content = QFrame()
            cdq_layout = QVBoxLayout(cdq_content)
            cdq_layout.setContentsMargins(0, 0, 0, 0)
            cdq_layout.setSpacing(0)
            cdq_layout.addWidget(CDQMatchingWidget())
            self._cdq_matching_content_widget = cdq_content
            self._cdq_matching_content_widget.hide()
            body_layout.addWidget(self._cdq_matching_content_widget, 1)

        # ---- 新增：SIS自主化检测 ----
        self._sis_detection_content_widget = QFrame()
        self._sis_detection_content_widget.setObjectName("moduleContainer")
        sis_layout = QVBoxLayout(self._sis_detection_content_widget)
        sis_layout.setContentsMargins(0, 0, 0, 0)
        sis_layout.setSpacing(0)
        sis_layout.addWidget(self._sis_widget)
        self._sis_detection_content_widget.hide()
        body_layout.addWidget(self._sis_detection_content_widget, 1)

        # ---- 新增：在线SIL验证 ----
        self._sil_validation_content_widget = QFrame()
        self._sil_validation_content_widget.setObjectName("moduleContainer")
        sil_layout = QVBoxLayout(self._sil_validation_content_widget)
        sil_layout.setContentsMargins(0, 0, 0, 0)
        sil_layout.setSpacing(0)
        sil_layout.addWidget(self._sil_widget)
        self._sil_validation_content_widget.hide()
        body_layout.addWidget(self._sil_validation_content_widget, 1)

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
        menu_items = self._nav_menu_map.get(button, [])
        nav_title = self._nav_title_map.get(button, "")
        if self._dropdown.isVisible() and self._active_dropdown_button is button:
            self._dropdown.hide()
            return
        self._active_dropdown_button = button
        for nav_btn in self._nav_menu_map:
            nav_btn.setChecked(nav_btn is button)
        if not menu_items:
            self._on_empty_module_clicked(nav_title)
            return
        self._set_dropdown_items(nav_title, menu_items)
        self._position_dropdown()
        self._dropdown.show()
        self._dropdown.raise_()

    def _on_empty_module_clicked(self, nav_title):
        self._dropdown.hide()
        if self._content_title_label is not None:
            self._content_title_label.setText(nav_title)
        if self._active_dropdown_button:
            items = self._nav_menu_map.get(self._active_dropdown_button, [])
            if items:
                self._on_submodule_clicked(nav_title, items[0])

    def _on_submodule_clicked(self, nav_title, submodule_title):
        self._dropdown.hide()
        if self._content_title_label is not None:
            self._content_title_label.setText(f"{nav_title} - {submodule_title}")

        # 隐藏所有容器
        for widget in (
            self._anomaly_content_widget,
            self._correlation_content_widget,
            self._second_order_content_widget,
            self._error_class_content_widget,
            self._auto_score_content_widget,
            self._cdq_matching_content_widget,
            self._process_training_content_widget,
            self._process_mpc_content_widget,
            self._sis_detection_content_widget,
            self._sil_validation_content_widget,
        ):
            if widget is not None:
                widget.hide()

        # 显示目标容器
        if submodule_title == "基于移动目标防御的异常检测":
            self._anomaly_content_widget.show()
        elif submodule_title == "二阶非线性动态系统":
            self._second_order_content_widget.show()
        elif submodule_title == "潜在安全威胁识别与自动分类":
            self._error_class_content_widget.show()
        elif submodule_title == "多评估准则融合的风险学习分析":
            self._auto_score_content_widget.show()
        elif submodule_title == "风险场景动态匹配与适配方案生成算法":
            self._cdq_matching_content_widget.show()
        elif submodule_title == "控制模型训练评估":
            self._process_training_content_widget.show()
        elif submodule_title == "优化控制仿真验证":
            self._process_mpc_content_widget.show()
        elif submodule_title == "SDG-HAZOP":
            self._sis_detection_content_widget.show()
            if not hasattr(self._sis_widget, '_signal_connected'):
                self._sis_widget.request_sil_validation.connect(self._on_request_sil)
                self._sis_widget._signal_connected = True
        elif submodule_title == "基于GSPN-MC模型的动态化SIL验证方法":
            self._sil_validation_content_widget.show()
        else:
            if self._correlation_content_widget is not None:
                self._correlation_content_widget.show()

    def _on_request_sil(self, node_id, prob, sev):
        self._on_submodule_clicked("在线SIL验证", "SIL 验证")
        if hasattr(self._sil_widget, 'set_recommended_params'):
            self._sil_widget.set_recommended_params(node_id, prob, sev)
        QMessageBox.information(
            self,
            "SIL 验证请求",
            f"节点 {node_id} 需要 SIL 验证\n概率: {prob:.4e}\n严重性: {sev}\n已切换到 SIL 验证模块。"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_dropdown()

    def closeEvent(self, event):
        for widget in (self._sis_widget, self._sil_widget):
            if widget is not None:
                checker = getattr(widget, "has_running_worker", None)
                if callable(checker) and checker():
                    QMessageBox.warning(self, "后台任务运行中", "请等待当前计算任务完成后再关闭。")
                    event.ignore()
                    return
        super().closeEvent(event)

    def showEvent(self, event):
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
                font-size: 42px;
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
                font-size: 21px;
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
                font-size: 29px;
                font-weight: 700;
                color: #eef6ff;
            }
            QFrame#contentBody {
                border: 1px solid rgba(106, 154, 222, 0.28);
                border-top: none;
                border-radius: 0 0 14px 14px;
                background: transparent;
            }
            /* 模块容器背景深色 */
            QFrame#moduleContainer {
                background: rgba(13, 31, 59, 0.95);
                border: none;
            }
            QGroupBox {
                border: 1px solid rgba(123, 176, 247, 0.3);
                border-radius: 8px;
                margin-top: 1ex;
                color: #d8e7ff;
                background: rgba(13, 31, 59, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #c3d8f6;
            }
            QLineEdit, QComboBox, QTextEdit {
                background: rgba(13, 31, 59, 0.8);
                border: 1px solid rgba(123, 176, 247, 0.3);
                border-radius: 4px;
                padding: 4px;
                color: #d8e7ff;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: rgba(123, 176, 247, 0.8);
            }
            QPushButton {
                background: rgba(62, 109, 165, 0.3);
                border: 1px solid rgba(123, 176, 247, 0.3);
                border-radius: 6px;
                padding: 6px 12px;
                color: #d8e7ff;
            }
            QPushButton:hover {
                background: rgba(99, 157, 230, 0.4);
            }
            QLabel {
                color: #d8e7ff;
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


if __name__ == "__main__":
    main()