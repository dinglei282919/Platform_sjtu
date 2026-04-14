import random
from datetime import datetime

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SharedParameterStore:
    _correlation_params = None

    @classmethod
    def set_correlation_params(cls, percent_min, percent_max, sigma1, sigma2):
        cls._correlation_params = {
            "percent_min": float(percent_min),
            "percent_max": float(percent_max),
            "sigma1": float(sigma1),
            "sigma2": float(sigma2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    @classmethod
    def get_correlation_params(cls):
        if cls._correlation_params is None:
            return None
        return dict(cls._correlation_params)


class CorrelationAnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._percent_min_input = None
        self._percent_max_input = None
        self._sigma1_input = None
        self._sigma2_input = None
        self._status_label = None
        self._build_ui()
        self.generate_random_values()

    def _build_ui(self):
        root = QFrame()
        root.setObjectName("corrRoot")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        panel = QGroupBox("参数设定")
        form = QFormLayout(panel)
        form.setVerticalSpacing(12)

        percent_holder = QFrame()
        percent_layout = QHBoxLayout(percent_holder)
        percent_layout.setContentsMargins(0, 0, 0, 0)
        percent_layout.setSpacing(8)
        self._percent_min_input = self._new_spin(0.05, 0.5, 0.05)
        self._percent_max_input = self._new_spin(0.05, 0.5, 0.1)
        percent_layout.addWidget(self._percent_min_input, 1)
        percent_layout.addWidget(QLabel("~"))
        percent_layout.addWidget(self._percent_max_input, 1)
        form.addRow("攻击幅度 [0.05, 0.5]:", percent_holder)

        self._sigma1_input = self._new_spin(0.01, 0.3, 0.2)
        form.addRow("实部噪声 [0.01, 0.3]:", self._sigma1_input)

        self._sigma2_input = self._new_spin(0.01, 0.3, 0.1)
        form.addRow("虚部噪声 [0.01, 0.3]:", self._sigma2_input)

        actions = QHBoxLayout()
        actions.addStretch()
        random_btn = QPushButton("随机生成一组参数")
        random_btn.clicked.connect(self.generate_random_values)
        actions.addWidget(random_btn)
        actions.addStretch()
        form.addRow("", actions)

        layout.addWidget(panel)

        self._status_label = QLabel()
        self._status_label.setObjectName("corrStatus")
        layout.addWidget(self._status_label)
        layout.addStretch()

        self.setStyleSheet(
            """
            QFrame#corrRoot {
                border: 1px solid rgba(126, 175, 245, 0.18);
                border-radius: 14px;
                background: qlineargradient(x1:0,y1:0,x2:0.9,y2:1,
                    stop:0 #1d3559, stop:1 #172b47);
            }
            QLabel#corrStatus {
                font-size: 18px;
                color: #9bd0ff;
                padding-left: 2px;
            }
            QGroupBox {
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background: rgba(31, 49, 70, 0.88);
                color: #d4e8ff;
                font-size: 18px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #d9ecff;
            }
            QLabel {
                color: #d2e5fb;
                font-size: 18px;
            }
            QDoubleSpinBox {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.95);
                color: #e7f2ff;
                min-height: 32px;
                padding: 2px 8px;
                font-size: 18px;
            }
            QPushButton {
                border: 1px solid rgba(101, 175, 235, 0.52);
                border-radius: 8px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(35, 90, 132, 0.95),
                    stop: 1 rgba(31, 71, 112, 0.95)
                );
                color: #dff2ff;
                padding: 8px 14px;
                font-weight: 700;
                font-size: 17px;
            }
            QPushButton:hover {
                background: rgba(62, 122, 174, 0.95);
                color: #ffffff;
            }
            """
        )

    def _new_spin(self, minimum, maximum, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        return spin

    def generate_random_values(self):
        percent_min = round(random.uniform(0.05, 0.5), 3)
        percent_max = round(random.uniform(percent_min, 0.5), 3)
        sigma1 = round(random.uniform(0.01, 0.3), 3)
        sigma2 = round(random.uniform(0.01, 0.3), 3)

        self._percent_min_input.setValue(percent_min)
        self._percent_max_input.setValue(percent_max)
        self._sigma1_input.setValue(sigma1)
        self._sigma2_input.setValue(sigma2)

        self._status_label.setText(
            f"已生成参数: 攻击幅度=[{percent_min:.3f}, {percent_max:.3f}]，"
            f"实部噪声={sigma1:.3f}，虚部噪声={sigma2:.3f}"
        )
        SharedParameterStore.set_correlation_params(percent_min, percent_max, sigma1, sigma2)
