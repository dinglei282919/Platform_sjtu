# auto_score.py
import random
import numpy as np

import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFrame
)


class AutoScoreWidget(QWidget):
    """自动评分页面，负责多维指标数据的随机生成、动态加权评分及雷达图呈现。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_def = [
            "热电比",
            "供电标煤耗",
            "供热标煤耗",
            "汽机负荷率",
            "能量转换比",
            "自发电占比"
        ]
        self._val_labels = {}
        self._weight_spins = {}
        self._current_data = {}
        self._build_ui()

        # 设置中文字体以支持Matplotlib正常显示中文标签
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei UI']
        matplotlib.rcParams['axes.unicode_minus'] = False

    def _build_ui(self):
        self.setObjectName("autoScoreRoot")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        left_panel = QGroupBox("数据与评分规则设定")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        headers = ["指标名称", "当前数值", "权重配置"]
        for col, text in enumerate(headers):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #a9c6e2; font-weight: bold;")
            grid_layout.addWidget(lbl, 0, col)

        for row, metric in enumerate(self.metrics_def, start=1):
            name_lbl = QLabel(metric)
            name_lbl.setAlignment(Qt.AlignCenter)

            val_lbl = QLabel("等待生成")
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet(
                "background: rgba(21, 35, 52, 0.95); border: 1px solid rgba(143, 182, 220, 0.35); border-radius: 4px; padding: 4px;")
            self._val_labels[metric] = val_lbl

            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.0, 10.0)
            weight_spin.setSingleStep(0.1)
            weight_spin.setValue(1.0)
            self._weight_spins[metric] = weight_spin

            grid_layout.addWidget(name_lbl, row, 0)
            grid_layout.addWidget(val_lbl, row, 1)
            grid_layout.addWidget(weight_spin, row, 2)

        left_layout.addLayout(grid_layout)
        left_layout.addStretch()

        actions = QHBoxLayout()
        self._generate_btn = QPushButton("随机生成数据")
        self._generate_btn.clicked.connect(self._generate_random_data)
        self._score_btn = QPushButton("执行加权评分")
        self._score_btn.clicked.connect(self._execute_scoring)
        self._score_btn.setEnabled(False)

        actions.addWidget(self._generate_btn)
        actions.addWidget(self._score_btn)
        left_layout.addLayout(actions)

        right_panel = QGroupBox("评估报告与六维蛛网图")
        right_layout = QVBoxLayout(right_panel)

        self._figure = Figure(dpi=100)
        self._figure.patch.set_facecolor('#1a2635')
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111, polar=True)
        self._ax.set_facecolor('#152334')
        self._ax.tick_params(colors='#d4e8ff')
        right_layout.addWidget(self._canvas, 3)

        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        self._result_box.setPlaceholderText("点击左侧执行加权评分后，此处将展示各项得分、加权总分及安全评分。")
        right_layout.addWidget(self._result_box, 2)

        content_row.addWidget(left_panel, 1)
        content_row.addWidget(right_panel, 2)
        main_layout.addLayout(content_row, 1)

        status_bar = QFrame()
        status_bar.setObjectName("autoScoreStatusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 2, 10, 2)
        self._status_label = QLabel("状态：等待生成数据")
        status_layout.addWidget(self._status_label)
        status_layout.addStretch()
        main_layout.addWidget(status_bar)

        self.setStyleSheet(
            """
            QWidget#autoScoreRoot {
                background: #1a2635;
            }
            QGroupBox {
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                background: rgba(31, 49, 70, 0.88);
                color: #d4e8ff;
                font-size: 16px;
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
                font-size: 14px;
            }
            QTextEdit, QDoubleSpinBox {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.95);
                color: #e7f2ff;
                padding: 2px 8px;
                min-height: 28px;
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
                padding: 7px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(62, 122, 174, 0.95);
            }
            QPushButton:disabled {
                color: #87a2bd;
                background: rgba(33, 55, 77, 0.75);
            }
            QFrame#autoScoreStatusBar {
                border: 1px solid rgba(126, 168, 208, 0.35);
                border-radius: 6px;
                background: rgba(25, 38, 55, 0.95);
            }
            """
        )

    def _generate_random_data(self):
        self._current_data = {
            "热电比": round(random.uniform(0.50, 0.70), 3),
            "供电标煤耗": round(random.uniform(190, 230), 2),
            "供热标煤耗": round(random.uniform(36, 42), 2),
            "汽机负荷率": round(random.uniform(0.65, 0.95), 3),
            "能量转换比": round(random.uniform(0.45, 0.75), 3),
            "自发电占比": round(random.uniform(0.55, 0.85), 3)
        }

        for metric in self.metrics_def:
            self._val_labels[metric].setText(str(self._current_data[metric]))

        self._score_btn.setEnabled(True)
        self._status_label.setText("状态：数据已生成，等待评分")

    def _calculate_single_score(self, metric, val):
        score = 60.0
        if metric == "热电比":
            if 0.58 <= val <= 0.62:
                score = 100.0
            elif val < 0.58:
                score = 100.0 - 40.0 * ((0.58 - val) / (0.58 - 0.45))
            else:
                score = 100.0 - 40.0 * ((val - 0.62) / (0.75 - 0.62))

        elif metric == "供电标煤耗":
            if val <= 200:
                score = 100.0
            else:
                score = 100.0 - 40.0 * ((val - 200) / (250 - 200))

        elif metric == "供热标煤耗":
            if val <= 38:
                score = 100.0
            else:
                score = 100.0 - 40.0 * ((val - 38) / (45 - 38))

        elif metric == "汽机负荷率":
            if val >= 0.8:
                score = 100.0
            else:
                score = 100.0 - 40.0 * ((0.8 - val) / (0.8 - 0.6))

        elif metric == "能量转换比":
            if val >= 0.6:
                score = 100.0
            else:
                score = 100.0 - 40.0 * ((0.6 - val) / (0.6 - 0.4))

        elif metric == "自发电占比":
            if 0.65 <= val <= 0.75:
                score = 100.0
            elif val < 0.65:
                score = 100.0 - 40.0 * ((0.65 - val) / (0.65 - 0.5))
            else:
                score = 100.0 - 40.0 * ((val - 0.75) / (0.9 - 0.75))

        return max(60.0, min(100.0, score))

    def _execute_scoring(self):
        scores = []
        weights = []

        for metric in self.metrics_def:
            val = self._current_data[metric]
            pts = self._calculate_single_score(metric, val)
            scores.append(pts)
            weights.append(self._weight_spins[metric].value())

        total_weight = sum(weights)
        if total_weight <= 0:
            total_weight = 1.0
            weights = [1.0] * len(self.metrics_def)

        normalized_weights = [w / total_weight for w in weights]
        weighted_total = sum(s * w for s, w in zip(scores, normalized_weights))

        safety_score = 1.0 - (weighted_total / 100.0)

        self._plot_radar_chart(scores)

        report = "【各项指标得分明细】\n"
        for i, metric in enumerate(self.metrics_def):
            report += f"{metric} ({self._current_data[metric]}): {scores[i]:.2f} 分 (权重: {normalized_weights[i]:.2%})\n"

        report += f"\n【综合评估】\n"
        report += f"加权总分: {weighted_total:.2f} / 100.00\n"
        report += f"潜在危险分数: {safety_score*100:.4f}\n"

        self._result_box.setText(report)
        self._status_label.setText("状态：评分完成")

    def _plot_radar_chart(self, scores):
        self._ax.clear()
        self._ax.set_facecolor('#152334')

        angles = np.linspace(0, 2 * np.pi, len(self.metrics_def), endpoint=False).tolist()
        scores_plot = scores + [scores[0]]
        angles_plot = angles + [angles[0]]

        self._ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='#63b9ff')
        self._ax.fill(angles_plot, scores_plot, alpha=0.3, color='#63b9ff')

        self._ax.set_thetagrids(np.degrees(angles), self.metrics_def, color='#d4e8ff', fontsize=11)
        self._ax.set_ylim(0, 100)
        self._ax.set_yticks([20, 40, 60, 80, 100])
        self._ax.set_yticklabels(["20", "40", "60", "80", "100"], color='#87a2bd', fontsize=9)
        self._ax.grid(color='#466385', linestyle='--', linewidth=0.5)
        self._ax.spines['polar'].set_color('#466385')

        self._canvas.draw()