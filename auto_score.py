# auto_score.py
import numpy as np

from platform_core.scoring import METRIC_CONFIGS, evaluate, random_values, single_score

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
        self.metric_configs = {
            name: {
                "range": (config["minimum"], config["maximum"]),
                "default": config["default"],
                "decimals": config["decimals"],
                "step": config["step"],
            }
            for name, config in METRIC_CONFIGS.items()
        }
        self.metrics_def = list(self.metric_configs)
        self._value_spins = {}
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

            config = self.metric_configs[metric]
            value_spin = QDoubleSpinBox()
            value_spin.setRange(*config["range"])
            value_spin.setDecimals(config["decimals"])
            value_spin.setSingleStep(config["step"])
            value_spin.setValue(config["default"])
            value_spin.setAlignment(Qt.AlignCenter)
            self._value_spins[metric] = value_spin

            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.0, 10.0)
            weight_spin.setSingleStep(0.1)
            weight_spin.setValue(1.0)
            self._weight_spins[metric] = weight_spin

            grid_layout.addWidget(name_lbl, row, 0)
            grid_layout.addWidget(value_spin, row, 1)
            grid_layout.addWidget(weight_spin, row, 2)

        left_layout.addLayout(grid_layout)
        left_layout.addStretch()

        actions = QHBoxLayout()
        self._generate_btn = QPushButton("随机生成数据")
        self._generate_btn.clicked.connect(self._generate_random_data)
        self._score_btn = QPushButton("执行加权评分")
        self._score_btn.clicked.connect(self._execute_scoring)
        self._score_btn.setEnabled(True)

        actions.addWidget(self._generate_btn)
        actions.addWidget(self._score_btn)
        left_layout.addLayout(actions)

        right_panel = QGroupBox("评估报告与六维蛛网图")
        right_layout = QVBoxLayout(right_panel)

        # ==========================================
        # 新增水平布局，将蛛网图与大分数值左右并排
        # ==========================================
        chart_and_score_layout = QHBoxLayout()

        self._figure = Figure(dpi=100)
        self._figure.patch.set_facecolor('#1a2635')
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111, polar=True)
        self._ax.set_facecolor('#152334')
        self._ax.tick_params(colors='#d4e8ff')
        chart_and_score_layout.addWidget(self._canvas, 3)  # 蛛网图占据比例为3

        # 大分数值展示面板
        self._big_score_frame = QFrame()
        self._big_score_frame.setObjectName("bigScoreFrame")
        big_score_layout = QVBoxLayout(self._big_score_frame)
        big_score_layout.setAlignment(Qt.AlignCenter)
        big_score_layout.setSpacing(10)

        self._lbl_total_title = QLabel("综合加权总分")
        self._lbl_total_title.setAlignment(Qt.AlignCenter)
        self._lbl_total_title.setStyleSheet(
            "color: #a9c6e2; font-size: 16px; font-weight: bold; border: none; background: transparent;")

        self._lbl_total_val = QLabel("--")
        self._lbl_total_val.setAlignment(Qt.AlignCenter)
        self._lbl_total_val.setStyleSheet(
            "color: #63b9ff; font-size: 42px; font-weight: bold; border: none; background: transparent;")

        self._lbl_risk_title = QLabel("潜在危险分数")
        self._lbl_risk_title.setAlignment(Qt.AlignCenter)
        self._lbl_risk_title.setStyleSheet(
            "color: #a9c6e2; font-size: 16px; font-weight: bold; margin-top: 30px; border: none; background: transparent;")

        self._lbl_risk_val = QLabel("--")
        self._lbl_risk_val.setAlignment(Qt.AlignCenter)
        self._lbl_risk_val.setStyleSheet(
            "color: #ff6b6b; font-size: 36px; font-weight: bold; border: none; background: transparent;")

        big_score_layout.addWidget(self._lbl_total_title)
        big_score_layout.addWidget(self._lbl_total_val)
        big_score_layout.addWidget(self._lbl_risk_title)
        big_score_layout.addWidget(self._lbl_risk_val)

        chart_and_score_layout.addWidget(self._big_score_frame, 1)  # 分数面板占据比例为1
        right_layout.addLayout(chart_and_score_layout, 3)

        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        self._result_box.setPlaceholderText("点击左侧执行加权评分后，此处将展示各项得分权重明细。")
        right_layout.addWidget(self._result_box, 2)

        content_row.addWidget(left_panel, 1)
        content_row.addWidget(right_panel, 2)
        main_layout.addLayout(content_row, 1)

        status_bar = QFrame()
        status_bar.setObjectName("autoScoreStatusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 2, 10, 2)
        self._status_label = QLabel("状态：可手动修改数值并执行评分")
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
            QFrame#bigScoreFrame {
                background: rgba(21, 35, 52, 0.6);
                border: 1px solid rgba(101, 175, 235, 0.3);
                border-radius: 8px;
            }
            """
        )

    def _generate_random_data(self):
        generated_data = random_values()

        for metric in self.metrics_def:
            self._value_spins[metric].setValue(generated_data[metric])

        self._status_label.setText("状态：数据已生成，等待评分")

    def _calculate_single_score(self, metric, val):
        return single_score(metric, val)

    def _execute_scoring(self):
        self._current_data = {
            metric: self._value_spins[metric].value()
            for metric in self.metrics_def
        }
        result = evaluate(
            self._current_data,
            {metric: self._weight_spins[metric].value() for metric in self.metrics_def},
        )
        scores = result["radar"]["scores"]
        normalized_weights = [item["normalized_weight"] for item in result["items"]]
        weighted_total = result["total_score"]
        safety_score = result["danger_score"] / 100.0

        self._plot_radar_chart(scores)

        # 更新右侧大分数值显示
        self._lbl_total_val.setText(f"{weighted_total:.2f}")
        self._lbl_risk_val.setText(f"{safety_score * 100:.2f}")

        report = "【各项指标得分明细】\n"
        for i, metric in enumerate(self.metrics_def):
            report += f"{metric} ({self._current_data[metric]}): {scores[i]:.2f} 分 (权重: {normalized_weights[i]:.2%})\n"

        report += f"\n【综合评估】\n"
        report += f"加权总分: {weighted_total:.2f} / 100.00\n"
        report += f"潜在危险分数: {safety_score * 100:.4f}\n"

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

        # 在图上为每个数据点标注单项得分
        # Matplotlib does not accept CSS-style 'rgba(...)' strings for colors.
        # Use an RGBA tuple with normalized 0-1 values instead.
        bbox_props = dict(boxstyle="round,pad=0.3", fc=(21/255, 35/255, 52/255, 0.8), ec="#63b9ff", lw=1)
        for angle, score in zip(angles, scores):
            # 将文字偏移量向外推，防止遮挡线条
            self._ax.text(angle, score + 8, f"{score:.1f}",
                          color='#ffffff', fontsize=10, ha='center', va='center', bbox=bbox_props)

        self._ax.set_thetagrids(np.degrees(angles), self.metrics_def, color='#d4e8ff', fontsize=11)

        # 将上限放宽至115，避免最外围的分数标签被图表边缘裁切
        self._ax.set_ylim(0, 115)
        self._ax.set_yticks([20, 40, 60, 80, 100])
        self._ax.set_yticklabels(["20", "40", "60", "80", "100"], color='#87a2bd', fontsize=9)
        self._ax.grid(color='#466385', linestyle='--', linewidth=0.5)
        self._ax.spines['polar'].set_color('#466385')

        self._canvas.draw()
