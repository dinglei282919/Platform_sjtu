# -*- coding: utf-8 -*-
"""
SIL 验证工具 (PySide6 版本) - 仅使用GSPN-MC仿真
支持全局共因失效 (Total β) 和部分共因失效 (Partial β)
左侧面板采用滚动区域，适应大量输入框
"""
import sys
import numpy as np
import heapq
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.stats import gamma as gamma_dist
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QLineEdit, QTextEdit, QFrame,
    QGroupBox, QFormLayout, QMessageBox, QProgressBar,
    QScrollArea
)
from PySide6.QtCore import Qt

# ==================== 配置常量 ====================
HOURS_PER_YEAR = 8760

# ==================== 参数估计模块 ====================
class BayesianLambdaEstimator:
    def __init__(self, T: float, k: float, confidence: float = 0.95):
        self.T = T
        self.k = k
        self.confidence = confidence
        self.alpha_post = k + 0.5
        self.beta_post = T

    def point_estimate(self) -> float:
        if self.alpha_post > 1/3:
            return (self.alpha_post - 1/3) / self.beta_post
        return 0.0

    def confidence_interval(self) -> Tuple[float, float]:
        alpha_tail = (1 - self.confidence) / 2
        lower = gamma_dist.ppf(alpha_tail, self.alpha_post) / self.beta_post
        upper = gamma_dist.ppf(1 - alpha_tail, self.alpha_post) / self.beta_post
        return lower, upper

    def get_fit(self) -> Tuple[float, float, float]:
        median = self.point_estimate() * 1e9
        low, high = self.confidence_interval()
        return median, low*1e9, high*1e9

# ==================== GSPN 仿真引擎 (MooN) ====================
@dataclass
class SimParams:
    N: int = 4
    M: int = 2
    TI: float = 8760
    MRT: float = 8
    LAMBDA_DU: float = (107 + 4.11) * 1e-9
    BETA1: float = 1.0
    beta_list: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    SIM_YEARS: int = 10000
    WARMUP_YEARS: int = 1000
    NUM_SIM: int = 500

    def __post_init__(self):
        expected_len = self.N - 1
        if len(self.beta_list) != expected_len:
            if len(self.beta_list) < expected_len:
                self.beta_list += [0.0] * (expected_len - len(self.beta_list))
            else:
                self.beta_list = self.beta_list[:expected_len]
        total = self.BETA1 + sum(self.beta_list)
        if total > 0 and abs(total - 1.0) > 1e-12:
            self.BETA1 /= total
            for i in range(len(self.beta_list)):
                self.beta_list[i] /= total
        self.beta_dict = {k: self.beta_list[k-2] for k in range(2, self.N+1)}
        self.BETA_TOTAL = 1 - self.BETA1

    @property
    def SIM_HOURS(self) -> float:
        return self.SIM_YEARS * HOURS_PER_YEAR

    @property
    def WARMUP_HOURS(self) -> float:
        return self.WARMUP_YEARS * HOURS_PER_YEAR

class State(Enum):
    OK = 0
    DU = 1
    TEST = 2

@dataclass(order=True)
class Event:
    time: float
    type: str
    channel: int
    seq: int = field(default=0, compare=False)

class GSPN_MooN_Base:
    def __init__(self, params: SimParams):
        self.p = params
        self.N = params.N
        self.M = params.M
        self.reset()

    def reset(self):
        self.states = [State.OK for _ in range(self.N)]
        self.t = 0.0
        self.queue = []
        self._seq = 0
        for i in range(self.N):
            self._schedule_fail(i)
        self._schedule_initial_ccf()

    def _add_event(self, time: float, type: str, channel: int):
        self._seq += 1
        heapq.heappush(self.queue, Event(time, type, channel, self._seq))

    def _schedule_fail(self, channel: int):
        interval = np.random.exponential(1 / self.p.LAMBDA_DU)
        self._add_event(self.t + interval, 'fail', channel)

    def _schedule_test(self, channel: int):
        next_test = np.ceil(self.t / self.p.TI) * self.p.TI
        if abs(next_test - self.t) < 1e-10:
            next_test += self.p.TI
        self._add_event(next_test, 'test', channel)

    def _schedule_repair(self, channel: int):
        interval = np.random.exponential(self.p.MRT)
        self._add_event(self.t + interval, 'repair', channel)

    def _clear_channel_events(self, channel: int):
        new_queue = []
        for ev in self.queue:
            if ev.channel != channel:
                new_queue.append(ev)
        heapq.heapify(new_queue)
        self.queue = new_queue

    def _get_ok_count(self) -> int:
        return sum(1 for s in self.states if s == State.OK)

    def is_danger(self) -> bool:
        return self._get_ok_count() < self.M

    def _apply_ccf_to_channels(self, k: int, current_time: float):
        ok_list = [i for i, s in enumerate(self.states) if s == State.OK]
        if not ok_list:
            return
        n_affect = min(k, len(ok_list))
        chosen = random.sample(ok_list, n_affect)
        for ch in chosen:
            self.states[ch] = State.DU
            self._clear_channel_events(ch)
            self._schedule_test(ch)

    def _schedule_initial_ccf(self):
        raise NotImplementedError

    def simulate(self) -> float:
        self.reset()
        danger_time = 0.0
        while self.t < self.p.SIM_HOURS:
            if not self.queue:
                dt = self.p.SIM_HOURS - self.t
                if self.t >= self.p.WARMUP_HOURS and self.is_danger():
                    danger_time += dt
                break
            next_event = self.queue[0]
            dt = next_event.time - self.t
            if self.t >= self.p.WARMUP_HOURS and self.is_danger():
                danger_time += dt
            self.t = next_event.time
            heapq.heappop(self.queue)
            ch = next_event.channel
            et = next_event.type

            if et == 'fail':
                if self.states[ch] == State.OK:
                    self.states[ch] = State.DU
                    self._clear_channel_events(ch)
                    self._schedule_test(ch)
            elif et == 'test':
                if self.states[ch] == State.DU:
                    self.states[ch] = State.TEST
                    self._clear_channel_events(ch)
                    self._schedule_repair(ch)
            elif et == 'repair':
                if self.states[ch] == State.TEST:
                    self.states[ch] = State.OK
                    self._clear_channel_events(ch)
                    self._schedule_fail(ch)
            elif et == 'ccf_total':
                self._handle_ccf_total()
            elif et.startswith('ccf'):
                k = int(et[3:])
                self._apply_ccf_to_channels(k, self.t)
                self._schedule_ccf(k)

        effective = self.p.SIM_HOURS - self.p.WARMUP_HOURS
        return danger_time / effective if effective > 0 else 1.0

    def _schedule_ccf(self, k: int):
        raise NotImplementedError

    def _handle_ccf_total(self):
        for i in range(self.N):
            if self.states[i] == State.OK:
                self.states[i] = State.DU
                self._clear_channel_events(i)
                self._schedule_test(i)

class GSPN_MooN_TotalBeta(GSPN_MooN_Base):
    def _schedule_initial_ccf(self):
        rate = self.p.BETA_TOTAL * self.p.LAMBDA_DU
        if rate > 0:
            interval = np.random.exponential(1 / rate)
            self._add_event(self.t + interval, 'ccf_total', -1)

    def _schedule_ccf(self, k: int):
        pass

    def _handle_ccf_total(self):
        super()._handle_ccf_total()
        self._schedule_initial_ccf()

class GSPN_MooN_MGL(GSPN_MooN_Base):
    def _schedule_initial_ccf(self):
        for k in range(2, self.N + 1):
            beta_k = self.p.beta_dict.get(k, 0.0)
            if beta_k > 0:
                rate = beta_k * self.p.LAMBDA_DU
                interval = np.random.exponential(1 / rate)
                self._add_event(self.t + interval, f'ccf{k}', -1)

    def _schedule_ccf(self, k: int):
        beta_k = self.p.beta_dict.get(k, 0.0)
        if beta_k > 0:
            rate = beta_k * self.p.LAMBDA_DU
            interval = np.random.exponential(1 / rate)
            self._add_event(self.t + interval, f'ccf{k}', -1)

def run_simulation(params: SimParams, model_type: str = 'MGL', n_sim: Optional[int] = None):
    if n_sim is None:
        n_sim = params.NUM_SIM
    results = []
    for _ in range(n_sim):
        if model_type == 'MGL':
            sim = GSPN_MooN_MGL(params)
        else:
            sim = GSPN_MooN_TotalBeta(params)
        results.append(sim.simulate())
    results = np.array(results)
    mean = np.mean(results)
    std = np.std(results, ddof=1) if n_sim > 1 else 0.0
    return results, mean, std

# ==================== SIL 等级映射 ====================
def sil_from_pfd(pfd: float) -> int:
    if pfd < 1e-4:
        return 4
    elif 1e-4 <= pfd < 1e-3:
        return 3
    elif 1e-3 <= pfd < 1e-2:
        return 2
    elif 1e-2 <= pfd < 1e-1:
        return 1
    else:
        return 0

# ==================== SIL 验证 Widget (PySide6) ====================
class SILValidationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._load_example_data()

    def _init_ui(self):
        # ========== 统一样式 ==========
        btn_style = """
            QPushButton {
                background: rgba(62, 109, 165, 0.3);
                border: 1px solid rgba(200, 200, 200, 0.3);
                border-radius: 6px;
                padding: 6px 12px;
                color: #d8e7ff;
            }
            QPushButton:hover {
                background: rgba(99, 157, 230, 0.4);
            }
        """
        input_style = """
            QLineEdit, QComboBox {
                background: rgba(13, 31, 59, 0.8);
                border: 1px solid rgba(200, 200, 200, 0.3);
                border-radius: 4px;
                padding: 4px;
                color: #d8e7ff;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: rgba(123, 176, 247, 0.8);
            }
        """
        groupbox_style = """
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
        """
        scroll_style = """
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 10px;
                background: rgba(13, 31, 59, 0.5);
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(200, 200, 200, 0.4);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(200, 200, 200, 0.6);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
        # ==================================

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧控制面板 - 深色背景，包含滚动区域
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setMinimumWidth(380)
        left_panel.setStyleSheet("background-color: rgba(13, 31, 59, 0.95); border: none;")
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(scroll_style)

        # 滚动内容容器
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(10)

        # ---- 将原有的所有控件放入 scroll_layout ----

        # 1. 表决架构
        group1 = QGroupBox("1. 表决架构 (MooN)")
        group1.setStyleSheet(groupbox_style)
        form1 = QFormLayout()
        self.entry_m = QLineEdit("2")
        self.entry_m.setStyleSheet(input_style)
        self.entry_n = QLineEdit("4")
        self.entry_n.setStyleSheet(input_style)
        self.entry_n.textChanged.connect(self._update_partial_beta_inputs)  # N变化时更新部分共因输入
        form1.addRow("M (表决阈值):", self.entry_m)
        form1.addRow("N (通道总数):", self.entry_n)
        group1.setLayout(form1)
        scroll_layout.addWidget(group1)

        # 2. 失效率配置
        group2 = QGroupBox("2. 失效率配置")
        group2.setStyleSheet(groupbox_style)
        form2 = QFormLayout()
        self.lambda_source = "direct"
        # 直接输入框
        self.direct_frame = QFrame()
        direct_layout = QHBoxLayout(self.direct_frame)
        direct_layout.setContentsMargins(0,0,0,0)
        self.direct_lam = QLineEdit("111.11")
        self.direct_lam.setStyleSheet(input_style)
        direct_layout.addWidget(QLabel("λ (FIT):"))
        direct_layout.addWidget(self.direct_lam)
        form2.addRow(self.direct_frame)

        # 数据估计框（默认隐藏）
        self.est_frame = QFrame()
        self.est_frame.hide()
        est_layout = QFormLayout(self.est_frame)
        self.entry_T = QLineEdit("876000")
        self.entry_T.setStyleSheet(input_style)
        self.entry_k = QLineEdit("5")
        self.entry_k.setStyleSheet(input_style)
        self.entry_low = QLineEdit("20")
        self.entry_low.setStyleSheet(input_style)
        self.entry_high = QLineEdit("80")
        self.entry_high.setStyleSheet(input_style)
        est_layout.addRow("总运行时间 (h):", self.entry_T)
        est_layout.addRow("失效次数 (等效):", self.entry_k)
        est_layout.addRow("低限阈值:", self.entry_low)
        est_layout.addRow("高限阈值:", self.entry_high)

        btn_import = QPushButton("📥 导入示例数据")
        btn_import.setStyleSheet(btn_style)
        btn_import.clicked.connect(self._import_sample_data)
        btn_estimate = QPushButton("📊 估计 λ (手动T/k)")
        btn_estimate.setStyleSheet(btn_style)
        btn_estimate.clicked.connect(self._estimate_lambda)
        est_layout.addRow(btn_import)
        est_layout.addRow(btn_estimate)
        self.lambda_est_label = QLabel("")
        est_layout.addRow(self.lambda_est_label)
        form2.addRow(self.est_frame)

        # 切换按钮
        btn_toggle = QPushButton("切换到 运行数据估计")
        btn_toggle.setStyleSheet(btn_style)
        btn_toggle.clicked.connect(self._toggle_lambda_source)
        form2.addRow(btn_toggle)
        group2.setLayout(form2)
        scroll_layout.addWidget(group2)

        # 3. 共因失效模式
        group_ccf = QGroupBox("3. 共因失效模式")
        group_ccf.setStyleSheet(groupbox_style)
        form_ccf = QFormLayout()

        self.ccf_mode_combo = QComboBox()
        self.ccf_mode_combo.addItems(["全局共因 (Total β)", "部分共因 (Partial β)"])
        self.ccf_mode_combo.setStyleSheet(input_style)
        self.ccf_mode_combo.currentIndexChanged.connect(self._on_ccf_mode_changed)
        form_ccf.addRow("共因模式:", self.ccf_mode_combo)

        # 全局β输入
        self.total_beta_frame = QFrame()
        total_beta_layout = QHBoxLayout(self.total_beta_frame)
        total_beta_layout.setContentsMargins(0,0,0,0)
        self.entry_total_beta = QLineEdit("0.1")
        self.entry_total_beta.setStyleSheet(input_style)
        total_beta_layout.addWidget(QLabel("β (共因因子):"))
        total_beta_layout.addWidget(self.entry_total_beta)
        form_ccf.addRow(self.total_beta_frame)

        # 部分共因输入 (β2, β3, ...) 动态生成，放在一个独立的Frame中
        self.partial_beta_frame = QFrame()
        self.partial_beta_frame.hide()
        self.partial_beta_layout = QFormLayout(self.partial_beta_frame)
        self.partial_beta_layout.setContentsMargins(0,0,0,0)
        form_ccf.addRow(self.partial_beta_frame)

        group_ccf.setLayout(form_ccf)
        scroll_layout.addWidget(group_ccf)

        # 4. 仿真控制
        group4 = QGroupBox("4. 仿真控制参数")
        group4.setStyleSheet(groupbox_style)
        form4 = QFormLayout()
        self.entry_ti = QLineEdit("8760")
        self.entry_ti.setStyleSheet(input_style)
        self.entry_mrt = QLineEdit("8")
        self.entry_mrt.setStyleSheet(input_style)
        self.entry_nsim = QLineEdit("500")
        self.entry_nsim.setStyleSheet(input_style)
        self.entry_years = QLineEdit("10000")
        self.entry_years.setStyleSheet(input_style)
        form4.addRow("测试间隔 TI (h):", self.entry_ti)
        form4.addRow("平均修复 MRT (h):", self.entry_mrt)
        form4.addRow("仿真次数:", self.entry_nsim)
        form4.addRow("仿真年数:", self.entry_years)
        group4.setLayout(form4)
        scroll_layout.addWidget(group4)

        # 中部按钮与进度条（放在滚动区域底部）
        mid_frame = QFrame()
        mid_frame.setStyleSheet("background: transparent;")
        mid_layout = QHBoxLayout(mid_frame)
        self.btn_run = QPushButton("▶ 开始验证")
        self.btn_run.setStyleSheet(btn_style)
        self.btn_run.clicked.connect(self._run_validation)
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid rgba(200,200,200,0.3);
                border-radius: 4px;
                background: rgba(13,31,59,0.8);
                color: #d8e7ff;
                text-align: center;
            }
            QProgressBar::chunk {
                background: rgba(62,109,165,0.6);
                border-radius: 4px;
            }
        """)
        self.progress.setValue(0)
        self.status_label = QLabel("就绪")
        mid_layout.addWidget(self.btn_run)
        mid_layout.addWidget(self.progress)
        mid_layout.addWidget(self.status_label)
        scroll_layout.addWidget(mid_frame)

        scroll_layout.addStretch()

        # 设置滚动内容
        scroll_area.setWidget(scroll_content)
        left_panel_layout.addWidget(scroll_area)

        # 右侧结果区
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: rgba(13, 31, 59, 0.7); border: none;")
        right_layout = QVBoxLayout(right_panel)

        # 文本结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(200)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background: rgba(13, 31, 59, 0.8);
                border: 1px solid rgba(200,200,200,0.2);
                border-radius: 4px;
                color: #d8e7ff;
            }
        """)
        right_layout.addWidget(self.result_text)

        # 图表：白色背景绘图区
        self.fig = Figure(figsize=(5,3))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white')
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')
        self.ax.tick_params(colors='black')
        self.ax.title.set_color('black')
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        # 初始化部分共因输入框
        self._update_partial_beta_inputs()

    # ---------- 以下方法与之前相同，但调整了部分逻辑 ----------
    def _toggle_lambda_source(self):
        if self.lambda_source == "direct":
            self.lambda_source = "estimate"
            self.direct_frame.hide()
            self.est_frame.show()
            self.sender().setText("切换到 直接输入")
        else:
            self.lambda_source = "direct"
            self.direct_frame.show()
            self.est_frame.hide()
            self.sender().setText("切换到 运行数据估计")

    def _estimate_lambda(self):
        try:
            T = float(self.entry_T.text())
            k = float(self.entry_k.text())
        except:
            QMessageBox.warning(self, "输入错误", "运行时间和失效次数必须为有效数字")
            return
        if T <= 0 or k < 0:
            QMessageBox.warning(self, "参数错误", "运行时间>0，失效次数>=0")
            return
        estimator = BayesianLambdaEstimator(T, k, confidence=0.95)
        median, low, high = estimator.get_fit()
        self.lambda_est_label.setText(f"λ = {median:.2f} FIT  [95% CI: {low:.2f}, {high:.2f}]")
        self.direct_lam.setText(f"{median:.6f}")

    def _import_sample_data(self):
        try:
            low_th = float(self.entry_low.text())
            high_th = float(self.entry_high.text())
        except:
            QMessageBox.warning(self, "输入错误", "阈值必须为有效数字")
            return
        if low_th >= high_th:
            QMessageBox.warning(self, "参数错误", "低限阈值必须小于高限阈值")
            return

        M_DEVICES = 10
        YEARS = 10
        T_SINGLE = YEARS * HOURS_PER_YEAR
        DT = 12
        LAMBDA_TRUE = 30000 * 1e-9

        def soft_prob_low(level):
            if level <= 20.0 or level >= 35.0:
                return 0.0
            k = 0.7
            p = 1.0 / (1.0 + np.exp(k * (level - 27.5)))
            return np.clip(p, 0.0, 1.0)

        def soft_prob_high(level):
            if level <= 65.0 or level >= 80.0:
                return 0.0
            k = 0.7
            p = 1.0 / (1.0 + np.exp(-k * (level - 72.5)))
            return np.clip(p, 0.0, 1.0)

        REPAIR_MEAN = 5*24
        REPAIR_STD = 2*24
        REPAIR_MIN = 2*24
        REPAIR_MAX = 10*24
        def generate_repair_time():
            rt = np.random.normal(REPAIR_MEAN, REPAIR_STD)
            return np.clip(rt, REPAIR_MIN, REPAIR_MAX)

        def generate_single_device_stats(device_id):
            np.random.seed(42 + device_id)
            N = int(T_SINGLE / DT)
            t = np.arange(0, T_SINGLE, DT)
            mu_device = np.random.normal(50, 5)
            sigma_noise = np.random.gamma(2, 0.5)
            diurnal = 5*np.sin(2*np.pi*t/24)
            annual = 3*np.sin(2*np.pi*t/8760)
            drift = 0.1*t/T_SINGLE
            base_level = mu_device + diurnal + annual + drift
            n_expected = LAMBDA_TRUE * T_SINGLE
            n_failure_events = np.random.poisson(n_expected)
            faults = []
            if n_failure_events > 0:
                failure_times = np.sort(np.random.uniform(0, T_SINGLE, n_failure_events))
                failure_durations = np.random.exponential(24, n_failure_events)
                failure_types = np.random.choice(['low','high'], n_failure_events, p=[0.5,0.5])
                for ft, fd, ftype in zip(failure_times, failure_durations, failure_types):
                    start_idx = int(np.searchsorted(t, ft))
                    end_idx = int(np.searchsorted(t, ft+fd))
                    if end_idx > N: end_idx = N
                    repair_duration = generate_repair_time()
                    repair_end_idx = int(np.searchsorted(t, ft+fd+repair_duration))
                    if repair_end_idx > N: repair_end_idx = N
                    failure_level = (low_th-10) if ftype=='low' else (high_th+10)
                    faults.append({'start':start_idx, 'repair_end':repair_end_idx, 'type':ftype, 'level':failure_level})
            hard_count = len(faults)
            soft_equiv = 0.0
            for fault in faults:
                level = fault['level']
                p = soft_prob_low(level) if fault['type']=='low' else soft_prob_high(level)
                soft_equiv += (1.0 + 0.8*max(p,0.0))
            return hard_count, soft_equiv

        total_hard = 0
        total_soft = 0.0
        for dev_id in range(M_DEVICES):
            h, s = generate_single_device_stats(dev_id)
            total_hard += h
            total_soft += s
        k_eff = total_hard + total_soft
        T_total = M_DEVICES * T_SINGLE

        self.entry_T.setText(f"{T_total:.0f}")
        self.entry_k.setText(f"{k_eff:.4f}")
        self._estimate_lambda()

    def _on_ccf_mode_changed(self):
        mode = self.ccf_mode_combo.currentText()
        if "全局" in mode:
            self.total_beta_frame.show()
            self.partial_beta_frame.hide()
        else:
            self.total_beta_frame.hide()
            self.partial_beta_frame.show()
        self._update_partial_beta_inputs()

    def _update_partial_beta_inputs(self):
        """根据N值动态生成β2~βN输入框"""
        try:
            N = int(self.entry_n.text())
        except:
            N = 4
        # 清除原有布局
        while self.partial_beta_layout.count():
            item = self.partial_beta_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        # 为 k=2..N 创建输入框
        self.partial_beta_entries = {}
        for k in range(2, N+1):
            label = QLabel(f"β{k} (影响{k}个通道):")
            entry = QLineEdit()
            entry.setStyleSheet("""
                QLineEdit {
                    background: rgba(13, 31, 59, 0.8);
                    border: 1px solid rgba(200, 200, 200, 0.3);
                    border-radius: 4px;
                    padding: 4px;
                    color: #d8e7ff;
                }
                QLineEdit:focus {
                    border-color: rgba(123, 176, 247, 0.8);
                }
            """)
            default_val = 0.1 / (N-1) if N > 1 else 0.0
            entry.setText(f"{default_val:.4f}")
            self.partial_beta_entries[k] = entry
            self.partial_beta_layout.addRow(label, entry)

    def _get_beta_list(self) -> List[float]:
        mode = self.ccf_mode_combo.currentText()
        try:
            N = int(self.entry_n.text())
        except:
            N = 4

        if "全局" in mode:
            beta = float(self.entry_total_beta.text())
            if N > 1:
                beta_each = beta / (N - 1)
                return [beta_each] * (N - 1)
            else:
                return []
        else:
            beta_list = []
            for k in range(2, N+1):
                entry = self.partial_beta_entries.get(k)
                if entry:
                    val = float(entry.text())
                    beta_list.append(val)
                else:
                    beta_list.append(0.0)
            return beta_list

    def _run_validation(self):
        try:
            M = int(self.entry_m.text())
            N = int(self.entry_n.text())
            lam_fit = float(self.direct_lam.text())
            TI = float(self.entry_ti.text())
            MRT = float(self.entry_mrt.text())
            nsim = int(self.entry_nsim.text())
            years = int(self.entry_years.text())
            beta_list = self._get_beta_list()
        except Exception as e:
            QMessageBox.warning(self, "输入错误", f"请检查参数格式: {e}")
            return

        if M > N:
            QMessageBox.warning(self, "参数错误", "M 不能大于 N")
            return

        lam = lam_fit * 1e-9
        params = SimParams(
            N=N, M=M,
            TI=TI, MRT=MRT,
            LAMBDA_DU=lam,
            BETA1=1.0 - sum(beta_list),
            beta_list=beta_list,
            SIM_YEARS=years,
            WARMUP_YEARS=max(1000, years//10),
            NUM_SIM=nsim
        )

        self.status_label.setText("仿真运行中...")
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)

        results = []
        total = nsim
        for i in range(total):
            sim = GSPN_MooN_MGL(params)
            pfd = sim.simulate()
            results.append(pfd)
            if (i+1) % max(1, total//20) == 0 or i == total-1:
                self.progress.setValue(int((i+1)/total * 100))
                self.repaint()

        self.btn_run.setEnabled(True)
        results = np.array(results)
        mean_pfd = np.mean(results)
        std_pfd = np.std(results, ddof=1) if nsim > 1 else 0.0
        ci_low = np.percentile(results, 2.5)
        ci_high = np.percentile(results, 97.5)
        sil = sil_from_pfd(mean_pfd)

        self.result_text.clear()
        self.result_text.append("="*60)
        self.result_text.append(f"  表决架构: {M}oo{N}")
        self.result_text.append(f"  失效率 λ = {lam*1e9:.2f} FIT")
        self.result_text.append(f"  测试间隔 TI = {TI:.0f} h")
        self.result_text.append(f"  平均修复时间 MRT = {MRT:.0f} h")
        self.result_text.append(f"  共因模式: {self.ccf_mode_combo.currentText()}")
        if "全局" in self.ccf_mode_combo.currentText():
            self.result_text.append(f"  共因因子 β = {float(self.entry_total_beta.text()):.3f}")
        else:
            beta_str = ", ".join([f"β{k}={self.partial_beta_entries[k].text()}" for k in sorted(self.partial_beta_entries.keys())])
            self.result_text.append(f"  部分共因: {beta_str}")
        self.result_text.append(f"  仿真次数 = {nsim}")
        self.result_text.append(f"  仿真年数 = {years}")
        self.result_text.append("="*60 + "\n")
        self.result_text.append("【仿真结果】\n")
        self.result_text.append(f"  PFDavg = {mean_pfd:.4e}  (标准差 {std_pfd:.4e})\n")
        self.result_text.append(f"  95% 置信区间: [{ci_low:.4e}, {ci_high:.4e}]\n")
        self.result_text.append(f"  SIL 等级 = {sil}\n")

        # 绘制直方图 - 白色背景
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.ax.hist(results, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        if mean_pfd > 0:
            self.ax.axvline(mean_pfd, color='red', linestyle='--', label=f'均值 = {mean_pfd:.2e}')
        self.ax.set_xlabel('PFDavg', color='black')
        self.ax.set_ylabel('概率密度', color='black')
        self.ax.tick_params(colors='black')
        self.ax.legend(facecolor='white', edgecolor='black')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_label.setText("仿真完成")
        self.progress.setValue(100)

    def _load_example_data(self):
        self.entry_m.setText("2")
        self.entry_n.setText("4")
        self.direct_lam.setText("111.11")
        self.entry_ti.setText("8760")
        self.entry_mrt.setText("8")
        self.entry_nsim.setText("500")
        self.entry_years.setText("10000")
        self.entry_T.setText("876000")
        self.entry_k.setText("5")
        self.entry_low.setText("20")
        self.entry_high.setText("80")
        self.entry_total_beta.setText("0.1")
        self.ccf_mode_combo.setCurrentIndex(0)
        self.lambda_est_label.setText("")
        self.result_text.clear()
        self.result_text.append("示例参数已加载 (2oo4, λ=111.11 FIT, 全局β=0.1)")
        self._update_partial_beta_inputs()
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.canvas.draw()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle("SIL 验证测试")
    layout = QVBoxLayout(win)
    layout.addWidget(SILValidationWidget())
    win.resize(1100, 750)
    win.show()
    sys.exit(app.exec())