"""Responsive reactor console backed by a persistent closed-loop worker."""
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QBoxLayout, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from desktop.risk_control.dnn_mpc_plot_view import (
    ControlComboBox, ControlDoubleSpinBox, ControlScrollArea, control_page_style,
)


def _runtime_factory():
    from platform_core.risk_control.live_runtime import LiveControlRuntime
    from platform_core.risk_control.reactor.live import ReactorLiveSession
    return LiveControlRuntime(ReactorLiveSession)


class ReactorLiveWidget(QWidget):
    """The UI only reads cached snapshots; numerical integration stays off-thread."""

    def __init__(self, runtime_factory=None):
        super().__init__()
        from desktop.risk_control.reactor_page import Diagram
        self.setObjectName("reactorLivePage")
        self.setFont(QFont("Microsoft YaHei", 10))
        self._runtime_factory = runtime_factory or _runtime_factory
        self.runtime = None
        self._active = False
        self._manual_paused = False
        self._closed = False
        self._last_plot = None
        self._event_text = ""
        self.latest = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 14)
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(12)
        sidebar = QFrame()
        sidebar.setProperty("controlPanel", True)
        controls = QVBoxLayout(sidebar)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(8)
        self.control_grid = QGridLayout()
        self.control_grid.setHorizontalSpacing(18)
        self.control_fields = []
        controls.addLayout(self.control_grid)
        self.inputs = {}
        self.apply_buttons = {}
        for key, label, default, button_text in (
            ("r1", "循环参数 R1", .3, "应用并复位"),
            ("r2", "循环参数 R2", .2, "应用并复位"),
        ):
            field = QWidget()
            form = QVBoxLayout(field)
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(5)
            form.addWidget(QLabel(label))
            row = QHBoxLayout()
            box = ControlDoubleSpinBox()
            box.setRange(0., 1. if key == "r1" else .95)
            box.setSingleStep(.05)
            box.setDecimals(3)
            box.setValue(default)
            box.setAccessibleName(label)
            box.setToolTip("R1 范围 [0,1]；R2 范围 [0,0.95]。应用后复位实验，历史曲线不混合不同参数。")
            self.inputs[key] = box
            row.addWidget(box, 1)
            button = QPushButton(button_text)
            button.clicked.connect(lambda checked=False, name=key: self.send_command(name, self.inputs[name].value()))
            self.apply_buttons[key] = button
            row.addWidget(button)
            form.addLayout(row)
            self.control_fields.append(field)
        field = QWidget()
        form = QVBoxLayout(field)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(5)
        form.addWidget(QLabel("系统模式"))
        row = QHBoxLayout()
        self.mode = ControlComboBox()
        self.mode.addItem("MATLAB 自动切换", 0)
        self.mode.addItem("固定模式 1", 1)
        self.mode.addItem("固定模式 2", 2)
        self.mode.setCurrentIndex(0)
        self.mode.setAccessibleName("系统模式")
        row.addWidget(self.mode, 1)
        self.mode_button = QPushButton("应用模式")
        self.mode_button.clicked.connect(lambda: self.send_command("mode", self.mode.currentData()))
        row.addWidget(self.mode_button)
        form.addLayout(row)
        self.control_fields.append(field)
        actions = QHBoxLayout()
        self.enabled = QCheckBox("控制器投入")
        self.enabled.setChecked(True)
        self.enabled.clicked.connect(lambda value: self.send_command("enabled", bool(value)))
        actions.addWidget(self.enabled)
        self.pause_button = QPushButton("暂停运行")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.reset_button = QPushButton("按当前设置重跑")
        self.reset_button.clicked.connect(self.reset)
        actions.addWidget(self.pause_button)
        actions.addWidget(self.reset_button)
        self.applied = QLabel("正在建立在线会话…")
        self.applied.setWordWrap(True)
        self.applied.setProperty("controlHint", True)
        actions.addWidget(self.applied, 1)
        controls.addLayout(actions)
        self.range_hint = QLabel("R1/R2 改变对象方程；应用后保持组分初态并重新开始实验。FDI 始终按论文的乘性反馈形式注入。")
        self.range_hint.setWordWrap(True)
        self.range_hint.setProperty("controlHint", True)
        controls.addWidget(self.range_hint)
        controls.addWidget(QLabel("FDI 攻击调节（在线生效，不复位状态）"))
        self.attack_inputs = {}
        attack_grid = QGridLayout()
        for column, (key, label, low, high, default, step) in enumerate((
            ("scale", "攻击强度 a", 0., 1.8, 1., .1),
            ("frequency", "角频率 ω / rad·s⁻¹", .01, 2., .25, .05),
            ("phase", "相位 φ / rad", -3.14159265, 3.14159265, 0., .1),
        )):
            box = ControlDoubleSpinBox(); box.setDecimals(8 if key == "phase" else 3)
            box.setRange(low, high); box.setSingleStep(step); box.setValue(default); box.setAccessibleName(label)
            self.attack_inputs[key] = box
            attack_grid.addWidget(QLabel(label), 0, column); attack_grid.addWidget(box, 1, column)
        controls.addLayout(attack_grid)
        attack_actions = QHBoxLayout()
        self.attack_apply = QPushButton("在线应用 FDI")
        self.attack_apply.clicked.connect(lambda: self.apply_attack())
        attack_actions.addWidget(self.attack_apply)
        self.attack_toggle = QCheckBox("FDI 攻击开启"); self.attack_toggle.setChecked(True)
        self.attack_toggle.clicked.connect(lambda value: self.send_command("attack_enabled", bool(value)))
        attack_actions.addWidget(self.attack_toggle)
        for label, preset in (("参数复位", dict(scale=1., frequency=.25, phase=0.)),
                              ("增强参数", dict(scale=1.8, frequency=.5, phase=0.))):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, values=preset: self.apply_attack(values))
            attack_actions.addWidget(button)
        controls.addLayout(attack_actions)
        self.attack_status = QLabel("默认论文攻击：a=1，ω=0.25，φ=0。")
        self.attack_status.setWordWrap(True); self.attack_status.setProperty("controlHint", True)
        controls.addWidget(self.attack_status)
        self._attack_initialized = False
        page.addWidget(sidebar)

        output = QFrame()
        output.setProperty("controlPanel", True)
        view = QVBoxLayout(output)
        view.setContentsMargins(12, 10, 12, 10)
        view.setSpacing(8)
        self.status = QLabel("正在建立在线会话…")
        self.status.setWordWrap(True)
        self.status.setProperty("controlSubheading", True)
        view.addWidget(self.status)
        objective = QLabel("控制目标：期望组分 <i>x</i><sub>s</sub>、<i>x</i><sub>d</sub> 趋于 0。")
        objective.setWordWrap(True)
        objective.setProperty("controlHint", True)
        view.addWidget(objective)
        self.monitor = QBoxLayout(QBoxLayout.LeftToRight)
        self.monitor.setSpacing(12)
        view.addLayout(self.monitor)
        self.diagram = Diagram()
        self.monitor.addWidget(self.diagram, 8, Qt.AlignTop)
        from platform_core.risk_control.reactor.coordinates import FDI_DESCRIPTION
        description = QLabel(FDI_DESCRIPTION + "\n在线 T=0.6、ell=0.9 支持持续交互；离线保留 MATLAB 的 T=1.5、ell=0.15。调节目标为组分零点邻域。乘性攻击对严格为零的反馈无额外激励；可按当前设置重跑比较初始恢复过程。控制输入 u 由右侧曲线展示。")
        description.setWordWrap(True)
        description.setProperty("controlHint", True)

        self.figure = Figure(figsize=(10, 3.2), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setAccessibleName("反应器实时控制曲线")
        self.canvas.setMinimumHeight(360)
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.plot_axes = self.figure.subplots(4, 1, sharex=True)
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        self.plot_font = (FontProperties(fname=str(font_path), size=9)
                          if font_path.is_file() else FontProperties(family="DejaVu Sans", size=9))
        self.plot_lines = []
        self.event_markers = []
        for ax, specs in zip(self.plot_axes, ((('xs',r'$x_s$'),('observer_s',r'$\hat{x}_s$'),('received_s',r'$\kappa x_s$')),
                                               (('xd',r'$x_d$'),('observer_d',r'$\hat{x}_d$'),('received_d',r'$\kappa\hat{x}_d$')),
                                               (('u','控制输入 u'),), (('k','FDI 增益 κ'),))):
            for i, (key, label) in enumerate(specs):
                line, = ax.plot([], [], linestyle='--' if i else '-', label=label, linewidth=1.3)
                self.plot_lines.append((key, line))
            ax.legend(loc='upper right', prop=self.plot_font, ncol=3)
            ax.grid(alpha=.2)
        self.plot_axes[-1].set_xlabel("仿真时间 / s", fontproperties=self.plot_font)
        self.canvas.setMinimumHeight(620)
        self.monitor.addWidget(self.canvas, 5)
        info_toggle = QPushButton("模型与 FDI 注入说明")
        info_toggle.setCheckable(True)
        info_toggle.setChecked(True)
        info_toggle.toggled.connect(description.setVisible)
        view.addWidget(info_toggle)
        view.addWidget(description)
        self.events_toggle = QPushButton("展开调节记录")
        self.events_toggle.setCheckable(True)
        view.addWidget(self.events_toggle, 0, Qt.AlignLeft)
        self.events = QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setAccessibleName("调节记录")
        self.events.setMaximumHeight(110)
        self.events.setVisible(False)
        self.events_toggle.toggled.connect(self.events.setVisible)
        self.events_toggle.toggled.connect(lambda checked: self.events_toggle.setText("收起调节记录" if checked else "展开调节记录"))
        view.addWidget(self.events)
        page.addWidget(output)
        page.addStretch()
        layout.addWidget(ControlScrollArea(content))
        self._compact_layout = None
        self._arrange_monitor()
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh)
        self.setStyleSheet(control_page_style("reactorLivePage") +
                          '\nQWidget#reactorLivePage, QWidget#reactorLivePage QWidget { font-family: "Microsoft YaHei"; }'
                          '\nQWidget#reactorLivePage QCheckBox { color: #d9ecff; font-size: 14px; spacing: 8px; }'
                          '\nQWidget#reactorLivePage QComboBox { background: #0d2035; color: #d9ecff; border: 1px solid #365477; border-radius: 6px; padding: 5px 8px; min-height: 24px; font-size: 14px; }'
                          '\nQWidget#reactorLivePage QComboBox QAbstractItemView { background: #102640; color: #d9ecff; selection-background-color: #31678f; }')
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def _ensure_runtime(self):
        if self.runtime is None and not self._closed:
            self.runtime = self._runtime_factory()
            self.runtime.start()

    def snapshot_status(self):
        return self.runtime.snapshot().get("status", "paused") if self.runtime else "paused"

    def set_active(self, active):
        if self._closed:
            return
        active = bool(active)
        was_active = self._active
        self._active = active
        if active:
            self._ensure_runtime()
            if not was_active and not self._manual_paused:
                self.runtime.command("resume")
            self.timer.start()
            self.refresh()
        else:
            if self.runtime is not None and was_active:
                self.runtime.command("pause")
            self.timer.stop()
            self.diagram.set_animated(False)

    def send_command(self, kind, value=None):
        if self.runtime is None or self._closed:
            return
        try:
            self.runtime.command(kind, value)
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("调节指令已提交，等待计算步确认…")

    def apply_attack(self, preset=None):
        from platform_core.risk_control.reactor.attack import parse_attack
        try:
            values = parse_attack(preset if preset is not None else {k: box.value() for k, box in self.attack_inputs.items()})
        except ValueError as exc:
            self.status.setText(str(exc)); return
        if preset is not None:
            for key, value in values.items(): self.attack_inputs[key].setValue(value)
        self.send_command("attack", values)

    def toggle_pause(self):
        self._manual_paused = not self._manual_paused
        self.send_command("pause" if self._manual_paused else "resume")
        self.pause_button.setText("继续运行" if self._manual_paused else "暂停运行")

    def reset(self):
        self._manual_paused = False
        self.mode.setCurrentIndex(0)
        self.enabled.setChecked(True)
        self.send_command("reset")
        self.send_command("resume")
        self.pause_button.setText("暂停运行")

    def refresh(self):
        if self.runtime is None or self._closed:
            return
        snapshot = self.runtime.snapshot()
        self.latest = snapshot
        status = snapshot.get("status", "running")
        current = snapshot.get("controls", {})
        t = float(snapshot.get("t", 0.))
        mode = current.get("mode", 1)
        enabled = current.get("enabled", True)
        names = {"running": "正常运行" if enabled else "开环运行（控制器退出）",
                 "paused": "已暂停", "error": "运行异常", "closed": "会话已关闭"}
        self.status.setText(f"{names.get(status, status)} · 仿真时间 {t:.2f} s" +
                            (f" · {snapshot['message']}" if snapshot.get("message") else ""))
        self.pause_button.setText("暂停运行" if status == "running" else "继续运行")
        self.pause_button.setEnabled(status != "error")
        self.applied.setText(f"当前 R1：{float(current.get('r1', .3)):.3f}　R2：{float(current.get('r2', .2)):.3f}　"
                             f"系统模式：{'MATLAB 自动' if mode == 0 else mode}　控制器：{'投入' if enabled else '退出'}")
        self.attack_toggle.blockSignals(True)
        self.attack_toggle.setChecked(bool(current.get("attack_enabled", True)))
        self.attack_toggle.blockSignals(False)
        self.attack_status.setText(f"当前 FDI：{'开启' if current.get('attack_enabled', True) else '关闭'}；"
                                   f"a={current.get('attack_scale',1.):.3g}，ω={current.get('attack_frequency',.25):.3g}，φ={current.get('attack_phase',0.):.3g}")
        if not self._attack_initialized:
            for key, box in self.attack_inputs.items(): box.setValue(current.get("attack_"+key, box.value()))
            self._attack_initialized = True
        self.enabled.blockSignals(True)
        self.enabled.setChecked(bool(enabled))
        self.enabled.blockSignals(False)
        self.diagram.set_animated(status == "running" and self._active)
        data = snapshot.get("data", {})
        if len(data.get("t", [])):
            self.diagram.set_live_data(data)
            last = len(data["t"]) - 1
            plot_key = (float(data["t"][last]), len(data["t"]), float(data["xs"][last]),
                        float(data["xd"][last]), float(data["u"][last]), float(data["k"][last]), len(snapshot.get("events",[])))
            if self._last_plot != plot_key:
                self._last_plot = plot_key
                self._update_curves(data)
        self._show_events(snapshot.get("events", []))

    def _update_curves(self, data):
        times = data["t"]
        last_t = float(times[-1])
        first = next((i for i, value in enumerate(times) if value >= last_t - 60), 0)
        x = times[first:]
        for marker in self.event_markers: marker.remove()
        self.event_markers = []
        for event in self.latest.get("events", [])[-8:]:
            if event.get("kind", "").startswith("attack") and event.get("time",-1) >= x[0]:
                for ax in self.plot_axes:
                    self.event_markers.append(ax.axvline(event["time"],color="#d6a033",linestyle=":",linewidth=.8))
        for key, line in self.plot_lines:
            line.set_data(x, data[key][first:])
        for ax in self.plot_axes:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            ax.set_xlim(max(0., last_t - 60), max(10., last_t + .2))
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontproperties(self.plot_font)
        self.canvas.draw_idle()

    def _show_events(self, events):
        names = {"attack": "FDI 参数", "attack_enabled": "FDI 开关", "r1": "R1 应用并复位", "r2": "R2 应用并复位",
                 "mode": "系统模式", "enabled": "控制器投入", "reset": "复位实验",
                 "pause": "暂停运行", "resume": "继续运行"}
        lines = []
        for event in events[-12:]:
            if not isinstance(event, dict):
                lines.append(str(event))
                continue
            t = float(event.get("t", event.get("time", 0.)))
            kind = event.get("kind", "")
            message = event.get("message") or names.get(kind, kind)
            if not event.get("message") and event.get("value") is not None:
                message += f"：{event['value']}"
            lines.append(f"{t:.2f} s · {message}")
        text = "\n".join(lines) or "尚无调节操作。"
        if text != self._event_text:
            self._event_text = text
            self.events.setPlainText(text)
            bar = self.events.verticalScrollBar()
            bar.setValue(bar.maximum())

    def showEvent(self, event):
        super().showEvent(event)
        self.set_active(True)

    def _arrange_monitor(self):
        compact = self.width() < 1050
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        columns = 2 if compact else 4
        for index, field in enumerate(self.control_fields):
            self.control_grid.addWidget(field, index // columns, index % columns)
        for column in range(4):
            self.control_grid.setColumnStretch(column, 1 if column < columns else 0)
        self.monitor.setDirection(QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_compact_layout"):
            self._arrange_monitor()

    def hideEvent(self, event):
        self.set_active(False)
        super().hideEvent(event)

    def shutdown(self):
        self.timer.stop()
        self.diagram.animation.stop()
        if self.runtime is not None and not self.runtime.close(timeout=2):
            self.status.setText("正在停止在线计算，请稍后关闭")
            return False
        self._closed = True
        self._active = False
        return True

    def closeEvent(self, event):
        event.accept() if self.shutdown() else event.ignore()
