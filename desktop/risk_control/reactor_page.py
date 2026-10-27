"""Standalone Qt review interface; all simulation work runs off the UI thread."""
from pathlib import Path
from io import BytesIO
import time

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QByteArray, QRectF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QFileDialog, QMessageBox, QFrame, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from desktop.risk_control.dnn_mpc_plot_view import ControlTabWidget as QTabWidget
from desktop.risk_control.dnn_mpc_plot_view import ControlDoubleSpinBox as QDoubleSpinBox, ControlSlider as QSlider
from desktop.risk_control.dnn_mpc_plot_view import DnnMpcPlotView, ControlScrollArea, ControlTaskStatus, ControlRunLog, control_page_style, format_rmse

from platform_core.risk_control.control_paths import module_output_dir
from platform_core.risk_control.reactor.core import Config, SimulationCancelled, simulate
from platform_core.risk_control.reactor.reporting import export_result, load_result, populate_figure
from platform_core.risk_control.reactor.presentation import build_diagram_series, render_diagram

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = module_output_dir("reactor")


class Worker(QThread):
    progress = Signal(int)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            result = simulate(self.config, lambda p: self.progress.emit(round(p * 100)),
                              self.isInterruptionRequested)
            self.succeeded.emit(result)
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class Diagram(QWidget):
    """Render the shared process diagram at the selected simulation sample."""

    def __init__(self):
        super().__init__()
        # Keep the natural aspect ratio independent of animation frames.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setStyleSheet("background: white; border: none;")
        self.renderer = QSvgRenderer(self)
        self.data = None
        self.series = None
        self.sample_index = 0
        self.frame_index = 0
        self.svg = ""
        self.animation = QTimer(self)
        self.animation.setInterval(50)
        self.animation.setTimerType(Qt.PreciseTimer)
        self.animation.timeout.connect(self._advance_frame)
        self.animated = False
        self.setAccessibleName("化学反应器双路进料工艺框图")
        self._load_frame()

    def set_data(self, data):
        self.data = data
        self.series = build_diagram_series(data) if data is not None else None
        self.sample_index = 0
        self.frame_index = 0
        self._load_frame()

    def set_live_data(self, data):
        """Update the current online sample without restarting the flow phase."""
        self.data = data
        self.series = build_diagram_series(data)
        self.sample_index = max(0, len(data["t"]) - 1)
        self._load_frame()

    def set_sample(self, index):
        self.sample_index = (max(0, min(int(index), len(self.data["t"]) - 1))
                             if self.data is not None else 0)
        self._load_frame()

    def _load_frame(self):
        self.svg = render_diagram(data=self.data, index=self.sample_index,
                                  series=self.series, phase=self.frame_index / 120)
        if not self.renderer.load(QByteArray(self.svg.encode("utf-8"))):
            raise ValueError("反应器矢量框图资源无效")
        self.update()

    def _advance_frame(self):
        # Flow dashes/stirring are decorative. Only the playback slider changes
        # the selected sample and its numerical values, level and valve state.
        self.frame_index = (self.frame_index + 1) % 120
        self._load_frame()

    def set_animated(self, enabled):
        self.animated = bool(enabled)
        self.animation.stop()
        self.rescale()
        if self.animated:
            self.animation.start()

    def rescale(self):
        height = max(1, round(self.width() * 895 / 1600))
        if self.height() != height:
            self.setFixedHeight(height)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        painter.setRenderHint(QPainter.Antialiasing)
        self.renderer.render(painter, QRectF(self.contentsRect()))
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale()


class ReactorExperimentWidget(QWidget):
    busy_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("reactorPage")
        self.result = None
        self.worker = None
        self.setWindowTitle("化学反应器自适应控制仿真")
        self.resize(1320, 900)
        self.setMinimumSize(600, 450)


        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(18)
        sidebar = QFrame()
        sidebar.setProperty("controlPanel", True)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 20, 20, 20)
        side.setSpacing(12)
        heading = QLabel("仿真设置")
        heading.setProperty("controlHeading", True)
        side.addWidget(heading)
        objective = QLabel("控制目标：期望组分 <i>x</i><sub>s</sub>、<i>x</i><sub>d</sub> 趋于 0。")
        objective.setWordWrap(True)
        objective.setProperty("controlHint", True)
        side.addWidget(objective)
        form = QFormLayout()
        self.inputs = {}
        specs = [("duration", "仿真时长 / s", 0.1, 3, 3, 3),
                 ("sample_dt", "输出间隔 / s", 0.0001, 0.1, 0.002, 4),
                 ("prescribed_time", "预设时间 T / s", 0.1, 3, 1.5, 3),
                 ("ell", "增益参数 ell", 0.01, 1, 0.15, 3),
                 ("xs", "组分初态 xs", -4, 4, 1., 3),
                 ("xd", "组分初态 xd", -2, 2, 0.5, 3),
                 ("observer_s", "组分估计初态 xs", -4, 4, 0, 3),
                 ("observer_d", "组分估计初态 xd", -2, 2, 0.5, 3),
                 ("r1", "循环参数 R1", 0, 1, .3, 3),
                 ("r2", "循环参数 R2", 0, .95, .2, 3),
                 ("attack_scale", "FDI 强度 a（0 无攻击）", 0, 1.8, 1., 3),
                 ("attack_frequency", "FDI 角频率 ω / rad·s⁻¹", .01, 2, .25, 3),
                 ("attack_phase", "FDI 相位 φ / rad", -3.141592653589793, 3.141592653589793, 0., 8)]
        for name, label, lower, upper, default, decimals in specs:
            box = QDoubleSpinBox()
            box.setDecimals(decimals)
            box.setRange(lower, upper)
            box.setValue(default)
            box.setSingleStep({"duration": .1, "sample_dt": .001, "prescribed_time": .1, "ell": .01, "attack_scale": .1, "attack_frequency": .05, "attack_phase": .1}.get(name, .05))
            box.setAccessibleName(label)
            self.inputs[name] = box
            form.addRow(label.replace("xs", "<i>x</i><sub>s</sub>").replace("xd", "<i>x</i><sub>d</sub>"), box)
        side.addLayout(form)
        self.run_button = QPushButton("运行仿真")
        self.run_button.setProperty("controlPrimary", True)
        self.run_button.clicked.connect(self.start_run)
        side.addWidget(self.run_button)
        self.cancel_button = QPushButton("取消计算")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_run)
        side.addWidget(self.cancel_button)
        self.reset_button = QPushButton("恢复默认参数")
        self.reset_button.clicked.connect(self.reset)
        side.addWidget(self.reset_button)
        self.export_button = QPushButton("导出本次结果…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        side.addWidget(self.export_button)
        self.load_button = QPushButton("载入已有仿真结果…")
        self.load_button.clicked.connect(self.load)
        side.addWidget(self.load_button)
        self.task_status = ControlTaskStatus("化学反应器仿真")
        self.progress = self.task_status.progress
        self.status = self.task_status.message
        self.summary = QLabel("运行后显示终态与采样控制峰值。")
        self.summary.setWordWrap(True)
        side.addWidget(self.summary)
        from platform_core.risk_control.reactor.coordinates import FDI_DESCRIPTION
        note = QLabel(FDI_DESCRIPTION)
        note.setWordWrap(True)
        note.setProperty("controlHint", True)
        side.addWidget(note)
        side.addStretch()
        scroll = ControlScrollArea(sidebar, width=340)
        layout.addWidget(scroll)
        output = QFrame()
        output.setProperty("controlPanel", True)
        output_layout = QVBoxLayout(output)
        output_layout.setContentsMargins(20, 20, 20, 20)
        output_heading = QLabel("结果图示")
        output_heading.setProperty("controlHeading", True)
        output_layout.addWidget(output_heading)
        output_layout.addWidget(self.task_status)
        self.tabs = QTabWidget()
        output_layout.addWidget(self.tabs, 1)
        self.logs = ControlRunLog()
        output_layout.addWidget(self.logs)
        layout.addWidget(ControlScrollArea(output), 1)
        process = QWidget()
        view = QVBoxLayout(process)
        view.setContentsMargins(0, 0, 0, 0)
        view.setSpacing(12)
        self.diagram = Diagram()
        view.addWidget(self.diagram)
        diagram_note = QLabel("显示反变换后的组分 xs/xd 和控制输入 u；容器液面与冷却管路仅作工艺示意。")
        diagram_note.setWordWrap(True)
        diagram_note.setProperty("controlHint", True)
        view.addWidget(diagram_note)
        row = QHBoxLayout()
        self.animation_button = QPushButton("暂停工艺动画")
        self.animation_button.clicked.connect(self.toggle_animation)
        row.addWidget(self.animation_button)
        self.play_button = QPushButton("回放仿真数据")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)
        row.addWidget(self.play_button)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.show_sample)
        self.slider.sliderPressed.connect(self.pause_playback)
        self.slider.actionTriggered.connect(self.pause_playback)
        row.addWidget(self.slider, 1)
        view.addLayout(row)
        self.readout = QLabel("时间 t：—    切换模式 σ：—    组分 <i>x</i><sub>s</sub>：—    组分 <i>x</i><sub>d</sub>：—    控制输入 u：—")
        self.readout.setWordWrap(True)
        self.readout.setObjectName("resultPathLabel")
        view.addWidget(self.readout)
        view.addStretch(1)
        self.tabs.addTab(process, "工艺框图")
        plots = QWidget()
        plot_layout = QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 14, 0, 0)
        self.figure = Figure(figsize=(13, 10), dpi=150)
        FigureCanvasAgg(self.figure)
        self.plot_view = DnnMpcPlotView(aspect_ratio=13 / 10)
        self.plot_view.set_placeholder("完成仿真后显示曲线")
        self.plot_view.setAccessibleName("化学反应器仿真曲线")
        plot_layout.addWidget(self.plot_view)
        plot_layout.addStretch(1)
        self.tabs.addTab(plots, "仿真曲线")
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.advance)
        self.diagram.set_animated(True)
        self.setStyleSheet(control_page_style("reactorPage"))

    def has_running_worker(self):
        return self.worker is not None

    def reset(self):
        defaults = Config()
        for key, box in self.inputs.items():
            box.setValue(getattr(defaults, key))

    def set_busy(self, busy):
        for box in self.inputs.values():
            box.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.load_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.result is not None)
        self.busy_changed.emit(bool(busy))

    def start_run(self):
        if self.worker is not None:
            return
        c = Config(**{name: box.value() for name, box in self.inputs.items()})
        try:
            c.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self.timer.stop()
        self.logs.clear()
        self.logs.append("化学反应器计算开始。")
        self.resume_animation = self.diagram.animated
        self.diagram.set_animated(False)
        self.animation_button.setText("计算期间暂停工艺动画")
        self.animation_button.setEnabled(False)
        self.play_button.setText("回放仿真数据")
        self.play_button.setEnabled(False)
        self.slider.setEnabled(False)
        self.result = None
        self.diagram.set_data(None)
        self.figure.clear()
        self.plot_view.clear()
        self.readout.setText("计算中，等待本次结果…")
        self.summary.setText("计算中…")
        self.task_status.set_status("running", "计算中 · 工艺动画已暂停，可随时取消", 0)
        self.set_busy(True)
        self.worker = Worker(c, self)
        self.worker.progress.connect(lambda p: self.task_status.set_status("running", self.status.text(), p))
        self.worker.succeeded.connect(self.accept_result)
        self.worker.failed.connect(self.failed)
        self.worker.cancelled.connect(lambda: self.failed("本次计算已取消"))
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def cancel_run(self):
        if self.worker:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("正在取消…")

    def accept_result(self, result):
        self.result = result
        self.diagram.set_data(result["data"])
        populate_figure(self.figure, result)
        preview = BytesIO()
        self.figure.savefig(preview, format="svg")
        self.plot_view.load_svg(preview.getvalue())
        self.slider.setRange(0, len(result["data"]["t"]) - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.show_sample(0)
        m = result["metrics"]
        self.summary.setText(f"终态组分范数 = {format_rmse(m['final_composition_norm'])}\n观测误差 = {format_rmse(m['final_composition_observer_error'])}\n采样控制峰值 |u| = {format_rmse(m['peak_abs_u_sampled'])}\n输出点数 = {m['samples']}")
        elapsed = m.get("elapsed_seconds")
        duration_label = f" · 用时 {elapsed:.1f} s" if elapsed is not None else ""
        self.task_status.set_status("succeeded", f"计算完成{duration_label} · 可检查曲线、回放数据或导出", 100)
        self.logs.append("任务完成。")

    def failed(self, message):
        self.logs.append(message)
        self.task_status.set_status("cancelled" if "取消" in message else "failed", message)
        self.summary.setText("未生成完整结果。可调整参数后重试。")
        self.readout.setText("无本次仿真数据")

    def worker_finished(self):
        worker = self.worker
        self.worker = None
        self.set_busy(False)
        self.diagram.set_animated(self.resume_animation)
        self.animation_button.setText("暂停工艺动画" if self.resume_animation else "播放工艺动画")
        self.animation_button.setEnabled(True)
        worker.deleteLater()

    def show_sample(self, index):
        if self.result:
            d = self.result["data"]
            self.diagram.set_sample(index)
            self.readout.setText(f"时间 t：{d['t'][index]:.3f} s    切换模式 σ：{d['mode'][index]:.0f}    组分 <i>x</i><sub>s</sub>：{d['xs'][index]:.5g}    组分 <i>x</i><sub>d</sub>：{d['xd'][index]:.5g}    控制输入 u：{d['u'][index]:.5g}")

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("回放仿真数据")
        else:
            if self.slider.value() == self.slider.maximum():
                self.slider.setValue(0)
            self.play_start_t = float(self.result["data"]["t"][self.slider.value()])
            self.play_start_clock = time.monotonic()
            self.timer.start()
            self.play_button.setText("暂停数据回放")

    def pause_playback(self, *_):
        self.timer.stop()
        self.play_button.setText("回放仿真数据")

    def advance(self):
        import numpy as np
        t = self.play_start_t + time.monotonic() - self.play_start_clock
        index = min(int(np.searchsorted(self.result["data"]["t"], t)), self.slider.maximum())
        self.slider.setValue(index)
        if index == self.slider.maximum():
            self.timer.stop()
            self.play_button.setText("回放仿真数据")

    def toggle_animation(self):
        enabled = not self.diagram.animated
        self.diagram.set_animated(enabled)
        self.animation_button.setText("暂停工艺动画" if enabled else "播放工艺动画")

    def export(self):
        if not self.result:
            return
        target = QFileDialog.getExistingDirectory(self, "选择导出位置（将在其中新建结果文件夹）", str(OUTPUT if OUTPUT.exists() else ROOT))
        if target:
            from datetime import datetime
            directory = Path(target) / datetime.now().strftime("reactor_%Y%m%d_%H%M%S_%f")
            try:
                export_result(self.result, directory)
                self.status.setText(f"已导出到：{directory}")
            except Exception as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    def load(self):
        target = QFileDialog.getExistingDirectory(self, "选择含 summary.json 与 trajectory.npz 的结果文件夹", str(OUTPUT if OUTPUT.exists() else ROOT))
        if not target:
            return
        try:
            self.load_saved(target)
        except Exception as exc:
            QMessageBox.warning(self,"载入失败",str(exc))

    def load_saved(self, target):
        result = load_result(target)
        self.logs.clear()
        self.logs.append("正在载入已有化学反应器仿真结果。")
        self.timer.stop()
        self.play_button.setText("回放仿真数据")
        for key, box in self.inputs.items():
            box.setValue(result["config"][key])
        self.accept_result(result)
        self.export_button.setEnabled(True)
        self.status.setText(f"已载入结果 · {result['config']['duration']:g} s · {result['config']['method']}\n重算将使用当前默认求解器设置。")
        self.status.setToolTip(str(target))

    def closeEvent(self, event):
        if self.worker:
            self.worker.requestInterruption()
            self.status.setText("正在停止计算，请稍后关闭")
            event.ignore()
            self.worker.finished.connect(self.close)
        else:
            self.timer.stop()
            self.diagram.animation.stop()
            event.accept()


class ReactorWidget(QWidget):
    """Online operation by default, with the original experiment tools retained."""

    def __init__(self, runtime_factory=None):
        super().__init__()
        from desktop.risk_control.reactor_live_page import ReactorLiveWidget
        self.setObjectName("reactorWorkspace")
        self.setWindowTitle("化学反应器自适应控制仿真")
        self.resize(1320, 980)
        self.setMinimumSize(600, 450)
        self.setFont(QFont("Microsoft YaHei", 10))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.live = ReactorLiveWidget(runtime_factory=runtime_factory)
        self.experiment = ReactorExperimentWidget()
        self.tabs.addTab(self.live, "在线运行")
        self.tabs.addTab(self.experiment, "离线实验与导出")
        layout.addWidget(self.tabs)
        self._experiment_animated = True
        self._mode_index = None
        self._closed = False
        self.tabs.currentChanged.connect(self._select_mode)
        self.experiment.busy_changed.connect(self._experiment_busy)
        self.mode_timer = QTimer(self)
        self.mode_timer.setInterval(100)
        self.mode_timer.timeout.connect(self._sync_modes)
        self.setStyleSheet(control_page_style("reactorWorkspace"))
        self._select_mode(0)

    @property
    def diagram(self):
        return self.live.diagram

    def has_running_worker(self):
        return self.experiment.has_running_worker()

    def _experiment_busy(self, busy):
        self.tabs.setTabEnabled(0, not busy)

    def _select_mode(self, index):
        if index == 0 and self.has_running_worker():
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(1)
            self.tabs.blockSignals(False)
            index = 1
        if index == 0:
            if self._mode_index == 1:
                self._experiment_animated = self.experiment.diagram.animated
            self.experiment.pause_playback()
            self.experiment.diagram.set_animated(False)
            self.live.set_active(self.isVisible())
        else:
            self.live.set_active(False)
            self.experiment.diagram.set_animated(self._experiment_animated and self.isVisible())
        self._mode_index = index
        self._sync_modes()

    def _sync_modes(self):
        # Pause is acknowledged by the worker before offline work is enabled.
        ready = self.live.runtime is None or self.live.snapshot_status() != "running"
        self.experiment.setEnabled(self.tabs.currentIndex() != 1 or ready)
        self.tabs.setTabEnabled(0, not self.has_running_worker())

    def showEvent(self, event):
        super().showEvent(event)
        if not self._closed:
            self.mode_timer.start()
            self._select_mode(self.tabs.currentIndex())

    def hideEvent(self, event):
        self.mode_timer.stop()
        self.live.set_active(False)
        self.experiment.pause_playback()
        if self.tabs.currentIndex() == 1:
            self._experiment_animated = self.experiment.diagram.animated
        self.experiment.diagram.set_animated(False)
        super().hideEvent(event)

    def shutdown(self):
        if self.has_running_worker():
            return False
        stopped = self.live.shutdown()
        if stopped:
            self._closed = True
            self.mode_timer.stop()
            self.experiment.timer.stop()
            self.experiment.diagram.animation.stop()
        return stopped

    def closeEvent(self, event):
        if self.has_running_worker():
            self.experiment.cancel_run()
            self.experiment.status.setText("正在停止离线计算，请稍后关闭")
            event.ignore()
            return
        if self.shutdown():
            event.accept()
        else:
            event.ignore()


def create_application():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    font_file = Path("C:/Windows/Fonts/msyh.ttc")
    if font_file.is_file():
        QFontDatabase.addApplicationFont(str(font_file))
        app.setFont(QFont("Microsoft YaHei", 9))
    palette = QPalette()
    for role, color in ((QPalette.Window,"#f4f8fa"), (QPalette.WindowText,"#203746"),
                        (QPalette.Base,"#ffffff"), (QPalette.AlternateBase,"#eaf3f7"),
                        (QPalette.Text,"#203746"), (QPalette.Button,"#e7eff4"),
                        (QPalette.ButtonText,"#203746"), (QPalette.Highlight,"#167c9d"),
                        (QPalette.HighlightedText,"#ffffff")):
        palette.setColor(role,QColor(color))
    app.setPalette(palette)
    return app
