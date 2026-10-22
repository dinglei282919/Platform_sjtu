"""Standalone Qt review interface; all simulation work runs off the UI thread."""
from pathlib import Path
from io import BytesIO
import time

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QColor, QFont, QFontDatabase, QMovie, QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QDoubleSpinBox,
    QTabWidget, QFileDialog, QMessageBox, QSlider, QFrame, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from dnn_mpc_plot_view import DnnMpcPlotView, ControlScrollArea, ControlTaskStatus, control_page_style

from platform_core.control_paths import module_output_dir
from platform_core.reactor.core import Config, SimulationCancelled, simulate
from platform_core.reactor.reporting import export_result, load_result, populate_figure

ROOT = Path(__file__).resolve().parent
DIAGRAM = ROOT / "assets" / "reactor"
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


class Diagram(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 190)
        # Scaled movie frames must not become the parent layout's size hint.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background: white; border: 1px solid #dce6ed; border-radius: 8px")
        self.still = QPixmap(str(DIAGRAM / "reactor_dual_feed.png"))
        self.animation = QMovie(str(DIAGRAM / "reactor_dual_feed.gif"))
        self.animation.setCacheMode(QMovie.CacheNone)
        self.animated = False
        self.setAccessibleName("化学反应器双路进料工艺框图")
        if self.still.isNull() or not self.animation.isValid():
            raise FileNotFoundError(f"Diagram assets missing: {DIAGRAM}")

    def set_animated(self, enabled):
        self.animated = enabled
        if enabled:
            self.setMovie(self.animation)
            # Set the decoding size before the first frame is displayed.
            self.rescale()
            self.animation.start()
        else:
            self.animation.stop()
            self.rescale()

    def rescale(self):
        size = QSize(1600, 895).scaled(self.size() - QSize(16, 16), Qt.KeepAspectRatio)
        if self.animated:
            if self.animation.scaledSize() != size:
                self.animation.setScaledSize(size)
        else:
            self.setPixmap(self.still.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale()


class ReactorWidget(QWidget):
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
        form = QFormLayout()
        self.inputs = {}
        specs = [("duration", "仿真时长 / s", 0.1, 3, 3, 3),
                 ("sample_dt", "输出间隔 / s", 0.0001, 0.1, 0.002, 4),
                 ("prescribed_time", "预设时间 T / s", 0.1, 3, 1.5, 3),
                 ("ell", "增益参数 ell", 0.01, 1, 0.15, 3),
                 ("x1", "初态 x1", -1, 1, 0.5, 3),
                 ("x2", "初态 x2", -1, 1, 0.5, 3),
                 ("observer1", "估计初态 xhat1", -1, 1, 0, 3),
                 ("observer2", "估计初态 xhat2", -1, 1, 0.5, 3)]
        for name, label, lower, upper, default, decimals in specs:
            box = QDoubleSpinBox()
            box.setDecimals(decimals)
            box.setRange(lower, upper)
            box.setValue(default)
            box.setSingleStep(0.001 if name == "sample_dt" else 0.05)
            box.setAccessibleName(label)
            self.inputs[name] = box
            form.addRow(label, box)
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
        note = QLabel("状态与输入沿用原模型符号，未推定物理单位。\n\n框图为双路连续进料示意；液位与流速不随仿真数值变化。")
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
        layout.addWidget(ControlScrollArea(output), 1)
        process = QWidget()
        view = QVBoxLayout(process)
        self.diagram = Diagram()
        view.addWidget(self.diagram, 1)
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
        row.addWidget(self.slider, 1)
        view.addLayout(row)
        self.readout = QLabel("t = —    σ = —    x1 = —    x2 = —    u = —")
        self.readout.setWordWrap(True)
        self.readout.setObjectName("resultPathLabel")
        view.addWidget(self.readout)
        self.tabs.addTab(process, "工艺框图")
        plots = QWidget()
        plot_layout = QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 14, 0, 0)
        self.figure = Figure(figsize=(13, 7), dpi=150)
        FigureCanvasAgg(self.figure)
        self.plot_view = DnnMpcPlotView(aspect_ratio=13 / 7)
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
        self.resume_animation = self.diagram.animated
        self.diagram.set_animated(False)
        self.animation_button.setText("计算期间暂停工艺动画")
        self.animation_button.setEnabled(False)
        self.play_button.setText("回放仿真数据")
        self.play_button.setEnabled(False)
        self.slider.setEnabled(False)
        self.result = None
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
        self.summary.setText(f"终态 ||x|| = {m['final_x_norm']:.3g}\n观测误差 = {m['final_observer_error']:.3g}\n采样 max |u| = {m['peak_abs_u_sampled']:.3g}\n输出点数 = {m['samples']}")
        elapsed = m.get("elapsed_seconds")
        duration_label = f" · 用时 {elapsed:.1f} s" if elapsed is not None else ""
        self.task_status.set_status("succeeded", f"计算完成{duration_label} · 可检查曲线、回放数据或导出", 100)

    def failed(self, message):
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
            self.readout.setText(f"t = {d['t'][index]:.3f} s    σ = {d['mode'][index]:.0f}    x1 = {d['x1'][index]:.5g}    x2 = {d['x2'][index]:.5g}    u = {d['u'][index]:.5g}")

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
