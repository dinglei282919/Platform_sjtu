"""Desktop adapter for the shared 重放攻击 simulator; constructed lazily by the menu."""
from pathlib import Path
import shutil
from threading import Event
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QDoubleSpinBox, QTabWidget,
                               QFrame, QFileDialog, QMessageBox, QLayout)
from dnn_mpc_plot_view import DnnMpcPlotView, ControlPlotRefresh, ControlRmseMetrics, ControlRunLog, ControlScrollArea, ControlTaskStatus, control_page_style
from platform_core.replay.config import PLOT_GROUPS, defaults

PLOT_DESCRIPTIONS = {
    "tracking": "参考轨迹、系统输出与跟踪误差",
    "disturbance": "失配扰动与估计误差；绿色阴影为恢复窗口，虚线为 1.8627 秒恢复时限",
    "state_control": "系统状态与控制输入；阴影表示重放时段",
    "switching_replay": "上方分别显示两路切换函数，下方对比实时状态与重放信号",
}


class ReplayWorker(QObject):
    succeeded = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, str)
    logged = Signal(str)
    done = Signal()

    def __init__(self, config, run_dir, cancel_event):
        super().__init__()
        self.config, self.run_dir, self.cancel_event = config, run_dir, cancel_event

    @Slot()
    def run(self):
        try:
            from platform_core.replay.runner import run
            result = run(self.config, self.logged.emit, run_dir=self.run_dir, progress=self.progress.emit,
                         cancelled=self.cancel_event.is_set)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class ReplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("replayPage")
        self._thread = None
        self._worker = None
        self._cancel = Event()
        self._run_dir = None
        self._result = None
        self.views = {}
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(18)
        controls = QFrame()
        controls.setObjectName("replayControls")
        controls.setProperty("controlPanel", True)
        left = QVBoxLayout(controls)
        left.setContentsMargins(20, 20, 20, 20)
        left.setSpacing(12)
        heading = QLabel("仿真设置")
        heading.setObjectName("replayHeading")
        heading.setProperty("controlHeading", True)
        left.addWidget(heading)
        left.addWidget(QLabel("仿真时长（秒）"))
        self.duration = QDoubleSpinBox()
        self.duration.setAccessibleName("重放攻击仿真时长")
        self.duration.setRange(.1, 50.)
        self.duration.setDecimals(1)
        self.duration.setSingleStep(.1)
        self.duration.setValue(defaults()["sim_time"])
        left.addWidget(self.duration)
        self.run_button = QPushButton("开始重放攻击仿真")
        self.run_button.setProperty("controlPrimary", True)
        self.run_button.clicked.connect(self.start)
        left.addWidget(self.run_button)
        self.cancel_button = QPushButton("取消仿真")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        left.addWidget(self.cancel_button)
        description = QLabel("初始状态\nx₁ = 0，x₂ = 0\n\n参考轨迹\nyᵈ = sin(0.5t) + 0.5sin(t)\n\n重放时段 / 回放延迟（秒）\n5–9 / 4.8\n12–14 / 2.5\n18–21 / 3.5\n27–32 / 5.5\n36–39 / 3.7\n42–45 / 3.0\n\n橙色阴影表示重放攻击时段；扰动误差图的绿色阴影表示恢复窗口。")
        description.setWordWrap(True)
        left.addWidget(description)
        self.metrics = ControlRmseMetrics(columns=2)
        left.addWidget(self.metrics)
        self.export_button = QPushButton("导出四组图与数据")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        left.addWidget(self.export_button)
        left.addStretch()
        self._config_scroll = ControlScrollArea(controls, width=340)
        root.addWidget(self._config_scroll)
        right = QVBoxLayout()
        output = QFrame()
        output.setObjectName("replayOutput")
        output.setProperty("controlPanel", True)
        output_layout = QVBoxLayout(output)
        output_layout.setContentsMargins(20, 20, 20, 20)
        output_heading = QLabel("结果图示")
        output_heading.setObjectName("replayHeading")
        output_heading.setProperty("controlHeading", True)
        output_layout.addWidget(output_heading)
        self.task_status = ControlTaskStatus("重放攻击仿真")
        self.status = self.task_status.message
        self.progress = self.task_status.progress
        output_layout.addWidget(self.task_status)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setObjectName("replayPlotTabs")
        self.plot_tabs.setAccessibleName("选择输出图像")
        self.plot_tabs.tabBar().setExpanding(True)
        for name, title in PLOT_GROUPS:
            card = QFrame()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 14, 0, 0)
            plot_heading = QLabel(title)
            plot_heading.setObjectName("replayPlotHeading")
            plot_heading.setProperty("controlPlotHeading", True)
            card_layout.addWidget(plot_heading)
            description = QLabel(PLOT_DESCRIPTIONS[name])
            description.setProperty("controlHint", True)
            description.setWordWrap(True)
            card_layout.addWidget(description)
            view = DnnMpcPlotView(aspect_ratio=8.6 / 6.2)
            view.set_placeholder("运行仿真后显示结果图")
            view.setAccessibleName(title + "曲线")
            card_layout.addWidget(view, 1)
            self.views[name] = view
            self.plot_tabs.addTab(card, title)
        output_layout.addWidget(self.plot_tabs, 1)
        right.addWidget(output)
        self.logs = ControlRunLog()
        right.addWidget(self.logs)
        right.addStretch()
        content = QWidget()
        content.setAttribute(Qt.WA_StyledBackground, True)
        content.setProperty("controlSurface", True)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSizeConstraint(QLayout.SetMinimumSize)
        content.setLayout(right)
        self._result_scroll = ControlScrollArea(content)
        root.addWidget(self._result_scroll, 1)
        self.setStyleSheet(control_page_style("replayPage"))
        self._plot_refresh = ControlPlotRefresh(self.views, self)

    @Slot()
    def start(self):
        if self.has_running_worker():
            return
        from platform_core.replay.runner import new_run_dir
        try:
            self._run_dir = new_run_dir()
        except OSError as exc:
            self.task_status.set_status("failed", f"无法创建结果目录：{exc}", 0)
            return
        self._result = None
        self.logs.clear()
        self.logs.append("任务开始执行。")
        self.metrics.clear()
        for view in self.views.values():
            view.clear()
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.duration.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.task_status.set_status("running", "正在初始化仿真", 0)
        self._cancel = Event()
        self._thread = QThread(self)
        self._worker = ReplayWorker({"sim_time": self.duration.value()}, self._run_dir, self._cancel)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.logged.connect(self.logs.append)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.done.connect(self._thread.quit)
        self._worker.done.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._plot_refresh.start(self._run_dir)
        self._thread.start()

    @Slot()
    def cancel(self):
        self._plot_refresh.stop()
        self._cancel.set()
        self.cancel_button.setEnabled(False)
        self.task_status.set_status("running", "正在取消仿真…")

    @Slot(int, str)
    def _on_progress(self, percent, message):
        if not self._cancel.is_set():
            self.task_status.set_status("running", message, percent)

    @Slot(dict)
    def _on_success(self, result):
        self._plot_refresh.stop()
        try:
            for name, view in self.views.items():
                view.load(str(self._run_dir / f"{name}.svg"))
        except Exception as exc:
            self._on_failure("结果图加载失败：" + str(exc))
            return
        self._result = result
        self.task_status.set_status("succeeded", f"仿真完成 · {result['config']['sim_time']:g} 秒场景 · 用时 {result['elapsed_seconds']:.2f} 秒")
        self.metrics.set_values(result)
        self.export_button.setEnabled(True)
        self.logs.append("任务完成。")

    @Slot(str)
    def _on_failure(self, message):
        self._plot_refresh.stop()
        self.task_status.set_status("cancelled" if self._cancel.is_set() else "failed",
                                    message if self._cancel.is_set() else "执行失败：" + message)
        self.logs.append("仿真已取消。" if self._cancel.is_set() else "执行失败：" + message)

    @Slot()
    def _on_finished(self):
        self._plot_refresh.stop()
        self._worker = None
        self._thread = None
        self.run_button.setEnabled(True)
        self.duration.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def has_running_worker(self):
        return self._thread is not None and self._thread.isRunning()

    @Slot()
    def export(self):
        if not self._result:
            return
        selected = QFileDialog.getExistingDirectory(self, "选择 重放攻击结果导出目录")
        if not selected:
            return
        from platform_core.replay.runner import ARTIFACT_NAMES, artifact_path
        from uuid import uuid4
        target = Path(selected) / ("replay-" + uuid4().hex[:10])
        try:
            target.mkdir(exist_ok=False)
            for name in ARTIFACT_NAMES:
                shutil.copy2(artifact_path(self._run_dir, name), target / name)
            self.task_status.set_status("succeeded", "结果已导出到 " + str(target))
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def closeEvent(self, event):
        if self.has_running_worker():
            QMessageBox.warning(self, "后台任务运行中", "请等待 重放攻击仿真完成，或取消后再关闭。")
            event.ignore()
        else:
            super().closeEvent(event)
