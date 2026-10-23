"""Desktop adapter for the shared 自愈控制 simulator; constructed lazily by the menu."""
from pathlib import Path
import shutil
import html
from threading import Event
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt, QEvent
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton,
                               QFrame, QFileDialog, QMessageBox, QLayout)
from desktop.risk_control.dnn_mpc_plot_view import ControlTabWidget as QTabWidget, ControlTabBar as QTabBar
from desktop.risk_control.dnn_mpc_plot_view import ControlSpinBox as QSpinBox, ControlDoubleSpinBox as QDoubleSpinBox, ControlComboBox as QComboBox
from desktop.risk_control.dnn_mpc_plot_view import DnnMpcPlotView, ControlPlotRefresh, ControlRunLog, ControlScrollArea, ControlTaskStatus, control_page_style, format_rmse
from platform_core.risk_control.self_healing.config import PLOT_GROUPS, SCENARIOS, PARAMETERS, NOTICE, defaults, validate_config

PLOT_DESCRIPTIONS = {
    "outputs": "NORMAL 为正常工况，FAULT 为故障下仅底层控制；SCSC、CCSC 为两种独立自愈对照。竖虚线表示故障起点。",
    "inputs": "四组对照实际施加的输入设定值；修正在下一批生效。",
}


class SelfHealingWorker(QObject):
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
            from platform_core.risk_control.self_healing.runner import run
            result = run(self.config, self.logged.emit, run_dir=self.run_dir, progress=self.progress.emit,
                         cancelled=self.cancel_event.is_set)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class SelfHealingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("self_healingPage")
        self._thread = None
        self._worker = None
        self._cancel = Event()
        self._run_dir = None
        self._result = None
        self.views = {}
        self._scenario_states = {}
        self._active_scenario = next(iter(SCENARIOS))
        container = QVBoxLayout(self)
        container.setContentsMargins(0, 0, 0, 0)
        page = QWidget()
        page.setProperty("controlSurface", True)
        page.setAttribute(Qt.WA_StyledBackground, True)
        root = QVBoxLayout(page)
        root.setSizeConstraint(QLayout.SetMinimumSize)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(18)
        self.scenario_tabs = QTabBar()
        self.scenario_tabs.setDrawBase(False)
        self.scenario_tabs.setAccessibleName("自愈控制场景")
        self.scenario_tabs.setExpanding(True)
        for key, label in SCENARIOS.items():
            index = self.scenario_tabs.addTab(label)
            self.scenario_tabs.setTabData(index, key)
        root.addWidget(self.scenario_tabs)
        body = QHBoxLayout()
        body.setSpacing(18)
        root.addLayout(body)
        controls = QFrame()
        controls.setObjectName("self_healingControls")
        controls.setProperty("controlPanel", True)
        left = QVBoxLayout(controls)
        left.setContentsMargins(20, 20, 20, 20)
        left.setSpacing(12)
        heading = QLabel("仿真设置")
        heading.setObjectName("self_healingHeading")
        heading.setProperty("controlHeading", True)
        left.addWidget(heading)
        self.fields = {}
        for parameter in PARAMETERS:
            key = parameter["key"]
            left.addWidget(QLabel(parameter["label"]))
            if "choices" in parameter:
                field = QComboBox()
                for value, label in parameter["choices"].items():
                    field.addItem(label, value)
            else:
                field = QSpinBox() if parameter.get("integer") else QDoubleSpinBox()
                if isinstance(field, QDoubleSpinBox):
                    field.setDecimals(6)
                field.setRange(parameter["min"], parameter["max"])
                field.setSingleStep(parameter["step"])
            field.setAccessibleName(parameter["label"])
            self.fields[key] = field
            field.installEventFilter(self)
            left.addWidget(field)
        self._reset_config()
        self.run_button = QPushButton("开始自愈控制仿真")
        self.run_button.setProperty("controlPrimary", True)
        self.run_button.clicked.connect(self.start)
        left.addWidget(self.run_button)
        self.cancel_button = QPushButton("取消仿真")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        left.addWidget(self.cancel_button)
        description = QLabel(NOTICE)
        description.setWordWrap(True)
        left.addWidget(description)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setTextFormat(Qt.RichText)
        left.addWidget(self.metrics)
        self.export_button = QPushButton("导出对比图与数据")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        left.addWidget(self.export_button)
        left.addStretch()
        controls.setFixedWidth(380)
        body.addWidget(controls)
        right = QVBoxLayout()
        output = QFrame()
        output.setObjectName("self_healingOutput")
        output.setProperty("controlPanel", True)
        output_layout = QVBoxLayout(output)
        output_layout.setContentsMargins(20, 20, 20, 20)
        output_heading = QLabel("结果图示")
        output_heading.setObjectName("self_healingHeading")
        output_heading.setProperty("controlHeading", True)
        output_layout.addWidget(output_heading)
        self.task_status = ControlTaskStatus("自愈控制仿真")
        self.status = self.task_status.message
        self.progress = self.task_status.progress
        output_layout.addWidget(self.task_status)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setObjectName("self_healingPlotTabs")
        self.plot_tabs.setAccessibleName("选择输出图像")
        self.plot_tabs.tabBar().setExpanding(True)
        for name, title in PLOT_GROUPS:
            card = QFrame()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 14, 0, 0)
            plot_heading = QLabel(title)
            plot_heading.setObjectName("self_healingPlotHeading")
            plot_heading.setProperty("controlPlotHeading", True)
            card_layout.addWidget(plot_heading)
            description = QLabel(PLOT_DESCRIPTIONS[name])
            description.setProperty("controlHint", True)
            description.setWordWrap(True)
            card_layout.addWidget(description)
            view = DnnMpcPlotView(aspect_ratio=11 / 7.5)
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
        body.addWidget(content, 1)
        root.addStretch()
        self._page_scroll = ControlScrollArea(page)
        self._page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        container.addWidget(self._page_scroll)
        self.setStyleSheet(control_page_style("self_healingPage"))
        self._plot_refresh = ControlPlotRefresh(self.views, self)
        self.scenario_tabs.currentChanged.connect(self._change_scenario)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel and watched in self.fields.values():
            delta = event.angleDelta().y()
            if delta:
                bar = self._page_scroll.verticalScrollBar()
                bar.setValue(bar.value() - delta)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _change_scenario(self):
        self._scenario_states[self._active_scenario] = {
            "values": {key: field.currentData() if isinstance(field, QComboBox) else field.value()
                       for key, field in self.fields.items()},
            "result": self._result, "run_dir": self._run_dir,
            "metrics": self.metrics.text(), "logs": self.logs.text.toPlainText(),
            "logs_expanded": self.logs.toggle.isChecked(),
            "state": self.task_status.property("state"), "message": self.status.text(),
            "percent": self.progress.value(), "plot": self.plot_tabs.currentIndex(),
        }
        self._active_scenario = self.scenario_tabs.tabData(self.scenario_tabs.currentIndex())
        self._reset_config()
        self._result = None
        self._run_dir = None
        self.metrics.clear()
        self.logs.clear()
        for view in self.views.values():
            view.clear()
            view.set_placeholder("运行仿真后显示结果图")
        self.export_button.setEnabled(False)
        self.task_status.set_status("idle", "设置参数后开始运行", 0)
        saved = self._scenario_states.get(self._active_scenario)
        if saved is not None:
            for key, field in self.fields.items():
                if isinstance(field, QComboBox):
                    field.setCurrentIndex(field.findData(saved["values"][key]))
                else:
                    field.setValue(saved["values"][key])
            self._result, self._run_dir = saved["result"], saved["run_dir"]
            self.metrics.setText(saved["metrics"])
            if saved["logs"]:
                self.logs.append(saved["logs"])
            self.logs.toggle.setChecked(saved["logs_expanded"])
            self.task_status.set_status(saved["state"], saved["message"], saved["percent"])
            self.plot_tabs.setCurrentIndex(saved["plot"])
            if self._run_dir is not None:
                for name, view in self.views.items():
                    path = self._run_dir / f"{name}.svg"
                    if path.is_file():
                        view.load(str(path))
            self.export_button.setEnabled(self._result is not None)

    def _reset_config(self):
        values = defaults(self.scenario_tabs.tabData(self.scenario_tabs.currentIndex()))
        for key, field in self.fields.items():
            if isinstance(field, QComboBox):
                field.setCurrentIndex(field.findData(values[key]))
            else:
                field.setValue(values[key])

    def config(self):
        values = {key: field.currentData() if isinstance(field, QComboBox) else field.value()
                  for key, field in self.fields.items()}
        return validate_config({"scenario": self.scenario_tabs.tabData(self.scenario_tabs.currentIndex()), **values})

    def _set_controls_enabled(self, enabled):
        self.scenario_tabs.setEnabled(enabled)
        for field in self.fields.values():
            field.setEnabled(enabled)

    @Slot()
    def start(self):
        if self.has_running_worker():
            return
        from platform_core.risk_control.self_healing.runner import new_run_dir
        try:
            config = self.config()
            self._run_dir = new_run_dir()
        except (ValueError, OSError) as exc:
            self.task_status.set_status("failed", str(exc), 0)
            return
        self._result = None
        self.logs.clear()
        self.logs.append("任务开始执行。")
        self.metrics.clear()
        for view in self.views.values():
            view.clear()
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self._set_controls_enabled(False)
        self.cancel_button.setEnabled(True)
        self.task_status.set_status("running", "正在初始化仿真", 0)
        self._cancel = Event()
        self._thread = QThread(self)
        self._worker = SelfHealingWorker(config, self._run_dir, self._cancel)
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
        self.task_status.set_status("succeeded", f"仿真完成 · {SCENARIOS[result['config']['scenario']]} · {result['config']['duration']:g} 模型时间 · 用时 {result['elapsed_seconds']:.2f} 秒")
        methods = ("normal", "fault", "scsc", "ccsc")
        rows = ["<tr><th>输出</th>" + "".join(f"<th>{m.upper()}</th>" for m in methods) + "</tr>"]
        for j, label in enumerate(result["output_names"]):
            rows.append("<tr><td>" + html.escape(label) + "</td>" + "".join(
                f"<td>{format_rmse(result['metrics'][m]['tail_20_percent']['rmse'][j])}</td>" for m in methods) + "</tr>")
        self.metrics.setText("<p>最后 20% 时域 RMSE（各输出自身单位）</p><table cellspacing='5'>" + "".join(rows) + "</table>")
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
        self._set_controls_enabled(True)
        self.cancel_button.setEnabled(False)

    def has_running_worker(self):
        return self._thread is not None and self._thread.isRunning()

    @Slot()
    def export(self):
        if not self._result:
            return
        selected = QFileDialog.getExistingDirectory(self, "选择 自愈控制结果导出目录")
        if not selected:
            return
        from platform_core.risk_control.self_healing.runner import ARTIFACT_NAMES, artifact_path
        from uuid import uuid4
        target = Path(selected) / ("self_healing-" + uuid4().hex[:10])
        try:
            target.mkdir(exist_ok=False)
            for name in ARTIFACT_NAMES:
                shutil.copy2(artifact_path(self._run_dir, name), target / name)
            self.task_status.set_status("succeeded", "结果已导出到 " + str(target))
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def closeEvent(self, event):
        if self.has_running_worker():
            QMessageBox.warning(self, "后台任务运行中", "请等待 自愈控制仿真完成，或取消后再关闭。")
            event.ignore()
        else:
            super().closeEvent(event)
