"""Desktop pages for the shared Python DNN training and MPC implementation."""
import json
import shutil
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QFrame, QLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton, QLabel,
    QSplitter, QFileDialog, QMessageBox, QSizePolicy, QTabWidget)

from platform_core.risk_control.dnn_mpc.config import training_defaults, mpc_defaults, validate_config, model_info, MPC_PLOT_GROUPS
from platform_core.risk_control.dnn_mpc.runner import new_run_dir, run
from desktop.risk_control.dnn_mpc_plot_view import DnnMpcPlotView, ControlPanel, ControlRunLog, ControlScrollArea, ControlTaskStatus, control_page_style


MPC_PLOT_DESCRIPTIONS = {
    "dnn_mpc": "四个状态偏差与控制输入的整体响应",
    "mpc_states": "位置、速度、角度与角速度相对参考值的偏差",
    "mpc_control": "控制力随仿真时间的变化",
}


class ProcessControlDnnMpcWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    logged = Signal(str)

    def __init__(self, mode, config, run_dir):
        super().__init__()
        self.mode, self.config, self.run_dir = mode, config, run_dir

    def run(self):
        try:
            self.finished.emit(run(self.mode, self.config, self.logged.emit, run_dir=self.run_dir))
        except Exception as exc:
            self.failed.emit(str(exc))


class ProcessControlDnnMpcWidget(QWidget):
    def __init__(self, page_mode="training", parent=None):
        super().__init__(parent)
        self.page_mode = page_mode if page_mode in ("training", "mpc") else "training"
        self._thread = self._worker = None
        self._outputs = {}
        self._payload = None
        self._run_dir = None
        self._model_path_user_selected = False
        self._model_signature = None
        self._last_image_revision = {}
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(500)
        self._progress_timer.timeout.connect(self._poll_progress)
        self._build_ui()

    def _build_ui(self):
        self.setObjectName("processControlRoot")
        self.setStyleSheet(control_page_style("processControlRoot"))
        defaults = training_defaults() if self.page_mode == "training" else mpc_defaults()
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(18)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(12)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)
        self._config_group = ControlPanel("训练设置" if self.page_mode == "training" else "仿真设置")
        layout = self._config_group.body
        paths = ControlPanel("路径设置", secondary=True)
        paths_form = QFormLayout()
        paths.body.addLayout(paths_form)
        parameters = ControlPanel("训练参数" if self.page_mode == "training" else "仿真参数", secondary=True)
        parameter_layout = parameters.body
        form = QFormLayout()
        for fields in (paths_form, form):
            fields.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            fields.setRowWrapPolicy(QFormLayout.WrapAllRows)
            fields.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
            fields.setVerticalSpacing(8)
        parameter_layout.addLayout(form)
        self._output_dir_input = QLineEdit(defaults["output_dir"])
        self._model_path_input = QLineEdit(defaults["model_path"])
        paths_form.addRow("输出目录", self._path_row(self._output_dir_input, "directory"))
        paths_form.addRow("模型文件", self._path_row(self._model_path_input, "save-npz" if self.page_mode == "training" else "open-npz"))
        self._model_path_input.textEdited.connect(self._model_edited)
        self._output_dir_input.textEdited.connect(self._output_edited)
        if self.page_mode == "training":
            self._sample_count_input = QSpinBox()
            self._sample_count_input.setRange(100, 100000)
            self._sample_count_input.setSingleStep(1000)
            self._sample_count_input.setValue(defaults["sample_count"])
            form.addRow("生成样本数", self._parameter_field(self._sample_count_input,
                        "范围：100～100000（整数）；仅自动生成时生效。"))
            self._dataset_path_input = QLineEdit(defaults["dataset_path"])
            self._dataset_path_input.setPlaceholderText("留空则按原状态方程生成数据集")
            self._epochs_input = QSpinBox()
            self._epochs_input.setRange(1, 5000)
            self._epochs_input.setValue(defaults["epochs"])
            self._lr_input = QDoubleSpinBox()
            self._lr_input.setDecimals(6)
            self._lr_input.setRange(.000001, .1)
            self._lr_input.setSingleStep(.0001)
            self._lr_input.setValue(defaults["lr"])
            self._hidden_layers_input = QLineEdit(defaults["hidden_layers"])
            form.addRow("训练轮数", self._parameter_field(self._epochs_input, "范围：1～5000（整数）。"))
            form.addRow("学习率", self._parameter_field(self._lr_input, "范围：0.000001～0.1。"))
            form.addRow("两个隐藏层", self._parameter_field(self._hidden_layers_input,
                        "每层 1～512 个节点；用英文逗号分隔，例如 64,64。"))
            form.addRow("外部数据集", self._path_row(self._dataset_path_input, "open-mat", clearable=True))
            self._dataset_note = QLabel()
            self._dataset_note.setWordWrap(True)
            parameter_layout.addWidget(self._dataset_note)
            self._dataset_path_input.textChanged.connect(self._dataset_changed)
            self._dataset_changed()
            description = "tanh / Adam；70% 训练、15% 验证、15% 测试。曲线为标准化 RMSE。"
        else:
            self._sim_time_input = QDoubleSpinBox()
            self._sim_time_input.setRange(.2, 20.)
            self._sim_time_input.setSingleStep(.02)
            self._sim_time_input.setValue(defaults["sim_time"])
            self._horizon_input = QSpinBox()
            self._horizon_input.setRange(1, 60)
            self._horizon_input.setValue(defaults["prediction_horizon"])
            form.addRow("仿真时长 (s)", self._parameter_field(self._sim_time_input, "范围：0.2～20 s。"))
            form.addRow("预测步长", self._parameter_field(self._horizon_input,
                        "范围：1～60（整数）；每步 0.02 s。"))
            latest = QPushButton("使用最近训练模型")
            latest.clicked.connect(self._use_latest_model)
            paths_form.addRow(latest)
            description = "Δt=0.02 s；Q=diag(300,1,300,1)，R=0.01；P 由模型计算；控制输入 ±10 N。状态由同一预测网络更新，属于模型内闭环验证。"
        for field in [*parameters.findChildren(QSpinBox), *parameters.findChildren(QDoubleSpinBox)]:
            field.setMinimumWidth(0)
            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        note = QLabel(description)
        note.setWordWrap(True)
        note.setProperty("controlHint", True)
        parameter_layout.addWidget(note)
        self._run_btn = QPushButton("运行训练模块" if self.page_mode == "training" else "运行 MPC 仿真")
        self._run_btn.setProperty("controlPrimary", True)
        self._run_btn.clicked.connect(lambda: self._run_mode(self.page_mode))
        parameter_layout.addWidget(self._run_btn)
        layout.addWidget(paths)
        layout.addWidget(parameters)
        result_paths = ControlPanel("结果文件", secondary=True)
        result_paths_layout = result_paths.body
        self._result_path_label = QLabel("运行后显示本次结果路径。")
        self._result_path_label.setObjectName("resultPathLabel")
        self._result_path_label.setWordWrap(True)
        self._result_path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._result_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        result_paths_layout.addWidget(self._result_path_label)
        layout.addWidget(result_paths)
        layout.addStretch()
        scroll = ControlScrollArea(self._config_group)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(380)
        scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._config_scroll = scroll
        for child in [self._config_group, *self._config_group.findChildren(QWidget)]:
            child.installEventFilter(self)
        splitter.addWidget(scroll)
        self._result_content = QWidget()
        self._result_content.setAttribute(Qt.WA_StyledBackground, True)
        self._result_content.setProperty("controlSurface", True)
        result_column = QVBoxLayout(self._result_content)
        result_column.setContentsMargins(0, 0, 0, 0)
        result_column.setSpacing(12)
        result_column.setSizeConstraint(QLayout.SetMinimumSize)
        self._result_group = ControlPanel("结果图示")
        result_layout = self._result_group.body
        self._task_status = ControlTaskStatus("训练" if self.page_mode == "training" else "MPC 仿真")
        self._status_label = self._task_status.message
        self._progress_bar = self._task_status.progress
        result_layout.addWidget(self._task_status)
        self._images = {}
        if self.page_mode == "mpc":
            self._plot_tabs = QTabWidget()
            self._plot_tabs.setObjectName("mpcPlotTabs")
            self._plot_tabs.setAccessibleName("MPC 输出图像")
            self._plot_tabs.tabBar().setExpanding(True)
            for name, title, key in MPC_PLOT_GROUPS:
                panel = QWidget()
                panel_layout = QVBoxLayout(panel)
                panel_layout.setContentsMargins(0, 12, 0, 0)
                self._add_plot_heading(panel_layout, title, MPC_PLOT_DESCRIPTIONS[name])
                view = DnnMpcPlotView(aspect_ratio=8.6 / 6.5)
                view.setAccessibleName(title + "曲线")
                panel_layout.addWidget(view, 1)
                self._images[key] = view
                self._plot_tabs.addTab(panel, title)
            self._image = self._images["simulation_figure"]
            result_layout.addWidget(self._plot_tabs, 1)
        else:
            self._add_plot_heading(result_layout, "训练 / 验证 RMSE", "训练集与验证集的标准化误差随轮次变化")
            self._image = DnnMpcPlotView(aspect_ratio=2.35)
            self._image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._images["training_figure"] = self._image
            result_layout.addWidget(self._image, 1)
        divider = QFrame()
        divider.setProperty("controlDivider", True)
        divider.setFixedHeight(1)
        result_layout.addSpacing(6)
        result_layout.addWidget(divider)
        self._metrics_label = QLabel()
        self._metrics_label.setWordWrap(True)
        result_layout.addWidget(self._metrics_label)
        caption = QLabel("曲线为标准化 RMSE；参数修改后需重新训练。" if self.page_mode == "training" else
                         "x₁：位置(m)，x₂：速度(m/s)，x₃=θ−π(rad)，x₄：角速度(rad/s)，u：控制力(N)。图示相对参考值的偏差；综合 RMSE 混合量纲，累计代价不代表实际能耗。")
        caption.setWordWrap(True)
        caption.setProperty("controlHint", True)
        self._caption = caption
        result_layout.addWidget(caption)
        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.hide()
        result_layout.addWidget(self._warning_label)
        self._downloads_layout = QGridLayout()
        self._download_columns = 0
        files = [("SVG", "training_rmse.svg"), ("PDF", "training_rmse.pdf"), ("模型", "trained_model.npz"), ("训练记录", "training_history.csv"), ("数据集", "training_dataset.mat")] if self.page_mode == "training" else [("SVG", "dnn_mpc.svg"), ("PDF", "dnn_mpc.pdf"), ("结果 ZIP", "dnn_mpc_results.zip")]
        self._download_buttons = []
        for label, name in files:
            button = QPushButton(("导出当前图 " if self.page_mode == "mpc" and label in ("SVG", "PDF") else "导出 ") + label)
            button.clicked.connect(lambda checked=False, filename=name: self._export(self._selected_plot_file(filename)))
            button.setEnabled(False)
            self._download_buttons.append(button)
        result_layout.addLayout(self._downloads_layout)
        result_column.addWidget(self._result_group)
        self._logs = ControlRunLog()
        result_column.addWidget(self._logs)
        result_column.addStretch()
        self._result_scroll = ControlScrollArea(self._result_content)
        # Keep the viewport width stable when wrapped results/logs change height.
        # Qt 6.11's as-needed scrollbar sizing can recursively resize this content
        # at the overflow threshold, eventually overflowing the GUI thread stack.
        self._result_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._result_content.installEventFilter(self)
        self._layout_downloads()
        splitter.addWidget(self._result_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 1050])

    def _add_plot_heading(self, layout, title, description):
        heading = QLabel(title)
        heading.setProperty("controlPlotHeading", True)
        layout.addWidget(heading)
        hint = QLabel(description)
        hint.setProperty("controlHint", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _layout_downloads(self):
        columns = max(1, min(len(self._download_buttons), (self._result_content.width() - 40) // 150))
        if columns != self._download_columns:
            self._download_columns = columns
            for index, button in enumerate(self._download_buttons):
                self._downloads_layout.addWidget(button, index // columns, index % columns)

    def _parameter_field(self, field, text):
        widget = QWidget()
        column = QVBoxLayout(widget)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(field)
        hint = QLabel(text)
        hint.setObjectName("parameterRangeHint")
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        field.setAccessibleDescription(text)
        column.addWidget(hint)
        return widget

    def _path_row(self, field, kind, clearable=False):
        field.setMinimumWidth(0)
        field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(field, 1)
        button = QPushButton("选择")
        button.setFixedWidth(56)
        button.clicked.connect(lambda: self._browse(field, kind))
        row.addWidget(button)
        if clearable:
            clear = QPushButton("清空")
            clear.setFixedWidth(56)
            clear.clicked.connect(field.clear)
            row.addWidget(clear)
        return widget

    def _dataset_changed(self):
        supplied = bool(self._dataset_path_input.text().strip())
        self._sample_count_input.setEnabled(not supplied)
        self._dataset_note.setText("使用完整 MAT 数据；清空路径可启用自动生成。" if supplied else
                                   "按原状态方程生成仿真数据后训练；dt=0.02 s，随机种子 42。")

    def eventFilter(self, watched, event):
        if watched is getattr(self, "_result_content", None):
            if event.type() == QEvent.Resize:
                self._layout_downloads()
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta:
                bar = self._config_scroll.verticalScrollBar()
                bar.setValue(bar.value() - delta)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _browse(self, field, kind):
        if kind == "directory":
            path = QFileDialog.getExistingDirectory(self, "选择输出目录", field.text())
        else:
            filters = "NPZ 模型 (*.npz)" if kind.endswith("npz") else "MAT 数据 (*.mat)"
            picker = QFileDialog.getSaveFileName if kind.startswith("save") else QFileDialog.getOpenFileName
            path, _ = picker(self, "选择文件", field.text(), filters)
        if path:
            field.setText(path)
            if field is self._model_path_input:
                self._model_edited()
            elif field is self._output_dir_input:
                self._output_edited()

    def _model_edited(self):
        self._model_path_user_selected = True
        if self.page_mode == "mpc":
            self._clear_result()

    def _output_edited(self):
        if self.page_mode == "training" and not self._model_path_user_selected:
            self._model_path_input.setText(str(Path(self._output_dir_input.text()) / "trained_model.npz"))

    def _use_latest_model(self):
        self._model_path_user_selected = False
        self._apply_latest_training_defaults()

    def _apply_latest_training_defaults(self):
        if self.page_mode != "mpc" or self.has_running_worker():
            return
        if not self._model_path_user_selected:
            defaults = mpc_defaults()
            self._model_path_input.setText(defaults["model_path"])
        path = Path(self._model_path_input.text())
        try:
            signature = (str(path), model_info(str(path))["sha256"])
        except (ValueError, OSError):
            signature = (str(path), None)
        if self._model_signature != signature:
            self._clear_result()
            self._model_signature = signature

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_latest_training_defaults()

    def _config(self):
        values = dict(output_dir=self._output_dir_input.text(), model_path=self._model_path_input.text())
        if self.page_mode == "training":
            values.update(dataset_path=self._dataset_path_input.text(), epochs=self._epochs_input.value(),
                          lr=self._lr_input.value(), hidden_layers=self._hidden_layers_input.text(),
                          sample_count=self._sample_count_input.value())
        else:
            values.update(sim_time=self._sim_time_input.value(), prediction_horizon=self._horizon_input.value())
        return validate_config(self.page_mode, values)

    def _clear_result(self):
        self._task_status.set_status("idle", "设置参数后开始运行", 0)
        self._outputs, self._payload = {}, None
        self._last_image_revision = {}
        for view in self._images.values():
            view.clear()
            view.set_placeholder("运行后显示本次计算结果")
        self._metrics_label.clear()
        self._warning_label.clear()
        self._warning_label.hide()
        self._logs.clear()
        self._result_path_label.setText("运行后显示本次结果路径。")
        for button in self._download_buttons:
            button.setEnabled(False)

    def _run_mode(self, mode):
        if self.has_running_worker():
            return
        self._apply_latest_training_defaults()
        try:
            config = self._config()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        self._clear_result()
        self._run_dir = new_run_dir(config)
        self._config_group.setEnabled(False)
        self._task_status.set_status("running", "正在启动任务", 0)
        for view in self._images.values():
            view.set_placeholder("计算中，等待图像更新…")
        self._thread = QThread(self)
        self._worker = ProcessControlDnnMpcWorker(mode, config, self._run_dir)
        self._worker.moveToThread(self._thread)
        self._logs.append("任务开始执行。")
        self._worker.logged.connect(self._logs.append)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._handle_success)
        self._worker.failed.connect(self._handle_failure)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker)
        self._progress_timer.start()
        self._thread.start()

    def _poll_progress(self):
        if self._run_dir is None:
            return
        try:
            value = json.loads((self._run_dir / "progress.json").read_text(encoding="utf-8"))
            self._task_status.set_status("running", value["message"], value["percent"])
            self._outputs.update(value.get("outputs", {}))
            self._refresh_image()
        except (OSError, ValueError, KeyError):
            pass

    def _refresh_image(self):
        for key, view in self._images.items():
            filename = self._outputs.get(key)
            if not filename or not Path(filename).is_file():
                continue
            revision = (filename, Path(filename).stat().st_mtime_ns)
            if revision != self._last_image_revision.get(key):
                view.load(str(filename))
                view.show()
                self._last_image_revision[key] = revision

    def _selected_plot_file(self, filename):
        if self.page_mode == "mpc" and filename in ("dnn_mpc.svg", "dnn_mpc.pdf"):
            return MPC_PLOT_GROUPS[self._plot_tabs.currentIndex()][0] + Path(filename).suffix
        return filename

    def _handle_success(self, payload):
        self._progress_timer.stop()
        self._payload, self._outputs = payload, payload["outputs"]
        result = payload["result"]
        self._task_status.set_status("succeeded", f"执行完成 · 用时 {result['elapsed_seconds']:.2f} 秒")
        metrics = result["metrics"]
        if self.page_mode == "training":
            self._metrics_label.setText(f"测试 RMSE（原始量纲综合）：{metrics['test_rmse']:.6f}；最佳轮次：{metrics['best_epoch']}")
        else:
            self._metrics_label.setText(f"跟踪 RMSE：{metrics['tracking_rmse']:.4f}；最大 |u|：{metrics['max_abs_control']:.2f} N；输入越界：{metrics['input_violations']}；累计代价：{metrics['total_cost']:.2f}")
        self._warning_label.setText(result.get("warning", ""))
        self._warning_label.setVisible(bool(result.get("warning")))
        self._logs.append("任务完成。")
        self._result_path_label.setText(payload["json_path"])
        self._refresh_image()
        for button in self._download_buttons:
            button.setEnabled(True)

    def _handle_failure(self, error):
        self._progress_timer.stop()
        self._logs.append("任务失败：" + error)
        self._task_status.set_status("failed", "执行失败：" + error)
        for view in self._images.values():
            view.set_placeholder("运行后显示本次计算结果")
        QMessageBox.warning(self, "执行失败", error)

    def _clear_worker(self):
        self._thread = self._worker = None
        self._config_group.setEnabled(True)

    def has_running_worker(self):
        return self._thread is not None and self._thread.isRunning()

    def closeEvent(self, event):
        if self.has_running_worker():
            QMessageBox.warning(self, "任务运行中", "请等待 DNN-MPC 任务完成后关闭。")
            event.ignore()
        else:
            super().closeEvent(event)

    def _export(self, name):
        if self._payload is None:
            return
        source = Path(self._payload["artifact_dir"]) / name
        destination, _ = QFileDialog.getSaveFileName(self, "导出结果", name)
        if destination and source.resolve() != Path(destination).resolve():
            try:
                shutil.copyfile(source, destination)
            except OSError as exc:
                QMessageBox.warning(self, "导出失败", str(exc))


class DnnMpcWorkflowWidget(QTabWidget):
    """One module with persistent training and simulation stages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("DNN/MPC 控制训练与优化")
        self.training_page = ProcessControlDnnMpcWidget("training")
        self.mpc_page = ProcessControlDnnMpcWidget("mpc")
        self.addTab(self.training_page, "1. 模型训练")
        self.addTab(self.mpc_page, "2. MPC 仿真")
        self.tabBar().setExpanding(True)
        self.currentChanged.connect(self._stage_changed)

    def _stage_changed(self, index):
        if index == 1:
            self.mpc_page._apply_latest_training_defaults()

    def has_running_worker(self):
        return self.training_page.has_running_worker() or self.mpc_page.has_running_worker()

    def closeEvent(self, event):
        if self.has_running_worker():
            QMessageBox.warning(self, "任务运行中", "请等待 DNN-MPC 任务完成后关闭。")
            event.ignore()
        else:
            super().closeEvent(event)
