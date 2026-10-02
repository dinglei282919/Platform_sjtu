"""Desktop pages for the shared Python DNN training and MPC implementation."""
import json
import shutil
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton, QLabel, QProgressBar,
    QScrollArea, QSplitter, QFileDialog, QMessageBox, QSizePolicy)

from platform_core.dnn_mpc.config import training_defaults, mpc_defaults, validate_config, model_info
from platform_core.dnn_mpc.runner import new_run_dir, run
from dnn_mpc_plot_view import DnnMpcPlotView


class ProcessControlDnnMpcWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, mode, config, run_dir):
        super().__init__()
        self.mode, self.config, self.run_dir = mode, config, run_dir

    def run(self):
        try:
            self.finished.emit(run(self.mode, self.config, run_dir=self.run_dir))
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
        self._last_image_revision = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(500)
        self._progress_timer.timeout.connect(self._poll_progress)
        self._build_ui()

    def _build_ui(self):
        self.setObjectName("processControlRoot")
        self.setStyleSheet("""
            QWidget#processControlRoot { background: #1a2635; }
            QWidget#processControlRoot QGroupBox {
                background: rgba(31, 49, 70, .88); border: 1px solid rgba(123, 167, 210, .35);
                border-radius: 8px; margin-top: 10px; padding-top: 12px;
                color: #d4e8ff; font-size: 15px; font-weight: 600;
            }
            QWidget#processControlRoot QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #d9ecff;
            }
            QWidget#processControlRoot QScrollArea { border: none; background: transparent; }
            QWidget#processControlRoot QLabel { color: #d2e5fb; font-size: 14px; }
            QWidget#processControlRoot QLabel#parameterRangeHint { color: #acc6df; font-size: 13px; }
            QWidget#processControlRoot QLabel#processControlStatus { color: #9bd0ff; font-size: 15px; font-weight: 650; }
            QWidget#processControlRoot QLabel#resultPathLabel {
                border: 1px solid rgba(143, 182, 220, .35); border-radius: 5px;
                background: rgba(21, 35, 52, .72); color: #d6eaff; min-height: 32px;
                padding: 7px 9px; font-size: 13px; font-weight: 500;
            }
            QWidget#processControlRoot QLineEdit,
            QWidget#processControlRoot QSpinBox, QWidget#processControlRoot QDoubleSpinBox {
                background: rgba(21, 35, 52, .95); color: #e7f2ff;
                border: 1px solid rgba(143, 182, 220, .35); border-radius: 5px;
                min-height: 28px; padding: 2px 8px;
            }
            QWidget#processControlRoot QSpinBox:disabled { color: #87a2bd; }
            QWidget#processControlRoot QPushButton {
                border: 1px solid rgba(101, 175, 235, .52); border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(35,90,132,.95), stop:1 rgba(31,71,112,.95));
                color: #dff2ff; padding: 7px 14px; font-weight: 700;
            }
            QWidget#processControlRoot QPushButton:hover { background: rgba(62,122,174,.95); color: white; }
            QWidget#processControlRoot QPushButton:disabled {
                color: #87a2bd; border-color: rgba(110,140,170,.35); background: rgba(33,55,77,.75);
            }
            QWidget#processControlRoot QSplitter::handle {
                background: rgba(126,176,236,.28); border-left: 1px solid rgba(126,176,236,.42);
                border-right: 1px solid rgba(18,32,48,.65);
            }
            QWidget#processControlRoot QSplitter::handle:hover { background: rgba(126,176,236,.55); }
        """)
        defaults = training_defaults() if self.page_mode == "training" else mpc_defaults()
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(12)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)
        self._config_group = QGroupBox("DNNTrain" if self.page_mode == "training" else "MPC simulation")
        layout = QVBoxLayout(self._config_group)
        paths = QGroupBox("路径设置")
        paths_form = QFormLayout(paths)
        parameters = QGroupBox("训练参数" if self.page_mode == "training" else "仿真参数")
        parameter_layout = QVBoxLayout(parameters)
        form = QFormLayout()
        for fields in (paths_form, form):
            fields.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            fields.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            fields.setVerticalSpacing(8)
        parameter_layout.addLayout(form)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
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
            field.setMaximumWidth(180)
            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if self.page_mode == "training":
            self._hidden_layers_input.setMaximumWidth(180)
        note = QLabel(description)
        note.setWordWrap(True)
        parameter_layout.addWidget(note)
        self._run_btn = QPushButton("运行训练模块" if self.page_mode == "training" else "运行 MPC 仿真")
        self._run_btn.clicked.connect(lambda: self._run_mode(self.page_mode))
        action = QHBoxLayout()
        action.addStretch()
        action.addWidget(self._run_btn)
        parameter_layout.addLayout(action)
        layout.addWidget(paths)
        layout.addWidget(parameters)
        result_paths = QGroupBox("结果文件")
        result_paths_layout = QVBoxLayout(result_paths)
        self._result_path_label = QLabel("运行后显示本次结果路径。")
        self._result_path_label.setObjectName("resultPathLabel")
        self._result_path_label.setWordWrap(True)
        self._result_path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._result_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        result_paths_layout.addWidget(self._result_path_label)
        layout.addWidget(result_paths)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._config_group)
        scroll.setMinimumWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._config_scroll = scroll
        for child in [self._config_group, *self._config_group.findChildren(QWidget)]:
            child.installEventFilter(self)
        splitter.addWidget(scroll)
        result_group = QGroupBox("训练 / 验证 RMSE" if self.page_mode == "training" else "状态偏差与控制输入")
        result_layout = QVBoxLayout(result_group)
        self._status_label = QLabel("状态：待执行")
        self._status_label.setObjectName("processControlStatus")
        self._status_label.setWordWrap(True)
        result_layout.addWidget(self._status_label)
        self._progress_bar = QProgressBar()
        result_layout.addWidget(self._progress_bar)
        self._empty_label = QLabel("运行后显示本次计算结果")
        self._empty_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self._empty_label)
        self._image = DnnMpcPlotView()
        self._image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image.hide()
        result_layout.addWidget(self._image, 1)
        self._metrics_label = QLabel()
        self._metrics_label.setWordWrap(True)
        result_layout.addWidget(self._metrics_label)
        if self.page_mode == "mpc":
            caption = QLabel("x₁：位置偏差(m)，x₂：速度偏差(m/s)，x₃=θ−π(rad)，x₄：角速度偏差(rad/s)，u：控制力(N)。综合 RMSE 混合不同量纲，累计代价不代表实际能耗。")
            caption.setWordWrap(True)
            result_layout.addWidget(caption)
        downloads = QHBoxLayout()
        files = [("SVG", "training_rmse.svg"), ("PDF", "training_rmse.pdf"), ("模型", "trained_model.npz"), ("训练记录", "training_history.csv"), ("数据集", "training_dataset.mat")] if self.page_mode == "training" else [("SVG", "dnn_mpc.svg"), ("PDF", "dnn_mpc.pdf"), ("结果 ZIP", "dnn_mpc_results.zip")]
        self._download_buttons = []
        for label, name in files:
            button = QPushButton("导出 " + label)
            button.clicked.connect(lambda checked=False, filename=name: self._export(filename))
            button.setEnabled(False)
            downloads.addWidget(button)
            self._download_buttons.append(button)
        result_layout.addLayout(downloads)
        splitter.addWidget(result_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([620, 1120])

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
        self._outputs, self._payload = {}, None
        self._last_image_revision = None
        self._image.clear()
        self._image.hide()
        self._empty_label.show()
        self._metrics_label.clear()
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
        self._status_label.setText("状态：启动中")
        self._progress_bar.setValue(0)
        self._thread = QThread(self)
        self._worker = ProcessControlDnnMpcWorker(mode, config, self._run_dir)
        self._worker.moveToThread(self._thread)
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
            self._progress_bar.setValue(round(value["percent"]))
            self._status_label.setText(value["message"])
            self._outputs.update(value.get("outputs", {}))
            self._refresh_image()
        except (OSError, ValueError, KeyError):
            pass

    def _refresh_image(self):
        key = "training_figure" if self.page_mode == "training" else "simulation_figure"
        filename = self._outputs.get(key)
        if not filename or not Path(filename).is_file():
            return
        revision = (filename, Path(filename).stat().st_mtime_ns)
        if revision != self._last_image_revision:
            self._image.load(str(filename))
            self._image.show()
            self._empty_label.hide()
            self._last_image_revision = revision

    def _handle_success(self, payload):
        self._progress_timer.stop()
        self._payload, self._outputs = payload, payload["outputs"]
        self._progress_bar.setValue(100)
        result = payload["result"]
        self._status_label.setText(f"执行完成，用时 {result['elapsed_seconds']:.2f} 秒")
        metrics = result["metrics"]
        if self.page_mode == "training":
            self._metrics_label.setText(f"测试 RMSE（原始量纲综合）：{metrics['test_rmse']:.6f}；最佳轮次：{metrics['best_epoch']}")
        else:
            self._metrics_label.setText(f"跟踪 RMSE：{metrics['tracking_rmse']:.4f}；最大 |u|：{metrics['max_abs_control']:.2f} N；输入越界：{metrics['input_violations']}；累计代价：{metrics['total_cost']:.2f}。" + result.get("warning", ""))
        self._result_path_label.setText(payload["json_path"])
        self._refresh_image()
        for button in self._download_buttons:
            button.setEnabled(True)

    def _handle_failure(self, error):
        self._progress_timer.stop()
        self._status_label.setText("执行失败：" + error)
        self._progress_bar.setValue(0)
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
