import importlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap, QPixmapCache
from PySide6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


DEFAULT_DNN_MPC_PACKAGE_NAME = "dnnmpcpkg"


class ProcessControlDnnMpcWorker(QObject):
    """Background worker that calls the compiled MATLAB Runtime package."""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        mode: str,
        output_dir: Path,
        model_path: Path,
        sample_count: int,
        epochs: int,
        hidden_layer_config: str,
        dataset_path: Path,
        sim_time: float,
        prediction_horizon: int,
        package_dir: Path | None,
        package_name: str,
        mcr_root: Path | None,
    ):
        super().__init__()
        self.mode = mode
        self.output_dir = output_dir
        self.model_path = model_path
        self.sample_count = sample_count
        self.epochs = epochs
        self.hidden_layer_config = hidden_layer_config
        self.dataset_path = dataset_path
        self.sim_time = sim_time
        self.prediction_horizon = prediction_horizon
        self.package_dir = package_dir
        self.package_name = package_name
        self.mcr_root = mcr_root

    def run(self):
        """Run work in a background thread so the Qt window remains responsive."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if not self._python_package_can_be_used():
                raise ModuleNotFoundError(
                    f"未找到 MATLAB Python Package: {self.package_name}。"
                    "请先安装 dnn_mpc/build_python,或把 Python包目录指向导出目录。"
                )
            payload = self._run_with_python_package()
            self.finished.emit(payload)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())

    def _python_package_can_be_used(self):
        if not self.package_name:
            return False
        if self.package_dir is not None and (self.package_dir / self.package_name).is_dir():
            return True
        return importlib.util.find_spec(self.package_name) is not None

    def _run_with_python_package(self):
        runtime_library = self._prepare_runtime_paths()
        if self.package_dir is not None and self.package_dir.is_dir():
            build_path = str(self.package_dir)
            if build_path not in sys.path:
                sys.path.insert(0, build_path)

        package_module = importlib.import_module(self.package_name)
        handle = package_module.initialize()
        try:
            result_json = self._call_package_handle(handle)
        finally:
            try:
                handle.terminate()
            except Exception:  # noqa: BLE001
                pass

        return self._make_payload(
            result_json,
            backend="matlab_python_package",
            backend_detail={
                "package_name": self.package_name,
                "package_dir": str(self.package_dir) if self.package_dir is not None else "",
                "mcr_root": str(self.mcr_root),
                "runtime_library": str(runtime_library),
            },
        )

    def _prepare_runtime_paths(self):
        if self.mcr_root is None or not str(self.mcr_root).strip():
            raise FileNotFoundError("使用 MATLAB Python Package 时必须填写 MCR_ROOT。")

        system = platform.system()
        if system == "Windows":
            arch = "win64"
            path_var = "PATH"
            runtime_file = "mclmcrrt24_2.dll"
        elif system == "Linux":
            arch = "glnxa64"
            path_var = "LD_LIBRARY_PATH"
            runtime_file = "libmwmclmcrrt.so.24.2"
        elif system == "Darwin":
            arch = "maca64" if platform.mac_ver()[-1] == "arm64" else "maci64"
            path_var = "DYLD_LIBRARY_PATH"
            runtime_file = "libmwmclmcrrt.24.2.dylib"
        else:
            raise RuntimeError(f"不支持的操作系统: {system}")

        runtime_dir = self.mcr_root / "runtime" / arch
        bin_dir = self.mcr_root / "bin" / arch
        extern_dir = self.mcr_root / "extern" / "bin" / arch
        runtime_library = runtime_dir / runtime_file
        if not runtime_library.exists():
            raise FileNotFoundError(f"未找到 MATLAB Runtime R2024b 运行库: {runtime_library}")

        current = os.environ.get(path_var, "")
        current_parts = current.split(os.pathsep) if current else []
        prepend_parts = []
        for part in (runtime_dir, bin_dir, extern_dir):
            part_str = str(part)
            if part.exists() and not any(part_str.lower() == p.lower() for p in current_parts):
                prepend_parts.append(part_str)
        if prepend_parts:
            os.environ[path_var] = os.pathsep.join(prepend_parts + current_parts)
        return runtime_library

    def _call_package_handle(self, handle):
        dataset_path = "" if self.dataset_path is None else str(self.dataset_path)
        if self.mode == "training":
            return str(
                handle.run_process_control_training(
                    str(self.output_dir),
                    float(self.sample_count),
                    float(self.epochs),
                    self.hidden_layer_config,
                    dataset_path,
                )
            )
        if self.mode == "mpc":
            return str(
                handle.run_process_control_mpc_validation(
                    str(self.output_dir),
                    str(self.model_path),
                    float(self.sim_time),
                    float(self.prediction_horizon),
                )
            )
        return str(
            handle.run_process_control_pipeline(
                str(self.output_dir),
                float(self.sample_count),
                float(self.epochs),
                float(self.sim_time),
                float(self.prediction_horizon),
                self.hidden_layer_config,
                dataset_path,
            )
        )


    def _make_payload(self, result_json: str, backend: str, backend_detail: dict):
        result = json.loads(result_json)
        outputs = result.get("outputs", {})
        result["execution_backend"] = backend
        result["execution_backend_detail"] = backend_detail
        return {
            "mode": self.mode,
            "backend": backend,
            "backend_detail": backend_detail,
            "result": result,
            "result_json": result_json,
            "json_path": outputs.get("summary_json", str(self.output_dir / "summary.json")),
            "outputs": outputs,
        }


class ProcessControlDnnMpcWidget(QWidget):
    """Process-control DNN-MPC page backed by the compiled Runtime package."""

    def __init__(self, page_mode: str = "training", parent=None):
        super().__init__(parent)
        self.page_mode = page_mode if page_mode in ("training", "mpc") else "training"
        self.workspace_dir = Path(__file__).resolve().parent
        self._latest_training_state_path = self.workspace_dir / "dnn_mpc" / "latest_training_state.json"
        self._model_path_user_selected = False
        self._package_dir_input = None
        self._package_name_input = None
        self._mcr_root_input = None
        self._output_dir_input = None
        self._model_path_input = None
        self._dataset_path_input = None
        self._sample_count_input = None
        self._epochs_input = None
        self._hidden_layers_input = None
        self._sim_time_input = None
        self._horizon_input = None
        self._train_btn = None
        self._mpc_btn = None
        self._refresh_btn = None
        self._status_label = None
        self._progress_bar = None
        self._result_path_label = None
        self._image_title_label = None
        self._image_label = None
        self._image_buttons = {}
        self._config_scroll = None
        self._splitter = None
        self._current_image_mode = "trajectory"
        self._last_image_path = None
        self._outputs = {}
        self._progress_path = None
        self._last_progress_mtime = None
        self._thread = None
        self._worker = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1200)
        self._progress_timer.timeout.connect(self._poll_progress)
        self._build_ui()
        if self.page_mode == "mpc":
            self._apply_latest_training_defaults()
        self._set_image_mode("training" if self.page_mode == "training" else "trajectory")

    def _build_ui(self):
        self.setObjectName("processControlRoot")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        page_title = "DNNTrain" if self.page_mode == "training" else "MPC simulation"
        config_group = QGroupBox(page_title)
        config_group.setMinimumWidth(0)
        config_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 10, 10, 10)
        config_layout.setSpacing(7)

        path_group = QGroupBox("路径设置")
        path_form = QFormLayout(path_group)
        path_form.setVerticalSpacing(8)
        path_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        path_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._package_dir_input = QLineEdit(str(self.workspace_dir / "dnn_mpc" / "build_python"))
        self._package_dir_input.setPlaceholderText("可留空；留空时使用当前 Python 环境已安装的包")
        path_form.addRow("Python包目录:", self._make_path_row(self._package_dir_input))

        self._package_name_input = QLineEdit(DEFAULT_DNN_MPC_PACKAGE_NAME)
        path_form.addRow("Python包名:", self._package_name_input)

        self._mcr_root_input = QLineEdit(r"E:\MATLAB2024")
        self._mcr_root_input.setToolTip("MATLAB R2024b 或 MATLAB Runtime R2024b 根目录。")
        path_form.addRow("MCR_ROOT:", self._make_path_row(self._mcr_root_input))

        default_output_dir = self.workspace_dir / "dnn_mpc" / "output"
        self._output_dir_input = QLineEdit(str(default_output_dir))
        path_form.addRow("输出目录:", self._make_path_row(self._output_dir_input))

        self._model_path_input = QLineEdit(str(self._default_model_path(default_output_dir)))
        self._model_path_input.textEdited.connect(self._handle_model_path_edited)
        self._output_dir_input.textChanged.connect(self._handle_output_dir_changed)
        path_form.addRow("模型文件:", self._make_file_path_row(self._model_path_input))
        config_layout.addWidget(path_group)

        if self.page_mode == "training":
            training_group = QGroupBox("DNNTrain")
            training_form = QFormLayout(training_group)
            training_form.setVerticalSpacing(8)
            training_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            training_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

            self._sample_count_input = self._new_int_spin(100, 100000, 1000, 1000)
            self._sample_count_input.setToolTip("留空外部数据集时用于自动生成训练样本；使用外部数据集时以文件实际样本数为准。")
            training_form.addRow("训练样本数:", self._sample_count_input)

            self._epochs_input = self._new_int_spin(1, 5000, 50, 50)
            training_form.addRow("训练轮数:", self._epochs_input)

            self._hidden_layers_input = QLineEdit("64,64")
            self._hidden_layers_input.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            training_form.addRow("隐藏层规模:", self._hidden_layers_input)

            self._dataset_path_input = QLineEdit("")
            self._dataset_path_input.setPlaceholderText("留空则自动生成数据集；选择 .mat 则使用外部 X_data/Y_data")
            self._dataset_path_input.setToolTip("外部 .mat 需包含 X_data 和 Y_data，支持 5xN/4xN 或 Nx5/Nx4。")
            training_form.addRow("外部数据集:", self._make_file_path_row(self._dataset_path_input, allow_clear=True))

            train_row = QHBoxLayout()
            train_row.addStretch()
            self._train_btn = QPushButton("运行训练模块")
            self._train_btn.clicked.connect(lambda _checked=False: self._run_mode("training"))
            train_row.addWidget(self._train_btn)
            training_form.addRow("", train_row)
            config_layout.addWidget(training_group)
        else:
            mpc_group = QGroupBox("MPC simulation")
            mpc_form = QFormLayout(mpc_group)
            mpc_form.setVerticalSpacing(8)
            mpc_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            mpc_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

            self._sim_time_input = self._new_float_spin(0.2, 20.0, 1.0, 0.2)
            mpc_form.addRow("仿真时长(s):", self._sim_time_input)

            self._horizon_input = self._new_int_spin(1, 60, 5, 1)
            mpc_form.addRow("预测步长:", self._horizon_input)

            mpc_row = QHBoxLayout()
            mpc_row.addStretch()
            self._mpc_btn = QPushButton("运行 MPC 仿真")
            self._mpc_btn.clicked.connect(lambda _checked=False: self._run_mode("mpc"))
            mpc_row.addWidget(self._mpc_btn)
            mpc_form.addRow("", mpc_row)
            config_layout.addWidget(mpc_group)

        self._status_label = QLabel("状态：待执行")
        self._status_label.setObjectName("processControlStatus")

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("待执行")

        result_group = QGroupBox("结果文件")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(10, 8, 10, 10)
        self._result_path_label = QLabel("运行后显示 JSON 结果文件路径。")
        self._result_path_label.setObjectName("resultPathLabel")
        self._result_path_label.setWordWrap(True)
        self._result_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        result_layout.addWidget(self._result_path_label)
        config_layout.addWidget(result_group)

        image_group = QGroupBox("结果图")
        image_group.setMinimumWidth(260)
        image_group.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(10, 12, 10, 10)
        image_layout.setSpacing(7)

        status_panel = QFrame()
        status_panel.setObjectName("processControlStatusPanel")
        status_layout = QHBoxLayout(status_panel)
        status_layout.setContentsMargins(10, 7, 10, 7)
        status_layout.setSpacing(12)
        status_layout.addWidget(self._status_label, 2)
        status_layout.addWidget(self._progress_bar, 3)
        image_layout.addWidget(status_panel)

        switch_row = QHBoxLayout()
        switch_row.setSpacing(8)
        image_modes = (
            (("训练曲线", "training"), ("预测误差", "prediction"))
            if self.page_mode == "training"
            else (("状态轨迹", "trajectory"), ("控制输入", "control"), ("跟踪误差", "tracking"), ("代价曲线", "cost"))
        )
        for text, mode in image_modes:
            btn = self._make_image_switch(text, mode)
            self._image_buttons[mode] = btn
            switch_row.addWidget(btn)
        switch_row.addStretch()
        self._refresh_btn = QPushButton("刷新图像")
        self._refresh_btn.clicked.connect(self._refresh_image)
        switch_row.addWidget(self._refresh_btn)
        image_layout.addLayout(switch_row)

        initial_title = "训练曲线 (training_performance.png)" if self.page_mode == "training" else "状态轨迹 (process_control_trajectory.png)"
        self._image_title_label = QLabel(initial_title)
        self._image_title_label.setObjectName("imagePanelTitle")
        image_layout.addWidget(self._image_title_label)

        self._image_label = QLabel("暂无图像")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(220, 220)
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setObjectName("processControlImage")
        image_layout.addWidget(self._image_label, 1)

        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QFrame.NoFrame)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        config_scroll.setMinimumWidth(420)
        config_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        config_scroll.verticalScrollBar().setSingleStep(28)
        config_scroll.setWidget(config_group)
        self._config_scroll = config_scroll
        config_scroll.installEventFilter(self)
        config_scroll.viewport().installEventFilter(self)
        self._install_wheel_scroll_forwarding(config_group)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("processControlSplitter")
        splitter.setHandleWidth(12)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(config_scroll)
        splitter.addWidget(image_group)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([620, 1120])
        self._splitter = splitter
        root_layout.addWidget(splitter, 1)

        self.setStyleSheet(
            """
            QWidget#processControlRoot {
                background: #1a2635;
            }
            QWidget#processControlRoot QGroupBox {
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 12px;
                background: rgba(31, 49, 70, 0.88);
                color: #d4e8ff;
                font-size: 15px;
                font-weight: 600;
            }
            QWidget#processControlRoot QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #d9ecff;
            }
            QWidget#processControlRoot QLabel {
                color: #d2e5fb;
                font-size: 14px;
            }
            QWidget#processControlRoot QLabel#processControlStatus {
                color: #9bd0ff;
                font-size: 15px;
                font-weight: 650;
            }
            QWidget#processControlRoot QLabel#resultPathLabel {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.72);
                color: #d6eaff;
                min-height: 32px;
                padding: 7px 9px;
                font-size: 13px;
                font-weight: 500;
            }
            QWidget#processControlRoot QFrame#processControlStatusPanel {
                border: 1px solid rgba(108, 170, 226, 0.38);
                border-radius: 7px;
                background: rgba(18, 36, 56, 0.86);
            }
            QWidget#processControlRoot QLabel#imagePanelTitle {
                font-size: 20px;
                color: #e4f2ff;
                font-weight: 650;
                padding: 2px 0 2px 2px;
            }
            QWidget#processControlRoot QLineEdit,
            QWidget#processControlRoot QSpinBox,
            QWidget#processControlRoot QDoubleSpinBox {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.95);
                color: #e7f2ff;
                min-height: 28px;
                padding: 2px 8px;
            }
            QWidget#processControlRoot QProgressBar {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 6px;
                background: rgba(21, 35, 52, 0.95);
                color: #dff2ff;
                min-height: 26px;
                text-align: center;
                font-weight: 700;
            }
            QWidget#processControlRoot QProgressBar::chunk {
                border-radius: 5px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(71, 168, 226, 0.98),
                    stop: 1 rgba(108, 215, 188, 0.98)
                );
            }
            QWidget#processControlRoot QPushButton {
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
            QWidget#processControlRoot QPushButton#imgSwitch {
                min-width: 70px;
                padding: 6px 7px;
            }
            QWidget#processControlRoot QPushButton#imgSwitch:checked {
                border: 1px solid rgba(146, 212, 255, 0.78);
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(68, 147, 206, 0.98),
                    stop: 1 rgba(39, 99, 153, 0.98)
                );
                color: #ffffff;
            }
            QWidget#processControlRoot QPushButton:hover {
                background: rgba(62, 122, 174, 0.95);
                color: #ffffff;
            }
            QWidget#processControlRoot QPushButton:disabled {
                color: #87a2bd;
                border-color: rgba(110, 140, 170, 0.35);
                background: rgba(33, 55, 77, 0.75);
            }
            QWidget#processControlRoot QLabel#processControlImage {
                border: 1px solid rgba(129, 169, 206, 0.45);
                background: rgba(18, 32, 48, 0.92);
                border-radius: 5px;
                color: #bcd8f6;
            }
            QWidget#processControlRoot QSplitter::handle {
                background: rgba(126, 176, 236, 0.28);
                border-left: 1px solid rgba(126, 176, 236, 0.42);
                border-right: 1px solid rgba(18, 32, 48, 0.65);
                width: 8px;
            }
            QWidget#processControlRoot QSplitter::handle:hover {
                background: rgba(126, 176, 236, 0.55);
            }
            """
        )

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self.page_mode == "mpc":
            self._apply_latest_training_defaults()

    def _default_model_path(self, output_dir: Path | None = None):
        if output_dir is None:
            output_text = self._output_dir_input.text().strip() if self._output_dir_input is not None else ""
            output_dir = Path(output_text) if output_text else self.workspace_dir / "dnn_mpc" / "output"
        return output_dir / "process_control_nn_model.mat"

    def _same_path_text(self, left, right):
        return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(os.path.normpath(str(right)))

    def _handle_model_path_edited(self, _text: str):
        self._model_path_user_selected = True

    def _handle_output_dir_changed(self, text: str):
        if self.page_mode != "mpc" or self._model_path_input is None or self._model_path_user_selected:
            return
        output_dir = Path(text.strip()) if text.strip() else self.workspace_dir / "dnn_mpc" / "output"
        self._model_path_input.setText(str(self._default_model_path(output_dir)))

    def _save_latest_training_state(self, model_path: str, result: dict, json_path: str):
        if not model_path:
            return
        model = Path(model_path)
        payload = {
            "model_mat": str(model),
            "output_dir": str(model.parent),
            "summary_json": json_path,
            "timestamp": result.get("timestamp", ""),
        }
        try:
            self._latest_training_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._latest_training_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _read_latest_training_state(self):
        try:
            payload = json.loads(self._latest_training_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _apply_latest_training_defaults(self):
        if self.page_mode != "mpc" or self._model_path_input is None or self._output_dir_input is None:
            return
        state = self._read_latest_training_state()
        model_text = str(state.get("model_mat", "")).strip()
        if not model_text:
            return
        current_model_text = self._model_path_input.text().strip()
        if self._model_path_user_selected and current_model_text:
            return

        output_text = str(state.get("output_dir", "")).strip()
        current_output_text = self._output_dir_input.text().strip()
        default_output = self.workspace_dir / "dnn_mpc" / "output"
        if output_text and (not current_output_text or self._same_path_text(current_output_text, default_output)):
            self._output_dir_input.setText(output_text)

        self._model_path_input.setText(model_text)
        self._model_path_user_selected = False

    def _install_wheel_scroll_forwarding(self, root_widget: QWidget):
        root_widget.installEventFilter(self)
        for child in root_widget.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QEvent.Type.Wheel and self._config_scroll is not None:
            delta = event.angleDelta().y()
            if delta:
                scrollbar = self._config_scroll.verticalScrollBar()
                scrollbar.setValue(scrollbar.value() - delta)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _make_path_row(self, line_edit: QLineEdit):
        line_edit.setMinimumWidth(0)
        line_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        row = QFrame()
        row.setMinimumHeight(30)
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse_btn = QPushButton("选择")
        browse_btn.setFixedWidth(56)
        browse_btn.clicked.connect(lambda _checked=False, edit=line_edit: self._browse_dir(edit))
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse_btn)
        return row

    def _make_file_path_row(self, line_edit: QLineEdit, allow_clear: bool = False):
        line_edit.setMinimumWidth(0)
        line_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        row = QFrame()
        row.setMinimumHeight(30)
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse_btn = QPushButton("选择")
        browse_btn.setFixedWidth(56)
        browse_btn.clicked.connect(lambda _checked=False, edit=line_edit: self._browse_file(edit))
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse_btn)
        if allow_clear:
            clear_btn = QPushButton("清空")
            clear_btn.setFixedWidth(56)
            clear_btn.clicked.connect(lambda _checked=False, edit=line_edit: edit.clear())
            layout.addWidget(clear_btn)
        return row

    def _browse_dir(self, line_edit: QLineEdit):
        current = line_edit.text().strip() or str(self.workspace_dir)
        selected = QFileDialog.getExistingDirectory(self, "选择目录", current)
        if selected:
            line_edit.setText(selected)

    def _browse_file(self, line_edit: QLineEdit):
        current = line_edit.text().strip() or str(self.workspace_dir)
        selected, _ = QFileDialog.getOpenFileName(self, "选择 MAT 文件", current, "MAT 文件 (*.mat)")
        if selected:
            line_edit.setText(selected)

    def _new_int_spin(self, minimum, maximum, value, step):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMinimumWidth(0)
        spin.setMaximumWidth(180)
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    def _new_float_spin(self, minimum, maximum, value, step):
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMinimumWidth(0)
        spin.setMaximumWidth(180)
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    def _make_image_switch(self, text: str, mode: str):
        btn = QPushButton(text)
        btn.setObjectName("imgSwitch")
        btn.setCheckable(True)
        btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        btn.clicked.connect(lambda _checked=False, m=mode: self._set_image_mode(m))
        return btn

    def _run_mode(self, mode: str):
        if mode == "mpc":
            self._apply_latest_training_defaults()
        if self.has_running_worker():
            QMessageBox.information(self, "任务运行中", "当前 DNN-MPC 任务尚未结束，请等待完成后再操作。")
            return

        output_dir = Path(self._output_dir_input.text().strip())
        model_path = Path(self._model_path_input.text().strip())
        dataset_text = self._dataset_path_input.text().strip() if self._dataset_path_input is not None else ""
        dataset_path = Path(dataset_text) if dataset_text else None
        package_text = self._package_dir_input.text().strip()
        package_dir = Path(package_text) if package_text else None
        package_name = self._package_name_input.text().strip() or DEFAULT_DNN_MPC_PACKAGE_NAME
        mcr_root_text = self._mcr_root_input.text().strip()
        mcr_root = Path(mcr_root_text) if mcr_root_text else None

        if not all(part.isidentifier() for part in package_name.split(".")):
            QMessageBox.warning(self, "包名错误", "MATLAB Python Package 名称必须是有效的 Python 模块名。")
            return

        package_available = False
        if package_dir is not None and (package_dir / package_name).is_dir():
            package_available = True
        elif importlib.util.find_spec(package_name) is not None:
            package_available = True

        if not package_available:
            QMessageBox.warning(
                self,
                "包缺失",
                f"未找到 MATLAB Python Package: {package_name}。\n"
                "请先安装 dnn_mpc/build_python，或把 Python包目录指向导出目录。",
            )
            return

        if mcr_root is None:
            QMessageBox.warning(self, "路径错误", "请填写 MATLAB Runtime R2024b 根目录 MCR_ROOT。")
            return

        if mode == "mpc" and not model_path.exists():
            QMessageBox.warning(self, "模型缺失", f"请先运行训练模块，或选择已有模型:\n{model_path}")
            return

        if mode in ("training", "pipeline") and dataset_path is not None and not dataset_path.exists():
            QMessageBox.warning(self, "数据集缺失", f"外部数据集文件不存在:\n{dataset_path}")
            return

        backend_label = "MATLAB Runtime Python Package"
        status_text = {
            "training": f"状态：{backend_label} 启动并运行 DNNTrain",
            "mpc": f"状态：{backend_label} 启动并运行 MPC simulation",
            "pipeline": f"状态：{backend_label} 启动与全流程运行中",
        }.get(mode, f"状态：{backend_label} 运行中")

        self._set_running(True)
        if mode in ("training", "pipeline"):
            self._set_image_mode("training")
        else:
            self._set_image_mode("trajectory")
        self._start_progress_polling(output_dir)
        self._status_label.setText(status_text)
        self._result_path_label.setText("运行中，完成后显示 JSON 结果文件路径。")
        self._image_label.setText("运行中...")
        self._image_label.setPixmap(QPixmap())

        self._thread = QThread(self)
        self._worker = ProcessControlDnnMpcWorker(
            mode,
            output_dir,
            model_path,
            self._sample_count_input.value() if self._sample_count_input is not None else 1000,
            self._epochs_input.value() if self._epochs_input is not None else 50,
            (self._hidden_layers_input.text().strip() if self._hidden_layers_input is not None else "64,64") or "64,64",
            dataset_path,
            self._sim_time_input.value() if self._sim_time_input is not None else 1.0,
            self._horizon_input.value() if self._horizon_input is not None else 5,
            package_dir,
            package_name,
            mcr_root,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._handle_success)
        self._worker.failed.connect(self._handle_failure)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker)
        self._thread.start()


    def _handle_success(self, payload: dict):
        mode = payload.get("mode", "pipeline")
        result = payload["result"]
        self._outputs = payload.get("outputs", {})
        json_path = payload.get("json_path", "")
        self._result_path_label.setText(f"结果已保存：{json_path}" if json_path else "结果已保存到输出目录。")
        self._stop_progress_polling()
        self._set_progress(100, "执行完成")

        model_path = self._outputs.get("model_mat")
        if model_path:
            self._model_path_input.setText(model_path)
            if mode == "training":
                self._save_latest_training_state(model_path, result, json_path)

        status_label = {
            "training": "训练模块完成",
            "mpc": "MPC simulation 完成",
            "pipeline": "全流程完成",
        }.get(mode, "执行完成")
        backend_label = {
            "matlab_python_package": "MATLAB Python Package",
        }.get(payload.get("backend"), payload.get("backend", ""))
        backend_suffix = f"（{backend_label}）" if backend_label else ""
        elapsed = result.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            self._status_label.setText(
                f"状态：{status_label}{backend_suffix}，用时 {elapsed:.2f} 秒，JSON输出 {json_path}"
            )
        else:
            self._status_label.setText(f"状态：{status_label}{backend_suffix}，JSON输出 {json_path}")

        if mode == "training":
            self._set_image_mode("training")
        else:
            self._set_image_mode("trajectory")
        self._set_running(False)

    def _handle_failure(self, error_text: str):
        self._outputs = {}
        self._stop_progress_polling()
        self._result_path_label.setText("执行失败，未生成 JSON 结果；错误详情见弹窗。")
        self._status_label.setText("状态：执行失败")
        self._set_progress(0, "执行失败")
        self._image_label.setText("暂无图像")
        self._image_label.setPixmap(QPixmap())
        self._set_running(False)
        QMessageBox.critical(self, "DNN-MPC 执行失败", error_text)

    def _clear_worker(self):
        self._thread = None
        self._worker = None

    def has_running_worker(self):
        return self._thread is not None and self._thread.isRunning()

    def closeEvent(self, event):  # noqa: N802
        if self.has_running_worker():
            QMessageBox.warning(self, "任务运行中", "DNN-MPC 任务仍在运行，完成后再关闭该页面。")
            event.ignore()
            return
        super().closeEvent(event)

    def _set_running(self, running: bool):
        widgets = [
            self._train_btn,
            self._mpc_btn,
        ]
        for widget in widgets:
            if widget is not None:
                widget.setEnabled(not running)

    def _start_progress_polling(self, output_dir: Path):
        self._progress_path = output_dir / "progress.json"
        self._last_progress_mtime = None
        if self._progress_path.exists():
            try:
                self._progress_path.unlink()
            except OSError:
                pass
        self._set_progress(0, "启动中")
        self._progress_timer.start()

    def _stop_progress_polling(self):
        self._progress_timer.stop()
        self._poll_progress()

    def _poll_progress(self):
        if self._progress_path is None or not self._progress_path.exists():
            return

        try:
            mtime = self._progress_path.stat().st_mtime
            if self._last_progress_mtime is not None and mtime == self._last_progress_mtime:
                return
            self._last_progress_mtime = mtime
            payload = json.loads(self._progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        percent = payload.get("percent", 0)
        message = payload.get("message", "运行中")
        module = payload.get("module", "")
        if module:
            self._status_label.setText(f"状态：{module} - {message}")
        else:
            self._status_label.setText(f"状态：{message}")
        self._set_progress(percent, message)

        outputs = payload.get("outputs")
        if isinstance(outputs, dict):
            self._outputs.update(outputs)
        self._refresh_image()

    def _set_progress(self, value, text: str):
        if self._progress_bar is None:
            return
        bounded = max(0, min(100, int(round(float(value)))))
        self._progress_bar.setValue(bounded)
        self._progress_bar.setFormat(f"{bounded}%  {text}")

    def _get_image_meta(self):
        output_dir = Path(self._output_dir_input.text().strip())
        mapping = {
            "training": (
                "训练曲线 (training_performance.png)",
                self._outputs.get("training_figure", str(output_dir / "training_performance.png")),
                "暂无训练曲线",
            ),
            "prediction": (
                "预测误差 (prediction_error.png)",
                self._outputs.get("prediction_error_figure", str(output_dir / "prediction_error.png")),
                "暂无预测误差图",
            ),
            "control": (
                "控制输入 (control_input.png)",
                self._outputs.get("control_figure", str(output_dir / "control_input.png")),
                "暂无控制输入图",
            ),
            "tracking": (
                "跟踪误差 (tracking_error.png)",
                self._outputs.get("tracking_error_figure", str(output_dir / "tracking_error.png")),
                "暂无跟踪误差图",
            ),
            "cost": (
                "代价曲线 (cost_curve.png)",
                self._outputs.get("cost_figure", str(output_dir / "cost_curve.png")),
                "暂无代价曲线",
            ),
        }
        return mapping.get(
            self._current_image_mode,
            (
                "状态轨迹 (process_control_trajectory.png)",
                self._outputs.get("trajectory_figure", str(output_dir / "process_control_trajectory.png")),
                "暂无状态轨迹图",
            ),
        )

    def _set_image_mode(self, mode: str):
        self._current_image_mode = mode
        for image_mode, button in self._image_buttons.items():
            button.setChecked(mode == image_mode)
        title, _, _ = self._get_image_meta()
        if self._image_title_label is not None:
            self._image_title_label.setText(title)
        self._refresh_image()

    def _refresh_image(self):
        _, image_path_text, empty_text = self._get_image_meta()
        image_path = Path(image_path_text)
        self._last_image_path = image_path
        if not image_path.exists():
            self._image_label.setText(empty_text)
            self._image_label.setPixmap(QPixmap())
            self._image_label.setToolTip(str(image_path))
            return

        QPixmapCache.clear()
        pixmap = QPixmap()
        pixmap.load(str(image_path))
        if pixmap.isNull():
            self._image_label.setText(f"图像加载失败:\n{image_path}")
            self._image_label.setPixmap(QPixmap())
            return

        scaled = pixmap.scaled(
            max(240, self._image_label.width() - 8),
            max(160, self._image_label.height() - 8),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setText("")
        self._image_label.setPixmap(scaled)
        self._image_label.setToolTip(str(image_path))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._last_image_path is not None:
            self._refresh_image()
