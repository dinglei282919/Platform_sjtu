import csv
import importlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, Signal
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
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from correlation_analysis import SharedParameterStore


class SecondOrderSimulationWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            payload = self._run_simulations()
            self.finished.emit(payload)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())

    def _run_simulations(self):
        model_name = self.config["model_name"]
        base_dir = Path(self.config["package_dir"]).resolve()
        output_dir = Path(self.config["output_dir"]).resolve()
        mcr_root = Path(self.config["mcr_root"]).resolve() if self.config.get("mcr_root") else None
        output_dir.mkdir(parents=True, exist_ok=True)

        if not base_dir.is_dir():
            raise FileNotFoundError(f"Python package directory does not exist: {base_dir}")
        if mcr_root is not None and not mcr_root.exists():
            raise FileNotFoundError(f"MCR_ROOT does not exist: {mcr_root}")

        base_dir_text = str(base_dir)
        if base_dir_text not in sys.path:
            sys.path.insert(0, base_dir_text)

        simulate_model = importlib.import_module("simulate_model")
        self.progress.emit(5, "Initializing MATLAB Runtime package")
        mdl = simulate_model.load_and_init_pkg(model_name, base_dir=base_dir, mcr_root=mcr_root)

        u = self._build_input_signal()
        tunable_params = {
            "dx2min": float(self.config["dx2min"]),
            "dx2max": float(self.config["dx2max"]),
        }
        scenarios = [
            ("default", "default parameter values", {}),
            ("limits", "dx2min/dx2max limits", {"TunableParameters": tunable_params}),
            ("input", "external input u", {"ExternalInput": u}),
            ("limits_input", "limits and external input u", {"TunableParameters": tunable_params, "ExternalInput": u}),
        ]

        results = {}
        try:
            for index, (key, label, extra_args) in enumerate(scenarios, start=1):
                self.progress.emit(10 + index * 18, f"Running simulation {index}/4: {label}")
                call_args = ["ModelName", model_name]
                for arg_name, arg_value in extra_args.items():
                    call_args.extend([arg_name, arg_value])
                raw_result = mdl.simulate(*call_args)
                results[key] = {
                    "label": label,
                    "signals": self._extract_signals(raw_result),
                }
        finally:
            try:
                mdl.terminate()
            except Exception:  # noqa: BLE001
                pass

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name,
            "parameters": {
                "dx2min": float(self.config["dx2min"]),
                "dx2max": float(self.config["dx2max"]),
                "positive_input": float(self.config["positive_input"]),
                "negative_input": float(self.config["negative_input"]),
                "input_segments": {
                    "leading_zero": int(self.config["leading_zero"]),
                    "positive_width": int(self.config["positive_width"]),
                    "middle_zero": int(self.config["middle_zero"]),
                    "negative_width": int(self.config["negative_width"]),
                    "trailing_zero": int(self.config["trailing_zero"]),
                },
            },
            "package_dir": str(base_dir),
            "mcr_root": str(mcr_root) if mcr_root is not None else "",
            "scenarios": results,
            "export_files": {
                "json": str(output_dir / "second_order_dynamic_system_results.json"),
                "csv": str(output_dir / "second_order_dynamic_system_results.csv"),
                "plot_png": str(output_dir / "second_order_dynamic_system_plot.png"),
            },
        }
        self.progress.emit(92, "Exporting result data")
        self._write_json(payload)
        self._write_csv(payload)
        self.progress.emit(100, "Simulation complete")
        return payload

    def _build_input_signal(self):
        lead = np.zeros(int(self.config["leading_zero"]))
        positive = float(self.config["positive_input"]) * np.ones(int(self.config["positive_width"]))
        middle = np.zeros(int(self.config["middle_zero"]))
        negative = float(self.config["negative_input"]) * np.ones(int(self.config["negative_width"]))
        tail = np.zeros(int(self.config["trailing_zero"]))
        return np.concatenate([lead, positive, middle, negative, tail])

    def _extract_signals(self, result):
        signals = {}
        for name in ("x1", "x2", "u"):
            try:
                signal = result[name]
                signals[name] = {
                    "time": self._to_float_list(signal["Time"]),
                    "data": self._to_float_list(signal["Data"]),
                }
            except Exception:  # noqa: BLE001
                continue
        return signals

    def _to_float_list(self, value):
        array = np.asarray(value, dtype=float).reshape(-1)
        return [float(item) for item in array.tolist()]

    def _write_json(self, payload):
        path = Path(payload["export_files"]["json"])
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_csv(self, payload):
        path = Path(payload["export_files"]["csv"])
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scenario", "scenario_label", "signal", "index", "time", "data"])
            for scenario_key, scenario in payload["scenarios"].items():
                for signal_name, signal in scenario.get("signals", {}).items():
                    times = signal.get("time", [])
                    data = signal.get("data", [])
                    for index, value in enumerate(data):
                        time_value = times[index] if index < len(times) else ""
                        writer.writerow([scenario_key, scenario.get("label", ""), signal_name, index, time_value, value])


class SecondOrderDynamicSystemWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace_dir = Path(__file__).resolve().parent
        self.example_dir = self.workspace_dir / "run-deployed-simulations-using-python"
        self._model_name_input = None
        self._package_dir_input = None
        self._mcr_root_input = None
        self._output_dir_input = None
        self._dx2min_input = None
        self._dx2max_input = None
        self._positive_input = None
        self._negative_input = None
        self._leading_zero_input = None
        self._positive_width_input = None
        self._middle_zero_input = None
        self._negative_width_input = None
        self._trailing_zero_input = None
        self._run_btn = None
        self._status_label = None
        self._progress_bar = None
        self._result_box = None
        self._figure = Figure(figsize=(8, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._thread = None
        self._worker = None
        self._last_payload = None
        self._build_ui()
        self._draw_empty_plot()

    def _build_ui(self):
        self.setObjectName("secondOrderRoot")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("secondOrderSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(12)

        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QFrame.NoFrame)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setMinimumWidth(420)
        config_scroll.setWidget(self._build_config_panel())

        result_panel = self._build_result_panel()
        splitter.addWidget(config_scroll)
        splitter.addWidget(result_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([560, 1000])
        root_layout.addWidget(splitter, 1)

        self.setStyleSheet(
            """
            QWidget#secondOrderRoot {
                background: #1a2635;
            }
            QWidget#secondOrderRoot QGroupBox {
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 12px;
                background: rgba(31, 49, 70, 0.88);
                color: #d4e8ff;
                font-size: 15px;
                font-weight: 600;
            }
            QWidget#secondOrderRoot QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #d9ecff;
            }
            QWidget#secondOrderRoot QLabel {
                color: #d2e5fb;
                font-size: 14px;
            }
            QWidget#secondOrderRoot QLineEdit,
            QWidget#secondOrderRoot QSpinBox,
            QWidget#secondOrderRoot QDoubleSpinBox,
            QWidget#secondOrderRoot QTextEdit {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.95);
                color: #e7f2ff;
                min-height: 30px;
                padding: 2px 8px;
                font-size: 14px;
            }
            QWidget#secondOrderRoot QPushButton {
                border: 1px solid rgba(101, 175, 235, 0.52);
                border-radius: 8px;
                background: rgba(35, 90, 132, 0.95);
                color: #dff2ff;
                padding: 7px 12px;
                font-weight: 700;
                font-size: 14px;
            }
            QWidget#secondOrderRoot QPushButton:hover {
                background: rgba(62, 122, 174, 0.95);
            }
            QWidget#secondOrderRoot QPushButton:disabled {
                background: rgba(52, 67, 84, 0.8);
                color: #8da3b9;
            }
            QWidget#secondOrderRoot QProgressBar {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.95);
                color: #e7f2ff;
                text-align: center;
                min-height: 26px;
            }
            QWidget#secondOrderRoot QProgressBar::chunk {
                background: rgba(88, 166, 255, 0.88);
                border-radius: 4px;
            }
            QWidget#secondOrderRoot QSplitter::handle {
                background: rgba(126, 168, 208, 0.1);
            }
            """
        )

    def _build_config_panel(self):
        panel = QGroupBox("二阶非线性动态系统")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        path_group = QGroupBox("路径设置")
        path_form = QFormLayout(path_group)
        path_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        path_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._model_name_input = QLineEdit("model1")
        path_form.addRow("模型名:", self._model_name_input)

        self._package_dir_input = QLineEdit(str(self.example_dir))
        path_form.addRow("Python包目录:", self._make_dir_row(self._package_dir_input))

        self._mcr_root_input = QLineEdit(r"E:\MATLAB2024")
        path_form.addRow("MCR_ROOT:", self._make_dir_row(self._mcr_root_input))

        self._output_dir_input = QLineEdit(str(self.example_dir / "output"))
        path_form.addRow("输出目录:", self._make_dir_row(self._output_dir_input))
        layout.addWidget(path_group)

        parameter_group = QGroupBox("仿真参数")
        parameter_form = QFormLayout(parameter_group)
        parameter_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        parameter_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._dx2min_input = self._new_float_spin(-20.0, 20.0, -3.0, 0.1)
        self._dx2max_input = self._new_float_spin(-20.0, 20.0, 4.0, 0.1)
        parameter_form.addRow("dx2min:", self._dx2min_input)
        parameter_form.addRow("dx2max:", self._dx2max_input)

        self._positive_input = self._new_float_spin(-20.0, 20.0, 2.0, 0.1)
        self._negative_input = self._new_float_spin(-20.0, 20.0, -2.0, 0.1)
        parameter_form.addRow("正输入幅值:", self._positive_input)
        parameter_form.addRow("负输入幅值:", self._negative_input)

        self._leading_zero_input = self._new_int_spin(0, 100, 1, 1)
        self._positive_width_input = self._new_int_spin(1, 100, 1, 1)
        self._middle_zero_input = self._new_int_spin(0, 100, 3, 1)
        self._negative_width_input = self._new_int_spin(1, 100, 2, 1)
        self._trailing_zero_input = self._new_int_spin(0, 100, 1, 1)
        parameter_form.addRow("前置零时长:", self._leading_zero_input)
        parameter_form.addRow("正输入时长:", self._positive_width_input)
        parameter_form.addRow("中间零时长:", self._middle_zero_input)
        parameter_form.addRow("负输入时长:", self._negative_width_input)
        parameter_form.addRow("尾部零时长:", self._trailing_zero_input)
        layout.addWidget(parameter_group)

        self._run_btn = QPushButton("运行四组仿真")
        self._run_btn.clicked.connect(self._start_simulation)
        layout.addWidget(self._run_btn)

        self._status_label = QLabel("状态: 待执行")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("待执行")
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
        layout.addStretch()
        return panel

    def _build_result_panel(self):
        panel = QGroupBox("仿真结果")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas, 1)

        self._result_box = QTextEdit()
        self._result_box.setReadOnly(True)
        self._result_box.setMinimumHeight(105)
        self._result_box.setText("运行完成后显示 JSON/CSV/PNG 导出路径。")
        layout.addWidget(self._result_box)
        return panel

    def _make_dir_row(self, line_edit: QLineEdit):
        line_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        row = QFrame()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        browse_btn = QPushButton("选择")
        browse_btn.setFixedWidth(56)
        browse_btn.clicked.connect(lambda _checked=False, edit=line_edit: self._browse_dir(edit))
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse_btn)
        return row

    def _browse_dir(self, line_edit: QLineEdit):
        current = line_edit.text().strip() or str(self.workspace_dir)
        selected = QFileDialog.getExistingDirectory(self, "选择目录", current)
        if selected:
            line_edit.setText(selected)

    def _new_float_spin(self, minimum, maximum, value, step):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    def _new_int_spin(self, minimum, maximum, value, step):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    def _start_simulation(self):
        if self.has_running_worker():
            QMessageBox.information(self, "任务运行中", "当前仿真任务尚未结束，请等待完成后再操作。")
            return

        model_name = self._model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "模型名缺失", "请填写模型名。")
            return

        config = {
            "model_name": model_name,
            "package_dir": self._package_dir_input.text().strip(),
            "mcr_root": self._mcr_root_input.text().strip(),
            "output_dir": self._output_dir_input.text().strip(),
            "dx2min": self._dx2min_input.value(),
            "dx2max": self._dx2max_input.value(),
            "positive_input": self._positive_input.value(),
            "negative_input": self._negative_input.value(),
            "leading_zero": self._leading_zero_input.value(),
            "positive_width": self._positive_width_input.value(),
            "middle_zero": self._middle_zero_input.value(),
            "negative_width": self._negative_width_input.value(),
            "trailing_zero": self._trailing_zero_input.value(),
        }
        if not config["package_dir"] or not config["output_dir"]:
            QMessageBox.warning(self, "路径缺失", "请填写 Python 包目录和输出目录。")
            return

        self._set_running(True)
        self._set_progress(0, "启动中")
        self._status_label.setText("状态: MATLAB Runtime 仿真启动中")
        self._result_box.setText("仿真运行中...")

        self._thread = QThread(self)
        self._worker = SecondOrderSimulationWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._set_progress)
        self._worker.finished.connect(self._handle_success)
        self._worker.failed.connect(self._handle_failure)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker)
        self._thread.start()

    def _handle_success(self, payload: dict):
        self._last_payload = payload
        self._draw_payload(payload)
        plot_path = Path(payload["export_files"]["plot_png"])
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(plot_path, dpi=160)
        SharedParameterStore.set_second_order_simulation_result(payload)
        self._result_box.setText(
            "仿真完成，结果已导出:\n"
            f"JSON: {payload['export_files']['json']}\n"
            f"CSV: {payload['export_files']['csv']}\n"
            f"PNG: {payload['export_files']['plot_png']}"
        )
        self._status_label.setText(f"状态: 仿真完成，时间 {payload.get('timestamp', '')}")
        self._set_progress(100, "完成")
        self._set_running(False)

    def _handle_failure(self, error_text: str):
        self._status_label.setText("状态: 仿真失败")
        self._result_box.setText("仿真失败，详细错误已在弹窗显示。")
        self._set_progress(0, "失败")
        self._set_running(False)
        QMessageBox.critical(self, "二阶非线性动态系统仿真失败", error_text)

    def _draw_empty_plot(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_title("Second-order nonlinear dynamic system")
        ax.set_xlabel("Time")
        ax.set_ylabel("x1")
        ax.grid(True, alpha=0.35)
        ax.text(0.5, 0.5, "No simulation result", ha="center", va="center", transform=ax.transAxes)
        self._canvas.draw()

    def _draw_payload(self, payload: dict):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("Time")
        ax.set_ylabel("x1")
        ax.set_title("x1 from four simulations")

        colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        plotted_u = False
        ax_u = None
        for index, (key, scenario) in enumerate(payload.get("scenarios", {}).items()):
            x1 = scenario.get("signals", {}).get("x1")
            if x1:
                ax.plot(
                    x1.get("time", []),
                    x1.get("data", []),
                    linewidth=2,
                    color=colors[index % len(colors)],
                    label=scenario.get("label", key),
                )
            u = scenario.get("signals", {}).get("u")
            if u and not plotted_u:
                ax_u = ax.twinx()
                ax_u.step(
                    u.get("time", []),
                    u.get("data", []),
                    where="post",
                    linewidth=1.8,
                    color=colors[(index + 2) % len(colors)],
                    label="input u",
                )
                ax_u.set_ylabel("u")
                plotted_u = True

        handles, labels = ax.get_legend_handles_labels()
        if ax_u is not None:
            handles_u, labels_u = ax_u.get_legend_handles_labels()
            handles += handles_u
            labels += labels_u
        if handles:
            ax.legend(handles, labels, loc="best")
        self._canvas.draw()

    def _set_progress(self, value, text: str):
        bounded = max(0, min(100, int(value)))
        self._progress_bar.setValue(bounded)
        self._progress_bar.setFormat(f"{bounded}%  {text}")
        self._status_label.setText(f"状态: {text}")

    def _set_running(self, running: bool):
        if self._run_btn is not None:
            self._run_btn.setEnabled(not running)

    def _clear_worker(self):
        self._thread = None
        self._worker = None

    def has_running_worker(self):
        return self._thread is not None and self._thread.isRunning()

    def closeEvent(self, event):  # noqa: N802
        if self.has_running_worker():
            QMessageBox.warning(self, "任务运行中", "二阶非线性动态系统仿真仍在运行，请完成后再关闭。")
            event.ignore()
            return
        super().closeEvent(event)
