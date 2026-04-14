import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
# 参数传递见_apply_correlation_params函数
from correlation_analysis import SharedParameterStore


class MultiScenarioAnomalyDetectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace_dir = Path.cwd()
        self.core_output_dir = self.workspace_dir / "gridattackpkg_core_output"
        self.figure_dir = self.workspace_dir / "output_figures"
        self.last_result = None

        self._mcr_root_input = None
        self._percent_min_input = None
        self._percent_max_input = None
        self._sigma1_input = None
        self._sigma2_input = None
        self._use_correlation_params_checkbox = None
        self._apply_correlation_params_btn = None
        self._run_btn = None
        self._refresh_btn = None
        self._export_btn = None
        self._status_left_label = None
        self._status_right_label = None
        self._image_title_label = None
        self._image_view_label = None
        self._topology_switch_btn = None
        self._detection_switch_btn = None
        self._current_image_mode = "topology"
        self._result_preview = None

        self._build_ui()
        self._refresh_images()
        self._set_status("待执行")

    def _build_ui(self):
        self.setObjectName("anomalyRoot")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        left_root = QGroupBox("参数与导出设置")
        left_root_layout = QVBoxLayout(left_root)
        left_root_layout.setContentsMargins(10, 12, 10, 10)
        left_root_layout.setSpacing(10)

        param_group = QGroupBox("参数设置")
        param_layout = QFormLayout(param_group)
        param_layout.setVerticalSpacing(10)

        self._mcr_root_input = QLineEdit(r"D:\MATLAB\MATLAB Runtime\R2023a")
        param_layout.addRow("MATLAB Runtime路径:", self._mcr_root_input)

        percent_holder = QFrame()
        percent_layout = QHBoxLayout(percent_holder)
        percent_layout.setContentsMargins(0, 0, 0, 0)
        percent_layout.setSpacing(6)
        self._percent_min_input = self._new_spin(0.05, 0.5, 0.05)
        self._percent_max_input = self._new_spin(0.05, 0.5, 0.10)
        percent_layout.addWidget(self._percent_min_input, 1)
        percent_layout.addWidget(QLabel("~"))
        percent_layout.addWidget(self._percent_max_input, 1)
        param_layout.addRow("攻击幅度 [0.05, 0.5]:", percent_holder)

        self._sigma1_input = self._new_spin(0.01, 0.3, 0.2)
        param_layout.addRow("实部噪声 [0.01, 0.3]:", self._sigma1_input)

        self._sigma2_input = self._new_spin(0.01, 0.3, 0.1)
        param_layout.addRow("虚部噪声 [0.01, 0.3]:", self._sigma2_input)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        self._use_correlation_params_checkbox = QCheckBox("从异构数据治理-关联分析读取生成参数")
        source_row.addWidget(self._use_correlation_params_checkbox, 1)
        self._apply_correlation_params_btn = QPushButton("读取并应用")
        self._apply_correlation_params_btn.clicked.connect(self._apply_correlation_params)
        source_row.addWidget(self._apply_correlation_params_btn)
        param_layout.addRow("参数来源:", source_row)

        run_row = QHBoxLayout()
        run_row.addStretch()
        self._run_btn = QPushButton("运行异常结果")
        self._run_btn.clicked.connect(self._run_detection)
        run_row.addWidget(self._run_btn)
        run_row.addStretch()
        param_layout.addRow("", run_row)
        left_root_layout.addWidget(param_group)

        export_group = QGroupBox("导出设置")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(8)

        self._result_preview = QTextEdit()
        self._result_preview.setReadOnly(True)
        self._result_preview.setMinimumHeight(260)
        self._result_preview.setPlaceholderText(
            "执行完成后显示输出路径、运行时间与核心结果摘要。"
        )
        export_layout.addWidget(self._result_preview, 1)

        export_btn_row = QHBoxLayout()
        self._export_btn = QPushButton("导出结果JSON")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_result)
        export_btn_row.addWidget(self._export_btn)
        export_btn_row.addStretch()
        export_layout.addLayout(export_btn_row)
        left_root_layout.addWidget(export_group, 1)

        right_root = QGroupBox("结果图示")
        right_root_layout = QVBoxLayout(right_root)
        right_root_layout.setContentsMargins(10, 12, 10, 10)
        right_root_layout.setSpacing(10)

        switch_row = QHBoxLayout()
        switch_row.setSpacing(8)
        self._topology_switch_btn = QPushButton("网络拓扑图")
        self._topology_switch_btn.setCheckable(True)
        self._topology_switch_btn.setObjectName("imgSwitch")
        self._topology_switch_btn.clicked.connect(
            lambda _checked=False: self._set_image_mode("topology")
        )
        self._detection_switch_btn = QPushButton("检测概率图")
        self._detection_switch_btn.setCheckable(True)
        self._detection_switch_btn.setObjectName("imgSwitch")
        self._detection_switch_btn.clicked.connect(
            lambda _checked=False: self._set_image_mode("detection")
        )
        switch_row.addWidget(self._topology_switch_btn)
        switch_row.addWidget(self._detection_switch_btn)
        switch_row.addStretch()
        right_root_layout.addLayout(switch_row)

        image_panel, self._image_title_label, self._image_view_label = self._create_image_panel(
            "网络拓扑图 (topology.png)"
        )
        right_root_layout.addWidget(image_panel, 1)

        refresh_row = QHBoxLayout()
        refresh_row.addStretch()
        self._refresh_btn = QPushButton("刷新图像")
        self._refresh_btn.clicked.connect(self._refresh_images)
        refresh_row.addWidget(self._refresh_btn)
        right_root_layout.addLayout(refresh_row)

        self._set_image_mode("topology")

        content_row.addWidget(left_root, 1)
        content_row.addWidget(right_root, 1)
        main_layout.addLayout(content_row, 1)

        status_bar = QFrame()
        status_bar.setObjectName("anomalyStatusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 2, 10, 2)
        status_layout.setSpacing(6)
        self._status_left_label = QLabel("状态：待执行")
        self._status_left_label.setObjectName("statusLeft")
        self._status_right_label = QLabel("Status: Idle")
        self._status_right_label.setObjectName("statusRight")
        status_layout.addWidget(self._status_left_label)
        status_layout.addStretch()
        status_layout.addWidget(self._status_right_label)
        main_layout.addWidget(status_bar)

        self.setStyleSheet(
            """
            QWidget#anomalyRoot {
                background: #1a2635;
            }
            QWidget#anomalyRoot QGroupBox {
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                background: rgba(31, 49, 70, 0.88);
                color: #d4e8ff;
                font-size: 16px;
                font-weight: 600;
            }
            QWidget#anomalyRoot QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #d9ecff;
            }
            QWidget#anomalyRoot QLabel {
                color: #d2e5fb;
                font-size: 14px;
            }
            QWidget#anomalyRoot QCheckBox {
                color: #d2e5fb;
                font-size: 14px;
            }
            QWidget#anomalyRoot QLineEdit,
            QWidget#anomalyRoot QTextEdit,
            QWidget#anomalyRoot QDoubleSpinBox {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.95);
                color: #e7f2ff;
                min-height: 28px;
                padding: 2px 8px;
            }
            QWidget#anomalyRoot QPushButton {
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
            QWidget#anomalyRoot QPushButton#imgSwitch {
                min-width: 150px;
                padding: 6px 12px;
            }
            QWidget#anomalyRoot QPushButton#imgSwitch:checked {
                border: 1px solid rgba(146, 212, 255, 0.78);
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(68, 147, 206, 0.98),
                    stop: 1 rgba(39, 99, 153, 0.98)
                );
                color: #ffffff;
            }
            QWidget#anomalyRoot QPushButton:hover {
                background: rgba(62, 122, 174, 0.95);
                color: #ffffff;
            }
            QWidget#anomalyRoot QPushButton:disabled {
                color: #87a2bd;
                border-color: rgba(110, 140, 170, 0.35);
                background: rgba(33, 55, 77, 0.75);
            }
            QWidget#anomalyRoot QFrame#anomalyStatusBar {
                border: 1px solid rgba(126, 168, 208, 0.35);
                border-radius: 6px;
                background: rgba(25, 38, 55, 0.95);
            }
            QWidget#anomalyRoot QLabel#statusLeft,
            QWidget#anomalyRoot QLabel#statusRight {
                color: #cfe3fb;
                font-size: 13px;
                font-weight: 600;
            }
            QWidget#anomalyRoot QLabel#imagePanelTitle {
                font-size: 22px;
                color: #e4f2ff;
                font-weight: 650;
                padding: 2px 0 2px 2px;
            }
            """
        )

    def _new_spin(self, minimum, maximum, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        return spin

    def _create_image_panel(self, title):
        panel = QFrame()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("imagePanelTitle")
        panel_layout.addWidget(title_label)

        image_label = QLabel("暂无图像")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumHeight(470)
        image_label.setStyleSheet(
            "border: 1px solid rgba(129, 169, 206, 0.45);"
            "background: rgba(18, 32, 48, 0.92);"
            "border-radius: 5px;"
            "color: #bcd8f6;"
        )
        panel_layout.addWidget(image_label, 1)
        return panel, title_label, image_label

    def _set_status(self, text):
        if self._status_left_label is not None:
            self._status_left_label.setText(f"状态：{text}")
        if self._status_right_label is not None:
            self._status_right_label.setText(f"Status: {text}")

    def _get_image_meta(self):
        if self._current_image_mode == "detection":
            return (
                "检测概率图 (detection_probability.png)",
                self.figure_dir / "detection_probability.png",
                "暂无检测概率图",
            )
        return (
            "网络拓扑图 (topology.png)",
            self.figure_dir / "topology.png",
            "暂无拓扑图",
        )

    def _set_image_mode(self, mode):
        self._current_image_mode = "detection" if mode == "detection" else "topology"
        if self._topology_switch_btn is not None:
            self._topology_switch_btn.setChecked(self._current_image_mode == "topology")
        if self._detection_switch_btn is not None:
            self._detection_switch_btn.setChecked(self._current_image_mode == "detection")
        title, _, _ = self._get_image_meta()
        if self._image_title_label is not None:
            self._image_title_label.setText(title)
        self._refresh_images()

    def _set_running(self, running):
        self._run_btn.setEnabled(not running)
        self._refresh_btn.setEnabled(not running)
        self._export_btn.setEnabled((not running) and self.last_result is not None)
        if self._topology_switch_btn is not None:
            self._topology_switch_btn.setEnabled(not running)
        if self._detection_switch_btn is not None:
            self._detection_switch_btn.setEnabled(not running)

    def _prepare_runtime_paths(self, mcr_root: Path):
        runtime_dir = mcr_root / "runtime" / "win64"
        bin_dir = mcr_root / "bin" / "win64"
        extern_dir = mcr_root / "extern" / "bin" / "win64"
        dll_path = runtime_dir / "mclmcrrt9_14.dll"

        if not dll_path.exists():
            raise FileNotFoundError(f"未找到 MATLAB Runtime 9.14 DLL: {dll_path}")

        current = os.environ.get("PATH", "")
        current_parts = current.split(os.pathsep) if current else []
        prepend_parts = []
        for part in (runtime_dir, bin_dir, extern_dir):
            part_str = str(part)
            if not any(part_str.lower() == p.lower() for p in current_parts):
                prepend_parts.append(part_str)
        if prepend_parts:
            os.environ["PATH"] = os.pathsep.join(prepend_parts + current_parts)
        return dll_path

    def _validate_inputs(self):
        percent_min = self._percent_min_input.value()
        percent_max = self._percent_max_input.value()
        sigma1 = self._sigma1_input.value()
        sigma2 = self._sigma2_input.value()

        if percent_min > percent_max:
            raise ValueError("攻击幅度最小值不能大于最大值。")
        return percent_min, percent_max, sigma1, sigma2

    def _apply_correlation_params(self, notify_if_missing=True):
        params = SharedParameterStore.get_correlation_params()
        if not params:
            if notify_if_missing:
                QMessageBox.information(
                    self,
                    "无可用参数",
                    "尚未从“异构数据治理 - 关联分析”模块生成参数。",
                )
            return False

        self._percent_min_input.setValue(params["percent_min"])
        self._percent_max_input.setValue(params["percent_max"])
        self._sigma1_input.setValue(params["sigma1"])
        self._sigma2_input.setValue(params["sigma2"])
        self._set_status(f"已读取关联分析参数 ({params['timestamp']})")
        return True

    def _import_runtime_modules(self):
        candidate_paths = []

        if getattr(sys, "frozen", False):
            meipass = Path(getattr(sys, "_MEIPASS", ""))
            if str(meipass):
                candidate_paths.extend(
                    [
                        meipass,
                        meipass / "gridattackpkg",
                        Path(sys.executable).resolve().parent,
                    ]
                )

        candidate_paths.extend(
            [
                self.workspace_dir / "build_python",
                self.workspace_dir / ".venv_fw8_mcr914" / "Lib" / "site-packages",
            ]
        )

        for path in candidate_paths:
            path_str = str(path)
            if path_str and path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)

        try:
            gridattackpkg = importlib.import_module("gridattackpkg")
        except ModuleNotFoundError as exc:
            checked = [str(p) for p in candidate_paths if str(p)]
            raise ModuleNotFoundError(
                "未找到模块 gridattackpkg。已尝试路径: "
                + " | ".join(checked)
            ) from exc

        matlab = importlib.import_module("matlab")
        return gridattackpkg, matlab

    def _run_detection_with_external_python(self, mcr_root: Path, percent_min, percent_max, sigma1, sigma2):
        runner = self.workspace_dir / "run_core_plot_with_three_params.py"
        if not runner.exists():
            raise FileNotFoundError(f"未找到外部运行脚本: {runner}")

        python_candidates = [
            self.workspace_dir / ".venv_fw8_mcr914" / "python.exe",
            Path(sys.executable).resolve().parent / ".venv_fw8_mcr914" / "python.exe",
        ]
        python_exe = next((p for p in python_candidates if p.exists()), None)
        if python_exe is None:
            checked = " | ".join(str(p) for p in python_candidates)
            raise FileNotFoundError(f"未找到可用 Python 3.10 环境: {checked}")

        cmd = [
            str(python_exe),
            str(runner),
            "--percent-min",
            str(percent_min),
            "--percent-max",
            str(percent_max),
            "--sigma1",
            str(sigma1),
            "--sigma2",
            str(sigma2),
            "--mcr-root",
            str(mcr_root),
            "--core-output-dir",
            str(self.core_output_dir),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(self.workspace_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        merged_output = "\n".join(
            part for part in [proc.stdout.strip(), proc.stderr.strip()] if part
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"外部 Python 运行失败(退出码={proc.returncode})。\n{merged_output}"
            )
        return {
            "python_exe": str(python_exe),
            "script": str(runner),
            "output": merged_output,
        }

    def _run_detection(self):
        if (
            self._use_correlation_params_checkbox is not None
            and self._use_correlation_params_checkbox.isChecked()
        ):
            if not self._apply_correlation_params(notify_if_missing=True):
                return

        try:
            percent_min, percent_max, sigma1, sigma2 = self._validate_inputs()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        mcr_root_text = self._mcr_root_input.text().strip()
        if not mcr_root_text:
            QMessageBox.warning(self, "参数错误", "请填写 MATLAB Runtime 根目录。")
            return

        self._set_running(True)
        self._set_status("运行中")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        handle = None
        try:
            mcr_root = Path(mcr_root_text)
            dll_path = self._prepare_runtime_paths(mcr_root)
            self.core_output_dir.mkdir(parents=True, exist_ok=True)
            self.figure_dir.mkdir(parents=True, exist_ok=True)

            started = time.perf_counter()
            run_mode = "inprocess"
            external_info = None
            try:
                gridattackpkg, matlab = self._import_runtime_modules()

                overrides = {
                    "percent_range": matlab.double([percent_min, percent_max]),
                    "sigma1": sigma1,
                    "sigma2": sigma2,
                }

                handle = gridattackpkg.initialize()
                params = handle.make_default_params(overrides)
                core_out = handle.run_core(params, str(self.core_output_dir))
                handle.plot_results_from_core(core_out, nargout=0)
                run_core_output_type = str(type(core_out).__name__)
            except Exception as inner_exc:  # noqa: BLE001
                msg = str(inner_exc)
                if ("not supported" in msg and "Python" in msg) or "Python 3.11" in msg:
                    run_mode = "external_python"
                    external_info = self._run_detection_with_external_python(
                        mcr_root, percent_min, percent_max, sigma1, sigma2
                    )
                    run_core_output_type = "external_runner"
                else:
                    raise
            elapsed = time.perf_counter() - started

            stage2_file = self.core_output_dir / "stage2_results.mat"
            topology_png = self.figure_dir / "topology.png"
            detection_png = self.figure_dir / "detection_probability.png"
            missing = [str(p) for p in (stage2_file, topology_png, detection_png) if not p.exists()]
            if missing:
                raise RuntimeError(f"输出文件缺失: {missing}")

            self._refresh_images()
            self.last_result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mcr_root": str(mcr_root),
                "mcr_dll": str(dll_path),
                "overrides": {
                    "percent_range": [percent_min, percent_max],
                    "sigma1": sigma1,
                    "sigma2": sigma2,
                },
                "outputs": {
                    "stage2_results_mat": str(stage2_file),
                    "topology_png": str(topology_png),
                    "detection_probability_png": str(detection_png),
                },
                "run_mode": run_mode,
                "run_core_output_type": run_core_output_type,
                "elapsed_seconds": round(elapsed, 2),
            }
            if external_info is not None:
                self.last_result["external_runner"] = external_info
            self._result_preview.setText(
                json.dumps(self.last_result, ensure_ascii=False, indent=2)
            )
            self._set_status("执行完成")
        except Exception as exc:  # noqa: BLE001
            self.last_result = None
            self._result_preview.setText(f"执行失败: {type(exc).__name__}: {exc}")
            self._set_status("执行失败")
            QMessageBox.critical(self, "异常检测执行失败", f"{type(exc).__name__}: {exc}")
        finally:
            if handle is not None:
                try:
                    handle.terminate()
                except Exception:  # noqa: BLE001
                    pass
            self._set_running(False)
            QApplication.restoreOverrideCursor()

    def _load_image(self, label, image_path: Path, empty_tip: str):
        if not image_path.exists():
            self._set_black_placeholder(label, empty_tip)
            label.setToolTip("")
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._set_black_placeholder(label, f"{empty_tip}\n(加载失败)")
            label.setToolTip(str(image_path))
            return

        target_width = max(240, label.width() - 8)
        target_height = max(160, label.height() - 8)
        scaled = pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(scaled)
        label.setToolTip(str(image_path))

    def _set_black_placeholder(self, label, tip_text):
        width = max(320, label.width() if label.width() > 0 else 640)
        height = max(180, label.height() if label.height() > 0 else 220)
        canvas = QPixmap(width, height)
        canvas.fill(QColor("#132335"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#63b9ff"))
        painter.drawEllipse(QPointF(width * 0.15, height * 0.22), 2.2, 2.2)
        painter.drawEllipse(QPointF(width * 0.82, height * 0.78), 3.0, 3.0)
        painter.setPen(QColor("#8ecbff"))
        painter.drawLine(
            QPointF(width * 0.82 - 6, height * 0.78),
            QPointF(width * 0.82 + 6, height * 0.78),
        )
        painter.drawLine(
            QPointF(width * 0.82, height * 0.78 - 6),
            QPointF(width * 0.82, height * 0.78 + 6),
        )
        painter.end()
        label.setPixmap(canvas)
        label.setText(tip_text)

    def _refresh_images(self):
        if self._image_view_label is None:
            return
        _, image_path, empty_tip = self._get_image_meta()
        self._load_image(self._image_view_label, image_path, empty_tip)

    def _export_result(self):
        if not self.last_result:
            QMessageBox.information(self, "无可导出结果", "请先执行异常检测。")
            return

        default_name = f"anomaly_result_{datetime.now():%Y%m%d_%H%M%S}.json"
        default_path = str(self.workspace_dir / default_name)
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "导出结果JSON",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_name:
            return

        Path(file_name).write_text(
            json.dumps(self.last_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        QMessageBox.information(self, "导出成功", f"结果已导出到:\n{file_name}")
