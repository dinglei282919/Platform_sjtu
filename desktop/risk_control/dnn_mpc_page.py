"""DNN-MPC training and live control tabs for the desktop CSTR workspace."""
import json
import math
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QProgressBar, QSizePolicy, QWidget)

from platform_core.risk_control import presentation
from platform_core.risk_control.cstr import SAMPLE_MIN, TREND_INTERVALS, TREND_MAX_INTERVALS
from platform_core.risk_control.disturbance import options as disturbance_options
from platform_core.risk_control.dnn_mpc import dataset
from platform_core.risk_control.dnn_mpc.mpc import Settings
from platform_core.risk_control.service import STAGES, TRAIN_DEFAULTS

from .controls import ControlComboBox, ControlTabWidget
from .layout import FieldGrid, FlowRow, RunningValue, StagePage, StatusStrip, action_button, choice, field, label
from .process_diagram import ProcessDiagram
from .trends import Trend

STATUS = {"ready": "就绪", "running": "运行中", "paused": "已暂停", "ended": "已结束", "queued": "排队中",
          "succeeded": "已完成", "failed": "失败", "cancelled": "已取消"}


WINDOW_INTERVALS = (15, 30, 60, TREND_INTERVALS, 240, TREND_MAX_INTERVALS)
PARAMETER_HELP = json.loads(
    (Path(__file__).resolve().parents[2] / "assets/cstr/parameter-help.json").read_text(encoding="utf-8")
)


def describe_number(form, key, box):
    period_seconds = 60 * SAMPLE_MIN
    bounds = dict(min=f"{box.minimum():f}".rstrip("0").rstrip("."),
                  max=f"{box.maximum():f}".rstrip("0").rstrip("."),
                  period=f"{period_seconds:g}")
    template = PARAMETER_HELP[key]
    if key == "speed":
        hint = form.add_hint(box, "")

        def show_period(speed):
            seconds = round(period_seconds / speed, 2)
            text = template.format(**bounds, wall=f"，按此倍率约用 {seconds:g} 秒")
            hint.setText(text)
            box.setAccessibleDescription(text)

        show_period(box.value())
        box.valueChanged.connect(show_period)
    else:
        form.add_hint(box, template.format(**bounds))


class DnnMpcPage(ControlTabWidget):
    def __init__(self, service):
        super().__init__()
        self.setAccessibleName("深度神经网络优化控制工作流程")
        self.service = service
        self.session = None
        self.training_id = self.update_id = None
        self.rows = []
        self.pinned = None
        self.window_intervals = TREND_INTERVALS
        self.archive = None
        self.csv_path = None
        self.last_seq = -1
        self.last_run = None
        self.closed = False
        self._training_page()
        self._control_page()
        self.setCurrentIndex(1)
        self.currentChanged.connect(self._tab_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(300)

    def open_control(self):
        if self.currentIndex() == 1:
            self._tab_changed()
        else:
            self.setCurrentIndex(1)

    def _button(self, text, fn, role="primary"):
        return action_button(text, lambda: self._perform(fn), role)

    def _perform(self, fn):
        try:
            fn()
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            if self.currentIndex() == 0:
                self.train_msg.show_state("failed", "失败", str(exc))
            else:
                self.control_msg.setText(str(exc))
                self.control_msg.show()

    def _training_page(self):
        page = StagePage()
        self.addTab(page, STAGES[0])
        left, pane = page.parameters, page.display
        left.addWidget(label("模型训练配置", "heading"))
        self.csv_button = self._button("选择文件", self._choose_csv, "small")
        self.csv_button.setAccessibleName("外部训练数据（CSV，可选）")
        csv_label = label("外部训练数据（CSV，可选）", "fieldLabel")
        csv_label.setBuddy(self.csv_button)
        left.addWidget(csv_label)
        file_box = QFrame()
        file_box.setObjectName("fileField")
        file_row = QHBoxLayout(file_box)
        file_row.setContentsMargins(5, 3, 8, 3)
        file_row.setSpacing(8)
        file_row.addWidget(self.csv_button)
        self.file_name = label("尚未选择文件", "fieldLabel")
        self.file_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        file_row.addWidget(self.file_name, 1)
        left.addWidget(file_box)
        self.data_label = label("默认使用机理生成的 CSTR 轨迹。", "guide")
        self.data_label.setWordWrap(False)
        self.clear_csv = self._button("改用生成轨迹", lambda: self._set_csv(None), "small")
        self.clear_csv.hide()
        left.addLayout(FlowRow(self.data_label, self._button("下载 CSV 示例", self._save_example, "link"),
                               self.clear_csv))

        self.training_grid = form = FieldGrid(left)
        self.train_fields = {
            "trajectories": field(form, "轨迹数", 24, 600, TRAIN_DEFAULTS["trajectories"]),
            "length": field(form, "每条轨迹周期", 30, 120, TRAIN_DEFAULTS["length"]),
            "epochs": field(form, "训练轮数", 1, 2000, TRAIN_DEFAULTS["epochs"]),
            "lr": field(form, "学习率", 0.00001, 0.05, TRAIN_DEFAULTS["lr"], 0.0001, 5),
            "seed": field(form, "随机种子", 0, 2147483647, TRAIN_DEFAULTS["seed"])}
        for key, box in self.train_fields.items():
            describe_number(form, key, box)
        self.train_device = choice(form, "训练设备", ("auto", "cpu", "cuda"), wide=False)
        form.add_hint(self.train_device, PARAMETER_HELP["device"])
        self.submit_train = self._button("生成轨迹并训练", self._train)
        self.cancel_train = self._button("取消训练", lambda: self.service.cancel(self.training_id), "secondary")
        self.cancel_train.setEnabled(False)
        left.addLayout(FlowRow(self.submit_train, self.cancel_train))

        pane.addWidget(label("模型训练进度", "heading"))
        self.train_msg = StatusStrip()
        self.train_msg.show_state("ready", STATUS["ready"])
        pane.addWidget(self.train_msg)
        self.progress = QProgressBar()
        self.progress.setAccessibleName("模型训练进度")
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        pane.addWidget(self.progress)
        self.empty_loss = label("开始训练后，这里将显示训练集和验证集的预测误差。", "empty")
        self.empty_loss.setAlignment(Qt.AlignCenter)
        self.empty_loss.setMinimumHeight(94)
        pane.addWidget(self.empty_loss)
        self.loss = Trend("预测模型训练误差", ("训练 MSE", "验证 MSE"), "轮次")
        self.loss.setFixedHeight(205)
        for item in (self.loss.heading, self.loss):
            pane.addWidget(item)
            item.hide()
        self.metrics = label("", "readout")
        self.metrics.hide()
        pane.addWidget(self.metrics)
        self.last_report = None
        self.export_train = self._button("导出训练报告", self._export_training, "link")
        self.export_train.hide()
        pane.addWidget(self.export_train, 0, Qt.AlignLeft)

    def _control_page(self):
        page = StagePage()
        self.simulation_page = page
        page.add_simulation_bar(lambda kind: self._perform(lambda: self._command(kind)))
        self.addTab(page, STAGES[1])
        left, pane = page.parameters, page.display
        left.addWidget(label("工况扰动", "heading"))
        pulse = FieldGrid(left)
        scenarios = disturbance_options()
        self.disturbance_combo = choice(pulse, "扰动工况")
        for option in scenarios["presets"]:
            title = f"{option['label']}（{option['nominal']:g} → {option['target']:g} {option['unit']}）"
            self.disturbance_combo.addItem(title, option["id"])
        pulse.add_hint(self.disturbance_combo, PARAMETER_HELP["preset"])
        self.pulse_hint = label("", "guide")
        self.disturbance_cycles = field(pulse, "持续周期", 1, scenarios["max_cycles"], scenarios["default_cycles"],
                                        beside=self.pulse_hint)
        describe_number(pulse, "duration", self.disturbance_cycles)
        self.apply_pulse = self._button("施加扰动", lambda: self._command("disturbance"))
        self.clear_pulse = self._button("提前撤除", lambda: self._command("clear_disturbance"), "secondary")
        self.clear_pulse.setEnabled(False)
        left.addLayout(FlowRow(self.apply_pulse, self.clear_pulse))
        self.pulse_state = label("", "guide")
        self.pulse_state.hide()
        left.addWidget(self.pulse_state)
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        left.addWidget(divider)

        left.addWidget(label("闭环参数", "heading"))
        form = FieldGrid(left)
        self.model_combo = choice(form, "预测模型")
        form.add_hint(self.model_combo, PARAMETER_HELP["model"])
        self.reload_models()
        cfg = Settings(reference=433.0)
        self.controls = {
            "horizon": field(form, "预测时域", 3, 40, cfg.horizon),
            "flow_min": field(form, "最小流量 (L/min)", 95, 148.55, cfg.flow_min, 1, 2),
            "flow_max": field(form, "最大流量 (L/min)", 148.56, 200, cfg.flow_max, 1, 2),
            "move_weight": field(form, "流量变化权重", 0.005, 5, cfg.move_weight, 0.01, 3),
            "speed": field(form, "推进倍率", 0.25, 10, cfg.speed, 0.25, 2),
            "reference": field(form, "釜温设定 (K)", 428, 434, cfg.reference, 0.25, 3)}
        for key, box in self.controls.items():
            describe_number(form, key, box)
        self.running_values = {key: RunningValue(form, box) for key, box in self.controls.items()}
        self.reload_controls = False
        self.reference_queue = label("")
        form.fields[self.controls["reference"]].layout().addWidget(self.reference_queue)
        self.reference_queue.hide()
        self.applied_speed = None
        for box in (self.disturbance_cycles, self.controls["speed"]):
            box.lineEdit().textChanged.connect(self._update_pulse_hint)
            box.valueChanged.connect(self._update_pulse_hint)
            box.editingFinished.connect(self._update_pulse_hint)
        self._update_pulse_hint()
        left.addWidget(label(PARAMETER_HELP["apply"], "guide"))
        commands = (("model", "应用所选模型"),
                    ("reset", "应用配置并复位"), ("end", "结束"), ("reference", "更新设定值"))
        self.command_buttons = {key: self._button(title, lambda k=key: self._command(k)) for key, title in commands}
        left.addLayout(FlowRow(*self.command_buttons.values()))

        left.addWidget(label("在线模型更新", "subheading"))
        self.auto_update_box = QCheckBox("自动在线更新")
        self.auto_update_box.setChecked(True)
        left.addWidget(self.auto_update_box)
        self.auto_update_hint = label("", "guide")
        left.addWidget(self.auto_update_hint)
        update = FieldGrid(left)
        self.online_device = choice(update, "更新设备", ("auto", "cpu", "cuda"))
        update.add_hint(self.online_device, PARAMETER_HELP["device"])
        self.auto_update_box.toggled.connect(lambda _: self._perform(self._configure_updates))
        self.online_device.currentTextChanged.connect(lambda _: self._perform(self._configure_updates))
        self.online_state = label("", "guide")
        left.addWidget(self.online_state)
        self.update_button = self._button("手动更新一次", self._update, "secondary")
        self.cancel_update = self._button("取消本轮更新", lambda: self.service.cancel(self.update_id), "secondary")
        self.update_button.hide()
        self.cancel_update.hide()
        left.addLayout(FlowRow(self.update_button, self.cancel_update))
        left.addWidget(self._button("导出实验记录", self._export, "link"), 0, Qt.AlignLeft)

        pane.addWidget(label("在线闭环仿真", "heading"))
        self.status_label = StatusStrip()
        pane.addWidget(self.status_label)
        self.control_alarm = label("")
        self.control_alarm.setWordWrap(True)
        self.control_alarm.hide()
        pane.addWidget(self.control_alarm)
        self.svg = ProcessDiagram()
        pane.addWidget(self.svg)
        self.trial_label = label("", "statusStrip")
        self.trial_label.hide()
        pane.addWidget(self.trial_label)
        self.history_label = QLabel()
        self.history_label.setObjectName("guide")
        self.history_position = QProgressBar()
        self.history_position.setMinimumWidth(72)
        self.history_position.setTextVisible(False)
        self.history_position.setAccessibleName("历史时间窗口位置")
        self.span_selector = ControlComboBox()
        self.span_selector.setAccessibleName("时间窗口")
        for intervals in WINDOW_INTERVALS:
            self.span_selector.addItem(f"{intervals*SAMPLE_MIN:g} min", intervals)
        self.span_selector.setCurrentIndex(WINDOW_INTERVALS.index(self.window_intervals))
        self.span_selector.currentIndexChanged.connect(lambda: self._set_window_span(self.span_selector.currentData()))
        span_group = QWidget()
        span_row = QHBoxLayout(span_group)
        span_row.setContentsMargins(0, 0, 0, 0)
        span_row.setSpacing(6)
        span_row.addWidget(label("时间窗口", "fieldLabel"))
        span_row.addWidget(self.span_selector)
        self.reset_scale = self._button("重置比例尺", self._reset_scales, "small")
        self.reset_scale.setEnabled(False)
        self.follow = self._button("回到实时", lambda: self._pin_window(None), "small")
        pane.addLayout(FlowRow(self.history_label, self.history_position, span_group, self.reset_scale, self.follow,
                               grow=self.history_position))
        pane.addWidget(label("在绘图区内按住左键拖动可平移曲线，双击恢复默认实时视图。滚轮滚动页面，Shift + 滚轮缩放当前纵轴。", "guide"))
        self.temp = Trend("釜内温度 (K)", ("实际釜温", "设定值"), "仿真时间 (min)", "K")
        self.flow = Trend("冷却液流量 (L/min)", ("实际施加", "控制指令"), "仿真时间 (min)", "L/min")
        for chart in (self.temp, self.flow):
            chart.setMinimumHeight(260)
            chart.pan_to.connect(self._drag_time)
            chart.restore_live.connect(self._restore_live_view)
            chart.scale_changed.connect(self._sync_scale_button)
            pane.addWidget(chart.heading)
            pane.addWidget(chart)
        self.pulse_key = label("阴影为扰动施加时段，恒流对照保持施加前的冷却液流量。", "guide")
        self.pulse_key.hide()
        pane.addWidget(self.pulse_key)
        self.control_msg = label("", "guide")
        self.control_msg.hide()
        pane.addWidget(self.control_msg)

    def _update_pulse_hint(self):
        minutes = self.disturbance_cycles.preview_value() * SAMPLE_MIN
        speed = self.controls["speed"].preview_value()
        pending = self.applied_speed is not None and speed != self.applied_speed
        # Both clients round half a tenth of a second up.
        seconds = math.floor(minutes * 600 / speed + 0.5) / 10
        self.pulse_hint.setText(f"{minutes:.1f} 仿真分钟，按设定 {speed:g}× 约 {seconds:.1f} 秒"
                               + ("（待应用）" if pending else ""))

    def _sync_scale_button(self):
        self.reset_scale.setEnabled(self.window_intervals != TREND_INTERVALS
                                   or any(chart.manual_y_range is not None for chart in (self.temp, self.flow)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        chart_height = max(260, min(340, round(self.window().height()*.32)))
        for chart in (self.temp, self.flow):
            chart.setFixedHeight(chart_height)

    def reload_models(self):
        selected = self.model_combo.currentData()
        self.model_combo.clear()
        self.default_model_version = None
        for entry in self.service.models():
            if entry["id"] == "default":
                self.default_model_version = entry["version"]
            self.model_combo.addItem(entry["label"], entry["id"])
        i = self.model_combo.findData(selected)
        self.model_combo.setCurrentIndex(max(0, i))

    def _choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择外部轨迹", "", "CSV (*.csv)")
        if path:
            self._set_csv(Path(path))

    def _set_csv(self, path):
        self.csv_path = path
        self.file_name.setText(path.name if path else "尚未选择文件")
        self.file_name.setToolTip(str(path) if path else "")
        self.data_label.setVisible(path is None)
        self.clear_csv.setVisible(path is not None)
        self.submit_train.setText("使用外部数据训练" if path else "生成轨迹并训练")
        for key in ("trajectories", "length"):
            self.training_grid.show_field(self.train_fields[key], path is None)

    def _save_example(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV 示例", "cstr-trajectories.csv", "CSV (*.csv)")
        if path:
            Path(path).write_bytes(dataset.example_csv().encode("utf-8-sig"))

    def _train(self):
        options = {k: v.value() for k, v in self.train_fields.items()}
        content = None
        if self.csv_path:
            with self.csv_path.open("rb") as stream:
                content = stream.read(dataset.MAX_CSV_BYTES + 1)
            if not content or len(content) > dataset.MAX_CSV_BYTES:
                raise ValueError("请选择非空且不超过 32 MiB 的 CSV")
        self.training_id = self.service.train({**options, "device": self.train_device.currentText()}, content)
        self.submit_train.setEnabled(False)
        self.cancel_train.setEnabled(True)
        self.train_msg.show_state("queued", STATUS["queued"], "已提交训练")
        self.progress.setValue(0)
        self.empty_loss.show()
        self.loss.hide()
        self.loss.heading.hide()
        self.metrics.hide()
        self.last_report = None
        self.export_train.hide()

    def _export_training(self):
        if self.last_report is not None:
            self._save_json("导出训练报告", "cstr-training.json", lambda: self.last_report)

    def _settings(self):
        return {k: v.value() for k, v in self.controls.items()}

    def _ensure_session(self):
        if self.session is None:
            self.session = self.service.create(
                model_id=self.model_combo.currentData(), settings=self._settings(), auto_start=True)
        return self.session

    def _command(self, kind):
        s = self._ensure_session()
        config = self._settings()
        self.service.command(s, kind, uuid4().hex, value=config["reference"] if kind == "reference" else None,
                             model_id=self.model_combo.currentData() if kind in ("model", "reset") else None,
                             settings=config if kind == "reset" else None,
                             preset=self.disturbance_combo.currentData() if kind == "disturbance" else None,
                             duration_cycles=self.disturbance_cycles.value() if kind == "disturbance" else None)
        if kind in ("reference", "reset"):
            self.reload_controls = True
        self.control_msg.clear()
        self.control_msg.hide()
        self.refresh()

    def _configure_updates(self):
        try:
            self.service.configure_updates(self._ensure_session(), self.auto_update_box.isChecked(),
                                           self.online_device.currentText())
        finally:
            self.refresh()

    def _update(self):
        self.update_id = self.service.update_model(self._ensure_session(), self.online_device.currentText())
        self.control_msg.setText("后台更新已提交")
        self.control_msg.show()

    def _export(self):
        s = self._ensure_session()
        self._save_json("导出实验记录", "cstr-run.json", s.export)

    def _save_json(self, title, filename, get_data):
        path, _ = QFileDialog.getSaveFileName(self, title, str(
            self.service.root / filename), "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(get_data(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _tab_changed(self):
        if self.currentIndex() == 1:
            self._perform(self._ensure_session)
            self.refresh()

    def _pin_window(self, first):
        self.pinned = None if first is None else self._bounded_start(first)
        self._draw_window()

    def _bounded_start(self, first):
        # Keep at least one real sampling interval in a manually moved window.
        return max(1-self.window_intervals, min(first, max(0, self.last_seq-1)))

    def _drag_time(self, first_min):
        current = max(0, self.last_seq-self.window_intervals) if self.pinned is None else self.pinned
        first = self._bounded_start(round(first_min/SAMPLE_MIN))
        if first != current:
            self._pin_window(first)

    def _set_window_span(self, intervals):
        first = None if self.pinned is None else round(self.pinned + (self.window_intervals-intervals) / 2)
        self.window_intervals = intervals
        with QSignalBlocker(self.span_selector):
            self.span_selector.setCurrentIndex(WINDOW_INTERVALS.index(intervals))
        self._pin_window(first)
        self._sync_scale_button()

    def _reset_scales(self):
        for chart in (self.temp, self.flow):
            chart.reset_y_scale()
        self._set_window_span(TREND_INTERVALS)

    def _restore_live_view(self):
        self.pinned = None
        self._reset_scales()

    def _draw_window(self):
        if not self.rows:
            return
        span = self.window_intervals
        first = max(0, self.last_seq-span) if self.pinned is None else self.pinned
        domain = first*SAMPLE_MIN, (first+span)*SAMPLE_MIN
        data_first, data_last = max(0, first), min(self.last_seq, first+span)
        if data_first >= self.rows[0]["seq"]:
            rows = [r for r in self.rows if data_first <= r["seq"] <= data_last]
        else:
            key = self.last_run, data_first, data_last
            if not self.archive or self.archive[0] != key:
                self.archive = key, self.session.sample_window(self.last_run, data_first, data_last-data_first+1)
            rows = self.archive[1]
        bands, groups = presentation.trial_traces(rows, SAMPLE_MIN)
        self.pulse_key.setVisible(bool(bands or groups))
        for chart, actual, target, column in ((self.temp, "T", "T_ref", 1), (self.flow, "Qc", "Qc_cmd", 2)):
            curves = [[(point[0], point[column]) for point in group] for group in groups]
            chart.draw([(r["t"], r[actual], r[target]) for r in rows], domain, curves, bands, context=self.last_run)
        self.history_position.setRange(1-span, max(0, self.last_seq-1))
        self.history_position.setValue(first)
        self.follow.setEnabled(self.pinned is not None)
        mode = "跟随实时" if self.pinned is None else "历史回看"
        self.history_label.setText(f"{mode}  {domain[0]:.1f}–{domain[1]:.1f} min")

    def refresh(self):
        if self.closed:
            return
        if self.training_id:
            task = self.service.tasks.get(self.training_id)
            if task:
                progress = task["progress"]
                self.progress.setValue(progress.get("percent", 0))
                history = progress.get("history") or []
                self.empty_loss.setVisible(not history)
                self.loss.setVisible(bool(history))
                self.loss.heading.setVisible(bool(history))
                if history:
                    self.loss.draw(history, context=self.training_id)
                state = task["status"]
                self.train_msg.show_state(state, STATUS.get(state, state), task.get("error") or progress.get("message"))
                if state in ("succeeded", "failed", "cancelled"):
                    result = task.get("result") or {}
                    if result:
                        self.last_report = result
                        self.export_train.show()
                        m = result["test"]
                        reasons = result.get("validation", {}).get("reasons", [])
                        self.metrics.setText("\n".join((
                            "测试集 RMSE", f"单步 (C, T, Tc)　{format_values(m['one_rmse'])}",
                            f"滚动 (C, T, Tc)　{format_values(m['rollout_rmse'])}",
                            f"模型　{result['model'] or '验证未通过'}", *reasons)))
                        self.metrics.show()
                        self.reload_models()
                    self.submit_train.setEnabled(True)
                    self.cancel_train.setEnabled(False)
                    self.training_id = None
        if not self.isVisible() or self.currentIndex() != 1 or not self.session:
            return
        snap = self.session.snapshot(self.last_seq)
        if snap["run"] != self.last_run:
            self.rows = []
            self.pinned = self.archive = None
            self._reset_scales()
        if snap["run"] != self.last_run or snap["resync"]:
            snap = self.session.snapshot()
            self.rows = []
        restore = self.reload_controls or snap["run"] != self.last_run
        for key, hint in self.running_values.items():
            value = snap["reference"] if key == "reference" else snap["config"][key]
            hint.sync(value, restore=restore)
        self.reload_controls = False
        last_reference = next((e for e in reversed(snap["events"]) if e["event"].startswith("reference_")), None)
        queued = (last_reference and last_reference["event"] == "reference_queued"
                  and snap["status"] != "ended" and last_reference["value"] != snap["reference"])
        self.reference_queue.setText(PARAMETER_HELP["referenceQueued"].format(value=f"{last_reference['value']:g}") if queued else "")
        self.reference_queue.setVisible(bool(queued))
        self.last_run, self.last_seq = snap["run"], snap["seq"]
        self.rows = (self.rows+snap["samples"])[-(TREND_MAX_INTERVALS+1):]
        live = snap["latest"]
        self.svg.set_sample(live, snap["run"], snap["status"] == "running")
        self._draw_window()
        state = snap["status"]
        trial = snap["disturbance"]
        phase = trial["phase"] if trial else None
        self.apply_pulse.setEnabled(state != "ended" and phase not in ("queued", "active", "recovery"))
        self.clear_pulse.setEnabled(phase in ("queued", "active") and not trial["clear_requested"])
        display = presentation.trial_status(trial)
        speed = snap["config"]["speed"]
        if self.applied_speed != speed:
            self.applied_speed = speed
            self._update_pulse_hint()
        self.pulse_state.setText(display["status"] if display else "")
        self.pulse_state.setVisible(bool(self.pulse_state.text()))
        results = "   ·   ".join(filter(None, (display["timing"], display["metrics"]))) if display else ""
        self.trial_label.setVisible(bool(results))
        if results:
            self.trial_label.setText(results)
            self.trial_label.setToolTip("阴影为扰动施加区间。恒流对照保持施加前的冷却液流量。")
        model_title = "默认 CSTR 模型" if snap["model"] == self.default_model_version else "当前模型"
        control = presentation.control_status(snap)
        self.control_alarm.setText(control["message"])
        self.control_alarm.setStyleSheet("color: #ffbe78;" if control["active"] else "")
        self.control_alarm.setVisible(bool(control["message"]))
        self.status_label.show_state(state, STATUS[state], f"仿真时间 {live['t']:.2f} min",
                                     f"温度偏差 {live['T']-live['T_ref']:.2f} K",
                                     f"{model_title}　{snap['model']}", control["timing"])
        self.simulation_page.show_simulation_state(state, live["t"])
        allowed = {"ready": {"model", "reset", "reference", "end"}, "running": {"model", "reset", "reference", "end"},
                   "paused": {"model", "reset", "reference", "end"}, "ended": {"reset"}}
        for key, button in self.command_buttons.items():
            button.setEnabled(key in allowed[state])
        online = snap["online_update"]
        with QSignalBlocker(self.auto_update_box), QSignalBlocker(self.online_device):
            self.auto_update_box.setChecked(online["enabled"])
            self.online_device.setCurrentText(online["device"])
        self.auto_update_box.setEnabled(state != "ended")
        self.online_device.setEnabled(state != "ended")
        self.update_button.setVisible(not online["enabled"])
        self.update_button.setEnabled(online["manual_ready"])
        self.update_id = snap["update"]
        self.cancel_update.setVisible(bool(self.update_id))
        self.cancel_update.setEnabled(bool(self.update_id))
        display = presentation.online_update_status(online)
        self.online_state.setText("\n".join(filter(None, (display["status"], display["result"]))))
        update_hint = f"每新增 {online['batch_intervals']} 个运行区间，后台微调一次预测器。验证通过后在下一控制周期切换。"
        self.auto_update_hint.setText(update_hint)
        self.auto_update_box.setAccessibleDescription(update_hint)
        events = snap["events"]
        marked = [e for e in events if e["event"] in (
            "activated", "rollback", "reference_applied", "failed", "update_rejected")]
        if snap["error"]:
            self.control_msg.setText(snap["error"])
            self.control_msg.show()
        elif marked:
            e = marked[-1]
            names = {"activated": "模型已切换", "rollback": "模型已回退",
                     "reference_applied": "设定值已生效", "failed": "求解暂停", "update_rejected": "候选未通过"}
            self.control_msg.show()
            self.control_msg.setText(
                f"{names[e['event']]} · {e['t']:.2f} min · {e.get('new',e.get('value',e.get('reason','')))}")

    def stop_display(self):
        self.closed = True
        self.timer.stop()
        self.svg.set_running(False)
        if self.training_id:
            self.service.cancel(self.training_id)


def format_values(values):
    return " / ".join(f"{v:.4g}" for v in values)
