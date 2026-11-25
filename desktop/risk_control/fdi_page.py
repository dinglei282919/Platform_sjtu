"""Native views of the independently clocked measurement and execution trials."""
import json
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QByteArray, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHeaderView, QPushButton, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from platform_core.risk_control.fdi import BOOK, GROUPS, RECOVERY_STEPS, Config
from platform_core.risk_control.fdi_control import DT
from platform_core.risk_control.fdi_display import (DEMO_VIEWS, diagram_sample, loop_template, loop_tokens,
                                                   nominal_preview, preview_marker, view)
from platform_core.risk_control.presentation import marker_tokens
from .controls import ControlComboBox, ControlTabWidget
from .comparison_trends import ComparisonTrendWindow
from .layout import FieldGrid, FlowRow, RunningValue, StagePage, StatusStrip, label
from .process_diagram import ProcessDiagram
from .workspace import Trend, field


def choice(form, title, items):
    box = ControlComboBox()
    for key, name in items.items():
        box.addItem(name, key)
    form.add_field(title, box, wide=True)
    return box


class FdiTrialPage(StagePage):
    def __init__(self, side, service):
        super().__init__()
        self.side, self.service = side, service
        self.session = self.snap = self.historical = None
        self.rows, self.charts = [], []
        self.trend_window = ComparisonTrendWindow(lambda at: self.perform(lambda: self._show_sample(at)), self.restore)
        self.groups = GROUPS[side]
        self.demo = DEMO_VIEWS[side]
        self.add_simulation_bar(lambda kind: self.perform(lambda: self.command(kind)))
        self.parameters.addWidget(label("虚假数据注入攻击FDI Attack", "heading"))
        form = FieldGrid(self.parameters)
        self.scenario = choice(form, "预置样本", self.demo["presets"])
        self.scenario.setEnabled(self.scenario.count() > 1)
        notes = self.demo.get("preset_notes")
        self.preset_note = form.add_hint(self.scenario, notes[self.scenario.currentData()]) if notes else None
        self.begin = self.button("注入 FDI 样本", lambda: self.command("begin"))
        self.begin.setProperty("role", "primary")
        self.finish = self.button("结束本次注入", lambda: self.command("finish"))
        self.parameters.addLayout(FlowRow(self.begin, self.finish))
        self.parameters.addWidget(label(f"每次持续 {BOOK['length']*DT:g} 仿真分钟。生效期间再次注入，先撤除一拍，再施加所选样本。", "guide"))
        self.trust = label(self.demo["controller_note"], "guide")
        self.trust.setTextFormat(Qt.RichText)
        self.parameters.addWidget(self.trust)
        self.parameters.addWidget(label("仿真参数", "heading"))
        settings = FieldGrid(self.parameters)
        defaults = Config.parse({"side": side})
        self.seed = field(settings, "噪声与初态种子", 0, 2**31-1, defaults.seed)
        self.speed = field(settings, "推进倍率", 1, 1200, defaults.speed, 1, 1)
        settings.add_hint(self.seed, "相同种子复用同一组初态和噪声。")
        settings.add_hint(self.speed, "可输入范围 1 至 1200 倍。")
        self.running_seed = RunningValue(settings, self.seed)
        self.running_speed = RunningValue(settings, self.speed)
        self.apply = self.button("应用配置并复位", lambda: self.command("reset"))
        self.parameters.addWidget(self.apply)
        self.phase = label("", "guide")
        self.parameters.addWidget(self.phase)
        self.end = self.button("结束试验", lambda: self.command("end"))
        self.save = self.button("导出所选记录", self.export)
        self.parameters.addLayout(FlowRow(self.end, self.save))
        self.message = label("", "guide")
        self.parameters.addWidget(self.message)

        self.display.addWidget(label("闭环对照", "heading"))
        self.status = StatusStrip()
        self.display.addWidget(self.status)
        inspect = FieldGrid(self.display)
        self.runs = choice(inspect, "运行记录", {"": "本轮"})
        self.diagram = ProcessDiagram()
        self.display.addWidget(self.diagram)
        self.source = label("", "guide")
        self.display.addWidget(self.source)
        self.loop_toggle = QCheckBox("展开控制框图")
        self.display.addWidget(self.loop_toggle)
        self.loop = QSvgWidget()
        self.loop.setAccessibleName("所选分支的闭环信号图")
        self.loop.setFixedHeight(202)
        self.loop.hide()
        self.display.addWidget(self.loop)
        self._loop_values = None
        self.loop_toggle.toggled.connect(self.loop.setVisible)
        self.loop_toggle.toggled.connect(self.redraw)
        self.position = QSlider(Qt.Horizontal)
        self.position.setAccessibleName("查看采样点")
        self.follow = self.button("回到实时", self.restore)
        self.cursor_label = label("", "guide")
        self.display.addLayout(FlowRow(self.cursor_label, self.follow))
        self.display.addWidget(self.position)
        self.display.addWidget(label(self.demo["comparison"], "guide"))
        self.display.addWidget(self.trend_window.guide)
        self._build_charts()
        self.details_toggle = QCheckBox("试验详情")
        self.display.addWidget(self.details_toggle)
        details = QWidget()
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.display.addWidget(details)
        details.hide()
        self.details_toggle.toggled.connect(details.setVisible)
        self.group = choice(FieldGrid(detail_layout), "工艺图与采样详情分支", self.groups)
        self.group.setCurrentIndex(self.group.findData(self.demo["protected_group"]))
        self.readings = label("", "guide")
        detail_layout.addWidget(label("所选采样点", "subheading"))
        detail_layout.addWidget(self.readings)
        self.metric_title = label("", "subheading")
        detail_layout.addWidget(self.metric_title)
        self.table = QTableWidget(0, len(self.groups)+1)
        self.table.setHorizontalHeaderLabels(["指标", *self.groups.values()])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        detail_layout.addWidget(self.table)
        detail_layout.addWidget(label("恢复用时取真实釜温连续 10 拍处于设定值 ±0.10 K 内的首拍。请在相同的有效区间数下比较 IAE。", "guide"))
        self.events = label("", "guide")
        detail_layout.addWidget(label("采样事件", "subheading"))
        detail_layout.addWidget(self.events)
        self.timing = label("", "guide")
        detail_layout.addWidget(self.timing)
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(lambda: self.perform(self.refresh))
        self.scenario.currentIndexChanged.connect(self.redraw)
        self.group.currentIndexChanged.connect(self.redraw)
        self.details_toggle.toggled.connect(self.redraw)
        self.position.valueChanged.connect(lambda k: self.perform(lambda: self.inspect(k)))
        self.runs.currentIndexChanged.connect(lambda: self.perform(self.select_record))

    def button(self, title, fn):
        box = QPushButton(title)
        box.setProperty("role", "secondary")
        box.setCursor(Qt.PointingHandCursor)
        box.clicked.connect(lambda: self.perform(fn))
        return box

    def perform(self, fn):
        try:
            fn()
        except (KeyError, ValueError, RuntimeError, OSError, TypeError) as exc:
            self.message.setText(str(exc))

    def _build_charts(self):
        for spec in self.demo["charts"]:
            signals = spec["signals"]
            title = f"{spec['title']} ({spec['unit']})"
            chart = Trend(title, [s["name"] for s in signals], "仿真时间 (min)", spec["unit"])
            self.trend_window.attach(chart)
            for line, signal in zip(chart.lines, signals):
                width = 1 if signal["key"] == "T_ref" else 2
                style = Qt.DashLine if signal.get("dashed") else Qt.SolidLine
                line.setPen(QPen(QColor(signal["color"]), width, style))
            self.display.addWidget(chart.heading)
            if spec.get("note"):
                self.display.addWidget(label(spec["note"], "guide"))
            self.display.addWidget(chart)
            self.charts.append((chart, signals))

    def scenario_label(self, key):
        return self.demo["presets"].get(key, BOOK["cases"][key]["label"])

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        self.perform(self.refresh)

    def hideEvent(self, event):
        self.stop_display()
        super().hideEvent(event)

    def stop_display(self):
        self.timer.stop()
        self.diagram.set_running(False)

    def command(self, kind):
        if not self.snap:
            return
        fields = dict(side=self.side, scenario=self.scenario.currentData(), seed=int(self.seed.value()),
                      speed=self.speed.value()) if kind == "reset" else None
        if kind == "begin":
            fields = {"scenario": self.scenario.currentData()}
        self.session.command(kind, uuid4().hex, self.snap["run"], fields)
        self.message.clear()
        self.refresh()

    def export(self):
        data = self.historical or self.snap
        if not data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出所选记录", f"cstr-fdi-{self.side}-{data['run']}.json", "JSON (*.json)")
        if path:
            report = self.session.export(data["run"])
            Path(path).write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def ensure_session(self):
        if self.session is None:
            self.session = self.service.create(dict(side=self.side))

    def refresh(self):
        self.ensure_session()
        before = self.snap
        self.snap = self.session.snapshot(before["seq"] if before else -1, run=before["run"] if before else None)
        if before is None or before["run"] != self.snap["run"]:
            self.historical, self.rows = None, []
            self.running_seed.sync(self.snap["config"]["seed"], restore=True)
            self.running_speed.sync(self.snap["config"]["speed"], restore=True)
            with QSignalBlocker(self.scenario):
                self.scenario.setCurrentIndex(max(0, self.scenario.findData(self.snap["pending_scenario"] or self.snap["scenario"])))
            with QSignalBlocker(self.runs):
                self.runs.clear()
                self.runs.addItem("本轮", "")
                for record in reversed(self.snap["archives"]):
                    title = self.scenario_label(record["config"]["scenario"])
                    self.runs.addItem(title + " · " + record["run"][:8], record["run"])
                    if record["seq"] < 0:
                        self.runs.model().item(self.runs.count()-1).setEnabled(False)
        self.rows = (([] if self.snap["resync"] else self.rows) + self.snap["samples"])[-240:]
        self.redraw()

    def inspect(self, at):
        self.trend_window.first = None
        self._show_sample(at)

    def _show_sample(self, at):
        if self.snap and (self.snap["seq"] >= 0 or self.runs.currentData()):
            run = self.runs.currentData() or self.snap["run"]
            first = max(0, at-119)
            self.historical = self.session.window(run, first, at-first+1)
            self.redraw()

    def select_record(self):
        run = self.runs.currentData()
        if not run:
            self.restore()
        else:
            record = next(r for r in self.snap["archives"] if r["run"] == run)
            if record["seq"] >= 0:
                self.inspect(record["seq"])

    def restore(self):
        self.trend_window.first = None
        self.historical = None
        with QSignalBlocker(self.runs):
            self.runs.setCurrentIndex(0)
        self.redraw()

    def redraw(self):
        if self.preset_note is not None:
            note = self.demo["preset_notes"][self.scenario.currentData()]
            if self.preset_note.text() != note:
                self.preset_note.setText(note)
                self.scenario.setAccessibleDescription(note)
        if not self.snap:
            return
        live = self.historical is None
        data = self.snap if live else self.historical
        row = data["latest"]
        last = self.snap["seq"] if live or data["run"] == self.snap["run"] else data["seq"]
        domain = self.trend_window.update(data["run"], row["seq"] if row else -1, last, live)
        self.follow.setEnabled(not live)
        rendered = view(data, live=live)
        self.show_simulation_state(self.snap["status"], max(0, self.snap["seq"])*DT)
        config = data["config"]
        self.status.show_state(data["status"], rendered["phase"], self.scenario_label(data["scenario"]),
                               f"设定 {config['speed']:g}×")
        draft = self.scenario.currentData() != self.snap["scenario"]
        needs_reset = self.snap["config"]["scenario"] not in self.demo["presets"]
        ended = self.snap["status"] == "ended"
        pending = self.snap["pending"]
        self.begin.setText("再次注入 FDI" if self.snap["segment"] else "注入 FDI 样本")
        self.begin.setEnabled(live and not ended and not self.snap["error"] and not pending and not needs_reset)
        self.finish.setText("取消待注入样本" if pending == "begin" else "结束本次注入")
        self.finish.setEnabled(live and not ended and (pending == "begin" or bool(row and row["phase"] == "active" and not pending)))
        self.end.setEnabled(not ended)
        self.phase.setText(self.snap["error"] or ("应用配置并复位可从验收工况切换到演示样本。" if needs_reset else
                           f"每次注入单独评测，恢复观察最长 {RECOVERY_STEPS*DT:g} 仿真分钟。"))
        if not row:
            preview = nominal_preview(self.side, self.scenario.currentData())
            if not data["preview"] and not draft:
                preview["signal_marker"] = {}
            self.diagram.set_sample(preview, data["run"], False)
            self.draw_loop(preview, self.group.currentData())
            self.source.setText("名义工况预览 · 正在准备观测器")
            self.cursor_label.clear()
            self.readings.clear()
            self.events.clear()
            self.metric_title.clear()
            self.timing.clear()
            self.table.setRowCount(0)
            for chart, _ in self.charts:
                chart.draw([], domain)
                for line in chart.lines:
                    line.clear()
            return
        group = self.group.currentData()
        sample = row["groups"][group]
        diagram = diagram_sample(data, row, group, live=live)
        if diagram:
            if live and draft and row["phase"] == "healthy" and pending is None:
                diagram["signal_marker"] = preview_marker(self.side, self.scenario.currentData())
            self.diagram.set_sample(diagram, (data["run"], group), live and not ended and
                                    data["status"] == "running" and sample["valid"])
            self.draw_loop(diagram, group)
        self.source.setText(f"工艺图 · {self.groups[group]}　反馈来源 · " + rendered["sources"][group] + "\n" + rendered["timeline"])
        self.cursor_label.setText(("实时" if live else "历史") + f" · #{row['seq']} · {row['t']:.1f} min")
        with QSignalBlocker(self.position):
            self.position.setRange(0, last)
            self.position.setValue(row["seq"])
        rows = [r for r in (self.rows if live else data["samples"]) if r["seq"] >= self.trend_window.start]
        for chart, signals in self.charts:
            pts = [[r["t"], *[r["groups"][s["group"]].get(s["key"]) if r["groups"][s["group"]]["valid"] else None
                              for s in signals]] for r in rows]
            chart.draw(pts, domain, bands=data["bands"], context=data["run"])
        if not self.details_toggle.isChecked():
            return
        self.readings.setText("　　".join(f"{name}  {value}" for name,value in rendered["readings"][group]))
        self.metric_title.setText(rendered["metrics_title"])
        self.table.setRowCount(len(rendered["table"]))
        for i, cells in enumerate(rendered["table"]):
            for j, cell in enumerate(cells):
                self.table.setItem(i, j, QTableWidgetItem(cell))
        self.table.setFixedHeight(self.table.verticalHeader().length()+46)
        self.events.setText("\n".join(rendered["events"][-8:]) or "暂无事件")
        timing = data["timing"]
        self.timing.setText(f"周期预算 {timing['budget_ms']:.1f} ms　计算 P95 " +
                            ("—" if timing["p95_ms"] is None else f"{timing['p95_ms']:.2f} ms") +
                            f"　超时 {timing['overruns']} / {timing['intervals']}　实测推进 " +
                            ("—" if timing["effective_speed"] is None else f"{timing['effective_speed']:.1f}×"))

    def draw_loop(self, diagram, arm):
        if not self.loop_toggle.isChecked():
            return
        values = {**loop_tokens(arm), **marker_tokens(diagram["signal_marker"])}
        if values == self._loop_values:
            return
        svg = loop_template()
        for key, value in values.items():
            svg = svg.replace("@@"+key+"@@", value)
        self.loop.load(QByteArray(svg.encode("utf-8")))
        self.loop.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        self._loop_values = values


class FdiPage(ControlTabWidget):
    def __init__(self, service):
        super().__init__()
        for side, title in (("measurement", "测量端"), ("execution", "执行端")):
            self.addTab(FdiTrialPage(side, service), title)

    def showEvent(self, event):
        super().showEvent(event)
        for i in range(self.count()):
            trial = self.widget(i)
            trial.perform(trial.ensure_session)

    def stop_display(self):
        for i in range(self.count()):
            self.widget(i).stop_display()
