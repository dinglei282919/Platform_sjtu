"""Native live view for the shared four-branch CSTR comparison."""
import json
from pathlib import Path
from uuid import uuid4

from PySide6.QtCharts import QScatterSeries
from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QFileDialog, QHeaderView, QTableWidget, QTableWidgetItem

from platform_core.risk_control.healing import (COMPENSATION_PARAMETERS, Config, GROUPS, MODES, POST_FAULT_MIN,
                                               SCENARIOS, SIGNALS, STATE_NAMES, TRIGGER_LIMITS)
from platform_core.risk_control.healing_display import COLORS, HEADINGS, METHODS, SPAN, event_label, view
from platform_core.risk_control.cstr import FLOW_LIMITS, SAMPLE_MIN
from .layout import FieldGrid, FlowRow, RunningValue, StagePage, StatusStrip, action_button, choice, field, label
from .process_diagram import ProcessDiagram
from .trends import Trend


class EffectChart(Trend):
    def __init__(self):
        super().__init__("", ("正常参考", "仅 PI", "PI + SCSC"), "仿真时间 (min)")
        self.setMinimumHeight(245)
        self.milestones = []
        self.lines[0].setPen(QPen(QColor(COLORS[4]), 1, Qt.DashLine))
        self.lines[1].setPen(QPen(QColor(COLORS[1]), 2))

    def update_effect(self, snap, display, method):
        info = display["methods"][method]
        scsc = method == "scsc"
        self.heading.setText(display["title"])
        self.setAccessibleName(display["title"])
        self.lines[2].setName("PI + " + method.upper())
        self.lines[2].setPen(QPen(QColor(COLORS[2 if scsc else 3]), 2.5))
        self.milestones = [(snap["fault_min"], "退化开始", COLORS[1]),
                           (info["first_min"], "退化后首次补偿", COLORS[2 if scsc else 3])]
        self.sample_unit = display["unit"]
        self.sample_digits = 6 if snap["effect"]["signal"] == "C" else 4
        self.draw([[p[0], 0., p[1], p[2 if scsc else 3]] for p in snap["effect"]["points"]], context=(snap["run"], method))
        left, right = display["domain"]
        self.lines[0].replace([QPointF(left, 0.), QPointF(right, 0.)])
        self.axes[0].setRange(left, right)
        self.axes[1].setLabelFormat("%.5f" if snap["effect"]["signal"] == "C" else "%.3f")
        self.axes[1].setTickCount(1 + round((info["limits"][1] - info["limits"][0]) / info["tick"]))
        self.axes[1].setRange(*info["limits"])
        self.viewport().update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not getattr(self, "milestones", None):
            return
        area = self.chart().mapRectToScene(self.chart().plotArea())
        left, right = self.axes[0].min(), self.axes[0].max()
        if right <= left or area.isEmpty():
            return
        painter.save()
        painter.setClipRect(area, Qt.IntersectClip)
        for i, (at, title, color) in enumerate(self.milestones):
            if at is None or not left <= at <= right:
                continue
            x = area.left() + (at-left)*area.width()/(right-left)
            painter.setPen(QPen(QColor(color), 1.5, Qt.DashLine))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            width = painter.fontMetrics().horizontalAdvance(title) + 8
            y = area.top()+2 if i == 0 else area.bottom()-22
            box = QRectF(min(x+3, area.right()-width), y, width, 20)
            painter.fillRect(box, QColor("#10253f"))
            painter.drawText(box, Qt.AlignCenter, title)
        painter.restore()


class HealingPage(StagePage):
    def __init__(self, service):
        super().__init__()
        self.service, self.session, self.snap = service, None, None
        self.add_simulation_bar(lambda kind: self.perform(lambda: self.command(kind)))
        self.rows, self.pinned, self.archive = [], None, None
        self.status = StatusStrip()
        self.display.addWidget(self.status)
        self.parameters.addWidget(label("自愈控制配置", "heading"))
        form = FieldGrid(self.parameters)
        self.scenario = choice(form, "退化场景", SCENARIOS)
        self.mode = choice(form, "补偿调度", MODES)
        self.degrade = self.button("施加退化", lambda: self.command("degrade"), "primary")
        self.buttons = {"reset": self.button("应用配置并复位", lambda: self.command("reset"))}
        self.parameters.addLayout(FlowRow(self.degrade, self.buttons["reset"]))
        self.fault_help = label("", "guide")
        self.parameters.addWidget(self.fault_help)
        self.parameters.addWidget(label("自愈触发范围", "subheading"))
        self.parameters.addWidget(label("参考点用于越限判断和入口补偿。应用配置并复位后，曲线显示本轮生效的上下限。", "guide"))
        defaults = Config()
        self.references, self.widths, self.bounds = [], [], []
        self.running_references, self.running_widths = [], []
        for i, key in enumerate(STATE_NAMES):
            bands = FieldGrid(self.parameters)
            unit = SIGNALS[key][1]
            step = 0.0001 if i == 0 else 0.1
            limits = TRIGGER_LIMITS[key]
            ref_min, ref_max = limits["reference"]
            center = field(bands, f"{key} 参考点 ({unit})", ref_min, ref_max, defaults.references[i], step, 12)
            bands.add_hint(center, f"可输入范围 {ref_min:g} 至 {ref_max:g} {unit}")
            box = field(bands, f"{key} 半宽 ({unit})", *limits["half_width"], defaults.widths[i], step, 12)
            note = bands.add_hint(box, "")
            self.references.append(center)
            self.widths.append(box)
            self.bounds.append(note)
            self.running_references.append(RunningValue(bands, center))
            self.running_widths.append(RunningValue(bands, box))
            for editor in (center, box):
                editor.valueChanged.connect(self.update_bounds)
                editor.lineEdit().textChanged.connect(self.update_bounds)
                editor.editingFinished.connect(self.update_bounds)
        self.update_bounds()
        self.parameters.addWidget(label("补偿参数", "subheading"))
        tune = FieldGrid(self.parameters)
        self.fields = {}
        for kind, title in (("interval", "最小间隔 (min)"), ("gain", "增益")):
            spec = COMPENSATION_PARAMETERS[kind]
            for method in ("scsc", "ccsc"):
                name = f"{method}_{kind}"
                box = field(tune, f"{method.upper()} {title}", spec["min"], spec["max"],
                            getattr(defaults, name), spec["step"], spec["decimals"])
                self.fields[name] = box
                box.setAccessibleDescription(spec["hint"])
            tune.add_item(label(spec["hint"]), wide=True)
        for name, title in (("seed", "正常数据拟合种子"), ("speed", "推进倍率")):
            spec = COMPENSATION_PARAMETERS[name]
            self.fields[name] = field(tune, title, spec["min"], spec["max"], getattr(defaults, name),
                                      spec["step"], spec["decimals"])
            tune.add_hint(self.fields[name], spec["hint"])
        self.running_fields = {name: RunningValue(tune, box) for name, box in self.fields.items()}
        self.scenario.currentIndexChanged.connect(self.set_preset)
        self.mode.currentIndexChanged.connect(self.set_preset)
        self.parameters.addWidget(label(f"配置在复位后生效。手动施加退化后继续观测 {POST_FAULT_MIN:g} 仿真分钟。", "guide"))
        for key, text in (("end", "结束"), ("export", "导出实验记录")):
            self.buttons[key] = self.button(text, lambda k=key: self.command(k))
        self.parameters.addLayout(FlowRow(self.buttons["end"], self.buttons["export"]))
        self.message = label("", "guide")
        self.parameters.addWidget(self.message)
        self.display.addWidget(label("自愈效果对照", "heading"))
        self.method = choice(FieldGrid(self.display), "对照的自愈方法", METHODS)
        self.phase = label("", "guide")
        self.display.addWidget(self.phase)
        self.effect_chart = EffectChart()
        self.display.addWidget(self.effect_chart.heading)
        self.display.addWidget(self.effect_chart)
        self.reference = label("", "guide")
        self.display.addWidget(self.reference)
        self.comparison = label("", "statusStrip")
        self.display.addWidget(self.comparison)
        display_form = FieldGrid(self.display)
        self.group = choice(display_form, "工艺图与入口对照组", GROUPS)
        self.group.setCurrentIndex(2)
        self.diagram = ProcessDiagram()
        self.display.addWidget(self.diagram)
        self.readout = label("", "guide")
        self.display.addWidget(self.readout)
        self.display.addWidget(label("四组恢复指标", "subheading"))
        self.recovery_note = label("", "guide")
        self.display.addWidget(self.recovery_note)
        self.table = QTableWidget(4, len(HEADINGS))
        self.table.setHorizontalHeaderLabels(HEADINGS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setFixedHeight(166)
        self.display.addWidget(self.table)
        self.trend_set = choice(FieldGrid(self.display), "趋势信号", {"outputs":"输出与触发范围", "inputs":"实际入口与冷却流量"})
        self.follow = self.button("回到实时", self.restore)
        self.history = label("", "guide")
        self.display.addLayout(FlowRow(self.history, self.follow))
        self.charts, self.markers = {}, {}
        for key, (name, unit) in SIGNALS.items():
            output = key in STATE_NAMES
            labels = [*GROUPS.values(), "下限", "上限"] if output else (["实际流量", "下限", "上限"] if key=="Qc" else ["实际入口"])
            chart = Trend(f"{name} ({unit})", labels, "仿真时间 (min)", unit)
            chart.setMinimumHeight(220)
            for j, series in enumerate(chart.lines):
                bound = (output and j >= 4) or (key=="Qc" and j>0)
                series.setPen(QPen(QColor(COLORS[4] if bound else COLORS[j]), 1 if bound else 2,
                                   Qt.DashLine if bound else Qt.SolidLine))
            if output:
                marker = QScatterSeries()
                marker.setName("所选组触发")
                marker.setMarkerSize(8)
                marker.setColor(QColor("#ff8f79"))
                chart.chart().addSeries(marker)
                for axis in chart.axes:
                    marker.attachAxis(axis)
                self.markers[key] = marker
            self.charts[key] = chart
            self.display.addWidget(chart.heading)
            self.display.addWidget(chart)
            chart.pan_to.connect(self.pan)
            chart.restore_live.connect(self.restore)
        self.display.addWidget(label("补偿事件", "subheading"))
        self.events = label("", "guide")
        self.display.addWidget(self.events)
        self.timing = label("", "guide")
        self.display.addWidget(self.timing)
        self.method.currentIndexChanged.connect(lambda: self.select_group(self.method.currentData()))
        self.group.currentIndexChanged.connect(lambda: self.select_group(self.group.currentData()))
        self.scenario.currentIndexChanged.connect(self.redraw)
        self.trend_set.currentIndexChanged.connect(self.redraw)
        self.timer = QTimer(self, interval=350)
        self.timer.timeout.connect(self.refresh)
        self.set_preset()

    def button(self, text, fn, role="secondary"):
        return action_button(text, lambda: self.perform(fn), role)

    def select_group(self, key):
        with QSignalBlocker(self.group), QSignalBlocker(self.method):
            self.group.setCurrentIndex(self.group.findData(key))
            if key in METHODS:
                self.method.setCurrentIndex(self.method.findData(key))
        self.redraw()

    def perform(self, fn):
        try:
            fn()
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            self.message.setText(str(exc))

    def update_bounds(self):
        for key, ref, box, note in zip(STATE_NAMES, self.references, self.widths, self.bounds):
            center, w = ref.preview_value(), box.preview_value()
            lo, hi = (f"{n:.12f}".rstrip("0").rstrip(".") for n in (center-w, center+w))
            unit = SIGNALS[key][1]
            endpoints = (f"下限 {lo}　上限 {hi} {unit}" if key == "C"
                         else f"下限 {lo} {unit}\n上限 {hi} {unit}")
            text = f"可输入范围 {box.minimum():g} 至 {box.maximum():g} {unit}\n{endpoints}"
            note.setText(text)
            box.setAccessibleDescription(text)

    def set_preset(self):
        cfg = Config.parse(dict(scenario=self.scenario.currentData(), mode=self.mode.currentData()))
        for name in ("scsc_interval", "scsc_gain", "ccsc_interval", "ccsc_gain"):
            self.fields[name].setValue(getattr(cfg, name))
            if name.endswith("interval"):
                self.fields[name].setEnabled(cfg.mode=="event")

    def settings(self):
        widths = [ref if abs(box.value()-ref)<5.1e-13 else box.value() for box, ref in zip(self.widths, Config().widths)]
        return dict(scenario=self.scenario.currentData(), mode=self.mode.currentData(), widths=widths,
                    references=[box.value() for box in self.references],
                    **{k: box.value() for k, box in self.fields.items()})

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        self.refresh()

    def hideEvent(self, event):
        self.stop_display()
        super().hideEvent(event)

    def stop_display(self):
        self.timer.stop()
        self.diagram.set_running(False)

    def command(self, key):
        if self.session is None:
            self.refresh()
        if self.session is None:
            return
        if key == "degrade" and self.scenario.currentData() != self.snap["config"]["scenario"]:
            raise ValueError("场景已修改，请先应用配置并复位。")
        if key == "export":
            path, _ = QFileDialog.getSaveFileName(self, "导出实验记录", "cstr-self-healing.json", "JSON (*.json)")
            if path:
                Path(path).write_text(json.dumps(self.session.export(), ensure_ascii=False, allow_nan=False), encoding="utf-8")
            return
        self.session.command(key, uuid4().hex, self.snap["run"], self.settings() if key=="reset" else None)
        self.message.clear()
        self.refresh()

    def refresh(self):
        self.perform(self._refresh)

    def _refresh(self):
        if self.session is None:
            self.session = self.service.create()
        snap = self.session.snapshot(self.snap["seq"] if self.snap else -1, run=self.snap["run"] if self.snap else None)
        changed = not self.snap or snap["run"] != self.snap["run"]
        if changed:
            self.pinned, self.archive = None, None
            config = snap["config"]
            with QSignalBlocker(self.scenario), QSignalBlocker(self.mode):
                self.scenario.setCurrentIndex(self.scenario.findData(config["scenario"]))
                self.mode.setCurrentIndex(self.mode.findData(config["mode"]))
            self.set_preset()
            for hint, center in zip(self.running_references, config["references"]):
                hint.sync(center, restore=True)
            for hint, width in zip(self.running_widths, config["widths"]):
                hint.sync(width, restore=True)
            for name, hint in self.running_fields.items():
                hint.sync(config[name], restore=True)
        self.rows = (([] if changed or snap["resync"] else self.rows)+snap["samples"])[-640:]
        self.snap = snap
        status = snap["status"]
        self.show_simulation_state(status, snap["latest"]["t"])
        self.buttons["end"].setEnabled(status != "ended")
        self.redraw()
        if snap["error"]:
            self.message.setText(snap["error"])

    def pan(self, start):
        if self.snap:
            self.pinned = max(0, min(max(0, self.snap["seq"]-1), round(start/SAMPLE_MIN)))
            self.redraw()

    def restore(self):
        self.pinned = self.archive = None
        for chart in self.charts.values():
            chart.reset_y_scale()
        self.redraw()

    def redraw(self):
        if self.snap is None:
            return
        snap = self.snap
        group = self.group.currentData()
        display = view(snap)
        self.recovery_note.setText(display["recovery_help"])
        self.status.show_state(snap["status"], display["state"], display["status"], display["speed"])
        effect = display["effect"]
        method = self.method.currentData()
        info = effect["methods"][method]
        self.phase.setText(effect["phase"] + " " + info["step"])
        self.reference.setText(effect["reference"])
        self.comparison.setText(info["readout"])
        self.effect_chart.update_effect(snap, effect, method)
        changed_scenario = self.scenario.currentData() != snap["config"]["scenario"]
        self.degrade.setEnabled(snap["fault_min"] is None and snap["status"] != "ended" and not changed_scenario)
        self.fault_help.setText("场景已修改，请先应用配置并复位。" if changed_scenario else effect["instruction"])
        latest = snap["latest"]
        v = latest["groups"][group]
        self.diagram.set_sample(dict(v, seq=latest["seq"]), (snap["run"],group), snap["status"]=="running")
        first_update = snap["metrics"]["groups"][group]["first_update"]
        first_label = "—" if first_update is None else f"{first_update:.1f} min"
        self.readout.setText(f"当前越限 {'、'.join(v['outside']) or '无'}　退化后首次触发 {first_label}　"
                             f"活性系数 {v['a']:.4f}　换热系数 {v['b']:.4f}")
        for i, row in enumerate(display["metrics"]):
            for j, item in enumerate(row):
                existing = self.table.item(i,j)
                if existing is None:
                    self.table.setItem(i,j,QTableWidgetItem(item))
                elif existing.text()!=item:
                    existing.setText(item)
        self.timing.setText("　".join(display["timing"]))
        self.events.setText("\n".join(event_label(e) for e in reversed(snap["recent"][-6:])) or "暂无补偿事件")
        start = max(0, snap["seq"]-SPAN) if self.pinned is None else self.pinned
        end = start + SPAN
        rows = self.rows
        if rows and start < rows[0]["seq"]:
            key = (snap["run"],start,min(end,snap["seq"]))
            if not self.archive or self.archive[0] != key:
                self.archive = key, self.session.history(snap["run"],start,SPAN+1)
            rows = self.archive[1]
        rows = [r for r in rows if start <= r["seq"] <= end]
        self.history.setText(f"{'跟随实时' if self.pinned is None else '历史回看'}　{start*SAMPLE_MIN:.1f}–{end*SAMPLE_MIN:.1f} min")
        self.follow.setEnabled(self.pinned is not None)
        for key, chart in self.charts.items():
            output = key in STATE_NAMES
            show = output == (self.trend_set.currentData()=="outputs")
            chart.setVisible(show)
            chart.heading.setVisible(show)
            if not show:
                continue
            points = []
            marks = []
            if output:
                i = STATE_NAMES.index(key)
                width = snap["config"]["widths"][i]
                center = snap["config"]["references"][i]
                for r in rows:
                    points.append([r["t"], *(r["groups"][g][key] for g in GROUPS), center-width, center+width])
                    marks.extend(QPointF(e["at"], center+e["error"][i]) for e in r["compensation"] if e["group"]==group and key in e["channels"])
                self.markers[key].replace(marks)
            else:
                bounds = FLOW_LIMITS if key=="Qc" else ()
                points = [[r["t"], r["groups"][group][key], *bounds] for r in rows]
            chart.draw(points, (start*SAMPLE_MIN, end*SAMPLE_MIN), context=(snap["run"], group))
