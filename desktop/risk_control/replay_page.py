"""Qt presentation of the independent replay-observer experiment."""
from uuid import uuid4

from PySide6.QtCore import QByteArray, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QCheckBox, QSlider

from platform_core.risk_control.replay import ARMS, DEMO_CASES, DT
from platform_core.risk_control.replay_display import (
    CASE_GUIDES, CHARTS, GUIDE, HEAT_CHART, OBSERVATIONS, QUANTITIES, SENSORS, diagram_sample, number)
from .controls import ControlComboBox
from .comparison_trends import ComparisonTrendWindow
from .layout import FieldGrid, FlowRow, RunningValue, StagePage, action_button, choice, field, label
from .process_diagram import ProcessDiagram
from .trends import Trend


MAIN_ARMS = ("raw", "guard", "candidate", "continuous")


class ReplayPage(StagePage):
    def __init__(self, service):
        super().__init__()
        self.service, self.session, self.snap = service, None, None
        self.at = None
        self.trend_window = ComparisonTrendWindow(self._show_sample, self.live)
        self.add_simulation_bar(self.command)
        self.parameters.addWidget(label("重放攻击 Replay Attack", "heading"))
        self.parameters.addWidget(label(GUIDE, "guide"))
        form = FieldGrid(self.parameters)
        self.scenario = choice(form, "本次重放的热负荷工况", DEMO_CASES)
        self.case_label = label(CASE_GUIDES[self.scenario.currentData()], "guide")
        self.parameters.addWidget(self.case_label)
        self.scenario.currentIndexChanged.connect(lambda: self.case_label.setText(CASE_GUIDES[self.scenario.currentData()]))
        self.begin = action_button("注入测温重放", lambda: self.command("begin"), "primary")
        self.cancel = action_button("取消待注入", lambda: self.command("cancel"))
        self.finish = action_button("提前结束本段", lambda: self.command("finish"))
        self.parameters.addLayout(FlowRow(self.begin, self.cancel, self.finish))
        self.phase_label = label("", "guide")
        self.parameters.addWidget(self.phase_label)
        settings = FieldGrid(self.parameters)
        self.speed = field(settings, "推进倍率", 1, 1200, 30)
        self.running_speed = RunningValue(settings, self.speed)
        self.reset = action_button("应用配置并复位", lambda: self.command("reset"))
        self.parameters.addLayout(FlowRow(self.reset))
        self.parameters.addWidget(label("每轮先生成 0–4.9 min 的正常测温记录，3.0 min 起额外放热。复位会新开一轮。", "guide"))
        self.message = label("", "guide")
        self.parameters.addWidget(self.message)

        self.display.addWidget(label("反应釜工艺与测温支路", "heading"))
        inspect = FieldGrid(self.display)
        self.records, self.arm = ControlComboBox(), ControlComboBox()
        self.records.addItem("本轮", "")
        for key in MAIN_ARMS:
            self.arm.addItem(ARMS[key],key)
        self.arm.setCurrentIndex(self.arm.findData("guard"))
        inspect.add_field("运行记录", self.records)
        inspect.add_field("查看分支", self.arm)
        self.process_diagram = ProcessDiagram()
        self.process_diagram.hide()
        self.display.addWidget(self.process_diagram)
        self.loop_toggle = QCheckBox("展开控制框图")
        self.display.addWidget(self.loop_toggle)
        self.diagram = QSvgWidget()
        self.diagram.setAccessibleName("测温副本至热负荷观测器的链路")
        self.diagram.setFixedHeight(180)
        self.display.addWidget(self.diagram)
        self.diagram.hide()
        self.loop_toggle.toggled.connect(self.diagram.setVisible)
        self.signals = label("", "guide")
        self.flow = label("", "subheading")
        self.display.addWidget(self.signals)
        self.display.addWidget(self.flow)
        self.position = QSlider(Qt.Horizontal)
        self.position.setAccessibleName("查看采样点")
        self.cursor_label = label("", "guide")
        self.follow = action_button("回到实时", self.live)
        self.locate = action_button("定位本段首拍", self.locate_start)
        self.display.addLayout(FlowRow(self.cursor_label,self.locate,self.follow))
        self.display.addWidget(self.position)
        self.references = QCheckBox("展开参照组")
        self.display.addWidget(self.references)
        self.sensor = choice(FieldGrid(self.display), "温度测点", SENSORS, wide=False)
        self.display.addWidget(label("阴影为观测支路收到旧测温的时段。切换分支可核对本地测温、观测副本及当拍的阀指令。", "guide"))
        self.display.addWidget(self.trend_window.guide)
        self.charts = []
        self.details = QCheckBox("评测指标与指令明细")
        for spec in [*CHARTS, HEAT_CHART]:
            chart = Trend(f"{spec['title']} ({spec['unit']})", [s["name"] for s in spec["lines"]], "仿真时间 (min)", spec["unit"])
            self.trend_window.attach(chart)
            for series, line in zip(chart.lines,spec["lines"]):
                series.setPen(QPen(QColor(line["color"]), 2, Qt.DashLine if line["dashed"] else Qt.SolidLine))
            if spec is HEAT_CHART:
                self.display.addWidget(self.details)
                self.heat_chart = chart
                for widget in (chart.heading, chart):
                    widget.hide()
                    self.details.toggled.connect(widget.setVisible)
            else:
                self.charts.append(chart)
            self.display.addWidget(chart.heading)
            self.display.addWidget(chart)
        self.metric_text = label("", "guide")
        self.display.addWidget(self.metric_text)
        self.metric_text.hide()
        self.details.toggled.connect(self.metric_text.setVisible)
        self.references.toggled.connect(self.show_references)
        self.arm.currentIndexChanged.connect(self.refresh)
        self.sensor.currentIndexChanged.connect(self.refresh)
        self.position.valueChanged.connect(self.inspect)
        self.records.currentIndexChanged.connect(self.select_record)
        self.timer = QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.refresh)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        self.refresh()

    def hideEvent(self, event):
        self.stop_display()
        super().hideEvent(event)

    def stop_display(self):
        self.timer.stop()
        self.process_diagram.set_running(False)

    def command(self, kind):
        if not self.snap:
            return
        try:
            settings = dict(scenario=self.scenario.currentData(), speed=self.speed.value()) if kind=="reset" else None
            if kind == "begin":
                settings = {"scenario": self.scenario.currentData()}
            self.session.command(kind,uuid4().hex,self.snap["live_run"],settings)
            if kind == "reset":
                self.live()
            else:
                self.refresh()
        except (KeyError,ValueError,RuntimeError,OSError) as exc:
            self.message.setText(str(exc))

    def inspect(self, seq):
        self.trend_window.first = None
        self._show_sample(seq)

    def _show_sample(self, seq):
        self.at = seq
        self.refresh()

    def select_record(self):
        self.trend_window.first = None
        self.at = None
        self.refresh()

    def live(self):
        with QSignalBlocker(self.records):
            self.records.setCurrentIndex(0)
        self.select_record()

    def locate_start(self):
        point = self.snap["point"] if self.snap else None
        if point and point["start_seq"] is not None:
            self.inspect(point["start_seq"])

    def set_arm_choices(self, available, selected):
        choices = [key for key in ARMS if key in available and (self.references.isChecked() or key in MAIN_ARMS)]
        if choices == [self.arm.itemData(i) for i in range(self.arm.count())] and selected == self.arm.currentData():
            return
        with QSignalBlocker(self.arm):
            self.arm.clear()
            for key in choices:
                self.arm.addItem(ARMS[key], key)
            self.arm.setCurrentIndex(self.arm.findData(selected if selected in choices else "guard"))

    def show_references(self):
        available = self.snap["available_arms"] if self.snap else ARMS
        self.set_arm_choices(available, self.arm.currentData())
        self.refresh()

    def refresh(self):
        try:
            if self.session is None:
                self.session = self.service.create()
            frame = self.session.frame(self.records.currentData() or None,self.at,arm=self.arm.currentData())
            self.set_arm_choices(frame["available_arms"], frame["arm"])
            run_changed = not self.snap or frame["live_run"] != self.snap["live_run"]
            self.running_speed.sync(frame["live_config"]["speed"], restore=run_changed)
            if run_changed:
                self.scenario.setCurrentIndex(self.scenario.findData(frame["live_config"]["scenario"]))
                with QSignalBlocker(self.records):
                    self.records.clear()
                    self.records.addItem("本轮","")
                    for entry in reversed(frame["archives"]):
                        self.records.addItem(DEMO_CASES[entry["scenario"]]+" · "+entry["run"][:8],entry["run"])
            self.snap = frame
            self.message.setText(frame["error"])
            self.draw(frame)
        except (KeyError,ValueError,TypeError,RuntimeError,OSError) as exc:
            self.message.setText(str(exc))

    def draw(self, data):
        self.show_simulation_state(data["status"],max(0,data["live_seq"])*DT)
        pinned = self.at is not None or bool(self.records.currentData())
        self.follow.setEnabled(pinned)
        self.begin.setText("再次注入测温重放" if data["segment"] else "注入测温重放")
        self.begin.setEnabled(not pinned and not data["pending"] and not data["error"])
        self.cancel.setVisible(data["pending"] == "begin" and not pinned)
        self.finish.setVisible(data["phase"] == "active" and not data["pending"] and not pinned)
        self.phase_label.setText("正在查看历史 · 回到实时后可注入" if pinned else data["phase_label"])
        point = data["point"]
        domain = self.trend_window.update(data["run"], point["seq"] if point else -1, data["last"], not pinned)
        self.locate.setEnabled(bool(point and point["start_seq"] is not None))
        arm = data["arm"]
        sample = diagram_sample(point, arm, data["display"])
        self.process_diagram.setVisible(sample is not None)
        if sample is not None:
            self.process_diagram.set_sample(sample, (data["run"], arm), not pinned and data["status"] == "running")
        else:
            self.process_diagram.set_running(False)
        if point:
            view = data["display"]
            self.diagram.load(QByteArray(view["svg"].encode("utf-8")))
            self.diagram.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
            signals = (f"{view['link']}　接收 {view['received']}　样本源 {view['source']}　{view['action']}\n"
                       f"釜温 · 本拍 {view['local']} / 观测器 {view['observed']}　"
                       f"夹套出口 · 本拍 {view['jacket']} / 观测器 {view['jacket_observed']}")
            if arm in ("candidate", "continuous"):
                signals += f"\n本地重估 · {view['local_mode']}"
                if view['local_prediction'] != "—":
                    signals += f"　釜温预测 {view['local_prediction']}　实测减预测 {view['local_residual']}"
                    if arm == "candidate":
                        signals += f"　同向越限 {view['local_streak']}"
            self.signals.setText(signals)
            self.flow.setText(f"补偿前（PI） {view['base']}　附加 {view['comp']}　补偿后（下发） {view['final']}")
            self.cursor_label.setText(("历史" if pinned else "实时")+f" · #{point['seq']} · {point['t']:.1f} min　"
                                      +data["phase_label"]+"　"+DEMO_CASES[data["scenario"]])
            with QSignalBlocker(self.position):
                self.position.setRange(0,data["last"])
                self.position.setValue(point["seq"])
            metrics = data["metrics"].get(arm,{})
            score = ("尚未注入重放" if data["scored_through"] is None else
                     f"第 {data['segment']} 段评测至 {data['scored_through']*DT:.1f} min")
            self.metric_text.setText(f"限幅前请求 {view['request']}　补偿权重 {view['weight']}\n"+score+"　"+
                                     "　".join(f"{label_} {number(metrics.get(k))}" for k,label_ in QUANTITIES.items()))
        else:
            self.diagram.load(QByteArray(b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'))
            self.signals.setText("正在准备首个采样点")
            for widget in (self.flow,self.cursor_label,self.metric_text):
                widget.clear()
        observation = OBSERVATIONS[self.sensor.currentData()]
        observation_title = f"{observation['title']} ({observation['unit']})"
        if observation_title != self.charts[0].title:
            self.charts[0].title = observation_title
            self.charts[0].setAccessibleName(observation_title)
            self.charts[0].reset_y_scale()
        samples = [r for r in data["samples"] if r["t"] >= domain[0]-1e-9]
        for chart,spec in zip([*self.charts, self.heat_chart], [observation, *CHARTS[1:], HEAT_CHART]):
            visible = [(self.references.isChecked() or not s["optional"])
                       and ("arm" not in s or s["arm"] in data["available_arms"]) for s in spec["lines"]]
            for series,show in zip(chart.lines,visible):
                series.setVisible(show)
                for marker in chart.chart().legend().markers(series):
                    marker.setVisible(show)
            rows = [[r["t"], *[r.get(s["key"]) if show else None for s,show in zip(spec["lines"],visible)]] for r in samples]
            chart.draw(rows,domain,bands=data["bands"],context=(data["run"], arm, spec["title"]))
            if not rows:
                for series in chart.lines:
                    series.clear()
