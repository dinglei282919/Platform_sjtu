"""Time navigation for comparison pages with a shared per-sample inspection cursor."""
from platform_core.risk_control.cstr import SAMPLE_MIN, TREND_INTERVALS
from .layout import label


class ComparisonTrendWindow:
    def __init__(self, inspect, follow):
        self.inspect, self.follow = inspect, follow
        self.first, self.run, self.latest = None, None, -1
        self.start, self.charts = 0, []
        self.guide = label("", "guide")

    def attach(self, chart):
        self.charts.append(chart)
        chart.setMinimumHeight(220)
        chart.pan_to.connect(self.pan)
        chart.restore_live.connect(self.restore)

    def update(self, run, seq, latest, live):
        if self.run != run:
            self.run, self.first = run, None
            for chart in self.charts:
                chart.reset_y_scale()
        self.latest = latest
        self.start = max(0, seq-TREND_INTERVALS+1) if self.first is None else self.first
        domain = (self.start*SAMPLE_MIN, (self.start+TREND_INTERVALS)*SAMPLE_MIN)
        mode = "跟随实时" if live else "历史回看"
        self.guide.setText(f"{mode}　{domain[0]:.1f}–{domain[1]:.1f} min　"
                           "在绘图区内按住左键拖动可平移曲线，双击回到实时并恢复自动纵轴。"
                           "滚轮滚动页面，Shift + 滚轮缩放当前纵轴。")
        return domain

    def pan(self, first_min):
        first = max(1-TREND_INTERVALS, min(self.latest, round(first_min/SAMPLE_MIN)))
        if self.latest >= 0 and first != self.start:
            self.first = first
            self.inspect(min(self.latest, first+TREND_INTERVALS-1))

    def restore(self):
        for chart in self.charts:
            chart.reset_y_scale()
        self.follow()
