"""Temperature, flow and event histories for the CSTR self-healing comparisons."""
from collections import deque

import numpy as np

from .cstr import FLOW_LIMITS, NOMINAL, SAMPLE_MIN
from .healing import EFFECT_OUTPUT, END_STEP, GROUPS, STATE_NAMES, TOLERANCE


class Metrics:
    def __init__(self):
        self.count = 0
        self.groups = {key: dict(square=np.zeros(3), peak=None, hits=0, upper=0, lower=0, entry=None,
                                 updates=0, limited=0, first=None) for key in GROUPS}
        self.times = {kind: deque(maxlen=END_STEP) for kind in ("ordinary", "compensation")}
        self.spacing = {kind: deque(maxlen=END_STEP) for kind in self.times}
        self.cycles = dict.fromkeys(self.times, 0)
        self.max_ms = dict.fromkeys(self.times, 0.)
        self.overruns = dict.fromkeys(self.times, 0)
        self.wall = 0.

    def record(self, row, budget, fault_seq):
        kind = "compensation" if row["compensation"] else "ordinary"
        self.times[kind].append(row["cycle_ms"])
        self.spacing[kind].append(row["wall_ms"])
        self.cycles[kind] += 1
        self.max_ms[kind] = max(self.max_ms[kind], row["cycle_ms"])
        self.overruns[kind] += row["cycle_ms"] > budget
        self.wall += row["wall_ms"]
        if fault_seq is None or row["seq"] <= fault_seq:
            return
        for event in row["compensation"]:
            g = self.groups[event["group"]]
            g["updates"] += 1
            g["limited"] += any(event["step_limited"]) or any(event["inlet_limited"])
            if g["first"] is None:
                g["first"] = event["at"]
        self.count += 1
        for key, sample in row["groups"].items():
            g = self.groups[key]
            error = np.array([sample["C"], sample["T"], sample["Tc"]])-NOMINAL
            g["square"] += error*error
            g["peak"] = max(g["peak"] or sample["T"], sample["T"])
            inside = (NOMINAL[0]-TOLERANCE[0] <= sample["C"] <= NOMINAL[0]+TOLERANCE[0] and
                      NOMINAL[1]-TOLERANCE[1] <= sample["T"] <= NOMINAL[1]+TOLERANCE[1])
            g["hits"] += inside
            if not inside:
                g["entry"] = None
            elif g["entry"] is None:
                g["entry"] = row["t"]
            g["upper"] += sample["Qc"] >= FLOW_LIMITS[1]-1e-9
            g["lower"] += sample["Qc"] <= FLOW_LIMITS[0]+1e-9

    def summary(self):
        result = {}
        for key, g in self.groups.items():
            result[key] = dict(rmse=np.sqrt(g["square"]/self.count).tolist() if self.count else None,
                               peak_T=g["peak"], joint_percent=100*g["hits"]/self.count if self.count else None,
                               sustained_since=g["entry"], upper_min=float(g["upper"]*SAMPLE_MIN),
                               lower_min=float(g["lower"]*SAMPLE_MIN), updates=g["updates"],
                               limited=g["limited"], first_update=g["first"])
        timing = {kind: dict(count=self.cycles[kind], p95_ms=float(np.percentile(seq, 95)) if seq else None,
                             publish_p95_ms=float(np.percentile(self.spacing[kind], 95)) if seq else None,
                             max_ms=self.max_ms[kind] if seq else None,
                             overrun_percent=100*self.overruns[kind]/self.cycles[kind] if seq else None)
                  for kind, seq in self.times.items()}
        steps = sum(self.cycles.values())
        return dict(groups=result, timing=timing, post_fault_samples=self.count,
                    actual_speed=steps*SAMPLE_MIN*60000/self.wall if self.wall else None)


class EffectHistory:
    """Compact completed samples for the overview, including exact intervention boundaries."""
    def __init__(self, scenario, initial):
        self.signal = EFFECT_OUTPUT[scenario]
        self.reference = float(NOMINAL[STATE_NAMES.index(self.signal)])
        self.latest = self.point(initial)
        self.points = deque([self.latest])
        self.first = {"scsc": None, "ccsc": None}

    def point(self, row):
        return [row["t"], *(float(row["groups"][key][self.signal]) - self.reference
                            for key in ("fault", "scsc", "ccsc"))]

    def keep_latest(self):
        if not self.points or self.points[-1][0] != self.latest[0]:
            self.points.append(self.latest)

    def record(self, row, fault_seq):
        first_change = False
        if fault_seq is None:
            while self.points and self.points[0][0] < row["t"] - 10.:
                self.points.popleft()
        else:
            for event in row["compensation"]:
                group = event["group"]
                if self.first[group] is None and any(event["applied"]):
                    self.first[group] = event
                    first_change = True
        if first_change:
            self.keep_latest()  # The command acts at the start of the interval.
        self.latest = self.point(row)
        if first_change or row["seq"] % 10 == 0:
            self.keep_latest()

    def snapshot(self):
        points = list(self.points)
        if not points or points[-1][0] != self.latest[0]:
            points.append(self.latest)
        return dict(signal=self.signal, reference=self.reference, points=points, first=self.first.copy())
