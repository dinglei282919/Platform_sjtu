"""Streaming evaluation on the same left-boundary intervals as offline validation."""
import math

from . import cstr
from .fdi_control import DT, T_REF


class Scores:
    def __init__(self, previous=None):
        self.event_n = self.recovery_n = self.healthy_n = self.false_active = 0
        self.event_iae = self.recovery_iae = self.tv = self.saturation = self.capacity = 0.
        self.feedback_sq = self.witness_sq = self.vessel_sq = self.nominal_sq = self.adapted_sq = 0.
        self.feedback_n = self.witness_n = self.predictions = self.false_entries = 0
        self.event_peak = self.recovery_peak = self.concentration_peak = None
        self.first_switch = self.recovery_min = None
        self.recovery_streak = 0
        self.previous, self.previous_phase = previous, None
        self.last_valid = None

    def record(self, sample, row, stop):
        if not sample["valid"] or row["phase"] == "monitoring":
            return
        phase, k = row["phase"], row["seq"]
        # The endpoint scores the state; its outgoing interval belongs to live monitoring.
        endpoint = row.get("evaluation_end", row["terminal"])
        self.last_valid = k
        error = abs(sample["T"]-T_REF)
        previous = self.previous
        if (phase == "healthy" or row["link"] is None) and k >= 10 and not endpoint:
            self.healthy_n += 1
            active = sample.get("active", False)
            self.false_active += int(active)
            self.false_entries += int(active and previous is not None and not previous.get("active"))
        if phase == "active":
            self.event_n += 1
            self.event_iae += error*DT
            self.event_peak = max(self.event_peak or 0., error)
            if sample.get("active") and self.first_switch is None and previous and not previous.get("active"):
                self.first_switch = row["t"]
            if sample.get("feedback") is not None:
                self.feedback_sq += (sample["feedback"]-sample["T"])**2
                self.feedback_n += 1
            if sample.get("witness") is not None:
                self.witness_sq += (sample["witness"]-sample["T"])**2
                self.witness_n += 1
        elif phase == "recovery":
            self.recovery_peak = max(self.recovery_peak or 0., error)
            if not endpoint:
                self.recovery_n += 1
                self.recovery_iae += error*DT
            self.recovery_streak = self.recovery_streak+1 if error <= 0.10 else 0
            if self.recovery_streak == 10 and self.recovery_min is None:
                self.recovery_min = round((k-9-stop)*DT, 10)
        if self.previous_phase == "active" and previous.get("dnn_Tc") is not None and sample["Tcm"] is not None:
            effect = sample["Tcm"]-previous["dnn_Tc"]
            self.nominal_sq += effect**2
            self.adapted_sq += (effect-previous["rbf"])**2
            self.vessel_sq += (sample["load_estimate"]-previous["load"])**2
            self.predictions += 1
        if phase != "healthy":
            self.concentration_peak = max(self.concentration_peak or 0., abs(sample["C"]-cstr.NOMINAL[0]))
            if not endpoint and sample["Qc"] is not None:
                if previous and previous["Qc"] is not None:
                    self.tv += abs(sample["Qc"]-previous["Qc"])
                self.saturation += DT * (sample["Qc"] in cstr.FLOW_LIMITS)
                self.capacity += DT * sample["capacity_hit"]
        self.previous, self.previous_phase = sample, phase

    def report(self):
        def rmse(total, count):
            return math.sqrt(total/count) if count else None
        return dict(event_intervals=self.event_n, recovery_intervals=self.recovery_n, last_valid_seq=self.last_valid,
                    iae_event=self.event_iae, iae_recovery=self.recovery_iae, iae=self.event_iae+self.recovery_iae,
                    event_peak=self.event_peak, recovery_peak=self.recovery_peak, recovery_min=self.recovery_min,
                    concentration_peak=self.concentration_peak, tv=self.tv, saturated_min=self.saturation,
                    capacity_min=self.capacity, feedback_rmse=rmse(self.feedback_sq, self.feedback_n),
                    witness_rmse=rmse(self.witness_sq, self.witness_n), first_switch_min=self.first_switch,
                    healthy_takeover_min=self.false_active*DT, healthy_switches=self.false_entries,
                    healthy_takeover_fraction=self.false_active/self.healthy_n if self.healthy_n else None,
                    jacket_nominal_rmse=rmse(self.nominal_sq, self.predictions),
                    jacket_adapted_rmse=rmse(self.adapted_sq, self.predictions),
                    load_rmse=rmse(self.vessel_sq, self.predictions))
