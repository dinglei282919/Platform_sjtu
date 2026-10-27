"""Parameterized version of the paper's multiplicative feedback attack."""
import math

ATTACK_DEFAULTS = dict(scale=1., frequency=.25, phase=0.)


def validate_attack(scale, frequency, phase):
    for name, value, low, high in (("强度 a", scale, 0., 1.8),
                                    ("角频率 ω", frequency, .01, 2.),
                                    ("相位 φ", phase, -math.pi, math.pi)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"FDI {name}范围 [{low:g}, {high:g}]，必须为有限数值。")


def parse_attack(value):
    if not isinstance(value, dict) or set(value) != set(ATTACK_DEFAULTS):
        raise ValueError("FDI 参数必须包含且仅包含 scale、frequency、phase。")
    validate_attack(**value)
    return {key: float(value[key]) for key in ATTACK_DEFAULTS}


def multiplier(t, scale=1., frequency=.25, phase=0.):
    return 1 + scale + 1.5 * scale * math.sin(frequency * t + phase)
