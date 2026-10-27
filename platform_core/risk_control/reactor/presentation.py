"""Shared process illustration with signed inverse-coordinate readouts."""
from functools import lru_cache
from pathlib import Path
import math
import re


FLOW1, FLOW2 = 0.3, 0.2
TEMPLATE = Path(__file__).resolve().parents[3] / "assets" / "reactor" / "reactor_process.svg"
FLOW_PATHS = (
    ((20, 19), (583, 19), (583, 133)),
    ((20, 60), (244, 60), (244, 133)),
    ((237, 990), (237, 1096), (855, 1096), (855, 244), (1003, 244)),
    ((583, 990), (583, 1065), (787, 1065), (787, 198), (1003, 198)),
    ((1111, 221), (1207, 221), (1304, 268)),
    ((1485, 908), (1485, 1096)),
)


def _flow_arrows(phase):
    """Small triangles following the original pipe centerlines in both renderers."""
    pieces = []
    for points in FLOW_PATHS:
        distance = 24 + (phase % 1) * 64
        for (x, y), (end_x, end_y) in zip(points, points[1:]):
            length = math.hypot(end_x - x, end_y - y)
            dx, dy = (end_x - x) / length, (end_y - y) / length
            while distance < length:
                cx, cy = x + dx * distance, y + dy * distance
                pieces.append(f"M{cx + 5 * dx:.3f} {cy + 5 * dy:.3f} "
                              f"L{cx - 4 * dx - 3 * dy:.3f} {cy - 4 * dy + 3 * dx:.3f} "
                              f"L{cx - 4 * dx + 3 * dy:.3f} {cy - 4 * dy - 3 * dx:.3f} Z")
                distance += 64
            distance -= length
    return " ".join(pieces)


def _clip(value, low=0.0, high=1.0):
    return min(high, max(low, value))


def build_diagram_series(data):
    """Fixed decorative fills; composition and flow readouts use actual run data."""
    from .coordinates import composition_series
    c = composition_series(data)
    return {**{k: v.tolist() for k,v in c.items()},
            "level1": [.5] * len(data["t"]), "level2": [.5] * len(data["t"])}


def diagram_tokens(data=None, index=0, series=None, phase=0.0):
    """Return trusted numeric/color/text tokens for the common SVG template."""
    ready = data is not None and len(data.get("t", [])) > 0
    if ready:
        index = min(len(data["t"]) - 1, max(0, int(index)))
        series = build_diagram_series(data) if series is None else series
        level1, level2 = (float(series[key][index]) for key in ("level1", "level2"))
        mode = int(data["mode"][index])
    else:
        level1 = level2 = .50
        mode = 0
        phase = 0.0
    signed_u = _clip(float(data["u"][index]) / 20., -1., 1.) if ready else 0.
    blade = 90 + 120 * abs(math.cos(2 * math.pi * phase))
    return {
        "XS": f"{float(series['xs'][index]):.4g}" if ready else "—",
        "XD": f"{float(series['xd'][index]):.4g}" if ready else "—",
        "KAPPA": f"{float(series['k'][index]):.4g}" if ready else "—",
        "TIME": f"{float(data['t'][index]):.3f} s" if ready else "等待仿真",
        "MODE": str(mode) if ready else "—",
        "FLOW1": f"{float(series['r1'][index]) if ready else FLOW1:.3f}", "FLOW2": f"{float(series['r2'][index]) if ready else FLOW2:.3f}",
        "LEVEL1": f"{math.floor(level1 * 100 + .5)}%" if ready else "—",
        "LEVEL2": f"{math.floor(level2 * 100 + .5)}%" if ready else "—",
        "FILL1_Y": f"{990 - 768 * level1:.3f}", "FILL1_H": f"{768 * level1:.3f}",
        "FILL2_Y": f"{990 - 768 * level2:.3f}", "FILL2_H": f"{768 * level2:.3f}",
        "SWITCH_Y": str(244 if mode == 1 else 198 if mode == 2 else 221),
        "ROUTE1": "#2563eb" if mode == 1 else "#cbd5e1",
        "ROUTE2": "#2563eb" if mode == 2 else "#cbd5e1",
        "U_BAR_X": f"{1420 + min(0., signed_u) * 180:.3f}",
        "U_BAR_WIDTH": f"{abs(signed_u) * 180:.3f}",
        "U_BAR_COLOR": "#d97706" if signed_u < 0 else "#167c9d",
        "U": f"{float(data['u'][index]):.4g}" if ready else "—",
        "LIQUID_COLOR": "#b7d9e5",
        "FLOW_OFFSET": f"{-phase * 48:.3f}",
        "FLOW_ARROWS": _flow_arrows(phase),
        "BLADE_HALF": f"{blade:.3f}",
        "BLADE_LEFT": f"{1485 - blade:.3f}", "BLADE_RIGHT": f"{1485 + blade:.3f}",
    }


@lru_cache(maxsize=1)
def _template():
    return TEMPLATE.read_text(encoding="utf-8")


def render_diagram(data=None, index=0, series=None, phase=0.0):
    tokens = diagram_tokens(data, index, series, phase)
    return re.sub(r"__([A-Z0-9_]+)__", lambda match: tokens[match[1]], _template())
