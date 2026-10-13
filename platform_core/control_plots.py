"""Bounded live SVG previews shared by the FDI and replay runners."""
from io import BytesIO
from pathlib import Path
from threading import Lock
import re
import time

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from .dnn_mpc.runner import atomic_write

_RENDER_LOCK = Lock()


def signal_legend(ax):
    """Reserve space above the axes so legends never cover live signal paths."""
    handles, _ = ax.get_legend_handles_labels()
    return ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02),
                     ncols=min(2, len(handles)), fontsize=9, framealpha=.92)


class _SvgPreview:
    """Lay out a fixed-axis figure once; refresh only its tagged signal paths.

    Axes, legends, clipping, colors, dash styles and fonts remain Matplotlib's
    SVG. Path cleanup uses its public API, including NaN gaps and step curves.
    Final exports still go through the normal full-quality figure renderer.
    """
    _SIGNAL = re.compile(r'(<g id="live-([^"\n]+)">\s*<path d=")[^"]*(")')

    def __init__(self, figure):
        self.figure = figure
        # SVG uses points. Keep transforms in those units after savefig returns.
        figure.set_dpi(72)
        buffer = BytesIO()
        figure.savefig(buffer, format="svg")
        self.template = buffer.getvalue().decode("utf-8")
        # Matplotlib omits an empty d attribute when a signal is initially
        # entirely outside the canvas (e.g. the FDI performance bound).
        self.template = re.sub(r'(<g id="live-[^"]+">\s*<path)(?!\s+d=)', r'\1 d=""', self.template)
        figure.set_layout_engine("none")
        self.lines = {line.get_gid()[5:]: line for ax in figure.axes for line in ax.lines
                      if line.get_gid() and line.get_gid().startswith("live-")}
        found = [name for _, name, _ in self._SIGNAL.findall(self.template)]
        if len(found) != len(self.lines) or set(found) != set(self.lines):
            raise ValueError("Live SVG signal paths do not match the figure")
        self.width, self.height = figure.get_size_inches() * 72

    def render(self, data):
        paths = {}
        for name, line in self.lines.items():
            negative = name.startswith("negative-")
            values = data[name[9:] if negative else name]
            values = -values if negative else values
            line.set_data(data["t"], np.where(np.isfinite(values), values, np.nan))
            transform = line.get_transform() + Affine2D().scale(1, -1).translate(0, self.height)
            # Qt SVG does not consistently honor clipPath. Trim geometry to the
            # signal's axes as well, so off-axis samples cannot paint over labels.
            bounds = line.axes.bbox
            clip = (bounds.x0, self.height - bounds.y1, bounds.x1, self.height - bounds.y0)
            commands = []
            for vertices, code in line.get_path().iter_segments(
                    transform=transform, clip=clip, curves=False):
                if code in (MplPath.MOVETO, MplPath.LINETO):
                    commands.append(f"{'M' if code == MplPath.MOVETO else 'L'} {vertices[0]:.6f} {vertices[1]:.6f}")
            paths[name] = "\n".join(commands)
        return self._SIGNAL.sub(lambda match: match[1] + paths[match[2]] + match[3], self.template).encode("utf-8")


class SimulationPlotPublisher:
    def __init__(self, directory, figures, duration, check_cancelled):
        self.directory = Path(directory)
        self.figures = figures
        self.duration = duration
        self.check_cancelled = check_cancelled
        self.outputs = {}
        self.image_revision = 0
        self._last_preview = None
        self._previews = {}

    def close(self):
        with _RENDER_LOCK:
            for preview in self._previews.values():
                preview.figure.clear()
            self._previews.clear()

    def preview(self, t, end, snapshot):
        """Build a snapshot only when due; never alter the solver's steps."""
        if t >= end or t < min(.1, end / 5):
            return
        if self._last_preview is not None and time.monotonic() - self._last_preview < .5:
            return
        self.check_cancelled()
        self.publish(snapshot())
        self._last_preview = time.monotonic()

    def publish(self, data, *, final=False, on_plot=None):
        # Matplotlib's math-text parser is shared across worker threads.
        with _RENDER_LOCK:
            self.check_cancelled()
            if not final:
                if not self._previews:
                    plots = self.figures(data, duration=self.duration)
                    try:
                        self._previews = {name: _SvgPreview(figure) for name, figure in plots.items()}
                    except Exception:
                        for figure in plots.values():
                            figure.clear()
                        raise
                for name, preview in self._previews.items():
                    self.check_cancelled()
                    atomic_write(self.directory / f"{name}.svg", preview.render(data))
                self.outputs = {name: str(self.directory / f"{name}.svg") for name in self._previews}
                self.image_revision += 1
                return
            plots = self.figures(data, duration=self.duration)
            try:
                for index, (name, figure) in enumerate(plots.items()):
                    self.check_cancelled()
                    # Use the same clipped paths in final exports and live views.
                    layout = figure.get_layout_engine()
                    atomic_write(self.directory / f"{name}.svg", _SvgPreview(figure).render(data))
                    figure.set_layout_engine(layout)
                    if final:
                        figure.savefig(self.directory / f"{name}.png", format="png", dpi=160)
                    if on_plot is not None:
                        on_plot(index)
                # Publish the revision only after all four SVGs are complete.
                self.outputs = {name: str(self.directory / f"{name}.svg") for name in plots}
                self.image_revision += 1
            finally:
                for figure in plots.values():
                    figure.clear()
