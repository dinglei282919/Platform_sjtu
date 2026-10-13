"""Ensure cached live SVGs agree with a normal Matplotlib redraw."""
from importlib import import_module
from io import BytesIO
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from platform_core.control_plots import _SvgPreview, SimulationPlotPublisher


def signal_paths(svg):
    document = ElementTree.fromstring(svg)
    result = {}
    for group in document.iter("{http://www.w3.org/2000/svg}g"):
        name = group.get("id", "")
        if name.startswith("live-"):
            path = group.find("{http://www.w3.org/2000/svg}path")
            result[name] = (re.findall(r"[MLQCZmlqcz]", path.get("d", "")),
                            [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", path.get("d", ""))],
                            path.get("style"), path.get("clip-path"))
    return result


class LiveSvgTests(unittest.TestCase):
    @staticmethod
    def plot_data(horizon):
        t = np.linspace(0, horizon, 501)
        # Full-horizon fixtures also exercise late attack and switching intervals.
        keys = ("yd", "x1", "DF", "e", "w1", "ew1", "ew_error", "x2",
                "ut", "sut", "L1", "sig", "K1", "K2", "xr1")
        return {"t": t, **{key: np.sin(t) for key in keys}}

    def test_legends_are_inside_canvas_and_outside_every_signal_axes(self):
        for mode, horizon in (("fdi", 10.), ("replay", 50.)):
            data = self.plot_data(horizon)
            factory = import_module(f"platform_core.{mode}.plots").figures
            for preview in (False, True):
                plots = factory({key: value[:3] for key, value in data.items()} if preview else data,
                                duration=horizon)
                try:
                    for name, figure in plots.items():
                        with self.subTest(module=mode, plot=name, preview=preview):
                            if preview:
                                _SvgPreview(figure).render(data)
                            canvas = FigureCanvasAgg(figure)
                            canvas.draw()
                            renderer = canvas.get_renderer()
                            for ax in figure.axes:
                                legend = ax.get_legend().get_window_extent(renderer)
                                self.assertTrue(figure.bbox.contains(legend.x0, legend.y0))
                                self.assertTrue(figure.bbox.contains(legend.x1, legend.y1))
                                for signal_axes in figure.axes:
                                    self.assertFalse(legend.overlaps(signal_axes.bbox))
                finally:
                    for figure in plots.values():
                        figure.clear()

    def test_replay_keeps_switching_signals_separate_and_measurements_available(self):
        from platform_core.replay.plots import figures
        plots = figures(self.plot_data(50.))
        try:
            axes = plots["switching_replay"].axes
            self.assertEqual(len(axes), 3)
            self.assertEqual([{line.get_gid() for line in ax.lines} for ax in axes],
                             [{"live-K1"}, {"live-K2"}, {"live-x1", "live-xr1"}])
            for ax in axes[:2]:
                self.assertLess(ax.get_ylim()[0], 0)
                self.assertGreater(ax.get_ylim()[1], 1)
            # The original -0.5 limit clipped the recorded -0.539 transient.
            self.assertLess(plots["disturbance"].axes[1].get_ylim()[0], -.539)
        finally:
            for figure in plots.values():
                figure.clear()

    def test_recovery_windows_and_limits_respect_the_display_horizon(self):
        from platform_core.replay.config import ATTACKS
        from platform_core.replay.plots import figures
        for horizon in (4., 8., 10., 50.):
            plots = figures(self.plot_data(horizon), duration=horizon)
            try:
                ax = plots["disturbance"].axes[1]
                windows = [p for p in ax.patches if (p.get_gid() or "").startswith("recovery-window-")]
                limits = [line for line in ax.lines if (line.get_gid() or "").startswith("recovery-limit-")]
                expected_ends = [end for _, end, _ in ATTACKS if end < horizon]
                self.assertEqual(len(windows), len(expected_ends))
                for window, end in zip(windows, expected_ends):
                    vertices = ax.transData.inverted().transform(
                        window.get_transform().transform(window.get_path().vertices))
                    np.testing.assert_allclose([vertices[:, 0].min(), vertices[:, 0].max()],
                                               [end, min(end + 1.8627, horizon)])
                np.testing.assert_allclose([line.get_xdata()[0] for line in limits],
                                           [end + 1.8627 for end in expected_ends if end + 1.8627 <= horizon])
            finally:
                for figure in plots.values():
                    figure.clear()

    def test_cached_paths_match_full_redraw_for_every_plot(self):
        for mode in ("fdi", "replay"):
            data = import_module(f"platform_core.{mode}.simulation").simulate({"sim_time": 1.})
            # Cover step changes, infinities and NaN gaps; clipping is checked below.
            if mode == "fdi":
                data["sig"] = np.where(data["t"] < .5, 1., 2.)
                data["DF"][5] = np.nan
            factory = import_module(f"platform_core.{mode}.plots").figures
            plots = factory({key: value[:3] for key, value in data.items()}, duration=1.)
            try:
                for name, figure in plots.items():
                    with self.subTest(module=mode, plot=name):
                        # Keep every finite sample on-axis for the direct SVG path
                        # comparison: normal SVG delegates clipping to the viewer.
                        for ax in figure.axes:
                            values = []
                            for line in ax.lines:
                                key = (line.get_gid() or "").removeprefix("live-")
                                if key in data:
                                    values.extend(data[key][np.isfinite(data[key])])
                                elif key == "negative-DF":
                                    values.extend(-data["DF"][np.isfinite(data["DF"])])
                            ax.set_ylim(min(values) - 1, max(values) + 1)
                        preview = _SvgPreview(figure)
                        initial = preview.render({key: value[:3] for key, value in data.items()})
                        cached = signal_paths(preview.render(data))
                        output = BytesIO()
                        figure.savefig(output, format="svg")
                        redrawn = signal_paths(output.getvalue())
                        self.assertTrue(cached)
                        self.assertNotEqual(signal_paths(initial), cached)
                        self.assertEqual(cached.keys(), redrawn.keys())
                        for key, (commands, coordinates, style, _) in cached.items():
                            self.assertEqual(commands, redrawn[key][0])
                            np.testing.assert_allclose(coordinates, redrawn[key][1], atol=1e-6, rtol=0)
                            self.assertEqual(style, redrawn[key][2])
            finally:
                for figure in plots.values():
                    figure.clear()

    def test_live_and_exported_svg_clip_off_axis_signals_for_qt(self):
        from matplotlib.figure import Figure

        def factory(data, *, duration=None):
            figure = Figure(constrained_layout=True)
            ax = figure.subplots()
            ax.set(xlim=(0, 2), ylim=(-1, 1))
            ax.plot(data["t"], data["y"], gid="live-y")
            return {"signal": figure}

        data = {"t": np.array([0., 1., 2.]), "y": np.array([-100., 0., 100.])}
        figure = factory(data)["signal"]
        preview = _SvgPreview(figure)
        try:
            svg = preview.render(data)
            _, coordinates, _, _ = signal_paths(svg)["live-y"]
            vertices = np.asarray(coordinates).reshape(-1, 2)
            bounds = figure.axes[0].bbox
            # Matplotlib's path cleanup retains a one-point stroke margin.
            self.assertTrue(np.all(vertices[:, 0] >= bounds.x0 - 1.01))
            self.assertTrue(np.all(vertices[:, 0] <= bounds.x1 + 1.01))
            self.assertTrue(np.all(vertices[:, 1] >= preview.height - bounds.y1 - 1.01))
            self.assertTrue(np.all(vertices[:, 1] <= preview.height - bounds.y0 + 1.01))
            with TemporaryDirectory() as directory:
                publisher = SimulationPlotPublisher(directory, factory, 2., lambda: None)
                publisher.publish(data, final=True)
                exported = signal_paths((Path(directory) / "signal.svg").read_bytes())
                np.testing.assert_allclose(exported["live-y"][1], coordinates, atol=1e-6)
        finally:
            figure.clear()

    def test_preview_reuses_figures_and_closes_cache(self):
        from platform_core.fdi.plots import figures
        from platform_core.fdi.simulation import simulate
        data = simulate({"sim_time": .2})
        with TemporaryDirectory() as directory, patch("platform_core.control_plots.time.monotonic", return_value=1.):
            with patch("platform_core.fdi.plots.figures", wraps=figures) as factory:
                publisher = SimulationPlotPublisher(directory, factory, .2, lambda: None)
                publisher.publish(data)
                publisher.publish(data)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(publisher.image_revision, 2)
                self.assertEqual(len(list(Path(directory).glob("*.svg"))), 4)
                self.assertFalse(list(Path(directory).glob("*.png")))
                publisher.close()
                self.assertFalse(publisher._previews)
