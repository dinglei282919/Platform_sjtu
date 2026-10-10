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
    def test_cached_paths_match_full_redraw_for_every_plot(self):
        for mode in ("fdi", "replay"):
            data = import_module(f"platform_core.{mode}.simulation").simulate({"sim_time": 1.})
            # Cover step changes, infinities, NaN gaps and off-axis signals.
            if mode == "fdi":
                data["sig"] = np.where(data["t"] < .5, 1., 2.)
                data["DF"][5] = np.nan
            factory = import_module(f"platform_core.{mode}.plots").figures
            plots = factory({key: value[:3] for key, value in data.items()}, duration=1.)
            try:
                for name, figure in plots.items():
                    with self.subTest(module=mode, plot=name):
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
