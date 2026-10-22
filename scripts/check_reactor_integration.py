"""Manual full-duration platform integration check (not part of the fast suite).

Run from the checkout: python -B scripts/check_reactor_integration.py
Outputs are placed in web_runtime/reactor_validation; no I-drive dependency.
"""
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase, QPalette, QColor
from main_interface import MainWindow
from platform_core.risk_control.reactor.reporting import export_result

app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)
if Path("C:/Windows/Fonts/msyh.ttc").is_file():
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    app.setFont(QFont("Microsoft YaHei", 9))
palette = app.palette()
for role in (QPalette.Window, QPalette.Base, QPalette.Button):
    palette.setColor(role, QColor("#10233e"))
for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
    palette.setColor(role, QColor("#e7eff8"))
app.setPalette(palette)
window = MainWindow()
window.show()
window._on_submodule_clicked("风险管控优化决策", "化学反应器自适应控制仿真")
page = window._reactor_content_widget
assert page.isVisible()
window._on_submodule_clicked("异构数据治理", "SDG-HAZOP")
assert not page.isVisible()
window._on_submodule_clicked("风险管控优化决策", "化学反应器自适应控制仿真")
assert window._reactor_content_widget is page
page.run_button.click()
assert page.has_running_worker() and not page.diagram.animated
start = time.monotonic()
last_log = start
while page.has_running_worker():
    app.processEvents()
    time.sleep(.01)
    if time.monotonic() - last_log > 30:
        print(f"elapsed={time.monotonic()-start:.1f}s progress={page.progress.value()}%", flush=True)
        last_log = time.monotonic()
    if time.monotonic() - start > 900:
        page.cancel_run()
        while page.has_running_worker():
            app.processEvents()
            time.sleep(.01)
        raise RuntimeError("Full desktop validation exceeded 15 minutes")
assert page.result is not None, page.status.text()
assert page.result["data"]["t"][-1] == 3
assert page.run_button.isEnabled() and page.export_button.isEnabled()
assert page.diagram.animated
directory = ROOT / "web_runtime" / "reactor_validation"
export_result(page.result, directory)
page.diagram.set_animated(False)
app.processEvents()
window.grab().save(str(directory / "desktop-diagram.png"))
page.tabs.setCurrentIndex(1)
app.processEvents()
window.grab().save(str(directory / "desktop-results.png"))
print(json.dumps({"desktop_full": page.result["metrics"], "wall_seconds": time.monotonic()-start}, indent=2), flush=True)
window.close()
app.processEvents()