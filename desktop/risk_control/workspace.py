"""Route desktop risk-control pages and manage their shared service."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QStackedWidget, QVBoxLayout, QWidget

from platform_core.risk_control import FDI_TITLE, HEALING_TITLE, NAVIGATION, REPLAY_TITLE
from platform_core.risk_control.service import ControlService
from platform_core.task_manager import TaskManager

from .controls import control_page_style
from .dnn_mpc_page import DnnMpcPage
from .layout import label


def overview_page():
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    card = QFrame()
    card.setObjectName("deferred")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 28, 18, 28)
    texts = (("综合仿真展示待接入", "heading"),
             ("从顶部“风险管控优化决策”菜单可分别运行 DNN-MPC、自愈控制、FDI 和重放控制试验。", "guide"))
    for text, role in texts:
        line = label(text, role)
        line.setAlignment(Qt.AlignCenter)
        layout.addWidget(line)
    outer.addWidget(card)
    outer.addStretch()
    return page


class RiskWorkspace(QWidget):
    def __init__(self, service=None):
        super().__init__()
        self.setObjectName("cstrWorkspace")
        self.setStyleSheet(control_page_style("cstrWorkspace"))
        self.manager = None if service else TaskManager()
        self.service = service or ControlService(self.manager)
        shell = QVBoxLayout(self)
        shell.setContentsMargins(16, 16, 16, 16)
        self.pages = QStackedWidget()
        shell.addWidget(self.pages)
        self.dnn_mpc_page = DnnMpcPage(self.service)
        self.pages.addWidget(overview_page())
        self.pages.addWidget(self.dnn_mpc_page)
        self.healing_page = None
        self.fdi_page = None
        self.replay_page = None
        for _ in NAVIGATION[2:]:
            self.pages.addWidget(QWidget())

    def select_page(self, name):
        index = NAVIGATION.index(name) if name in NAVIGATION else 0
        if ((name == HEALING_TITLE and self.healing_page is None) or
                (name == FDI_TITLE and self.fdi_page is None) or
                (name == REPLAY_TITLE and self.replay_page is None)):
            if name == HEALING_TITLE:
                from .healing_page import HealingPage
                page = self.healing_page = HealingPage(self.service.healing)
            elif name == FDI_TITLE:
                from .fdi_page import FdiPage
                page = self.fdi_page = FdiPage(self.service.fdi)
            else:
                from .replay_page import ReplayPage
                page = self.replay_page = ReplayPage(self.service.replay)
            previous = self.pages.widget(index)
            self.pages.removeWidget(previous)
            previous.deleteLater()
            self.pages.insertWidget(index, page)
        self.pages.setCurrentIndex(index)
        if name == NAVIGATION[1]:
            self.dnn_mpc_page.open_control()

    def shutdown(self):
        for page in (self.dnn_mpc_page, self.healing_page, self.fdi_page, self.replay_page):
            if page is not None:
                page.stop_display()
        self.service.close()
        if self.manager:
            self.manager.shutdown()
