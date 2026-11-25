"""HTTP adapter for the common self-healing session."""
from typing import Literal

from platform_core.risk_control.healing import options
from platform_core.risk_control.healing_display import event_label, view

from .comparison import SessionCommand, comparison_router, dispatch_command
from .control_http import Owner, call


class Command(SessionCommand):
    kind: Literal["start", "pause", "resume", "reset", "end", "degrade"]


def routes(service):
    router = comparison_router("/api/cstr/healing", service)

    @router.get("/config")
    def config():
        return options()

    @router.get("/sessions/{identifier}")
    def snapshot(identifier: str, after: int = -1, run: str | None = None, owner: Owner = None):
        data = call(service().session, identifier).snapshot(after, owner, run)
        data["view"] = view(data)
        data["recent"] = [dict(e, label=event_label(e)) for e in data["recent"]]
        return data

    @router.get("/sessions/{identifier}/history")
    def history(identifier: str, run: str, first: int, count: int):
        s = call(service().session, identifier)
        return dict(run=run, samples=call(s.history, run, first, count))

    @router.post("/sessions/{identifier}/commands")
    def command(identifier: str, request: Command, owner: Owner = None):
        return dispatch_command(service(), identifier, owner, request)

    @router.get("/sessions/{identifier}/export")
    def export(identifier: str):
        return call(service().session, identifier).export()

    return router
