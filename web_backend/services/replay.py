"""Local UI transport for independent CSTR observer trials."""
from typing import Literal

from platform_core.risk_control.presentation import tokens
from platform_core.risk_control.replay_display import diagram_sample
from platform_core.risk_control.replay_session import options

from .comparison import Owner, SessionCommand, comparison_router, dispatch_command


class Command(SessionCommand):
    kind: Literal["pause", "resume", "reset", "begin", "finish", "cancel"]


def routes(service, call, presence):
    router = comparison_router("/api/cstr/replay", service, call, presence)

    @router.get("/config")
    def config():
        return options()

    @router.get("/sessions/{identifier}")
    def frame(identifier: str, run: str | None = None, at: int | None = None, arm: str = "guard",
              owner: Owner = None):
        session = call(service().session, identifier)
        data = call(session.frame, run, at, owner, arm)
        sample = diagram_sample(data["point"], data["arm"], data["display"])
        data["process"] = {"labels": tokens(sample), "flow": sample["Qc"]} if sample else None
        return data

    @router.post("/sessions/{identifier}/commands")
    def command(identifier: str, body: Command, owner: Owner = None):
        return dispatch_command(service(), call, identifier, owner, body)

    return router
