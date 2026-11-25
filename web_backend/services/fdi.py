"""HTTP transport for the two independently owned module-3 trials."""
from typing import Literal

from platform_core.risk_control.fdi import GROUPS, SCENARIOS, options
from platform_core.risk_control.fdi_display import CHARTS, FLOW_CHART, loop_tokens, nominal_preview, preview_marker, view
from platform_core.risk_control.presentation import marker_tokens, tokens

from .comparison import Owner, SessionCommand, comparison_router, dispatch_command


class Command(SessionCommand):
    kind: Literal["start", "resume", "pause", "end", "reset", "begin", "finish"]


def routes(service, call, presence):
    router = comparison_router("/api/cstr/fdi", service, call, presence)

    @router.get("/config")
    def config():
        previews = {side: {case: marker_tokens(preview_marker(side, case)) for case in cases}
                    for side, cases in SCENARIOS.items()}
        nominal = nominal_preview("measurement", "healthy")
        return dict(**options(), charts=CHARTS, flow_chart=FLOW_CHART, previews=previews,
                    loop_labels={arm: loop_tokens(arm) for groups in GROUPS.values() for arm in groups},
                    nominal_display={side: tokens(nominal_preview(side, "healthy")) for side in GROUPS},
                    nominal_flow=nominal["Qc"])

    @router.get("/sessions/{identifier}")
    def snapshot(identifier: str, after: int = -1, run: str | None = None, owner: Owner = None):
        data = call(service().session, identifier).snapshot(after, owner, run)
        return dict(**data, view=view(data))

    @router.get("/sessions/{identifier}/history")
    def history(identifier: str, run: str, first: int, count: int):
        session = call(service().session, identifier)
        data = call(session.window, run, first, count)
        return dict(**data, view=view(data, live=False))

    @router.post("/sessions/{identifier}/commands")
    def command(identifier: str, request: Command, owner: Owner = None):
        return dispatch_command(service(), call, identifier, owner, request)

    @router.get("/sessions/{identifier}/export")
    def export(identifier: str, run: str | None = None):
        return call(call(service().session, identifier).export, run)

    return router
