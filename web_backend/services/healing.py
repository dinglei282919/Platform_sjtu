"""HTTP adapter for the common self-healing session."""
from typing import Any, Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel

from platform_core.risk_control.healing import options
from platform_core.risk_control.healing_display import event_label, view


class Create(BaseModel):
    settings: dict[str, Any] | None = None
    auto_start: bool = True


class Command(BaseModel):
    kind: Literal["start", "pause", "resume", "reset", "end", "degrade"]
    key: str
    run: str
    settings: dict[str, Any] | None = None


def routes(service, call, presence):
    router = APIRouter(prefix="/api/cstr/healing")
    owner_header = Header(None, alias="X-Control-Owner")

    @router.get("/config")
    def config():
        return options()

    @router.post("/sessions")
    def create(request: Create):
        s = call(service().create, request.settings, lease=True, auto_start=request.auto_start)
        return dict(id=s.id, owner=s.owner, run=s.run)

    @router.get("/sessions/{identifier}")
    def snapshot(identifier: str, after: int = -1, run: str | None = None, owner: str | None = owner_header):
        data = call(service().session, identifier).snapshot(after, owner, run)
        data["view"] = view(data)
        data["recent"] = [dict(e, label=event_label(e)) for e in data["recent"]]
        return data

    @router.get("/sessions/{identifier}/presence")
    async def connected(identifier: str, request: Request, owner: str | None = owner_header):
        return presence(call(service().session, identifier, owner, True), request)

    @router.get("/sessions/{identifier}/history")
    def history(identifier: str, run: str, first: int, count: int):
        s = call(service().session, identifier)
        return dict(run=run, samples=call(s.history, run, first, count))

    @router.post("/sessions/{identifier}/commands")
    def command(identifier: str, request: Command, owner: str | None = owner_header):
        s = call(service().session, identifier, owner, True)
        return call(s.command, request.kind, request.key, request.run, request.settings)

    @router.get("/sessions/{identifier}/export")
    def export(identifier: str):
        return call(service().session, identifier).export()

    @router.delete("/sessions/{identifier}")
    def release(identifier: str, owner: str | None = owner_header):
        s = call(service().session, identifier, owner, True)
        service().release(s)
        return {"released": True}

    return router
