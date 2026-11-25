"""HTTP session lifecycle shared by the independent CSTR comparison modules."""
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from .control_http import Owner, call, presence_stream


class Create(BaseModel):
    settings: dict[str, Any] | None = None
    auto_start: bool = True


class SessionCommand(BaseModel):
    kind: str
    key: str
    run: str
    settings: dict[str, Any] | None = None


def dispatch_command(service, identifier, owner, request: SessionCommand):
    session = call(service.session, identifier, owner, True)
    return call(session.command, request.kind, request.key, request.run, request.settings)


def comparison_router(prefix, service):
    """Resolve the current service on each request, including after app startup."""
    router = APIRouter(prefix=prefix)

    @router.post("/sessions")
    def create(request: Create):
        session = call(service().create, request.settings, lease=True, auto_start=request.auto_start)
        return dict(id=session.id, owner=session.owner, run=session.run)

    @router.get("/sessions/{identifier}/presence")
    async def connected(identifier: str, request: Request, owner: Owner = None):
        session = call(service().session, identifier, owner, True)
        return presence_stream(session, request)

    @router.delete("/sessions/{identifier}")
    def release(identifier: str, owner: Owner = None):
        registry = service()
        session = call(registry.session, identifier, owner, True)
        registry.release(session)
        return {"released": True}

    return router
