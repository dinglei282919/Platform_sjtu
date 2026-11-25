"""HTTP error mapping, owner headers and presence for CSTR sessions."""
import asyncio
from typing import Annotated

from fastapi import Header, HTTPException
from fastapi.responses import StreamingResponse


Owner = Annotated[str | None, Header(alias="X-Control-Owner")]


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


def presence_stream(session, request):
    async def connected():
        session.owner_connected()
        try:
            while not session.closed and not await request.is_disconnected():
                yield b": online\n\n"
                await asyncio.sleep(5)
        finally:
            session.owner_disconnected()

    return StreamingResponse(connected(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
