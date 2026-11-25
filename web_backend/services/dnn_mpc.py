"""HTTP adapter for DNN-MPC training, live control and model updates."""
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from platform_core.risk_control.dnn_mpc.dataset import MAX_CSV_BYTES, example_csv
from platform_core.risk_control.presentation import control_status, online_update_status, template, tokens, trial_status

from .control_http import Owner, call, presence_stream


class CstrCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = "default"
    auto_start: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)


class CstrCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["start", "pause", "resume", "reset", "end", "reference", "model", "disturbance", "clear_disturbance"]
    key: str = Field(min_length=1, max_length=128)
    preset: str | None = None
    duration_cycles: int | None = Field(default=None, strict=True)
    value: float | None = None
    model_id: str | None = None
    settings: dict[str, Any] | None = None


class CstrUpdate(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"


class CstrUpdatePolicy(CstrUpdate):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(strict=True)


def routes(service, task_manager):
    router = APIRouter(prefix="/api/cstr")

    @router.get("/config")
    def cstr_config():
        return {**service().config(), "diagram_template": template()}

    @router.post("/training")
    def cstr_train(options: dict[str, Any]):
        return {"task_id": call(service().train, options)}

    @router.get("/training/example.csv")
    def cstr_example():
        return Response(example_csv(), media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="cstr-trajectories.csv"'})

    @router.post("/training/csv")
    async def cstr_train_csv(request: Request, epochs: int = 320, seed: int = 42,
                             lr: float = 0.001, device: str = "auto"):
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_CSV_BYTES:
                raise HTTPException(413, "CSV 文件超过 32 MiB")
            content.extend(chunk)
        if not content:
            raise HTTPException(400, "请选择非空 CSV 文件")
        options = dict(epochs=epochs, seed=seed, lr=lr, device=device)
        return {"task_id": call(service().train, options, bytes(content))}

    @router.get("/training/{task_id}/report")
    def cstr_training_report(task_id: str):
        manager = task_manager()
        task = manager.get(task_id)
        if task is None:
            raise HTTPException(404, "任务不存在或任务记录已被移除")
        directory = manager.artifact_dir(task_id)
        if task["title"] != "CSTR 预测模型训练" or not directory or not (directory / "result.json").is_file():
            raise HTTPException(404, "训练报告尚不可用")
        return FileResponse(directory / "result.json", filename="cstr-training.json")

    @router.post("/sessions")
    def cstr_create(request: CstrCreate):
        session = call(service().create, model_id=request.model_id, settings=request.settings,
                       lease=True, auto_start=request.auto_start)
        return {"id": session.id, "owner": session.owner, "run": session.run}

    @router.get("/sessions/{identifier}")
    def cstr_snapshot(identifier: str, after: int = -1, owner: Owner = None):
        session = call(service().session, identifier)
        data = session.snapshot(after, owner)
        online = data["online_update"]
        return {
            **data,
            "control_display": control_status(data),
            "display": tokens(data["latest"]),
            "disturbance_display": trial_status(data["disturbance"]),
            "online_update": {**online, **online_update_status(online)},
        }

    @router.get("/sessions/{identifier}/presence")
    async def cstr_presence(identifier: str, request: Request, owner: Owner = None):
        session = call(service().session, identifier, owner, True)
        return presence_stream(session, request)

    @router.get("/sessions/{identifier}/history")
    def cstr_history(identifier: str, run: str, first: int, count: int):
        session = call(service().session, identifier)
        return {"run": run, "samples": call(session.sample_window, run, first, count)}

    @router.post("/sessions/{identifier}/commands")
    def cstr_command(identifier: str, request: CstrCommand, owner: Owner = None):
        session = call(service().session, identifier, owner, True)
        return call(service().command, session, request.kind, request.key, request.value,
                    request.model_id, request.settings, preset=request.preset,
                    duration_cycles=request.duration_cycles)

    @router.post("/sessions/{identifier}/updates")
    def cstr_update(identifier: str, request: CstrUpdate, owner: Owner = None):
        session = call(service().session, identifier, owner, True)
        return {"task_id": call(service().update_model, session, request.device)}

    @router.post("/sessions/{identifier}/updates/policy")
    def cstr_update_policy(identifier: str, request: CstrUpdatePolicy, owner: Owner = None):
        session = call(service().session, identifier, owner, True)
        policy = call(service().configure_updates, session, request.enabled, request.device)
        return {**policy, **online_update_status(policy)}

    @router.post("/sessions/{identifier}/updates/{task_id}/cancel")
    def cstr_cancel_update(identifier: str, task_id: str, owner: Owner = None):
        session = call(service().session, identifier, owner, True)
        with session.lock:
            if not session.update or session.update["id"] != task_id:
                raise HTTPException(409, "此更新已结束或不属于本次运行")
        return call(service().cancel, task_id)

    @router.get("/sessions/{identifier}/export")
    def cstr_export(identifier: str):
        session = call(service().session, identifier)
        return JSONResponse(session.export(), headers={"Content-Disposition": 'attachment; filename="cstr-run.json"'})

    @router.delete("/sessions/{identifier}")
    def cstr_release(identifier: str, owner: Owner = None):
        session = call(service().session, identifier, owner, True)
        service().release(session)
        return {"status": "closed"}

    return router
