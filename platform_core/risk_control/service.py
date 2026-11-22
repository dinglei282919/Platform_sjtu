"""Shared application entry points for the Qt and HTTP adapters."""
import json
import os
import re
from concurrent.futures import CancelledError
from dataclasses import asdict
from pathlib import Path
from threading import Event, RLock
from uuid import uuid4

import numpy as np
from .cstr import DEMO_INITIAL, TREND_INTERVALS, FEATURE_NAMES, NOMINAL, REFERENCE_LIMITS, SAMPLE_MIN
from .dnn_mpc import dataset, model, training, validation
from .dnn_mpc.mpc import Settings
from .session import Session
from .healing_session import HealingService
from .disturbance import options as disturbance_options
from . import NAVIGATION

STAGES = ("风险场景下控制系统智能模型训练算法", "风险场景下智能模型优化控制算法")
USER_DATA = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
TRAIN_DEFAULTS = dict(trajectories=240, length=48, epochs=320, seed=42, device="auto", lr=0.001)


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def train_options(options):
    opts = {**TRAIN_DEFAULTS, **options}
    if set(opts) != set(TRAIN_DEFAULTS):
        raise ValueError("训练参数名无效")
    ranges = dict(trajectories=(24, 600), length=(30, 120), epochs=(1, 2000), seed=(0, 2**31-1))
    for k, (lo, hi) in ranges.items():
        if type(opts[k]) is not int or not lo <= opts[k] <= hi:
            raise ValueError(f"{k} 应为 {lo}–{hi} 的整数")
    if isinstance(opts["lr"], bool) or not isinstance(opts["lr"], (int, float)) or not 0 < opts["lr"] <= 0.05:
        raise ValueError("学习率应在 (0, 0.05] 内")
    training.device_for(opts["device"])
    return opts


class ControlService:
    def __init__(self, tasks, root=None):
        self.tasks = tasks
        self.root = Path(root or os.environ.get("PLATFORM_CSTR_RUNTIME")
                         or USER_DATA/"Platform_sjtu"/"cstr")
        self.record_root = tasks.register_record_root(self.root/"tasks")
        self.healing = HealingService(self.root/"self-healing")
        self.lock = RLock()
        self.sessions = {}

    def models(self):
        default = model.load()
        result = [{"id": "default", "version": default.version, "label": "默认 CSTR 模型"}]
        for p in sorted((self.root/"models").glob("*.npz")):
            if re.fullmatch("[a-f0-9]{20}", p.stem):
                result.append({"id": p.stem, "version": p.stem, "label": p.stem})
        return result

    def config(self):
        return dict(models=self.models(), disturbances=disturbance_options(), settings=asdict(Settings(reference=433.0)), training=TRAIN_DEFAULTS,
                    navigation=NAVIGATION, stages=STAGES, reference_limits=REFERENCE_LIMITS,
                    period=SAMPLE_MIN, trend_intervals=TREND_INTERVALS, demo_initial=DEMO_INITIAL.tolist(), features=FEATURE_NAMES, nominal=NOMINAL.tolist())

    def get_model(self, identifier):
        if identifier == "default":
            return model.load()
        if not isinstance(identifier, str) or not re.fullmatch("[a-f0-9]{20}", identifier):
            raise ValueError("模型标识无效")
        path = self.root/"models"/(identifier+".npz")
        if not path.is_file():
            raise ValueError("找不到已发布模型")
        return model.load(path)

    def create(self, *, model_id="default", settings=None, lease=False, auto_start=False):
        defaults = {"reference": 433.0} if auto_start else {}
        config = Settings(**{**defaults, **(settings or {})})
        predictor = self.get_model(model_id)
        if not predictor.metadata.get("acceptance", {}).get("passed"):
            raise ValueError("模型缺少闭环验证记录")
        session = Session(self.root, predictor, config, lease, auto_start, submit_update=self.update_model)
        with self.lock:
            self.sessions = {k: s for k, s in self.sessions.items() if not s.closed}
            self.sessions[session.id] = session
        return session

    def session(self, identifier, owner=None, control=False):
        with self.lock:
            s = self.sessions.get(identifier)
        if s is None or s.closed:
            raise KeyError("会话不存在或已释放")
        if control and owner != s.owner:
            raise PermissionError("此会话处于只读模式")
        return s

    def command(self, s, kind, key, value=None, model_id=None, settings=None, *, preset=None, duration_cycles=None):
        if kind == "model":
            return s.command(kind, key, reset_model=self.get_model(model_id))
        if kind == "reset":
            predictor = self.get_model(model_id) if model_id else None
            cfg = Settings(**settings) if settings is not None else None
            return s.command(kind, key, reset_model=predictor, settings=cfg)
        return s.command(kind, key, value=value, preset=preset, duration_cycles=duration_cycles)

    def _job(self, title, work, stop=None):
        stop = stop or Event()
        directory = self.root/"jobs"/uuid4().hex
        directory.mkdir(parents=True)
        progress = directory/"progress.json"

        def execute(log):
            history = []

            def update(epoch, epochs, row):
                history.append(row)
                if epoch == 1 or epoch == epochs or epoch % max(1, epochs//50) == 0:
                    write_json(progress, dict(percent=round(75*epoch/epochs), message="训练中",
                                              history=history[-400:]))

            def stage(percent, message):
                write_json(progress, dict(percent=percent, message=message, history=history[-400:]))
            try:
                result = work(update, stage, stop)
                write_json(directory/"result.json", result)
                stage(100, "验证完成")
                return result
            except InterruptedError as exc:
                raise CancelledError() from exc
        return self.tasks.submit(title, execute, progress_path=progress, artifact_dir=directory, cancel_event=stop, record_root=self.record_root)

    def train(self, options, csv_data=None):
        opts = train_options(options)

        def work(update, stage, stop):
            stage(0, "读取外部 CSV 轨迹" if csv_data is not None else "生成 CSTR 轨迹")
            if csv_data is None:
                data = dataset.generate(opts["trajectories"], opts["length"], opts["seed"])
                source = dict(kind="generated", trajectories=len(data), length=opts["length"])
            else:
                data = dataset.read_csv(csv_data)
                source = dict(kind="csv", trajectories=len(data), intervals=sum(len(run)-1 for run in data.runs))
            source.update(dataset=data.identifier, period_min=SAMPLE_MIN)
            tr, va, te = dataset.split(data)
            net, history = training.fit(tr, va, epochs=opts["epochs"], device=opts["device"], lr=opts["lr"],
                                        seed=opts["seed"], progress=update, cancelled=stop.is_set)
            stage(80, "独立轨迹与机理闭环验证")
            check = validation.check(net, va, Settings(), cancelled=stop.is_set)
            if csv_data is not None:
                # A short external trajectory alone cannot establish model readiness for this plant.
                independent = validation.check(net, dataset.generate(12, 32, 812), Settings(), cancelled=stop.is_set)
                check["mechanistic_dataset"] = independent
                check["passed"] = check["passed"] and independent["passed"]
                check["reasons"].extend(independent["reasons"])
            result = dict(data_source=source, validation=check, test=training.errors(net, te), history=history,
                          model=None, device=net.metadata["training_device"])
            if stop.is_set():
                raise InterruptedError()
            if check["passed"]:
                net.metadata.update(acceptance={"passed": True, "validation_dataset": va.identifier,
                                                "limits_version": "cstr-stage1-v1"},
                                    test_errors=result["test"], training_options=opts, data_source=source)
                if csv_data is None:
                    net.metadata["data_generation"] = opts
                result["model"] = model.save(net, self.root/"models"/(net.version+".npz"))
            return result
        return self._job("CSTR 预测模型训练", work).id

    def configure_updates(self, s, enabled, device):
        if type(enabled) is not bool:
            raise ValueError("自动更新开关须为布尔值")
        training.device_for(device)
        with s.lock:
            cancel_id = s.update["id"] if not enabled and s.update and s.update["automatic"] else None
            policy = s.configure_updates(enabled, device)
        if cancel_id:
            self.tasks.cancel(cancel_id)
        return policy

    def update_model(self, s, device="auto", *, automatic=False):
        if not automatic:
            training.device_for(device)
        with s.update_submit_lock:
            with s.lock:
                if automatic:
                    if not s._update_due():
                        return None
                    device = s.update_device
                info, data = s.prepare_update(automatic=automatic)
                if data is None:
                    return info["id"]
                config = s.settings
            try:
                return self._submit_update(s, info, data, config, device)
            except Exception as exc:
                s.finish_update(info, None, {"reason": str(exc)})
                raise

    def _submit_update(self, s, info, data, config, device):
        base, trajectory = data
        registered = Event()

        def phase(name):
            with s.lock:
                if s.update is not info or info["cancel"].is_set():
                    raise InterruptedError("已取消")
                info["phase"] = name

        def work(update, stage, stop):
            registered.wait()
            try:
                phase("training")
                count = trajectory.features.shape[1]
                cut = int(0.7*count)
                anchors = dataset.generate(16, 30, 93)
                ax, ay = anchors.flat()
                # One unused transition separates fit and validation windows.
                rx = trajectory.features[0, :cut]
                ry = trajectory.targets[0, :cut]
                tr = dataset.Trajectories(np.concatenate(
                    (ax, rx))[None], np.concatenate((ay, ry))[None])
                recent = dataset.Trajectories(
                    trajectory.features[:, cut+1:], trajectory.targets[:, cut+1:])
                selection = dataset.generate(10, 30, 94)
                net, history = training.fit(tr, selection, initial=base, epochs=6, lr=0.00001, device=device,
                                            progress=update, cancelled=stop.is_set)
                phase("validating")
                stage(80, "对照活动版本并验证近期工况")
                independent = dataset.generate(12, 32, 812)
                check = validation.check(net, independent, config, active=base,
                                         recent=recent, cancelled=stop.is_set)
                if stop.is_set():
                    raise InterruptedError()
                net.metadata["acceptance"] = {"passed": check["passed"], "base": base.version}
                if check["passed"]:
                    path = self.root/"runs"/info["run"]/(net.version+".npz")
                    model.save(net, path)
                    candidate = model.load(path)
                else:
                    candidate = None
                ready = s.finish_update(info, candidate, check)
                return dict(ready=ready, version=net.version, base=base.version, cutoff=info["cutoff"],
                            validation=check, history=history)
            except BaseException as exc:
                s.finish_update(info, None, {"reason": str(exc) or "已取消"})
                raise

        task = self._job("CSTR 在线模型更新", work, info["cancel"])
        try:
            with s.lock:
                info["id"] = task.id
                if s.update is info:
                    s._event("update_requested", job=task.id, automatic=info["automatic"], device=device,
                             base=info["base"], first=info["first"], cutoff=info["cutoff"])
                else:
                    info["cancel"].set()
        except Exception:
            info["cancel"].set()
            raise
        finally:
            # The task may start as soon as submit() returns. Publish its ID before training.
            registered.set()
        if info["cancel"].is_set():
            self.tasks.cancel(task.id)
        return task.id

    def cancel(self, task_id):
        with self.lock:
            sessions = list(self.sessions.values())
        for s in sessions:
            with s.lock:
                if s.update and s.update["id"] == task_id:
                    s._event("update_cancelled", job=task_id)
                    s._cancel_update()
        return self.tasks.cancel(task_id)

    def release(self, s):
        with self.lock:
            self.sessions.pop(s.id, None)
        s.close()

    def close(self):
        self.healing.close()
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for s in sessions:
            s.close()
