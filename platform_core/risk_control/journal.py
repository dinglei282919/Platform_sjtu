"""Append-only run journal. The numerical thread enqueues immutable records."""
import json
import queue
from pathlib import Path
from threading import Event, Thread


class Journal:
    def __init__(self, root, run):
        self.path = Path(root) / "runs" / f"{run}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        self.queue = queue.Queue(2048)
        self.sample_offsets = []
        self.error = None
        self.thread = Thread(target=self._write, daemon=True, name="cstr-journal")
        self.thread.start()

    def append(self, kind, data):
        if self.error:
            raise OSError(self.error)
        try:
            self.queue.put_nowait((kind, data))
        except queue.Full as exc:
            raise RuntimeError("实验记录缓冲已满") from exc

    def _write(self):
        with self.stream as out:
            while True:
                kind, value = self.queue.get()
                if kind == "close":
                    return
                if kind == "flush":
                    try:
                        out.flush()
                    except OSError as exc:
                        self.error = str(exc)
                    value.set()
                    continue
                try:
                    if kind == "sample" and value["seq"] % 128 == 0:
                        self.sample_offsets.append(out.tell())
                    out.write(json.dumps({"kind": kind, **value}, ensure_ascii=False, allow_nan=False)+"\n")
                except (OSError, ValueError) as exc:
                    self.error = str(exc)

    def barrier(self):
        done = Event()
        if self.thread.is_alive():
            self.queue.put(("flush", done))
        else:
            done.set()
        return done

    def close(self):
        if self.thread.is_alive():
            self.queue.put(("close", None))
            self.thread.join()

    def samples_between(self, barrier, first, last):
        if not barrier.wait(10) or self.error:
            raise OSError(self.error or "读取历史样本超时")
        rows = []
        with self.path.open(encoding="utf-8") as stream:
            stream.seek(self.sample_offsets[first // 128])
            for line in stream:
                row = json.loads(line)
                if row["kind"] == "sample" and row["seq"] >= first:
                    rows.append(row)
                    if row["seq"] >= last:
                        break
        return rows

    def records(self, barrier, seq, event_number):
        if not barrier.wait(10):
            raise RuntimeError("保存实验记录超时")
        if self.error:
            raise OSError(self.error)
        with self.path.open(encoding="utf-8") as inp:
            records = []
            last_seq, last_event = -1, 0
            for line in inp:
                item = json.loads(line)
                records.append(item)
                if item["kind"] == "sample":
                    last_seq = item["seq"]
                else:
                    last_event = item["event_id"]
                if last_seq == seq and last_event == event_number:
                    break
        return records
