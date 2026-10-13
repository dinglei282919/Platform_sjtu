"""Native path dialogs for the local-only browser/server edition."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
PathDialogKind = Literal["directory", "open-mat", "save-mat", "open-npz", "save-npz"]
_DIALOG_LOCK = Lock()


def _initial_location(initial_path: str) -> tuple[Path, str]:
    text = initial_path.strip()
    candidate = Path(text) if text else ROOT
    if candidate.is_dir():
        return candidate, ""
    if candidate.parent.is_dir():
        return candidate.parent, candidate.name
    return ROOT, candidate.name if text else ""


def choose_local_path(kind: PathDialogKind, initial_path: str = "") -> str:
    """Open one native dialog and return an absolute local path or an empty string."""

    initial_dir, initial_name = _initial_location(initial_path)
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        with _DIALOG_LOCK:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            if kind == "directory":
                selected = filedialog.askdirectory(
                    parent=root,
                    title="\u9009\u62e9\u76ee\u5f55",
                    initialdir=str(initial_dir),
                    mustexist=True,
                )
            elif kind in ("open-mat", "open-npz"):
                extension = ".npz" if kind == "open-npz" else ".mat"
                selected = filedialog.askopenfilename(
                    parent=root,
                    title="选择 " + extension.upper() + " 文件",
                    initialdir=str(initial_dir),
                    initialfile=initial_name,
                    filetypes=((extension.upper() + " 文件", "*" + extension), ("所有文件", "*.*")),
                )
            elif kind in ("save-mat", "save-npz"):
                extension = ".npz" if kind == "save-npz" else ".mat"
                selected = filedialog.asksaveasfilename(
                    parent=root,
                    title="\u9009\u62e9\u6a21\u578b\u4fdd\u5b58\u4f4d\u7f6e",
                    initialdir=str(initial_dir),
                    initialfile=initial_name or ("trained_model.npz" if kind == "save-npz" else "process_control_nn_model.mat"),
                    defaultextension=extension,
                    filetypes=((extension.upper() + " 文件", "*" + extension), ("所有文件", "*.*")),
                )
            else:  # pragma: no cover - guarded by the API request model
                raise ValueError(f"\u4e0d\u652f\u6301\u7684\u8def\u5f84\u9009\u62e9\u7c7b\u578b\uff1a{kind}")
    except Exception as exc:  # pragma: no cover - depends on the desktop session
        raise RuntimeError(f"\u65e0\u6cd5\u6253\u5f00\u672c\u673a\u8def\u5f84\u9009\u62e9\u5bf9\u8bdd\u6846\uff1a{exc}") from exc
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    return str(Path(selected).resolve()) if selected else ""
