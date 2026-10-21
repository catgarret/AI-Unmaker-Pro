def _normalise_dropped_image_path(tk_interpreter, data):
    """Read a single native TkDND file-list entry without rewriting its name."""
    import os
    from pathlib import Path
    import re
    import tkinter
    from urllib.parse import unquote, urlsplit

    try:
        paths = tk_interpreter.splitlist(data)
    except (tkinter.TclError, TypeError, ValueError) as exc:
        raise ValueError("파일 경로를 읽지 못했습니다. 클릭해서 이미지를 선택해 주세요.") from exc
    if not paths:
        raise ValueError("이미지 파일을 드래그 앤 드롭해 주세요.")
    if len(paths) != 1:
        raise ValueError("이미지를 한 장씩 드래그 앤 드롭해 주세요.")
    value = str(paths[0])
    if not value or "\x00" in value:
        raise ValueError("올바른 이미지 파일을 선택해 주세요.")
    # Finder/TkDND normally sends native paths. Decode URI escapes only for an
    # actual file URI: a literal '%' or '%20' in a filename must stay literal.
    if value.lower().startswith("file:"):
        try:
            uri = urlsplit(value)
            if uri.netloc.lower() not in ("", "localhost") or uri.query or uri.fragment:
                raise ValueError("remote or ambiguous file URI")
            value = unquote(uri.path, encoding="utf-8", errors="strict")
        except (UnicodeError, ValueError) as exc:
            raise ValueError("컴퓨터에 저장된 이미지 파일을 선택해 주세요.") from exc
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", value):
            value = value[1:]
    elif re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) and not re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError("웹 주소 대신 컴퓨터에 저장된 이미지 파일을 선택해 주세요.")
    if "\x00" in value or value.startswith(("//", "\\\\")):
        raise ValueError("컴퓨터에 저장된 이미지 파일을 선택해 주세요.")
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("이미지 파일의 전체 경로를 확인하지 못했습니다.")
        if path.is_dir():
            raise ValueError("폴더 대신 이미지 파일을 선택해 주세요.")
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ValueError("PNG, JPG 또는 WEBP 이미지 파일을 선택해 주세요.")
        if not path.is_file():
            raise ValueError("이미지 파일을 찾을 수 없습니다. 파일 위치를 확인해 주세요.")
        return str(path.resolve(strict=True))
    except OSError as exc:
        raise ValueError("이미지 파일에 접근하지 못했습니다. 파일 위치와 권한을 확인해 주세요.") from exc


def _ui_local_model_progress(payload):
    """Describe the model/file named by an actual setup event."""
    title, detail = str(payload.get("title", "")), str(payload.get("detail", ""))
    phase = str(payload.get("phase", ""))
    text = title + " " + detail
    names = (("qwen", "Qwen 이미지 모델", "Frederic75/Qwen-Image-2512-GGUF"),
             ("lightning", "Lightning 가속 모델", "lightx2v/Qwen-Image-2512-Lightning"),
             ("vae", "이미지 변환 모델", "Qwen/Qwen-Image-2512"),
             ("embeddings", "처리 설정", "0xROOTPLS/DeSynth"))
    for index, (key, name, repo_hint) in enumerate(names, 1):
        repo = (globals().get("MODEL_SPECS", {}).get(key) or {}).get("repo_id", repo_hint)
        if repo not in text and repo_hint not in text:
            continue
        filename = detail.split(" · ", 1)[1] if " · " in detail else ""
        if phase == "retry":
            activity = "연결을 다시 시도합니다. 받은 파일은 이어서 사용합니다."
            friendly = name + " 다운로드를 이어가고 있습니다"
        elif phase == "connecting":
            activity = "다운로드 서버에 연결하고 있습니다."
            friendly = name + " 다운로드 연결 중"
        elif phase in ("verify", "ready"):
            activity = "받은 파일을 확인하고 있습니다."
            friendly = name + " 파일을 확인하고 있습니다"
        else:
            activity = "필요한 파일을 다운로드하고 있습니다."
            friendly = name + "을 다운로드하고 있습니다"
        return {"title": friendly, "headline": f"구성 {index}/{len(names)} · {name}",
                "activity": filename + "\n" + activity if filename else activity}
    if "모델" in text or (payload.get("total_bytes") is not None and phase in
                             ("download", "connecting", "retry", "verify", "ready")):
        return {"title": "이미지 처리 모델을 확인하고 있습니다",
                "headline": "Qwen · Lightning",
                "activity": detail or "모델 파일의 준비와 검증을 마친 뒤 사용할 수 있습니다."}
    return None


def _ui_metadata_evidence(scan):
    """Project observed file records into the existing summary/card widgets.

    The backend owns provenance interpretation. In particular, a preferred
    recorded application is a file-record observation, not a pixel-AI verdict.
    Old saved scan reports remain readable through the original field layout.
    """
    import re

    scan = scan if isinstance(scan, dict) else {}
    record = scan.get("metadata") or {}
    record = record if isinstance(record, dict) else {}
    fields = record.get("metadata") or {}
    fields = fields if isinstance(fields, dict) else {}
    observation = scan.get("metadata_observation") or {}
    observation = observation if isinstance(observation, dict) else {}
    labels = {"c2pa": "C2PA", "exif": "EXIF", "xmp": "XMP", "text": "텍스트",
              "jumbf": "JUMBF", "comment": "코멘트", "other": "기타"}
    # Presence is still grounded in the actual per-format parser result; a
    # summary/status flag must not manufacture a missing record category.
    categories = [key for key in labels
                  if isinstance(fields.get(key), dict) and fields[key].get("present") is True]
    declared = list(dict.fromkeys(str(item["name"])
        for item in (scan.get("synthid") or {}).get("declarations", [])
        if isinstance(item, dict) and item.get("name")))
    names = list(dict.fromkeys(declared + [labels[key] for key in categories]))
    apps = record.get("recorded_apps") or []
    apps = apps if isinstance(apps, list) else []
    preferred = observation.get("primary_app")
    ordered = ([preferred] if isinstance(preferred, dict) else []) + apps
    app_names = list(dict.fromkeys(re.sub(r"\s+", " ", str(item.get("name") or "")).strip()
        for item in ordered if isinstance(item, dict) and item.get("name")))
    c2pa = fields.get("c2pa") or {}
    incomplete = (observation.get("status") == "incomplete" or c2pa.get("status") in
                  ("error", "unavailable", "malformed", "invalid", "external_reference"))
    return {"names": names, "categories": categories, "apps": app_names,
            "present": bool(names), "incomplete": incomplete,
            "checked": bool(fields)}


def _ui_compact_evidence(values, limit=3):
    return " · ".join(values[:limit]) + (f" 외 {len(values)-limit}종" if len(values) > limit else "")


def launch_gui():
    import json
    import math
    import os
    from pathlib import Path
    import queue
    import re
    import sys
    import tempfile
    import threading
    app_title = "AI Unmarker Pro v" + APP_VERSION
    try:
        import tkinter as tk
        from tkinter import filedialog, font as tkfont
    except ImportError as exc:
        raise RuntimeError("기본 화면 구성 요소 tkinter가 없습니다. tkinter가 포함된 Python 또는 실행파일 배포본을 사용해 주세요.") from exc
    try:
        import customtkinter as ctk
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        bootstrap = globals().get("bootstrap_ui_dependencies")
        if not callable(bootstrap):
            raise RuntimeError("화면 실행에 필요한 구성 요소가 없습니다. 실행파일 배포본을 사용해 주세요.") from exc
        if not bootstrap():
            return
        try:
            import customtkinter as ctk
            from PIL import Image, ImageDraw, ImageOps
        except ImportError as retry_exc:
            raise RuntimeError("화면 구성 요소를 준비하지 못했습니다. 앱을 다시 실행해 주세요.") from retry_exc

    # TkinterDnD.require() supports an existing CustomTkinter root. Import and
    # initialise it after building the window so a missing runtime component
    # can be retried after setup, without fixing the root's class at startup.
    COPY, REFUSE_DROP = "copy", "refuse_drop"

    C = {"bg": "#F5F6F8", "surface": "#FFFFFF", "line": "#E3E6EB",
         "ink": "#101827", "sub": "#687180", "soft": "#F1F3F6",
         "green": "#179C79", "amber": "#AD650F", "red": "#B54747"}
    font_name = ui_font_family()

    def thumb(path):
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGBA")
            img.thumbnail((1400, 1200), Image.Resampling.LANCZOS)
            return img.copy()

    def size_text(value):
        value = max(0, float(value or 0))
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1000 or unit == "GB":
                return "%s %s" % (("%.1f" % value) if unit != "B" else str(int(value)), unit)
            value /= 1000

    def duration_text(seconds):
        if seconds is None:
            return None
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds < 0:
            return None
        if seconds < 60:
            return "약 %d초 남음" % max(1, round(seconds))
        return "약 %d분 남음" % max(1, math.ceil(seconds / 60))

    def progress_payload(*args, **kwargs):
        if args and isinstance(args[0], dict):
            return dict(args[0])
        return {"title": str(args[0]) if args else str(kwargs.get("status", "준비 중")),
                "detail": str(args[1]) if len(args) > 1 else str(kwargs.get("detail", ""))}

    def setup_error_message(payload):
        fallback = "준비를 마치지 못했습니다. 상세 로그를 확인해 주세요."
        if isinstance(payload, dict):
            cause = str(payload.get("cause") or "").strip()
            raw = cause or payload.get("error") or payload.get("detail") or fallback
        else:
            raw = payload
        lines = str(raw or "").splitlines()
        causes = [line.split(":", 1)[1].strip() for line in lines
                  if line.lstrip().startswith("원인:") and line.split(":", 1)[1].strip()]
        if causes:
            lines = causes + lines
        elif any(line.lstrip().startswith("Traceback") for line in lines):
            lines = list(reversed(lines))
        for line in lines:
            line = line.strip()
            if not line or line.startswith(("Traceback", "File \"", "File '", "로그 파일:", "설치 기록:")):
                continue
            line = re.sub(r"^[A-Za-z_][\w.]*(?:Error|Exception):\s*", "", line)
            path = re.search(r"https?://|[A-Za-z]:[\\/]|(?<!\w)/\S+", line)
            if path:
                line = line[:path.start()].rstrip(" :")
            if line:
                return line if len(line) <= 150 else line[:147].rstrip() + "…"
        return fallback

    class AIUnmarkerApp(ctk.CTk):
        def __init__(self):
            super().__init__()
            if (ctk.DrawEngine.preferred_drawing_method == "font_shapes"
                    and "CustomTkinter_shapes_font" not in tkfont.families(self)):
                ctk.DrawEngine.preferred_drawing_method = "polygon_shapes"
            self.app_identity = ui_apply_app_identity(self, app_title)
            self.geometry("1040x700")
            self.minsize(840, 610)
            self.configure(fg_color=C["bg"])
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=1)
            self.events = queue.Queue()
            self.close_event = threading.Event()
            self.cancel_event = threading.Event()
            self.setup_cancel = threading.Event()
            self.job = 0
            self.busy = False
            self.phase = "idle"
            self.input_path = None
            self.input_scan = None
            self.output_scan = None
            self.result = None
            self.result_dir = None
            self.result_path = None
            self.saved_path = None
            self.thumbnails = {}
            self.icon_images = {}
            self.preview_image = None
            transparent_pixel = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            self.empty_preview_image = ctk.CTkImage(
                light_image=transparent_pixel, dark_image=transparent_pixel, size=(1, 1))
            self.preview_timer = None
            self.poll_timer = None
            self.setup_window = None
            self.setup_check_timer = None
            self.setup_checked = False
            self.setup_ready = False
            self.setup_busy = False
            self.setup_stage = "checking"
            self.setup_check_started = False
            self.setup_estimate_data = {}
            self.setup_error = None
            self.resume_after_setup = False
            self.logs = []
            self.last_error = None
            self.work_stage = ""
            self.dnd_working = False
            self.dnd_error = ""
            self.dnd_targets = {}
            self.gauge_image = None
            self.processing_options = {
                "restore_mode": DETAIL_RESTORE_POLICY["default_mode"],
                "unsharp": DETAIL_RESTORE_POLICY["default_unsharp"],
            }
            self._build()
            self._set_state("idle")
            if not self.app_identity["icon_applied"]:
                self.logs.append("앱 아이콘 파일을 찾거나 적용하지 못했습니다. 확인 경로:\n" +
                    "\n".join(self.app_identity["searched"] + self.app_identity["errors"]))
            self.protocol("WM_DELETE_WINDOW", self._close)
            self._init_drag_and_drop()
            self.poll_timer = self.after(60, self._poll)
            # Paint first, then inspect cached models before cold native imports.
            # A quick healthy start needs no dialog; slow checks use the same
            # existing compact setup window without requesting a download.
            self.after_idle(self._check_startup)

        def _init_drag_and_drop(self):
            if self.close_event.is_set() or self.dnd_working:
                return self.dnd_working
            try:
                import importlib
                importlib.invalidate_caches()
                from tkinterdnd2 import TkinterDnD, DND_FILES
                require = getattr(TkinterDnD, "require", None) or TkinterDnD._require
                self.TkdndVersion = require(self)
                self._dnd_file_type = DND_FILES
                self._register_drop_targets()
                self.dnd_working = True
                self.dnd_error = ""
                self.logs.append("드래그 앤 드롭 준비 완료 · tkdnd " + str(self.TkdndVersion))
            except Exception as exc:
                import traceback
                detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
                if detail != self.dnd_error:
                    self.logs.append("드래그 앤 드롭 시작 오류:\n" + detail +
                        "\n앱을 다시 실행해 재확인할 수 있습니다. 계속되면 이 상세 로그를 공유해 주세요.")
                self.dnd_working = False
                self.dnd_error = detail
            prefix = "\n" if self.upload_icon_image else ""
            self.empty_preview_text = prefix + (
                "이미지를 드래그 앤 드롭하거나 클릭해 이미지를 첨부하세요.\nPNG · JPG · WEBP"
                if self.dnd_working else
                "드래그 앤 드롭을 사용할 수 없습니다.\n클릭해 이미지를 첨부하세요.\nPNG · JPG · WEBP")
            self._sync_setup_gate()
            return self.dnd_working

        def _register_drop_targets(self):
            # CTkFrame.winfo_children() hides its native drawing canvas.
            # Enumerate Tk's actual children so the empty label, its image and
            # every painted preview surface receive the same native events.
            pending = [self.preview_surface]
            while pending:
                widget = pending.pop()
                pending.extend(tk.Misc.winfo_children(widget))
                name = str(widget)
                if name in self.dnd_targets:
                    continue
                bindings = []
                try:
                    widget.drop_target_register(self._dnd_file_type)
                    for event, handler in (("<<DropEnter>>", self._drop_enter),
                                           ("<<DropPosition>>", self._drop_enter),
                                           ("<<Drop>>", self._drop)):
                        bindings.append((event, widget.dnd_bind(event, handler)))
                except Exception:
                    for event, funcid in bindings:
                        widget.tk.call("bind", widget._w, event, "")
                        if funcid:
                            widget.deletecommand(funcid)
                    try:
                        widget.drop_target_unregister()
                    except tk.TclError:
                        pass
                    raise
                self.dnd_targets[name] = widget

        def _drop_allowed(self):
            return (self.dnd_working and not self.close_event.is_set()
                    and self.setup_ready and not self.setup_busy and not self.busy)

        def _drop_enter(self, event):
            # Enter/position payloads need not include paths. Returning the
            # native action is required; returning None makes Finder refuse it.
            if not self._drop_allowed():
                return REFUSE_DROP
            actions = getattr(event, "actions", ())
            if isinstance(actions, str):
                try:
                    actions = self.tk.splitlist(actions)
                except tk.TclError:
                    return REFUSE_DROP
            return COPY if not actions or COPY in actions else REFUSE_DROP

        def _drop_message(self, message):
            self.logs.append("드래그 앤 드롭: " + str(message))
            # The caption is visible even when the idle right-hand panel is
            # intentionally hidden. Keep an existing image/result intact.
            self.caption.configure(text=str(message))

        def _remember_error(self, data, stage="이미지 처리"):
            payload = dict(data) if isinstance(data, dict) else {
                "stage": stage, "error": str(data), "detail": str(data)}
            payload["stage"] = str(payload.get("stage") or stage)
            payload["error"] = str(payload.get("error") or payload.get("detail") or "원인을 확인하지 못했습니다.")
            payload["detail"] = str(payload.get("detail") or payload["error"])
            self.last_error = payload
            self.logs.append("작업 오류:\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            raw = (payload["error"] + " " + payload["detail"]).lower()
            if "out of memory" in raw or "memoryerror" in raw:
                return "이미지를 처리할 메모리가 부족합니다. 다른 앱을 닫고 다시 시도해 주세요."
            if "permission" in raw or "access denied" in raw:
                return "파일에 접근하지 못했습니다. 저장 폴더의 권한을 확인해 주세요."
            if "no space" in raw or "disk full" in raw:
                return "저장 공간이 부족합니다. 여유 공간을 확보해 주세요."
            message = setup_error_message(payload)
            # Keep stack traces and native-library details in the details view.
            if re.search(r"[가-힣]", message):
                return message if len(message) <= 88 else message[:85].rstrip() + "…"
            return "이미지 처리 중 오류가 발생했습니다. 오류 상세에서 원인을 확인할 수 있습니다."

        def label(self, parent, text, size=13, bold=False, color=None, **kwargs):
            # CTkLabel defaults to a 28 px minimum, even for a single 11 px line.
            # Let text determine its height; gauges and controls opt in to a size.
            kwargs.setdefault("height", 0)
            return ctk.CTkLabel(parent, text=text, font=(font_name, size, "bold" if bold else "normal"),
                text_color=color or C["ink"], **kwargs)

        def button(self, parent, text, command, primary=False, **kwargs):
            return ctk.CTkButton(parent, text=text, command=command, corner_radius=8,
                font=(font_name, 13, "bold"), fg_color=C["ink"] if primary else C["soft"],
                hover_color="#293345" if primary else C["line"],
                text_color="#FFFFFF" if primary else C["ink"], **kwargs)

        def _icon_image(self, name, size, color):
            key = (name, size, color)
            if key not in self.icon_images:
                image = ui_icon_image(name, size, color)
                self.icon_images[key] = None if image is None else ctk.CTkImage(
                    light_image=image, dark_image=image, size=(size, size))
            return self.icon_images[key]

        def _set_preview_content(self, image=None, text=""):
            # CTk 5.2.2 does not clear the native image for image=None. Keep a
            # transparent CTkImage for empty states and replace the image via
            # the public API before changing text or releasing its old owner.
            display_image = self.empty_preview_image if image is None else image
            self.preview_label.configure(image=display_image)
            self.preview_image = display_image
            self.preview_label.configure(text=text)

        def _build(self):
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 16))
            title_row = ctk.CTkFrame(header, fg_color="transparent")
            title_row.pack(anchor="w")
            self.label(title_row, "AI Unmarker Pro", 26, True, height=32).pack(side="left")
            self.label(title_row, "v" + APP_VERSION, 11, True, color=C["sub"],
                fg_color=C["line"], corner_radius=5, width=60, height=24).pack(side="left", padx=(12, 0))
            self.label(header, "이미지를 처리하고 메타데이터를 삭제합니다.", 13,
                       color=C["sub"]).pack(anchor="w", pady=(6, 0))
            body = ctk.CTkFrame(self, fg_color="transparent")
            body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
            body.grid_columnconfigure(0, weight=1, minsize=340)
            body.grid_columnconfigure(1, weight=0, minsize=320)
            body.grid_rowconfigure(0, weight=1)
            preview = ctk.CTkFrame(body, fg_color=C["surface"], border_color=C["line"], border_width=1, corner_radius=12)
            self.preview_surface = preview
            preview.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
            preview.grid_columnconfigure(0, weight=1)
            preview.grid_rowconfigure(1, weight=1)
            preview_tools = ctk.CTkFrame(preview, fg_color="transparent")
            preview_tools.grid(row=0, column=0, sticky="ew", padx=16, pady=12)
            self.preview_tools = preview_tools
            self.tabs = ctk.CTkSegmentedButton(preview_tools, values=["원본"], command=self._select_tab,
                font=(font_name, 12), fg_color=C["soft"], selected_color=C["line"],
                selected_hover_color=C["line"], unselected_color=C["surface"],
                unselected_hover_color=C["soft"], text_color=C["ink"], height=28)
            self.tabs.pack(side="left")
            self.tabs.set("원본")
            self.change_button = self.button(preview_tools, "파일 변경", self._pick, width=82, height=28)
            self.change_button.pack(side="right")
            self.preview_area = ctk.CTkFrame(preview, fg_color=C["surface"], corner_radius=0)
            self.preview_area.grid(row=1, column=0, sticky="nsew", padx=16)
            self.preview_area.bind("<Configure>", self._schedule_preview)
            self.upload_icon_image = self._icon_image("upload-simple", 44, C["sub"])
            self.empty_preview_text = ("\n" if self.upload_icon_image else "") + \
                "이미지를 드래그 앤 드롭하거나 클릭해 이미지를 첨부하세요.\nPNG · JPG · WEBP"
            self.preview_label = self.label(self.preview_area,
                self.empty_preview_text, 14, image=self.upload_icon_image or self.empty_preview_image, compound="top",
                color=C["sub"], justify="center", wraplength=300, cursor="hand2")
            self.preview_label.place(relx=.5, rely=.5, anchor="center")
            self.preview_label.bind("<Button-1>", lambda _event: self._pick() if not self.input_path else None)
            self.preview_area.bind("<Button-1>", lambda _event: self._pick() if not self.input_path else None)
            self.caption = self.label(preview, "원본 파일은 그대로 보관됩니다.", 11, color=C["sub"], wraplength=470)
            self.caption.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 12))

            panel = ctk.CTkFrame(body, width=320, fg_color=C["surface"], border_color=C["line"], border_width=1, corner_radius=12)
            panel.grid(row=0, column=1, sticky="nsew")
            panel.grid_propagate(False)
            panel.grid_columnconfigure(0, weight=1)
            panel.grid_rowconfigure(1, weight=1)
            summary = ctk.CTkFrame(panel, fg_color="transparent")
            summary.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 0))
            self.summary_panel = summary
            self.state_icon = ctk.CTkFrame(summary, width=210, height=108, fg_color="transparent")
            self.state_icon.pack(pady=(0, 6))
            self.state_icon.pack_propagate(False)
            self.gauge_arc = self.label(self.state_icon, "", width=210, height=108)
            self.gauge_arc.place(x=0, y=0)
            self.gauge_value = self.label(self.state_icon, "대기", 29, True, height=36)
            self.gauge_value.place(relx=.5, y=64, anchor="center")
            self.gauge_caption = self.label(self.state_icon, "처리 상태", 10, color=C["sub"], height=16)
            self.gauge_caption.place(relx=.5, y=91, anchor="center")
            self.state_title = self.label(summary, "", 18, True, justify="center", wraplength=282)
            self.state_title.pack(fill="x")
            self.state_detail = self.label(summary, "", 12, color=C["sub"], justify="center", wraplength=280)
            self.state_detail.pack(fill="x", pady=(6, 0))

            self.idle_intro = ctk.CTkFrame(panel, fg_color="transparent")
            self.idle_intro.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=20, pady=(24, 0))
            self.label(self.idle_intro, "이미지를 넣고,\n한 번에 처리하세요.", 19, True,
                justify="left", anchor="w", wraplength=270).pack(fill="x")
            feature_list = ctk.CTkFrame(self.idle_intro, fg_color="transparent")
            feature_list.pack(fill="x", pady=(16, 0))
            feature_list.grid_columnconfigure(1, weight=1)
            for row, text in enumerate(("로컬 모델을 사용한 이미지 재생성", "메타데이터·C2PA 기록 확인 및 삭제",
                                        "등록된 가시 워터마크 검사 및 제거", "원본과 결과를 비교하고 저장")):
                ctk.CTkFrame(feature_list, width=4, height=4, corner_radius=2, fg_color=C["sub"]).grid(
                    row=row, column=0, sticky="n", padx=(0, 8), pady=(6, 0))
                self.label(feature_list, text, 12, color=C["sub"], anchor="w",
                    justify="left", wraplength=248).grid(row=row, column=1, sticky="ew", pady=(0, 8))

            evidence = ctk.CTkScrollableFrame(panel, width=268, height=160,
                fg_color=C["surface"], corner_radius=0, border_width=0,
                scrollbar_fg_color=C["surface"], scrollbar_button_color=C["line"],
                scrollbar_button_hover_color="#C7CDD6")
            evidence.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(12, 0))
            self.evidence_scroll = evidence
            evidence.grid_columnconfigure(0, weight=1)
            evidence_header = ctk.CTkFrame(evidence, fg_color="transparent")
            evidence_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
            self.evidence_title = self.label(evidence_header, "처리 안내", 11, True,
                color=C["sub"])
            self.evidence_title.pack(side="left")
            self.info_button = self.button(evidence_header, "상세", self._show_info, width=42, height=20)
            self.info_button.pack(side="right")
            self.synthid_button = self.button(evidence_header, "SynthID 공식 검사", self._open_synthid, width=112, height=20)
            self.synthid_button.pack(side="right", padx=(0, 6))
            self.evidence_labels = {}
            self.evidence_cards = {}
            for row, key in enumerate(("regeneration", "pixel", "metadata", "open_watermark", "visible"), 1):
                card = ctk.CTkFrame(evidence, fg_color=C["soft"], corner_radius=8)
                card.grid(row=row, column=0, sticky="ew", pady=(0, 6))
                card.grid_columnconfigure(0, weight=1)
                heading = self.label(card, "", 10, color=C["sub"], anchor="w")
                heading.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
                value = self.label(card, "", 13, True, anchor="w", justify="left", wraplength=244)
                value.grid(row=1, column=0, sticky="ew", padx=12)
                detail = self.label(card, "", 10, color=C["sub"], anchor="w", justify="left", wraplength=244)
                detail.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
                self.evidence_labels[key] = (heading, value, detail)
                self.evidence_cards[key] = card

            self.file_summary = self.label(preview, "", 11, color=C["sub"],
                justify="center", wraplength=470)
            self.file_summary.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 0))
            self.caption.grid(row=3, column=0, sticky="ew", padx=16, pady=(6, 12))
            actions = ctk.CTkFrame(panel, fg_color="transparent")
            actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(12, 16))
            self.action_panel = actions
            self.small_status = self.label(actions, "", 10, color=C["sub"],
                justify="center", wraplength=280)
            self.work_progress = ctk.CTkProgressBar(actions, mode="indeterminate", height=4,
                progress_color=C["ink"], fg_color=C["soft"])
            self.primary = self.button(actions, "이미지 선택", self._primary_action, primary=True, height=40)
            self.primary.pack(fill="x")
            self.options_button = self.button(actions, "처리 옵션 · 기본", self._show_processing_options, height=30)
            self.options_button.pack(fill="x", pady=(6, 0))
            self.secondary = self.button(actions, "다른 이미지", self._reset_and_pick, height=32)
            self.error_details = self.button(actions, "오류 상세", self._show_info, height=28)

        def _show_processing_options(self):
            """옵션은 다음 처리에 적용하며 기존 결과는 명시적 재처리 전까지 유지합니다."""
            if self.busy or not self.setup_ready:
                return
            dialog = ctk.CTkToplevel(self)
            dialog.title("이미지 처리 옵션")
            dialog.geometry("440x310")
            dialog.resizable(False, False)
            dialog.transient(self)
            dialog.grab_set()
            edge = tk.BooleanVar(dialog, self.processing_options["restore_mode"] == "edge")
            sharpen = tk.BooleanVar(dialog, self.processing_options["unsharp"] > 0)
            self.label(dialog, "디테일 복원 옵션", 18, True).pack(anchor="w", padx=24, pady=(22, 16))
            for title, variable in (("윤곽 보존 모드", edge), ("약한 선명화", sharpen)):
                ctk.CTkCheckBox(dialog, text=title, variable=variable, font=(font_name, 14),
                    fg_color=C["ink"], hover_color=C["sub"], text_color=C["ink"]).pack(
                        anchor="w", padx=24, pady=(0, 16))
            self.label(dialog, "두 옵션을 함께 사용할 수 있습니다.\n윤곽·질감 조정과 워터마크 검출 결과는 별개입니다.",
                12, color=C["sub"], justify="left", wraplength=390).pack(anchor="w", padx=24)
            footer = ctk.CTkFrame(dialog, fg_color="transparent")
            footer.pack(fill="x", padx=24, pady=(20, 20))
            def apply_options(process=False):
                self.processing_options.update(
                    restore_mode="edge" if edge.get() else "gaussian",
                    unsharp=DETAIL_RESTORE_POLICY["gentle_unsharp"] if sharpen.get() else 0.0)
                labels = (["윤곽 보존"] if edge.get() else []) + (["약한 선명화"] if sharpen.get() else [])
                self.options_button.configure(text="처리 옵션 · " + (" + ".join(labels) or "기본"))
                dialog.destroy()
                if process:
                    self._process()
            self.button(footer, "설정 저장", apply_options, height=34).pack(side="left")
            if self.input_path:
                self.button(footer, "적용 후 처리", lambda: apply_options(True),
                            primary=True, height=34).pack(side="right")

        def _set_small_status(self, text):
            self.small_status.configure(text=text)
            if text:
                before = self.work_progress if self.work_progress.winfo_manager() else self.primary
                self.small_status.pack(fill="x", pady=(0, 6), before=before)
            else:
                self.small_status.pack_forget()

        def _open_synthid(self):
            """제공사별 검사 범위를 표시하고 사용자가 선택한 검사만 실행합니다."""
            import webbrowser
            from tkinter import messagebox
            dialog = ctk.CTkToplevel(self)
            dialog.title("SynthID 공식 검사")
            dialog.geometry("580x470")
            dialog.transient(self)
            self.label(dialog, "제공사별 공식 검사", 18, True).pack(padx=20, pady=(20, 8))
            self.label(dialog,
                "원본과 저장한 처리본을 각각 검사해 주세요.\n"
                "C2PA 메타데이터와 SynthID 픽셀 판정은 별개의 결과입니다.\n"
                "페이지를 여는 것만으로 파일이 전송되지는 않습니다.",
                12, justify="left", wraplength=530).pack(padx=20, pady=8)
            for service in PROVENANCE_SERVICES.values():
                self.button(dialog, service["name"] + " 열기 · " + service["scope"],
                    lambda url=service["url"]: webbrowser.open(url, new=2), height=36).pack(fill="x", padx=20, pady=5)
            status = self.label(dialog, "미검출은 AI로 생성되지 않았다는 뜻이 아닙니다.", 12,
                justify="left", wraplength=530)
            status.pack(fill="x", padx=20, pady=12)
            responses = queue.Queue()

            def receive_result():
                if not dialog.winfo_exists():
                    return
                try:
                    result, error = responses.get_nowait()
                except queue.Empty:
                    dialog.after(150, receive_result)
                    return
                api_button.configure(state="normal")
                if error:
                    status.configure(text=error)
                    return
                labels = {"detected": "검출", "not_detected": "미검출", "not_tested": "검사 안 됨"}
                text = "OpenAI 공식 API 결과\n" + "\n".join(
                    name.upper() + " · " + labels[value["status"]] for name, value in result["signals"].items())
                status.configure(text=text + "\nGoogle 검사 결과는 별도로 확인해 주세요.")
                destination = filedialog.asksaveasfilename(parent=dialog, title="검사 결과 저장",
                    defaultextension=".json", initialfile="openai-verification.json",
                    filetypes=[("검사 결과", "*.json")])
                if destination:
                    try:
                        # 기존 파일은 덮어쓰지 않고 새 검사 기록으로 보존합니다.
                        with open(destination, "x", encoding="utf-8") as stream:
                            json.dump(result, stream, ensure_ascii=False, indent=2)
                    except OSError:
                        messagebox.showerror("저장", "다른 이름으로 저장해 주세요. 기존 파일은 덮어쓰지 않습니다.", parent=dialog)

            def run_api():
                path = filedialog.askopenfilename(parent=dialog, title="OpenAI에서 검사할 이미지 선택",
                    filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp")])
                if not path or not messagebox.askyesno("OpenAI 이미지 전송",
                    Path(path).name + " 파일을 OpenAI 공식 검사 API에 업로드합니다.\n"
                    "계정의 API 이용 조건과 요청 한도가 적용됩니다.\n전송하여 검사하시겠습니까?", parent=dialog):
                    return
                api_button.configure(state="disabled")
                status.configure(text="선택한 파일을 OpenAI에서 검사 중입니다.")

                def worker():
                    try:
                        responses.put((verify_openai_file(path, allow_upload=True), None))
                    except Exception as exc:
                        # 알려진 안전한 메시지만 화면에 표시하며 키나 본문을 기록하지 않습니다.
                        responses.put((None, str(exc) if isinstance(exc, ProvenanceError) else "검사를 완료하지 못했습니다."))
                threading.Thread(target=worker, daemon=True, name="unmark-provenance").start()
                dialog.after(150, receive_result)

            api_button = self.button(dialog, "선택한 이미지 OpenAI API 검사", run_api, height=36)
            api_button.pack(fill="x", padx=20, pady=5)
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                api_button.configure(state="disabled")
                self.label(dialog, "앱 내 API 검사는 OPENAI_API_KEY 설정 시 사용할 수 있습니다.\n"
                    "키가 없으면 위 공식 웹 검사를 이용해 주세요.", 11,
                    color=C["sub"], wraplength=530).pack(padx=20, pady=5)

        def _draw_icon(self, state, fraction=None):
            """A state indicator; a number is shown only for measured progress."""
            color = C["green"] if state in ("done", "saved") else (C["amber"] if state in ("error", "review") else C["sub"])
            valid = (state == "processing" and isinstance(fraction, (int, float))
                     and not isinstance(fraction, bool) and math.isfinite(fraction)
                     and 0 <= fraction <= 1)
            labels = {"idle": "대기", "loading": "확인", "ready": "확인", "processing": "처리",
                      "done": "완료", "saving": "저장", "saved": "저장", "error": "확인", "review": "검토"}
            self.gauge_value.configure(text=("%d%%" % int(fraction * 100)) if valid else labels.get(state, "확인"),
                text_color=color if state in ("review", "error", "done", "saved") else C["ink"])
            self.gauge_caption.configure(text="현재 단계 진행률" if valid else "처리 상태")
            scale = 3
            image = Image.new("RGBA", (210 * scale, 108 * scale), (255, 255, 255, 0))
            draw = ImageDraw.Draw(image)
            bounds = tuple(value * scale for value in (14, 10, 196, 192))
            draw.arc(bounds, 180, 360, fill=C["line"], width=11 * scale)
            if valid and fraction > 0:
                draw.arc(bounds, 180, 180 + 180 * fraction, fill=C["ink"], width=11 * scale)
            elif state in ("done", "saved", "error", "review"):
                draw.arc(bounds, 180, 360, fill=color, width=11 * scale)
            self.gauge_image = ctk.CTkImage(light_image=image, dark_image=image, size=(210, 108))
            self.gauge_arc.configure(image=self.gauge_image, text="")

        def _refresh_evidence(self):
            after = self.tabs.get() == "처리 후"
            scan = (self.output_scan if after else self.input_scan) or {}
            failure = self.last_error
            regeneration = (self.result or {}).get("regeneration") or {}

            def card(key, title, value, detail="", color=None):
                heading, primary, secondary = self.evidence_labels[key]
                heading.configure(text=title)
                primary.configure(text=value, text_color=color or C["ink"])
                secondary.configure(text=detail)

            self.info_button.configure(text="상세", width=42,
                state="normal" if scan or failure else "disabled")
            self.evidence_title.configure(text="처리 후 확인" if after else "확인한 정보")
            if self.phase == "processing":
                card("regeneration", "이미지 재생성", "처리 중", self.work_stage or "로컬 모델을 준비하고 있습니다.")
            elif failure and self.phase == "error":
                card("regeneration", "이미지 재생성", "처리를 마치지 못했습니다",
                    "오류 상세에서 원인을 확인할 수 있습니다.", C["amber"])
            elif after and regeneration.get("processed") is True:
                modelids = regeneration.get("modelids") or {}
                model_names = [name for key, name in (("qwen", "Qwen"),
                    ("lightning", "Lightning"))
                    if isinstance(modelids.get(key), str) and modelids[key]]
                unchanged = regeneration.get("status") == "unchanged" or regeneration.get("pixels_changed") is False
                value = "모델 실행 후 픽셀 변경 없음" if unchanged else "이미지 처리 완료"
                note = " · ".join(model_names) if model_names else str(regeneration.get("engine") or "이미지 재생성을 마쳤습니다.")
                card("regeneration", "이미지 재생성", value, note, C["amber"] if unchanged else C["ink"])
            elif after:
                card("regeneration", "이미지 재생성", "실행 내역 없음", "상세에서 처리 내용을 확인해 주세요.")
            else:
                card("regeneration", "이미지 처리", "처리 대기", "이미지 처리 후 원본 디테일을 복원합니다.")
            frequency = scan.get("frequency") or {}
            before_frequency = (self.input_scan or {}).get("frequency") or {}
            peak = frequency.get("peak_count")
            before_peak = before_frequency.get("peak_count")
            if isinstance(peak, int) and not isinstance(peak, bool):
                value = f"피크 {peak:,}개" if peak else "뚜렷한 주파수 피크 없음"
                note = "이미지의 주파수 특성을 측정했습니다."
                if after and isinstance(before_peak, int) and not isinstance(before_peak, bool):
                    note = f"원본 {before_peak:,} → 처리 후 {peak:,}"
                card("pixel", "주파수 패턴", value, note)
            elif frequency.get("status") in ("unavailable", "error"):
                card("pixel", "주파수 패턴", "확인하지 못했습니다", "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            else:
                card("pixel", "주파수 패턴", "확인 중" if self.phase == "loading" else "검사 정보 없음")

            names = {"c2pa": "C2PA", "exif": "EXIF", "xmp": "XMP", "text": "텍스트",
                     "jumbf": "JUMBF", "comment": "코멘트", "other": "기타"}
            metadata = (scan.get("metadata") or {}).get("metadata") or {}
            original = ((self.input_scan or {}).get("metadata") or {}).get("metadata") or {}
            metadata_evidence = _ui_metadata_evidence(scan)
            present = metadata_evidence["names"]
            removed = [name for key, name in names.items()
                       if (original.get(key) or {}).get("present") is True
                       and (metadata.get(key) or {}).get("present") is False]
            # Provenance declarations are metadata, not a pixel-verifier result.
            # Keep them in the existing metadata card, using the actual names
            # read from the file. Credentials must not disappear behind "외 N종".
            declared = list(dict.fromkeys(str(item["name"])
                for item in (scan.get("synthid") or {}).get("declarations", [])
                if isinstance(item, dict) and item.get("name")))
            before_declared = list(dict.fromkeys(str(item["name"])
                for item in ((self.input_scan or {}).get("synthid") or {}).get("declarations", [])
                if isinstance(item, dict) and item.get("name")))
            removed = list(dict.fromkeys([name for name in before_declared if name not in declared] + removed))
            metadata_incomplete = metadata_evidence["incomplete"]

            def compact(values, limit=2):
                return " · ".join(values[:limit]) + (f" 외 {len(values)-limit}종" if len(values) > limit else "")

            if metadata_incomplete and not present:
                card("metadata", "메타데이터", "확인하지 못했습니다",
                    "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            elif after and removed and not present:
                card("metadata", "메타데이터", compact(removed) + " 삭제 확인", "출력 파일을 다시 검사했습니다.", C["green"])
            elif present:
                app_names = metadata_evidence["apps"]
                note = ("기록된 앱: " + app_names[0][:30]) if app_names else "파일에 기록된 정보입니다."
                if after:
                    note = "출력 파일에 남아 있는 기록입니다."
                if metadata_incomplete:
                    note = "상세에서 검사 내용을 확인해 주세요."
                card("metadata", "메타데이터", _ui_compact_evidence(present), note,
                    C["amber"] if after or metadata_incomplete else None)
            else:
                card("metadata", "메타데이터", "삭제 대상 정보 없음" if metadata else "확인 중",
                    "색상 프로필 유지" if (metadata.get("icc") or {}).get("present") else
                    "EXIF · XMP · C2PA · 텍스트 확인" if metadata else "")

            public = scan.get("open_watermarks") or {}
            original_public = (self.input_scan or {}).get("open_watermarks") or {}
            public_status = public.get("status")
            was_public = original_public.get("status") == "detected"
            public_matches = public.get("matches") or original_public.get("matches") or []
            public_names = list(dict.fromkeys(str(item.get("name") or "공개 패턴")
                for item in public_matches if isinstance(item, dict)))
            if public_status == "detected":
                self.evidence_cards["open_watermark"].grid()
                card("open_watermark", "공개 비가시 워터마크", compact(public_names, 1) or "공개 패턴 확인",
                    "출력에서 다시 검출됐습니다." if after else "공개 코덱의 삽입 신호를 확인했습니다.", C["amber"])
            elif after and was_public:
                self.evidence_cards["open_watermark"].grid()
                if public_status == "not_detected":
                    card("open_watermark", "공개 비가시 워터마크", "재검사에서 검출되지 않음",
                        compact(public_names, 1) + " · 동일 검사 적용", C["green"])
                else:
                    card("open_watermark", "공개 비가시 워터마크", "재검사 확인 필요",
                        "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            elif public_status in ("error", "unavailable", "unsupported", "insufficient_data"):
                self.evidence_cards["open_watermark"].grid()
                card("open_watermark", "공개 비가시 워터마크", "검사 결과 확인 필요",
                    "상세에서 검사 내용을 확인해 주세요.", C["amber"])
            elif public_status == "not_detected":
                # A completed public-codec check is useful evidence even when
                # it misses. Do not hide it or let the metadata-only summary
                # imply that the untested proprietary signal is absent.
                self.evidence_cards["open_watermark"].grid()
                card("open_watermark", "공개 비가시 워터마크", "등록된 공개 패턴과 불일치",
                    "SynthID 픽셀 신호는 검사하지 않았습니다.")
            else:
                self.evidence_cards["open_watermark"].grid_remove()

            visible = scan.get("visible") or {}
            status = visible.get("status")
            result_visible = (self.result or {}).get("visible") or {}
            show_visible = status in ("detected", "error", "unavailable") or (
                after and result_visible.get("status") in ("cleaned", "partial", "unvalidated"))
            if not show_visible:
                self.evidence_cards["visible"].grid_remove()
                return
            self.evidence_cards["visible"].grid()
            labels = list(dict.fromkeys(str(mark.get("label") or mark.get("key") or "표식")
                for mark in visible.get("marks", []) if isinstance(mark, dict)))
            if status == "detected":
                card("visible", "가시 워터마크", (compact(labels, 1) or "등록된 표식")[:32],
                    "출력에서 다시 확인됐습니다." if after else "처리할 때 해당 영역을 복원합니다.", C["amber"])
            elif after and status == "not_detected" and result_visible.get("status") == "cleaned":
                card("visible", "가시 워터마크", "제거 후 재검사 완료", "등록된 패턴이 다시 검출되지 않았습니다.", C["green"])
            else:
                card("visible", "가시 워터마크", "검사 결과 확인 필요", "상세에서 검사 내용을 확인해 주세요.", C["amber"])

        def _show_work_progress(self, payload):
            # Do not infer completion from elapsed time, tile labels, scores,
            # or a step index that has no reported total.
            fraction = payload.get("progress")
            if not (isinstance(fraction, (int, float)) and not isinstance(fraction, bool)
                    and math.isfinite(fraction) and 0 <= fraction <= 1):
                completed, total = payload.get("completed"), payload.get("total")
                if (isinstance(completed, (int, float)) and not isinstance(completed, bool)
                        and isinstance(total, (int, float)) and not isinstance(total, bool)
                        and math.isfinite(completed) and math.isfinite(total) and 0 <= completed <= total and total > 0):
                    fraction = completed / total
                else:
                    fraction = None
            self._draw_icon("processing", fraction)
            self.work_progress.stop()
            if fraction is None:
                self.work_progress.configure(mode="indeterminate")
                self.work_progress.start()
            else:
                self.work_progress.configure(mode="determinate")
                self.work_progress.set(fraction)

        def _result_notice(self):
            if not self.result:
                return ""
            regeneration = self.result.get("regeneration") or {}
            preservation = regeneration.get("preservation_check") or {}
            if preservation.get("status") in ("review_required", "failed"):
                return "원본과 처리 결과를 비교해 주세요." + "\n" + "상세에서 검사 내용을 확인해 주세요."
            if regeneration.get("processed") is True and (regeneration.get("status") == "unchanged"
                    or regeneration.get("pixels_changed") is False):
                return "모델은 실행됐지만 픽셀 변경이 없습니다.\n처리 정보를 확인해 주세요."
            visible_status = self.result.get("visible", {}).get("status")
            if visible_status == "partial":
                return "가시 표식 일부가 남아 있습니다.\n원본과 결과를 비교해 주세요."
            if visible_status in ("unvalidated", "error"):
                return "가시 표식의 재검사를 마치지 못했습니다.\n상세에서 검사 내용을 확인해 주세요."
            # Rendering/encoding success is separate from the actual output
            # checks. Inspect their named result fields, never an arbitrary
            # detected=True flag or a missing legacy quality field.
            output_scan = self.result.get("output_scan") or getattr(self, "output_scan", None) or {}
            public_status = (output_scan.get("open_watermarks") or {}).get("status")
            if public_status == "detected":
                return ("공개 비가시 워터마크" + " · " + "출력에서 다시 검출됐습니다." + "\n" +
                        "상세에서 검사 내용을 확인해 주세요.")
            if public_status in ("error", "unavailable", "unsupported", "insufficient_data"):
                return ("공개 비가시 워터마크" + " · " + "재검사 확인 필요" + "\n" +
                        "상세에서 검사 내용을 확인해 주세요.")
            if (output_scan.get("metadata_observation") or {}).get("status") == "incomplete":
                return ("메타데이터" + " · " + "확인하지 못했습니다" + "\n" +
                        "상세에서 검사 내용을 확인해 주세요.")
            return ""

        def _set_state(self, state, detail=None):
            self.phase = state
            self.busy = state in ("loading", "processing", "saving")
            texts = {
                "idle": ("이미지를 선택해 주세요", "왼쪽에 이미지를 놓으면\n검사 결과를 확인할 수 있습니다.", "이미지 선택 후 처리"),
                "loading": ("이미지 확인 중", "선택한 이미지를 불러오고 있습니다.", "이미지 확인 중"),
                "ready": ("이미지 확인 완료", "이미지 재생성과 메타데이터 삭제를\n한 번에 실행합니다.", "이미지 처리"),
                "processing": ("이미지 처리 중", "로컬 모델로 처리하고 저장 파일을 확인합니다.", "이미지 처리 중"),
                "done": ("이미지 처리 완료", "원본과 결과를 비교한 뒤\n저장해 주세요.", "이미지 저장"),
                "saving": ("이미지 저장 중", "처리한 이미지를 저장하고 있습니다.", "이미지 저장 중"),
                "saved": ("저장 완료", "원본과 처리 결과를 비교해 주세요.", "이미지 저장"),
                "error": ("처리하지 못했습니다", "다시 시도하거나 처리 정보를\n확인해 주세요.", "다시 처리" if self.input_path else "이미지 선택"),
            }
            title, body, action = texts[state]
            notice = self._result_notice() if state in ("done", "saved") else ""
            visible_incomplete = (self.input_scan or {}).get("visible", {}).get("status") in ("unavailable", "error")
            public_incomplete = (self.input_scan or {}).get("open_watermarks", {}).get("status") in (
                "error", "unavailable", "unsupported", "insufficient_data")
            metadata_evidence = _ui_metadata_evidence(self.input_scan)
            credentials_incomplete = metadata_evidence["incomplete"]
            incomplete_scan = state == "ready" and (visible_incomplete or public_incomplete or credentials_incomplete)
            if incomplete_scan:
                title = "일부 검사가 필요합니다"
                body = ("메타데이터는 확인했습니다.\n처리 시 가시 워터마크를 다시 검사합니다."
                        if visible_incomplete and not credentials_incomplete else
                        "상세에서 검사 내용을 확인해 주세요.")
            if state == "ready" and metadata_evidence["present"]:
                # Keep the original heading, typography and actions. The
                # existing detail area now shows the actual scan evidence,
                # instead of repeating a generic processing instruction.
                observed = _ui_compact_evidence(metadata_evidence["names"])
                note = ("기록된 앱: " + metadata_evidence["apps"][0][:30]
                        if metadata_evidence["apps"] else "파일에 기록된 정보입니다.")
                if incomplete_scan:
                    note = "상세에서 검사 내용을 확인해 주세요."
                body = " · ".join(("메타데이터", observed)) + "\n" + note
            elif state == "ready" and metadata_evidence["checked"] and not incomplete_scan:
                public = (self.input_scan or {}).get("open_watermarks") or {}
                visible = (self.input_scan or {}).get("visible") or {}
                if public.get("status") != "detected" and visible.get("status") != "detected":
                    body = "메타데이터 · 삭제 대상 정보 없음" + "\n" + "SynthID · 검사 안 됨"
            if state == "done" and self.result:
                elapsed = self.result.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0:
                    body = f"{elapsed:.1f}초 만에 처리와 재검사를 마쳤습니다.\n원본과 결과를 비교해 주세요."
            if notice:
                if state == "done":
                    title = "결과 확인이 필요합니다"
                body = notice
            self.state_title.configure(text=title)
            self.state_detail.configure(text=detail or body)
            self.primary.configure(text=action, state="disabled" if self.busy or state == "idle" else "normal")
            self.change_button.configure(state="disabled" if self.busy else "normal")
            if self.input_path:
                self.preview_tools.grid()
                self.change_button.pack(side="right")
            else:
                self.preview_tools.grid_remove()
                self.change_button.pack_forget()
            self._draw_icon("review" if notice or incomplete_scan else state)
            self._refresh_evidence()
            if state in ("idle", "loading"):
                self._reset_evidence_scroll()
            self.secondary.pack_forget()
            self.error_details.pack_forget()
            self.work_progress.stop()
            self.work_progress.pack_forget()
            if self.busy:
                self.work_progress.pack(fill="x", pady=(0, 10), before=self.primary)
                self.work_progress.configure(mode="indeterminate")
                self.work_progress.start()
                if state != "saving":
                    self.secondary.configure(text="작업 중단", command=self._cancel, state="normal")
                    self.secondary.pack(fill="x", pady=(6, 0))
            elif state in ("done", "saved", "error"):
                self.secondary.configure(text="다른 이미지", command=self._reset_and_pick, state="normal")
                self.secondary.pack(fill="x", pady=(6, 0))
            if self.last_error and not self.busy:
                self.error_details.pack(fill="x", pady=(6, 0))
            self._set_small_status(Path(self.saved_path).name if state == "saved" and self.saved_path else "")
            self._sync_setup_gate()
            self._sync_idle_view()

        def _reset_evidence_scroll(self):
            parent = self.evidence_scroll.master
            while parent is not None and parent is not self:
                if isinstance(parent, tk.Canvas):
                    parent.yview_moveto(0)
                    return
                parent = getattr(parent, "master", None)

        def _sync_idle_view(self):
            if not self.input_scan:
                self.evidence_scroll.grid_remove()
                if self.phase in ("loading", "error"):
                    self.summary_panel.grid()
                    self.idle_intro.grid_remove()
                    self.action_panel.grid()
                else:
                    self.summary_panel.grid_remove()
                    self.idle_intro.grid()
                    self.action_panel.grid_remove()
            else:
                self.idle_intro.grid_remove()
                self.summary_panel.grid()
                self.evidence_scroll.grid()
                self.action_panel.grid()

        def _sync_setup_gate(self):
            locked = not self.setup_ready or self.setup_busy
            self.tabs.configure(state="disabled" if locked or self.busy else "normal")
            self.change_button.configure(state="disabled" if locked or self.busy else "normal")
            self.options_button.configure(state="disabled" if locked or self.busy else "normal")
            self.preview_label.configure(cursor="arrow" if locked else "hand2")
            if locked:
                checking = self.setup_stage == "checking"
                self.primary.configure(state="disabled", text="시작 준비 확인 중" if checking else "사용 준비 필요")
                self.state_title.configure(text="시작 준비 중" if checking else "사용 준비가 필요합니다")
                self.state_detail.configure(text="모델과 실행 구성요소 준비가 끝나면\n이미지를 선택할 수 있습니다.")
                self.secondary.configure(state="disabled")
                if not self.input_path:
                    self.preview_label.configure(text="사용 준비가 끝나면\n이미지를 선택할 수 있습니다.")
            elif not self.input_path and self.phase == "idle":
                self._set_preview_content(image=self.upload_icon_image, text=self.empty_preview_text)
            elif not self.input_path and self.phase == "error":
                self._set_preview_content(text="이미지를 열지 못했습니다.\n다른 파일을 선택해 주세요.")

        def _require_setup_ready(self):
            if self.close_event.is_set():
                return False
            if self.setup_ready and not self.setup_busy:
                return True
            self._sync_setup_gate()
            if self.setup_stage != "checking":
                self._open_setup()
            return False

        def _emit(self, job, kind, data=None):
            if self.close_event.is_set():
                if kind == "done" and data:
                    data[0].cleanup()
                return
            self.events.put((job, kind, data))

        def _check_startup(self):
            if self.close_event.is_set() or self.setup_check_started:
                return
            self.setup_check_started = True
            self.setup_stage = "checking"
            self.setup_ready = False
            self._set_small_status("앱 준비 상태 확인 중…")
            self._sync_setup_gate()
            # Cache presence does not mean cold imports finish immediately.
            # Show the existing progress window if checking takes long enough
            # to be visible; cancel this callback as soon as checking finishes.
            self.setup_check_timer = self.after(250, self._show_setup_checking)
            def worker():
                try:
                    # These checks read package metadata/cache headers only.
                    # A known missing model must show consent immediately,
                    # without waiting for a cold native-library import probe.
                    models = model_runtime_status()
                    estimate = setup_estimate()
                    snapshot = globals().get("runtime_dependency_snapshot")
                    dependencies = snapshot() if callable(snapshot) else {"metadata_only": True, "missing": []}
                    if models.get("cached") is not True or dependencies.get("missing") or dependencies.get("error_code"):
                        self._emit("setup", "checked", {"ready": False, "dependencies": dependencies,
                                   "models": models, "estimate": estimate})
                        return
                    if self.close_event.is_set() or self.setup_cancel.is_set():
                        return
                    dependencies = runtime_dependency_status(cancel_event=self.setup_cancel, include_gui=False)
                    ready = dependencies.get("ready") is True and models.get("cached") is True
                    self._emit("setup", "checked", {"ready": ready, "dependencies": dependencies,
                               "models": models, "estimate": estimate})
                except Exception as exc:
                    self._emit("setup", "checked", {"ready": False, "error": str(exc), "estimate": {}})
            def begin_check():
                if not self.close_event.is_set():
                    threading.Thread(target=worker, daemon=True, name="unmark-readiness").start()
            self.after(20, begin_check)

        def _show_setup_checking(self):
            self.setup_check_timer = None
            if (self.close_event.is_set() or self.setup_checked or self.setup_ready
                    or self.setup_stage != "checking"):
                return
            self._open_setup()

        def _cancel_setup_check_dialog(self):
            timer = self.setup_check_timer
            self.setup_check_timer = None
            if timer:
                try:
                    self.after_cancel(timer)
                except tk.TclError:
                    pass

        def _open_setup(self):
            if self.close_event.is_set() or self.setup_ready:
                return
            if self.setup_window and self.setup_window.winfo_exists():
                self.setup_window.lift()
                self._grab_setup(self.setup_window)
                return
            dialog = ctk.CTkToplevel(self)
            self.setup_window = dialog
            dialog.title(app_title + " · 시작 준비")
            dialog.geometry("560x380")
            dialog.minsize(560, 300)
            dialog.resizable(False, False)
            dialog.configure(fg_color=C["surface"])
            # A transient window is attached to its parent on macOS. Keep this
            # dialog independently movable; grab_set still enforces setup.
            dialog.grid_columnconfigure(0, weight=1)
            self.setup_title = self.label(dialog, "시작 준비를 확인하고 있습니다", 20, True,
                       wraplength=520, justify="left")
            self.setup_title.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 16))
            content = ctk.CTkFrame(dialog, fg_color=C["soft"], corner_radius=10)
            content.grid(row=1, column=0, sticky="ew", padx=20)
            self.setup_headline = self.label(content, "시작 중", 16, True, wraplength=488, justify="left")
            self.setup_headline.pack(anchor="w", padx=16, pady=(16, 6))
            self.setup_note = self.label(content, "", 12, color=C["sub"], wraplength=488, justify="left")
            self.setup_note.pack(anchor="w", padx=16)
            self.setup_activity = self.label(content, "", 12,
                color=C["sub"], wraplength=488, justify="left")
            self.setup_activity.pack(anchor="w", padx=16, pady=(12, 12))
            self.setup_bar = ctk.CTkProgressBar(content, height=6, mode="determinate",
                progress_color=C["ink"], fg_color=C["line"])
            self.setup_bar.set(0)
            self.setup_bar.pack(fill="x", padx=16, pady=(0, 8))
            self.setup_metrics = self.label(content, "", 11,
                color=C["sub"], wraplength=488, justify="left")
            self.setup_metrics.pack(anchor="w", padx=16, pady=(0, 16))
            footer = ctk.CTkFrame(dialog, fg_color="transparent")
            footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(16, 20))
            footer.grid_columnconfigure((0, 1), weight=1)
            self.setup_later = self.button(footer, "앱 닫기", self._defer_setup, height=40)
            self.setup_later.grid(row=0, column=0, sticky="ew", padx=(0, 6))
            self.setup_start = self.button(footer, "확인 중", self._start_setup, primary=True, height=40, state="disabled")
            self.setup_start.grid(row=0, column=1, sticky="ew", padx=(6, 0))
            self.setup_log = self.button(footer, "상세 로그", self._show_logs, height=28)
            dialog.protocol("WM_DELETE_WINDOW", self._close)
            self.after_idle(lambda: self._grab_setup(dialog))
            if self.setup_stage == "checking":
                self._setup_checking_controls()
            elif self.setup_busy:
                self._setup_running_controls()
            else:
                self._setup_consent_controls()
            # Size and center after the initial text is laid out. Subsequent
            # progress and parent movement must never reset this position.
            self._fit_setup_window()
            self._center_dialog(dialog)
            self._sync_setup_gate()

        def _setup_checking_controls(self):
            self.setup_title.configure(text="시작 준비를 확인하고 있습니다")
            self.setup_headline.configure(text="시작 중")
            self.setup_note.configure(text="실행 구성요소와 저장된 모델 파일을 확인합니다.")
            self.setup_activity.configure(text="확인이 끝나면 다음 단계를 진행합니다.", text_color=C["sub"])
            self.setup_metrics.configure(text="이 단계에서는 파일을 다운로드하지 않습니다.")
            self.setup_start.configure(text="확인 중", state="disabled")
            self.setup_later.configure(text="앱 닫기", state="normal")
            self.setup_log.grid_remove()
            self.setup_bar.configure(mode="indeterminate")
            self.setup_bar.start()
            self._fit_setup_window()

        def _setup_consent_controls(self):
            self.setup_title.configure(text="처음 한 번, 사용 준비가 필요합니다")
            amount = self.setup_estimate_data.get("remaining_bytes")
            amount_text = "약 9.0 GB" if amount is None else size_text(amount)
            if amount == 0:
                headline = "기존 모델 파일을 확인합니다."
                note = "실행 구성요소와 파일을 확인한 뒤 사용할 수 있습니다.\n부족한 구성요소는 추가로 다운로드합니다."
                action = "준비 시작"
            else:
                headline = "필요한 다운로드 · " + amount_text
                estimate_range = self.setup_estimate_data.get("estimated_seconds_range")
                time_text = "예상 전송 약 5~20분"
                if isinstance(estimate_range, (list, tuple)) and len(estimate_range) == 2:
                    low, high = estimate_range
                    if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(value) and value >= 0 for value in (low, high)):
                        lower, upper = max(1, math.ceil(low / 60)), max(1, math.ceil(high / 60))
                        time_text = f"예상 전송 약 {lower}~{upper}분" if lower != upper else f"예상 전송 약 {upper}분"
                note = time_text + " · 100~300Mbps 기준 가정\n설치·파일 확인 시간은 별도이며 연결 속도에 따라 달라집니다."
                action = "다운로드 시작"
            self.setup_headline.configure(text=headline)
            self.setup_note.configure(text=note)
            activity = "Qwen · Lightning 모델을 컴퓨터에 저장합니다."
            legacy_bytes = self.setup_estimate_data.get("legacy_cache_bytes", 0)
            if isinstance(legacy_bytes, (int, float)) and legacy_bytes > 0:
                activity += "\n첫 처리 성공 후 이전 모델 파일 " + size_text(legacy_bytes) + "를 삭제합니다."
            self.setup_activity.configure(text=activity, text_color=C["sub"])
            self.setup_metrics.configure(text="다운로드와 파일 확인이 끝나면 사용할 수 있습니다.")
            self.setup_bar.stop()
            self.setup_bar.configure(mode="determinate")
            self.setup_bar.set(0)
            self.setup_start.configure(text=action, state="normal")
            self.setup_later.configure(text="앱 닫기", state="normal")
            self.setup_log.grid_remove()
            if self.setup_stage == "interrupted":
                self.setup_title.configure(text="사용 준비가 중단되었습니다")
                self.setup_headline.configure(text="사용 준비를 이어서 진행합니다")
                self.setup_note.configure(text="받은 파일과 이어받기 정보를 확인한 뒤\n필요한 준비를 계속합니다.")
                self.setup_activity.configure(text="준비를 마쳐야 이미지를 열고 처리할 수 있습니다.", text_color=C["sub"])
                self.setup_metrics.configure(text="다시 시작하면 확인된 파일과 이어받기 정보를 사용합니다.")
                self.setup_start.configure(text="준비 계속하기")
            if self.setup_error:
                self.setup_title.configure(text="사용 준비를 마치지 못했습니다")
                self.setup_activity.configure(text=setup_error_message(self.setup_error), text_color=C["red"])
                self.setup_metrics.configure(text="자세한 내용은 상세 로그에서 확인할 수 있습니다.")
                self.setup_start.configure(text="다시 시도")
                self.setup_log.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            self._fit_setup_window()

        def _fit_setup_window(self):
            """Fit changing setup text without moving the user's window."""
            dialog = self.setup_window
            if not dialog or not dialog.winfo_exists():
                return
            dialog.update_idletasks()
            dimensions = re.match(r"(\d+)x(\d+)", dialog.geometry())
            if dimensions is None:
                return
            width, height = map(int, dimensions.groups())
            # The public CTk geometry is logical; Tk requested/actual sizes are
            # physical. Convert only the measured content height back to CTk.
            scale = dialog.winfo_height() / max(1, height)
            fitted_height = max(300, math.ceil(dialog.winfo_reqheight() / scale))
            if fitted_height != height:
                dialog.geometry(f"{width}x{fitted_height}")

        def _center_dialog(self, dialog, width=None, height=None):
            # CTk scales requested dimensions, while winfo reports the actual
            # window pixels. Read the size after applying geometry so HiDPI
            # windows center correctly without any private scaling methods.
            if width is not None and height is not None:
                dialog.geometry(f"{width}x{height}")
            dialog.update_idletasks()
            parent = dialog.master or self
            x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
            # Tk accepts '+-N' as an absolute negative coordinate. Do not clamp
            # to the primary monitor: secondary monitors may be to its left.
            dialog.geometry(f"+{x}+{y}")

        def _grab_setup(self, dialog):
            if not self.close_event.is_set() and dialog.winfo_exists():
                try:
                    dialog.lift()
                    dialog.grab_set()
                except tk.TclError:
                    pass

        def _setup_running_controls(self):
            self.setup_title.configure(text="실행 구성요소를 확인하고 있습니다")
            self.setup_headline.configure(text="실행 구성요소")
            self.setup_note.configure(text="확인이 끝나면 필요한 모델 파일을 다운로드합니다.")
            self.setup_start.configure(state="disabled", text="준비 중")
            self.setup_later.configure(text="준비 중단", state="normal")
            self.setup_log.grid_remove()
            self.setup_activity.configure(text="실행 환경과 설치된 파일을 확인합니다.", text_color=C["sub"])
            self.setup_metrics.configure(text="다운로드와 파일 확인이 끝나면 사용할 수 있습니다.")
            self.setup_bar.configure(mode="indeterminate")
            self.setup_bar.start()
            self._fit_setup_window()

        def _start_setup(self):
            if self.close_event.is_set() or self.setup_busy or not self.setup_checked or self.setup_stage == "checking":
                return
            self.setup_busy = True
            self.setup_stage = "running"
            self.setup_ready = False
            self.setup_error = None
            self.setup_cancel = threading.Event()
            cancel = self.setup_cancel
            self._setup_running_controls()
            self._sync_setup_gate()
            def progress(*args, **kwargs):
                payload = progress_payload(*args, **kwargs)
                self._emit("setup", "progress", payload)
            def worker():
                try:
                    dependency_result = ensure_runtime_dependencies(progress, cancel, include_gui=False)
                    if cancel.is_set() or (isinstance(dependency_result, dict) and dependency_result.get("cancelled")):
                        detail = dependency_result.get("error", "") if isinstance(dependency_result, dict) else ""
                        log_path = dependency_result.get("log_path", "") if isinstance(dependency_result, dict) else ""
                        self._emit("setup", "cancelled", detail + ("\n로그 파일: " + str(log_path) if log_path else ""))
                        return
                    if isinstance(dependency_result, dict) and dependency_result.get("restart_required"):
                        self._emit("setup", "restart", dependency_result)
                        return
                    if not isinstance(dependency_result, dict) or dependency_result.get("ready") is not True:
                        failure = dependency_result if isinstance(dependency_result, dict) else {
                            "error": "실행에 필요한 구성 요소를 준비하지 못했습니다. 다시 시도해 주세요."}
                        self._emit("setup", "error", failure)
                        return
                    prepare_models(progress, cancel)
                    if cancel.is_set():
                        self._emit("setup", "cancelled")
                        return
                    # ensure_runtime_dependencies already validated real imports
                    # in a new process. Do not repeat that slow native probe.
                    models = model_runtime_status()
                    if dependency_result.get("ready") is not True or models.get("cached") is not True:
                        raise RuntimeError("사용 준비 후 확인에 실패했습니다.\n" + json.dumps(
                            {"dependencies": dependency_result, "models": models}, ensure_ascii=False))
                    if cancel.is_set():
                        self._emit("setup", "cancelled")
                        return
                    self._emit("setup", "ready")
                except Exception as exc:
                    self._emit("setup", "cancelled" if cancel.is_set() else "error", str(exc))
            threading.Thread(target=worker, daemon=True, name="unmark-setup").start()

        def _defer_setup(self):
            self.resume_after_setup = False
            if self.setup_busy:
                self.setup_stage = "stopping"
                self.setup_cancel.set()
                self.setup_later.configure(state="disabled", text="중단 요청됨")
                self.setup_activity.configure(text="준비를 중단하고 있습니다. 잠시만 기다려 주세요.")
                return
            self._close()

        def _destroy_setup(self):
            self._cancel_setup_check_dialog()
            if self.setup_window and self.setup_window.winfo_exists():
                self.setup_bar.stop()
                try:
                    self.setup_window.grab_release()
                except tk.TclError:
                    pass
                self.setup_window.destroy()
            self.setup_window = None

        def _setup_event(self, kind, data):
            if self.close_event.is_set():
                return
            if kind == "checked":
                self._cancel_setup_check_dialog()
                self.setup_checked = True
                ready = data.get("ready") is True and not self.setup_cancel.is_set()
                self.setup_ready = False
                self.setup_estimate_data = data.get("estimate") or {}
                self.logs.append("준비 상태: " + json.dumps(data, ensure_ascii=False, default=str))
                if not self.busy:
                    self._set_small_status("")
                if not ready:
                    self.setup_stage = "consent"
                    self.setup_error = data.get("error")
                    self._open_setup()
                    self._setup_consent_controls()
                    self._sync_setup_gate()
                    if globals().get("INITIAL_SETUP_APPROVED"):
                        self._start_setup()
                else:
                    # The dependency worker deliberately creates no Tk root.
                    # Native TkDND must succeed on this actual desktop root
                    # before input/processing is unlocked.
                    if not self._init_drag_and_drop():
                        self._setup_event("error", self.dnd_error)
                        return
                    self.setup_ready = True
                    self.setup_stage = "ready"
                    self._destroy_setup()
                    self._set_state("idle")
            elif kind == "progress":
                title, detail = str(data.get("title", "준비 중")), str(data.get("detail", ""))
                log_entry = title + (" · " + detail if detail else "")
                # Byte updates still refresh the UI, but repeated file names
                # must not push connection failures out of the diagnostic log.
                if not self.logs or self.logs[-1] != log_entry:
                    self.logs.append(log_entry)
                if not self.setup_window or not self.setup_window.winfo_exists():
                    return
                if self.setup_cancel.is_set():
                    return
                phase = str(data.get("phase", ""))
                model_progress = _ui_local_model_progress(data)
                if model_progress:
                    friendly = model_progress["title"]
                    activity = model_progress["activity"]
                    self.setup_headline.configure(text=model_progress["headline"])
                    self.setup_note.configure(text="이미지 재생성에 필요한 가중치를 컴퓨터에 준비합니다.")
                elif phase == "retry":
                    friendly = "다운로드 연결을 다시 시도하고 있습니다."
                    activity = "확인된 파일은 보관하고 필요한 설치를 이어갑니다."
                elif "설치" in title + detail or "다운로드" in title + detail:
                    friendly = "필요한 실행 구성요소를 다운로드하고 있습니다."
                    activity = "다운로드 후 설치한 구성요소를 확인합니다."
                else:
                    friendly = "실행 구성요소를 확인하고 있습니다."
                    activity = "이미지를 열고 처리할 준비가 되었는지 확인합니다."
                if not model_progress:
                    self.setup_headline.configure(text="실행 구성요소 준비")
                    self.setup_note.configure(text="구성요소를 확인한 뒤 필요한 모델 파일을 준비합니다.")
                self.setup_title.configure(text=friendly)
                self.setup_activity.configure(text=activity, text_color=C["sub"])
                total, done = data.get("total_bytes"), data.get("downloaded_bytes")
                ratio = data.get("progress")
                if total and done is not None:
                    ratio = float(done) / float(total)
                if isinstance(ratio, (int, float)) and math.isfinite(ratio):
                    self.setup_bar.stop()
                    self.setup_bar.configure(mode="determinate")
                    self.setup_bar.set(max(0, min(1, ratio)))
                else:
                    self.setup_bar.configure(mode="indeterminate")
                    self.setup_bar.start()
                pieces = []
                if done is not None and total:
                    pieces.append(size_text(done) + " / " + size_text(total))
                speed = data.get("bytes_per_second")
                if phase not in ("retry", "connecting") and speed and float(speed) > 0:
                    pieces.append(size_text(speed) + "/s")
                eta = duration_text(data.get("eta_seconds")) if phase not in ("retry", "connecting") else ""
                if eta:
                    pieces.append(eta)
                if phase == "retry":
                    attempt, maximum = data.get("retry_attempt"), data.get("max_retries")
                    if isinstance(attempt, int) and isinstance(maximum, int) and maximum > 0:
                        pieces.append(f"자동 재시도 {attempt}/{maximum}")
                elif phase == "connecting":
                    pieces.append("서버 연결 중")
                fallback = ("받은 파일을 확인하고 있습니다. 아직 준비 중입니다." if phase in ("verify", "ready")
                            else "받은 파일을 보관한 채 자동으로 다시 연결합니다." if phase == "retry"
                            else "다운로드 서버의 응답을 기다리고 있습니다." if phase == "connecting"
                            else "전송 속도를 확인하고 있습니다…" if phase == "download"
                            else "구성요소 확인이 끝나면 이미지 작업 화면이 열립니다.")
                self.setup_metrics.configure(text=" · ".join(pieces) if pieces else fallback)
                self._fit_setup_window()
            elif kind == "restart":
                if self.setup_cancel.is_set():
                    self._setup_event("cancelled", None)
                    return
                self.logs.append("설치한 구성 요소를 적용하기 위해 앱을 다시 시작합니다.")
                if self.setup_window and self.setup_window.winfo_exists():
                    self.setup_bar.stop()
                    self.setup_activity.configure(text="설치한 구성요소를 적용한 뒤 모델 준비를 이어갑니다.", text_color=C["sub"])
                    self.setup_start.configure(state="disabled", text="다시 시작 중")
                    self.setup_later.configure(state="disabled")
                try:
                    # This handler runs on Tk's main thread. The launcher carries
                    # the existing setup consent to the new interpreter.
                    app_restart_after_setup()
                except Exception as exc:
                    self._setup_event("error", "앱을 다시 시작하지 못했습니다: " + str(exc))
                    return
                self._close()
            elif kind == "ready":
                # A final result may already be queued when the user cancels.
                # Progress phase='ready' and 100% downloaded never unlock UI.
                if self.setup_cancel.is_set() or not self.setup_busy:
                    self._setup_event("cancelled", None)
                    return
                if not self._init_drag_and_drop():
                    self._setup_event("error", self.dnd_error)
                    return
                self.setup_busy = False
                self.setup_stage = "ready"
                self.setup_ready = True
                self.setup_checked = True
                self.setup_error = None
                self._destroy_setup()
                self.resume_after_setup = False
                self._set_state("idle")
            elif kind == "cancelled":
                self.setup_busy = False
                self.setup_stage = "interrupted"
                self.setup_ready = False
                self.setup_error = None
                self.resume_after_setup = False
                self.logs.append("사용 준비 중단" + (": " + str(data) if data else ""))
                self._open_setup()
                self._setup_consent_controls()
                self._sync_setup_gate()
            elif kind == "error":
                if self.setup_cancel.is_set():
                    self._setup_event("cancelled", data)
                    return
                self.setup_busy = False
                self.setup_stage = "error"
                self.setup_ready = False
                self.setup_error = data
                raw_error = json.dumps(data, ensure_ascii=False, indent=2, default=str) if isinstance(data, dict) else str(data)
                self.logs.append("준비 오류: " + raw_error)
                self.resume_after_setup = False
                if not self.setup_window or not self.setup_window.winfo_exists():
                    self._open_setup()
                self._setup_consent_controls()
                self._sync_setup_gate()

        def _pick(self):
            if not self._require_setup_ready() or self.busy:
                return
            path = filedialog.askopenfilename(parent=self, title="이미지 선택",
                filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp"), ("모든 파일", "*")])
            if path:
                self._load(path)

        def _drop(self, event):
            if self._drop_enter(event) != COPY:
                return REFUSE_DROP
            try:
                path = _normalise_dropped_image_path(self.tk, getattr(event, "data", ""))
                # Confirm the actual image header before replacing an existing
                # result. A text/PDF file renamed to .png must not clear it.
                with Image.open(path) as candidate:
                    if candidate.format not in ("PNG", "JPEG", "WEBP"):
                        raise ValueError("PNG, JPG 또는 WEBP 이미지 파일을 선택해 주세요.")
            except (OSError, ValueError, tk.TclError, Image.DecompressionBombError) as exc:
                self._drop_message(str(exc) if isinstance(exc, ValueError) else
                    "이미지를 열지 못했습니다. 파일 형식과 읽기 권한을 확인해 주세요.")
                return REFUSE_DROP
            return COPY if self._load(path) else REFUSE_DROP

        def _clear_result(self):
            self.result = None
            self.result_path = None
            self.output_scan = None
            self.saved_path = None
            if self.result_dir is not None:
                self.result_dir.cleanup()
                self.result_dir = None
            self.thumbnails.pop("처리 후", None)
            self.tabs.configure(values=["원본"])
            self.tabs.set("원본")

        def _load(self, path):
            if not self._require_setup_ready() or self.busy:
                return False
            self.last_error = None
            self.work_stage = ""
            self._clear_result()
            self.job += 1
            job = self.job
            self.cancel_event = threading.Event()
            cancel = self.cancel_event
            self.input_path = None
            self.input_scan = None
            self.thumbnails.clear()
            self._set_preview_content(text="이미지를 불러오고 있습니다…")
            self.file_summary.configure(text="")
            self.caption.configure(text="원본 파일은 그대로 보관됩니다.")
            self._set_state("loading")
            def worker():
                try:
                    source = str(Path(path).expanduser().resolve(strict=True))
                    scan = app_scan_file(source)
                    image = thumb(source)
                    if cancel.is_set():
                        self._emit(job, "cancelled")
                    else:
                        self._emit(job, "loaded", (source, scan, image))
                except Exception as exc:
                    self._emit(job, "cancelled" if cancel.is_set() else "error",
                               app_error_payload(exc, "이미지 불러오기"))
            threading.Thread(target=worker, daemon=True, name="unmark-load").start()
            return True

        def _primary_action(self):
            if not self._require_setup_ready() or self.busy:
                return
            if self.result_path and self.phase in ("done", "saved"):
                self._save()
            elif self.input_path:
                self._process()
            else:
                self._pick()

        def _process(self):
            if not self._require_setup_ready() or self.busy or not self.input_path:
                return
            self.last_error = None
            self.work_stage = "이미지 처리 준비"
            self._clear_result()
            self.job += 1
            job = self.job
            source = self.input_path
            self.cancel_event = threading.Event()
            cancel = self.cancel_event
            options = {"remove_visible": True, "pixel_process": False,
                "regenerate": True, "preserve_color_profile": True}
            options.update(self.processing_options)
            self._select_tab("원본")
            self._set_state("processing")
            stage = [self.work_stage]
            def progress(*args, **kwargs):
                payload = progress_payload(*args, **kwargs)
                stage[0] = str(payload.get("title") or stage[0])
                self._emit(job, "progress", payload)
            def worker():
                directory = None
                try:
                    directory = tempfile.TemporaryDirectory(prefix="unmark-preview-")
                    output = str(Path(directory.name) / "processed.png")
                    result = app_process_isolated(source, output, options, progress, cancel)
                    if cancel.is_set():
                        directory.cleanup()
                        self._emit(job, "cancelled")
                        return
                    stage[0] = "결과 미리보기"
                    image = thumb(output)
                    scan = result.get("output_scan") or app_scan_file(output)
                    self._emit(job, "done", (directory, result, scan, image, output))
                except Exception as exc:
                    if directory is not None:
                        directory.cleanup()
                    self._emit(job, "cancelled" if cancel.is_set() else "error",
                               app_error_payload(exc, stage[0]))
            threading.Thread(target=worker, daemon=True, name="unmark-process").start()

        def _save(self):
            if not self._require_setup_ready() or self.busy or not self.result_path:
                return
            source = Path(self.input_path)
            candidate = source.with_name(source.stem + "_cleaned.png")
            number = 2
            while candidate.exists():
                candidate = source.with_name(source.stem + "_cleaned_%d.png" % number)
                number += 1
            output = filedialog.asksaveasfilename(parent=self, title="이미지 저장", defaultextension=".png",
                initialdir=str(source.parent), initialfile=candidate.name, filetypes=[("PNG", "*.png")])
            if not output:
                return
            if Path(output).exists():
                self._set_small_status("기존 파일을 보존할 수 있도록 다른 이름으로 저장해 주세요.")
                return
            job = self.job
            processed = self.result_path
            self._set_state("saving")
            def worker():
                try:
                    app_export_result(processed, output)
                    self._emit(job, "saved", output)
                except Exception as exc:
                    self._emit(job, "save_error", app_error_payload(exc, "처리 결과 저장"))
            threading.Thread(target=worker, daemon=True, name="unmark-export").start()

        def _reset_and_pick(self):
            if not self._require_setup_ready() or self.busy:
                return
            self.last_error = None
            self.work_stage = ""
            self._clear_result()
            self.input_path = None
            self.input_scan = None
            self.thumbnails.clear()
            self._set_preview_content(image=self.upload_icon_image, text=self.empty_preview_text)
            self.file_summary.configure(text="")
            self.caption.configure(text="원본 파일은 그대로 보관됩니다.")
            self._set_state("idle")
            self._pick()

        def _poll(self):
            if self.close_event.is_set():
                return
            for _ in range(100):
                try:
                    job, kind, data = self.events.get_nowait()
                except queue.Empty:
                    break
                if job == "setup":
                    self._setup_event(kind, data)
                    if self.close_event.is_set():
                        return
                    continue
                if job != self.job:
                    if kind == "done":
                        data[0].cleanup()
                    continue
                if kind == "loaded":
                    self.last_error = None
                    self.input_path, self.input_scan, image = data
                    self.thumbnails["원본"] = image
                    self._set_state("ready")
                    self._select_tab("원본")
                elif kind == "progress":
                    title, detail = str(data.get("title", "")), str(data.get("detail", ""))
                    self.work_stage = title or self.work_stage
                    self.logs.append(title + (" · " + detail if detail else ""))
                    phase = str(data.get("phase", ""))
                    friendly = "이미지 처리 중"
                    message = "처리한 파일의 주파수 패턴과 메타데이터를 다시 확인합니다."
                    if phase == "encode":
                        friendly = "이미지 변환 중"
                        message = "처리에 필요한 이미지 정보를 읽고 있습니다."
                    elif phase == "decode":
                        friendly = "결과 이미지 생성 중"
                        message = "처리한 데이터를 이미지로 변환합니다."
                    elif phase == "detail_restore":
                        friendly = "원본 디테일 복원 중"
                        message = "처리된 이미지에 원본의 세부 표현을 되살립니다."
                    elif phase in ("regenerate", "inference", "model", "model_load", "load") or "모델" in title:
                        friendly = "이미지 처리 중" if phase in ("regenerate", "inference") else "로컬 모델 준비 중"
                        message = "이미지는 이 컴퓨터에서 처리합니다."
                    elif "파일 확인" in title:
                        friendly = "이미지 확인 중"
                    elif phase == "metadata" or "메타" in title:
                        friendly = "메타데이터 삭제 중"
                    elif phase in ("pixel", "pixel_process") or "픽셀" in title:
                        friendly = "이미지 픽셀 처리 중"
                    elif "가시" in title or "워터마크" in title:
                        friendly = "가시 워터마크 처리 중"
                    elif "검증" in title:
                        friendly = "처리 결과 확인 중"
                    self.state_title.configure(text=friendly)
                    self._show_work_progress(data)
                    self.state_detail.configure(text=message)
                    self._refresh_evidence()
                    self._set_small_status("취소할 수 있습니다.")
                elif kind == "done":
                    self.last_error = None
                    self.result_dir, self.result, self.output_scan, image, self.result_path = data
                    self.input_scan = self.result.get("input_scan") or self.input_scan
                    self.thumbnails["처리 후"] = image
                    self.tabs.configure(values=["원본", "처리 후"])
                    self.tabs.set("처리 후")
                    self._set_state("done")
                    self._select_tab("처리 후")
                elif kind == "saved":
                    self.last_error = None
                    self.saved_path = data
                    self._set_state("saved")
                elif kind == "save_error":
                    detail = self._remember_error(data, "처리 결과 저장")
                    self._set_state("done", "저장 실패: " + detail)
                elif kind == "cancelled":
                    self._set_state("ready" if self.input_path else "idle")
                    self._set_small_status("작업을 중단했습니다.")
                elif kind == "error":
                    detail = self._remember_error(data)
                    self._set_state("error", detail)
            if len(self.logs) > 400:
                self.logs = self.logs[-400:]
            self.poll_timer = self.after(60, self._poll)

        def _select_tab(self, _value):
            if not self._require_setup_ready():
                return
            selected = self.tabs.get()
            scan = self.output_scan if selected == "처리 후" else self.input_scan
            path = self.result_path if selected == "처리 후" else self.input_path
            if scan:
                meta = scan.get("metadata", {})
                name = (Path(self.input_path).stem + " · 처리 후") if selected == "처리 후" else Path(self.input_path).name
                self.file_summary.configure(text=name if len(name) <= 64 else name[:61] + "…")
                self.caption.configure(text="%s · %s · %s × %s px" % (
                    selected, meta.get("format", ""), meta.get("width", "?"), meta.get("height", "?")))
            self._refresh_evidence()
            self._reset_evidence_scroll()
            self._render_preview()

        def _schedule_preview(self, _event):
            if self.preview_timer:
                self.after_cancel(self.preview_timer)
            self.preview_timer = self.after(90, self._render_preview)

        def _render_preview(self):
            self.preview_timer = None
            if self.close_event.is_set() or not self.setup_ready:
                return
            image = self.thumbnails.get(self.tabs.get())
            if image is None:
                return
            width, height = max(80, self.preview_area.winfo_width() - 8), max(80, self.preview_area.winfo_height() - 8)
            scale = min(width / image.width, height / image.height, 1)
            preview_image = ctk.CTkImage(light_image=image, dark_image=image,
                size=(max(1, int(image.width * scale)), max(1, int(image.height * scale))))
            self._set_preview_content(image=preview_image)

        def _cancel(self):
            if self.busy:
                self.cancel_event.set()
                self.secondary.configure(state="disabled", text="중단 요청됨")
                self._set_small_status("작업을 중단하고 있습니다…")

        def _text_window(self, title, content):
            parent = self.setup_window if self.setup_window and self.setup_window.winfo_exists() else self
            dialog = ctk.CTkToplevel(parent)
            dialog.title(title)
            self._center_dialog(dialog, 640, 490)
            dialog.transient(parent)
            dialog.grid_columnconfigure(0, weight=1)
            dialog.grid_rowconfigure(1, weight=1)
            self.label(dialog, title, 20, True).grid(row=0, column=0, sticky="w", padx=24, pady=(22, 14))
            textbox = ctk.CTkTextbox(dialog, font=(font_name, 12), wrap="word",
                takefocus=True, exportselection=False, undo=False)
            textbox.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=24, pady=(0, 8))
            textbox.insert("1.0", content)
            textbox.configure(state="disabled")
            copy_status = self.label(dialog, "", 11, color=C["sub"], height=18, anchor="w")
            copy_status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=24, pady=(0, 14))

            def copy_text(selection_only=False):
                try:
                    if selection_only:
                        if not textbox.tag_ranges("sel"):
                            copy_status.configure(text="복사할 텍스트를 선택해 주세요.", text_color=C["sub"])
                            return "break"
                        text = textbox.get("sel.first", "sel.last")
                    else:
                        # Keep the complete report, including intentional final
                        # newlines; Tk's implicit terminal newline is not data.
                        text = content
                    # The app owns the clipboard so closing this popup does not
                    # discard the copied report while the app remains open.
                    self.clipboard_clear()
                    self.clipboard_append(text)
                except (tk.TclError, RuntimeError):
                    copy_status.configure(text="복사하지 못했습니다. 다시 시도해 주세요.", text_color=C["red"])
                else:
                    copy_status.configure(text="선택한 내용을 복사했습니다." if selection_only else "전체 내용을 복사했습니다.",
                        text_color=C["green"])
                return "break"

            def select_all(_event=None):
                textbox.focus_set()
                textbox.tag_remove("sel", "1.0", "end")
                textbox.tag_add("sel", "1.0", "end-1c")
                return "break"

            self.button(dialog, "전체 복사", copy_text, width=96, height=30).grid(
                row=0, column=1, sticky="e", padx=(8, 24), pady=(22, 14))
            menu = tk.Menu(dialog, tearoff=False)
            menu.add_command(label="선택 복사", command=lambda: copy_text(True))
            menu.add_command(label="전체 복사", command=copy_text)
            menu.add_separator()
            menu.add_command(label="전체 선택", command=select_all)

            def show_menu(event):
                textbox.focus_set()
                menu.entryconfigure(0, state="normal" if textbox.tag_ranges("sel") else "disabled")
                try:
                    menu.tk_popup(event.x_root, event.y_root)
                finally:
                    menu.grab_release()
                return "break"

            # Disabled Text widgets can still select/copy. Explicit focus and
            # bindings make that work consistently without enabling edits or
            # reaching into CustomTkinter's private native widget.
            textbox.bind("<Button-1>", lambda _event: textbox.focus_set(), add="+")
            copy_events = ["<<Copy>>", "<Control-c>", "<Control-C>"]
            select_events = ["<<SelectAll>>", "<Control-a>", "<Control-A>"]
            menu_events = ["<Button-3>"]
            if dialog.tk.call("tk", "windowingsystem") == "aqua":
                copy_events += ["<Command-c>", "<Command-C>"]
                select_events += ["<Command-a>", "<Command-A>"]
                menu_events += ["<Button-2>", "<Control-Button-1>"]
            for sequence in copy_events:
                textbox.bind(sequence, lambda _event: copy_text(True), add="+")
            for sequence in select_events:
                textbox.bind(sequence, select_all, add="+")
            for sequence in menu_events:
                textbox.bind(sequence, show_menu, add="+")
            for sequence in ("<<Cut>>", "<<Paste>>", "<<PasteSelection>>", "<<Clear>>", "<<Undo>>", "<<Redo>>"):
                textbox.bind(sequence, lambda _event: "break", add="+")
            textbox.focus_set()
            def close_info():
                dialog.destroy()
                if self.setup_window and self.setup_window.winfo_exists():
                    self._grab_setup(self.setup_window)
            dialog.protocol("WM_DELETE_WINDOW", close_info)
            if parent is not self:
                self.after(60, lambda: self._grab_setup(dialog))

        def _show_logs(self):
            identity = "실행 정보\n" + json.dumps(app_identity(), ensure_ascii=False, indent=2, default=str)
            self._text_window("상세 로그", identity + "\n\n" + ("\n".join(self.logs[-200:]) or "기록된 내용이 없습니다."))

        def _show_info(self):
            identity = "실행 정보\n" + json.dumps(app_identity(), ensure_ascii=False, indent=2, default=str)
            if self.last_error:
                error = self.last_error
                text = ("실패 단계: " + str(error.get("stage") or "이미지 처리") +
                        "\n원인: " + str(error.get("error") or error.get("detail") or "원인 미확인"))
                if error.get("log_path"):
                    text += "\n로그 파일: " + str(error["log_path"])
                if error.get("traceback"):
                    text += "\n\n상세 오류 기록\n" + str(error["traceback"])
                self._text_window("처리 오류 정보", text + "\n\n" + identity)
                return
            selected = self.output_scan if self.tabs.get() == "처리 후" else self.input_scan
            if not selected:
                self._show_logs()
                return
            text = ("주파수 패턴은 이미지에서 측정한 통계이며 AI 생성 확률이나 SynthID 판정값이 아닙니다.\n"
                    "로컬 모델의 이미지 재생성 내역과 파일의 메타데이터·등록된 가시 표식을 확인합니다.\n")
            text += "\n" + identity
            if self.result:
                text += "\n\n이미지 재생성 실행 내역\n" + json.dumps(self.result.get("regeneration") or {},
                    ensure_ascii=False, indent=2, default=str)
                pixel = self.result.get("pixel_processing") or {}
                if pixel.get("processed") is True:
                    text += "\n\n추가 픽셀 처리 실행 내역\n" + json.dumps(pixel, ensure_ascii=False, indent=2, default=str)
                text += "\n\n처리 전후 검사\n" + json.dumps({
                    "before": (self.input_scan or {}).get("frequency"),
                    "after": (self.output_scan or {}).get("frequency"),
                    "metadata_removed": self.result.get("metadata", {}).get("removed_metadata", []),
                    "visible": self.result.get("visible"),
                }, ensure_ascii=False, indent=2, default=str)
            text += "\n\n선택한 이미지 검사 정보\n" + json.dumps(selected, ensure_ascii=False, indent=2, default=str)
            self._text_window("처리 정보" if self.result else "원본 검사 정보", text)

        def _close(self):
            self.close_event.set()
            self.cancel_event.set()
            self.setup_cancel.set()
            for timer in (self.poll_timer, self.preview_timer):
                if timer:
                    try:
                        self.after_cancel(timer)
                    except tk.TclError:
                        pass
            self._destroy_setup()
            while not self.events.empty():
                try:
                    _job, kind, data = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "done" and data:
                    data[0].cleanup()
            if self.result_dir is not None and self.phase != "saving":
                self.result_dir.cleanup()
                self.result_dir = None
            self.destroy()

    ctk.set_appearance_mode("Light")
    try:
        app = AIUnmarkerApp()
    except tk.TclError as exc:
        raise RuntimeError("데스크톱 화면을 열 수 없습니다. GUI를 지원하는 환경에서 실행해 주세요.") from exc
    app.mainloop()
