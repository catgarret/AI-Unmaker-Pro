import os

INITIAL_SETUP_APPROVED = os.environ.pop("UNMARK_SETUP_APPROVED", "") == "1"


def bootstrap_ui_dependencies():
    """Ask once, install in the app's private runtime, then open the desktop.

    Packaged releases already contain the UI; this path also makes a source
    launch recover missing Pillow/CustomTkinter without a terminal command.
    """
    global INITIAL_SETUP_APPROVED
    import queue
    import re
    import threading
    import tkinter as tk
    from tkinter import ttk

    font_name = ui_font_family()
    window = tk.Tk()
    ui_apply_app_identity(window, f"AI Unmarker Pro v{APP_VERSION} · 처음 사용 준비")
    window.geometry("560x380")
    window.minsize(560, 320)
    window.resizable(False, False)
    window.configure(bg="#F6F7F9")
    # Native Tk uses positive font sizes as points; CTk uses pixels. Match the
    # desktop's typography explicitly so this fallback does not appear larger.
    window.option_add("*Font", (font_name, -13))
    window.option_add("*Label.padX", 0)
    window.option_add("*Label.padY", 0)
    frame = tk.Frame(window, bg="white", padx=20, pady=20,
                     highlightbackground="#E2E5EA", highlightthickness=1)
    frame.pack(fill="both", expand=True, padx=20, pady=20)
    title = tk.Label(frame, text="시작 준비를 확인하고 있습니다", bg="white", fg="#17202D",
             font=(font_name, -20, "bold"), anchor="w", justify="left", wraplength=476)
    title.pack(fill="x", pady=(0, 12))
    message = tk.StringVar(value="이미지 처리에 필요한 실행 구성요소를 확인합니다.")
    tk.Label(frame, textvariable=message, bg="white", fg="#616B78", justify="left",
             anchor="w", wraplength=476).pack(fill="x")
    estimate = tk.Label(frame, text="시작 중", bg="white", fg="#17202D",
                        font=(font_name, -16, "bold"), anchor="w")
    estimate.pack(fill="x", pady=(16, 6))
    estimate_note = tk.StringVar(value="이 단계에서는 파일을 다운로드하지 않습니다.")
    tk.Label(frame, textvariable=estimate_note,
             bg="white", fg="#78818D", justify="left", anchor="w", wraplength=476,
             font=(font_name, -11)).pack(fill="x")
    status = tk.StringVar(value="")
    tk.Label(frame, textvariable=status, bg="white", fg="#616B78", anchor="w",
             wraplength=476, justify="left").pack(fill="x", pady=(12, 8))
    bar = ttk.Progressbar(frame, mode="indeterminate")
    actions = tk.Frame(frame, bg="white")
    actions.pack(side="bottom", fill="x", pady=(16, 0))
    events = queue.Queue()
    cancel = threading.Event()
    state = {"busy": False, "ready": False, "closed": False, "checking": True}

    def fit_window():
        # Long errors need more space than progress messages. Change only the
        # height so neither buttons nor text are squeezed and position stays put.
        window.update_idletasks()
        height = max(320, window.winfo_reqheight())
        if window.winfo_height() != height:
            window.geometry(f"560x{height}")

    def dismiss():
        state["closed"] = True
        cancel.set()
        window.destroy()

    def report(payload, detail=""):
        events.put(("progress", payload if isinstance(payload, dict) else {"title": str(payload), "detail": detail}))

    def setup_error_message(payload):
        fallback = "준비를 마치지 못했습니다. 오류 기록을 확인해 주세요."
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

    def start():
        if state["busy"] or state["checking"] or state["closed"]:
            return
        state["busy"] = True
        cancel.clear()
        start_button.configure(state="disabled", text="준비 중…")
        later_button.configure(text="중단하고 닫기")
        title.configure(text="실행 구성요소 준비")
        message.set("이미지 처리에 필요한 구성요소를 다운로드합니다.")
        status.set("실행 구성요소 확인 중…")
        bar.pack(fill="x", before=actions)
        bar.start(15)
        fit_window()

        def work():
            try:
                # The main desktop validates native DnD after this bootstrap
                # closes. A worker must not create a second macOS Tk app.
                result = ensure_runtime_dependencies(report, cancel, include_gui=False)
                if not isinstance(result, dict) or (not result.get("ready") and not result.get("restart_required")):
                    failure = result if isinstance(result, dict) else {
                        "error": "실행 구성요소의 준비 상태를 확인하지 못했습니다."}
                    events.put(("error", failure))
                    return
                events.put(("ready", result))
            except Exception as exc:
                events.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True, name="unmark-first-setup").start()

    start_button = tk.Button(actions, text="확인 중", command=start, state="disabled",
                             bg="#17202D", fg="white", activebackground="#2C3645",
                             activeforeground="white", relief="flat", bd=0,
                             font=(font_name, -13, "bold"), padx=20, pady=10)
    start_button.pack(side="right")
    later_button = tk.Button(actions, text="앱 닫기", command=dismiss,
                             bg="#F0F2F5", fg="#46505E", relief="flat", bd=0,
                             font=(font_name, -13), padx=20, pady=10)
    later_button.pack(side="right", padx=(0, 8))

    def poll():
        global INITIAL_SETUP_APPROVED
        if state["closed"]:
            return
        layout_changed = False
        try:
            while True:
                kind, payload = events.get_nowait()
                layout_changed = True
                if kind == "checked":
                    state["checking"] = False
                    title.configure(text="처음 한 번, 사용 준비가 필요합니다")
                    estimate.configure(text="실행 구성요소 · 예상 1~3분")
                    estimate_note.set("이미 설치된 파일은 다시 받지 않습니다.\n인터넷 연결 속도와 설치 환경에 따라 달라집니다.")
                    action = "다운로드 시작"
                    message.set("다운로드와 설치가 끝나면 사용할 수 있습니다.")
                    start_button.configure(state="normal", text=action)
                    if INITIAL_SETUP_APPROVED:
                        window.after_idle(start)
                elif kind == "progress":
                    detail = str(payload.get("detail", ""))
                    progress_title = str(payload.get("title", ""))
                    if "기다" in detail:
                        friendly = "다른 실행 창의 준비가 끝나기를 기다리고 있습니다."
                    elif any(word in progress_title + detail for word in ("확인", "검사", "검증", "준비했습니다")):
                        friendly = "필요한 실행 구성요소를 확인하고 있습니다."
                    else:
                        friendly = "이미지 처리에 필요한 구성요소를 다운로드하고 있습니다."
                    title.configure(text="실행 구성요소 준비")
                    current, total = payload.get("current"), payload.get("total")
                    if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
                        bar.stop()
                        bar.configure(mode="determinate", maximum=total, value=min(current, total))
                        status.set(f"{friendly} · {current / 1_000_000:.1f} / {total / 1_000_000:.1f} MB")
                    else:
                        bar.configure(mode="indeterminate")
                        bar.start(15)
                        status.set(friendly)
                elif kind == "ready":
                    if cancel.is_set():
                        dismiss()
                        return
                    try:
                        if isinstance(payload, dict) and payload.get("restart_required"):
                            app_restart_after_setup()
                            state["closed"] = True
                            window.destroy()
                            return
                        activate_runtime_packages()
                        INITIAL_SETUP_APPROVED = True
                    except Exception as exc:
                        events.put(("error", str(exc)))
                        continue
                    state["ready"] = True
                    state["closed"] = True
                    window.destroy()
                    return
                elif kind == "error":
                    if cancel.is_set():
                        dismiss()
                        return
                    state["busy"] = False
                    bar.stop()
                    bar.pack_forget()
                    message.set(setup_error_message(payload))
                    status.set("설치 기록은 앱의 오류 기록에 보관됩니다.")
                    try:
                        app_write_crash(RuntimeError(str(payload)))
                    except Exception:
                        pass
                    start_button.configure(state="normal", text="다시 시도")
                    later_button.configure(text="앱 닫기")
        except queue.Empty:
            pass
        if layout_changed:
            fit_window()
        window.after(80, poll)

    fit_window()
    window.protocol("WM_DELETE_WINDOW", dismiss)
    window.after(80, poll)
    def check_after_paint():
        def work():
            try:
                snapshot = globals().get("runtime_dependency_snapshot")
                result = snapshot() if callable(snapshot) else {}
                events.put(("checked", result if isinstance(result, dict) else {}))
            except Exception:
                # UI dependencies are already known missing. An unavailable
                # estimate must not be mistaken for permission to install.
                events.put(("checked", {}))
        if not state["closed"]:
            threading.Thread(target=work, daemon=True, name="unmark-first-check").start()
    window.after_idle(lambda: window.after(20, check_after_paint))
    window.mainloop()
    return state["ready"]


def app_restart_after_setup():
    """Start a fresh interpreter only when an active native dependency changed."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    environment = os.environ.copy()
    environment["UNMARK_SETUP_APPROVED"] = "1"
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    command = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, str(Path(__file__).resolve())]
    subprocess.Popen(command, env=environment, close_fds=True)
