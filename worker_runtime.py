"""Run one image job in a child process; release all model memory at its end."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


class _JobCancellation:
    def __init__(self, path):
        self.path = Path(path)

    def is_set(self):
        return self.path.exists()


def _app_process_worker(request_path):
    """Internal local IPC entry; the desktop does not send images to a service."""
    request_path = Path(request_path).resolve(strict=True)
    request = _rs_read_json(request_path)
    if not isinstance(request, dict) or request.get("schema") != 1:
        raise ValueError("이미지 처리 요청을 읽지 못했습니다.")
    directory = request_path.parent
    report_path = directory / "result.json"
    cancel = _JobCancellation(directory / "cancel")
    try:
        if hasattr(os, "nice"):
            try:
                os.nice(5)
            except OSError:
                pass
        with (directory / "progress.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
            def progress(title, detail=""):
                payload = title if isinstance(title, dict) else {"title": str(title), "detail": str(detail)}
                stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            result = app_process(request["source"], request["output"], request.get("options"), progress, cancel)
        _rs_atomic_json(report_path, {"ok": True, "result": result})
        return 0
    except (ProcessingCancelled, ModelCancelled, PixelProcessingCancelled) as exc:
        _rs_atomic_json(report_path, {"ok": False, "cancelled": True, "error": str(exc)})
        return 130
    except Exception as exc:
        error = app_error_payload(exc, "로컬 이미지 처리")
        _rs_atomic_json(report_path, {"ok": False, "error": str(exc), "details": error})
        return 1


def app_process_isolated(source, output, options=None, progress_callback=None, cancel_event=None):
    """Let one worker own the models while the UI background thread waits.

    The worker exits after every job, including failure and cancellation, so
    Torch/MPS allocations cannot remain attached to the next GUI operation.
    """
    _app_cancel(cancel_event)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("같은 이름의 파일이 있습니다. 다른 이름을 선택해 주세요.")
    with tempfile.TemporaryDirectory(prefix="unmarker-job-") as temporary:
        directory = Path(temporary)
        request = directory / "request.json"
        _rs_atomic_json(request, {"schema": 1, "source": str(Path(source).resolve(strict=True)),
                                  "output": str(output), "options": dict(options or {})})
        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command.append(str(Path(__file__).resolve()))
        command.extend(("--process-job", str(request)))
        scratch = directory / "tmp"
        scratch.mkdir()
        environment = os.environ.copy()
        environment.update({"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                            "TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch)})
        position, pending, cancel_started, process = 0, "", None, None
        try:
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        if cancel_started is None:
                            (directory / "cancel").touch()
                            cancel_started = time.monotonic()
                        if process.poll() is None and time.monotonic() - cancel_started > 3.0:
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                    progress_file = directory / "progress.jsonl"
                    if progress_file.exists():
                        with progress_file.open("r", encoding="utf-8") as reader:
                            reader.seek(position)
                            pending += reader.read()
                            position = reader.tell()
                        lines = pending.split("\n")
                        pending = lines.pop()
                        for line in lines:
                            if line and progress_callback is not None:
                                progress_callback(json.loads(line))
                    if process.poll() is not None:
                        break
                    time.sleep(0.08)
                process.wait()
            if cancel_started is not None or (cancel_event is not None and cancel_event.is_set()):
                raise ProcessingCancelled("작업을 취소했습니다.")
            report = directory / "result.json"
            if report.is_file():
                payload = _rs_read_json(report)
                if payload.get("ok") is True and process.returncode == 0:
                    result = payload["result"]
                    result["execution"] = {"isolation": "process", "worker_exit_code": 0,
                                           "model_memory_released": True}
                    return result
                if payload.get("cancelled"):
                    raise ProcessingCancelled(payload.get("error") or "작업을 취소했습니다.")
                detail = payload.get("error") or "이미지 처리에 실패했습니다."
                original_log = (payload.get("details") or {}).get("log_path")
                if original_log:
                    detail += "\n오류 기록: " + str(original_log)
                raise RuntimeError(detail)
            log = directory / "worker.log"
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4096))
                detail = _rs_safe_text(stream.read().decode("utf-8", errors="replace")).strip()
            raise RuntimeError(f"이미지 처리 작업이 종료되었습니다 (종료 코드 {process.returncode})."
                               + ("\n" + detail if detail else ""))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
