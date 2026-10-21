"""Startup window and worker ownership regressions; no model downloads.

The controller tests run real desktop methods with a deterministic Tk timer
queue. Native TkDND itself is covered by the full source/frozen build probes.
"""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "desktop_ui.py").read_text(encoding="utf-8"))
app_node = next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "AIUnmarkerApp")
method_names = {"_check_startup", "_show_setup_checking", "_cancel_setup_check_dialog",
                "_setup_event", "_destroy_setup"}


class Window:
    def __init__(self):
        self.alive = True
        self.destroyed = False

    def winfo_exists(self):
        return self.alive

    def grab_release(self):
        pass

    def destroy(self):
        self.alive = False
        self.destroyed = True


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.workers = []
        owner = self

        class DeferredThread:
            def __init__(self, target, **kwargs):
                self.target = target

            def start(self):
                owner.workers.append(self.target)

        self.env = {
            "threading": types.SimpleNamespace(Thread=DeferredThread),
            "tk": types.SimpleNamespace(TclError=RuntimeError),
            "json": json,
            "model_runtime_status": Mock(return_value={"cached": True}),
            "setup_estimate": Mock(return_value={"remaining_bytes": 0}),
            "runtime_dependency_snapshot": Mock(return_value={"missing": []}),
            "runtime_dependency_status": Mock(return_value={"ready": True}),
            "INITIAL_SETUP_APPROVED": False,
        }
        selected = [node for node in app_node.body
                    if isinstance(node, ast.FunctionDef) and node.name in method_names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / "desktop_ui.py"), "exec"), self.env)
        self.app = types.SimpleNamespace(
            close_event=threading.Event(), setup_cancel=threading.Event(),
            setup_checked=False, setup_ready=False, setup_busy=False,
            setup_check_started=False, setup_stage="checking", setup_check_timer=None,
            setup_window=None, setup_error=None, setup_estimate_data={},
            busy=False, logs=[], dnd_error="native DnD failed", resume_after_setup=False,
            setup_bar=types.SimpleNamespace(stop=Mock()),
        )
        for name in method_names:
            setattr(self.app, name, types.MethodType(self.env[name], self.app))
        self.timers = {}
        self.timer_id = 0
        self.events = []
        self.opened_stages = []
        self.app.after = self.after
        self.app.after_cancel = lambda key: self.timers.pop(key, None)
        self.app._set_small_status = Mock()
        self.app._sync_setup_gate = Mock()
        self.app._set_state = Mock()
        self.app._init_drag_and_drop = Mock(return_value=True)
        self.app._setup_consent_controls = Mock()
        self.app._start_setup = Mock()
        self.app._emit = lambda *payload: self.events.append(payload)
        self.app._open_setup = self.open_setup

    def after(self, delay, callback):
        self.timer_id += 1
        self.timers[self.timer_id] = (delay, callback)
        return self.timer_id

    def fire_delay(self, delay):
        pending = [(key, callback) for key, (value, callback) in list(self.timers.items()) if value == delay]
        for key, callback in pending:
            if self.timers.pop(key, None):
                callback()

    def open_setup(self):
        if self.app.setup_ready or self.app.close_event.is_set():
            return
        self.opened_stages.append(self.app.setup_stage)
        self.app.setup_window = Window()

    def run_check(self):
        self.fire_delay(20)
        self.assertEqual(len(self.workers), 1)
        self.workers.pop(0)()
        while self.events:
            job, kind, data = self.events.pop(0)
            self.assertEqual(job, "setup")
            self.app._setup_event(kind, data)

    def test_slow_cached_check_opens_existing_window_and_closes_when_ready(self):
        self.app._check_startup()
        self.fire_delay(250)
        self.assertEqual(self.opened_stages, ["checking"])
        self.assertFalse(self.app.setup_ready)
        dialog = self.app.setup_window
        self.run_check()
        self.assertTrue(dialog.destroyed)
        self.assertTrue(self.app.setup_ready)
        self.assertEqual(self.app.setup_stage, "ready")
        self.app._start_setup.assert_not_called()
        self.env["runtime_dependency_status"].assert_called_once_with(
            cancel_event=self.app.setup_cancel, include_gui=False)

    def test_fast_healthy_check_never_flashes_or_requests_download(self):
        self.app._check_startup()
        self.run_check()
        self.fire_delay(250)
        self.assertEqual(self.opened_stages, [])
        self.assertTrue(self.app.setup_ready)
        self.assertIsNone(self.app.setup_check_timer)
        self.app._start_setup.assert_not_called()

    def test_missing_models_show_consent_without_waiting_for_import_probe(self):
        self.env["model_runtime_status"].return_value = {"cached": False}
        self.app._check_startup()
        self.run_check()
        self.fire_delay(250)
        self.assertEqual(self.opened_stages, ["consent"])
        self.assertFalse(self.app.setup_ready)
        self.env["runtime_dependency_status"].assert_not_called()
        self.app._start_setup.assert_not_called()

    def test_parent_native_failure_never_unlocks_a_healthy_worker_result(self):
        self.app._init_drag_and_drop.return_value = False
        self.app._check_startup()
        self.run_check()
        self.assertFalse(self.app.setup_ready)
        self.assertEqual(self.app.setup_stage, "error")
        self.assertEqual(self.app.setup_error, "native DnD failed")
        self.app._set_state.assert_not_called()

    def test_parent_native_failure_after_installation_keeps_setup_locked(self):
        self.app.setup_checked = True
        self.app.setup_busy = True
        self.app.setup_stage = "running"
        self.app._init_drag_and_drop.return_value = False
        self.app._setup_event("ready", None)
        self.assertFalse(self.app.setup_ready)
        self.assertEqual(self.app.setup_stage, "error")
        self.app._set_state.assert_not_called()

    def test_success_already_queued_after_cancel_cannot_unlock(self):
        self.app.setup_busy = True
        self.app.setup_stage = "running"
        self.app.setup_cancel.set()
        self.app._setup_event("ready", None)
        self.assertFalse(self.app.setup_ready)
        self.assertEqual(self.app.setup_stage, "interrupted")
        self.app._init_drag_and_drop.assert_not_called()

    def test_closed_window_cancels_timer_and_suppresses_late_popup(self):
        self.app._check_startup()
        self.app.close_event.set()
        self.app._destroy_setup()
        self.fire_delay(250)
        self.fire_delay(20)
        self.assertEqual(self.opened_stages, [])
        self.assertEqual(self.workers, [])
        self.assertIsNone(self.app.setup_check_timer)


class RuntimeWorkerGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("startup_runtime_under_test", ROOT / "runtime_setup.py")
        cls.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runtime)

    def test_desktop_probe_does_not_create_a_second_tk_application(self):
        runtime = self.runtime
        with patch.object(runtime, "_rs_requirements", return_value={"tkinterdnd2": ("tkinterdnd2==0.6.4", "==0.6.4")}), \
             patch("tkinter.Tk.__init__", side_effect=AssertionError("second GUI root")) as window:
            result = runtime._rs_probe_actual(include_gui=False)
        window.assert_not_called()
        self.assertTrue(result["ready"])
        check = result["checks"]["tkinterdnd2"]
        self.assertFalse(check["native_checked"])
        self.assertEqual(check["native_check_stage"], "parent_gui")
        self.assertFalse(result["native_gui_check"])

    def test_standalone_probe_still_attempts_real_native_window(self):
        runtime = self.runtime
        with patch.object(runtime, "_rs_requirements", return_value={"tkinterdnd2": ("tkinterdnd2==0.6.4", "==0.6.4")}), \
             patch.dict(os.environ, {"DISPLAY": ":test"}), \
             patch("tkinter.Tk.__init__", side_effect=RuntimeError("native check attempted")) as window, \
             contextlib.redirect_stderr(io.StringIO()):
            result = runtime._rs_probe_actual()
        window.assert_called_once()
        self.assertFalse(result["ready"])
        self.assertTrue(result["native_gui_check"])
        self.assertIn("native check attempted", result["checks"]["tkinterdnd2"]["error"])

    def test_probe_command_only_defers_gui_when_explicitly_requested(self):
        runtime = self.runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = []

            def process(command, *_args, **_kwargs):
                commands.append(command)
                destination = Path(command[command.index("--result-file") + 1])
                destination.write_text(json.dumps({"ready": True, "missing": []}))
                return 0, ""

            with patch.object(runtime, "_rs_root", return_value=root), \
                 patch.object(runtime, "_rs_layers", return_value=[]), \
                 patch.object(runtime, "_rs_run_process", side_effect=process):
                runtime._rs_probe(log_path=root / "probe.log")
                runtime._rs_probe(log_path=root / "probe.log", include_gui=False)
            self.assertNotIn("--no-gui-probe", commands[0])
            self.assertIn("--no-gui-probe", commands[1])

    def test_cli_flag_preserves_default_native_verification(self):
        runtime = self.runtime
        for arguments, expected in (([], True), (["--no-gui-probe"], False)):
            with self.subTest(arguments=arguments), \
                 patch.object(runtime, "_rs_activate"), \
                 patch.object(runtime, "_rs_layers", return_value=[]), \
                 patch.object(runtime, "_rs_probe_actual", return_value={"ready": True}) as actual, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runtime.runtime_probe_cli(arguments), 0)
                actual.assert_called_once_with("core", include_gui=expected)


if __name__ == "__main__":
    unittest.main()
