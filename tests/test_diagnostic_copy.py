"""Report copy controls: exact text, selection, key routing and error recovery.

These tests isolate the popup methods through public widget doubles. Native Tk
clipboard/selection behavior is checked separately when a display is available.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
import unittest


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "desktop_ui.py").read_text(encoding="utf-8"))
APP = next(node for node in ast.walk(TREE)
           if isinstance(node, ast.ClassDef) and node.name == "AIUnmarkerApp")
METHODS = ast.Module(body=[node for node in APP.body if isinstance(node, ast.FunctionDef)
                          and node.name in ("_text_window", "_show_info", "_show_logs")], type_ignores=[])


class Widget:
    def __init__(self, **values):
        self.values = values

    def configure(self, **values):
        self.values.update(values)

    def grid(self, **_values):
        pass


class Textbox(Widget):
    def __init__(self, _parent, **values):
        super().__init__(**values)
        self.text = ""
        self.selection = None
        self.bindings = {}
        self.focused = False

    def insert(self, _index, content):
        self.text = content

    def get(self, first, last):
        if (first, last) != ("sel.first", "sel.last") or self.selection is None:
            raise AssertionError("Selection must be read through the public Text API")
        start, end = self.selection
        return self.text[start:end]

    def tag_ranges(self, _tag):
        return self.selection or ()

    def tag_remove(self, tag, first, last):
        assert (tag, first, last) == ("sel", "1.0", "end")
        self.selection = None

    def tag_add(self, tag, first, last):
        assert (tag, first, last) == ("sel", "1.0", "end-1c")
        self.selection = (0, len(self.text))

    def bind(self, sequence, callback, add):
        assert add == "+"
        self.bindings[sequence] = callback

    def focus_set(self):
        self.focused = True


class Dialog:
    def __init__(self, parent, window_system):
        self.master = parent
        self.tk = SimpleNamespace(call=lambda *_args: window_system)
        self.protocols = {}
        self.destroyed = False

    def title(self, _value):
        pass

    def transient(self, _value):
        pass

    def grid_columnconfigure(self, *_args, **_kwargs):
        pass

    def grid_rowconfigure(self, *_args, **_kwargs):
        pass

    def protocol(self, name, callback):
        self.protocols[name] = callback

    def destroy(self):
        self.destroyed = True


class Menu:
    def __init__(self, _parent, **_kwargs):
        self.entries = []
        self.popup_at = None
        self.released = False

    def add_command(self, **values):
        self.entries.append(values)

    def add_separator(self):
        self.entries.append({})

    def entryconfigure(self, index, **values):
        self.entries[index].update(values)

    def tk_popup(self, x, y):
        self.popup_at = (x, y)

    def grab_release(self):
        self.released = True


class App:
    setup_window = None

    def __init__(self, window_system="x11"):
        self.labels = []
        self.buttons = []
        self.clipboard = "previous clipboard"
        self.clipboard_error = False
        self.logs = []
        self.result = None
        self.last_error = None
        self.input_scan = None
        self.output_scan = None
        self.tabs = SimpleNamespace(get=lambda: "원본")
        namespace = {"tk": SimpleNamespace(Menu=self.new_menu, TclError=tk.TclError),
                     "ctk": SimpleNamespace(CTkToplevel=self.new_dialog, CTkTextbox=self.new_textbox),
                     "font_name": "Pretendard", "json": json,
                     "app_identity": lambda: {"app_version": "test", "frozen": False},
                     "C": {"sub": "sub", "green": "green", "red": "red", "soft": "soft", "ink": "ink"}}
        self.window_system = window_system
        exec(compile(METHODS, str(ROOT / "desktop_ui.py"), "exec"), namespace)
        for name in ("_text_window", "_show_info", "_show_logs"):
            setattr(self, name, namespace[name].__get__(self))

    def new_dialog(self, parent):
        self.dialog = Dialog(parent, self.window_system)
        return self.dialog

    def new_textbox(self, parent, **values):
        self.textbox = Textbox(parent, **values)
        return self.textbox

    def new_menu(self, parent, **values):
        self.menu = Menu(parent, **values)
        return self.menu

    def label(self, _parent, text, *_args, **values):
        value = Widget(text=text, **values)
        self.labels.append(value)
        return value

    def button(self, _parent, text, command, **values):
        value = Widget(text=text, command=command, **values)
        self.buttons.append(value)
        return value

    def _center_dialog(self, *_args):
        pass

    def clipboard_clear(self):
        if self.clipboard_error:
            raise tk.TclError("clipboard unavailable")
        self.clipboard = ""

    def clipboard_append(self, text):
        self.clipboard += text

    def copy_all(self):
        return self.buttons[-1].values["command"]()

    def key(self, sequence):
        return self.textbox.bindings[sequence](None)


class DiagnosticCopyTests(unittest.TestCase):
    def test_full_report_copy_preserves_unicode_json_and_intentional_newlines(self):
        content = "검사 정보\n" + json.dumps({"경로": "/작업/2_overlay.png", "기록": ["복사할 값"] * 12000},
                                              ensure_ascii=False, indent=2) + "\n\n"
        app = App()
        app._text_window("원본 검사 정보", content)
        app.textbox.selection = (0, 2)
        self.assertEqual(app.copy_all(), "break")
        self.assertEqual(app.clipboard, content)
        self.assertEqual(app.textbox.values["state"], "disabled")
        self.assertEqual(app.labels[-1].values["text_color"], "green")

    def test_selection_copy_is_exact_and_never_copies_unselected_private_context(self):
        app = App()
        app._text_window("처리 정보", "경로는 비공개\n선택한 내용\n다른 기록")
        app.textbox.selection = (8, 14)
        expected = app.textbox.text[8:14]
        self.assertEqual(app.key("<Control-c>"), "break")
        self.assertEqual(app.clipboard, expected)
        app.dialog.protocols["WM_DELETE_WINDOW"]()
        self.assertTrue(app.dialog.destroyed)
        self.assertEqual(app.clipboard, expected)

    def test_no_selection_does_not_overwrite_clipboard(self):
        app = App()
        app._text_window("상세 로그", "기록")
        app.key("<<Copy>>")
        self.assertEqual(app.clipboard, "previous clipboard")
        self.assertIn("선택", app.labels[-1].values["text"])

    def test_select_all_then_copy_respects_read_only_state(self):
        app = App()
        content = "선택\n{\"status\": null}\n"
        app._text_window("원본 검사 정보", content)
        self.assertEqual(app.key("<Control-a>"), "break")
        app.key("<<Copy>>")
        self.assertEqual(app.clipboard, content)
        for event in ("<<Cut>>", "<<Paste>>", "<<PasteSelection>>", "<<Clear>>", "<<Undo>>", "<<Redo>>"):
            self.assertEqual(app.key(event), "break")
        self.assertEqual(app.textbox.text, content)
        self.assertEqual(app.textbox.values["state"], "disabled")
        self.assertFalse(app.textbox.values["exportselection"])
        self.assertTrue(app.textbox.values["takefocus"])

    def test_aqua_command_shortcuts_and_mouse_menu_do_not_leak_to_x11(self):
        mac = App("aqua")
        mac._text_window("상세 로그", "macOS 기록")
        mac.key("<Command-a>")
        mac.key("<Command-c>")
        self.assertEqual(mac.clipboard, "macOS 기록")
        self.assertIn("<Control-Button-1>", mac.textbox.bindings)
        self.assertIn("<Button-2>", mac.textbox.bindings)
        linux = App("x11")
        linux._text_window("상세 로그", "Linux 기록")
        self.assertNotIn("<Command-c>", linux.textbox.bindings)
        self.assertNotIn("<Button-2>", linux.textbox.bindings)

    def test_context_menu_copy_follows_current_selection(self):
        app = App()
        app._text_window("원본 검사 정보", "앞부분\n선택\n뒷부분")
        event = SimpleNamespace(x_root=100, y_root=200)
        app.textbox.bindings["<Button-3>"](event)
        self.assertEqual(app.menu.entries[0]["state"], "disabled")
        app.textbox.selection = (4, 6)
        app.textbox.bindings["<Button-3>"](event)
        self.assertEqual(app.menu.entries[0]["state"], "normal")
        app.menu.entries[0]["command"]()
        self.assertEqual(app.clipboard, "선택")
        self.assertEqual(app.menu.popup_at, (100, 200))
        self.assertTrue(app.menu.released)

    def test_clipboard_failure_is_visible_and_can_retry(self):
        app = App()
        app._text_window("처리 오류 정보", "실패 단계와 전체 기록")
        app.clipboard_error = True
        app.copy_all()
        self.assertEqual(app.labels[-1].values["text_color"], "red")
        self.assertIn("다시 시도", app.labels[-1].values["text"])
        app.clipboard_error = False
        app.copy_all()
        self.assertEqual(app.clipboard, "실패 단계와 전체 기록")
        self.assertEqual(app.labels[-1].values["text_color"], "green")

    def test_shared_popup_copies_scan_process_error_and_log_reports(self):
        app = App()
        app.input_scan = {"metadata": {"path": "/원본.png"}, "synthid": {"detected": None}}
        app._show_info()
        app.copy_all()
        self.assertIn('"detected": null', app.clipboard)
        self.assertIn("선택한 이미지 검사 정보", app.clipboard)
        app.output_scan = {"metadata": {"path": "/처리.png"}}
        app.result = {"regeneration": {"processed": True}}
        app.tabs = SimpleNamespace(get=lambda: "처리 후")
        app._show_info()
        app.copy_all()
        self.assertIn("이미지 재생성 실행 내역", app.clipboard)
        self.assertIn("/처리.png", app.clipboard)
        app.last_error = {"stage": "다운로드", "error": "연결 종료", "traceback": "전체 예외 내역"}
        app._show_info()
        app.copy_all()
        self.assertIn("전체 예외 내역", app.clipboard)
        app.logs = [f"내역 {i}" for i in range(205)]
        app._show_logs()
        expected = app.textbox.text
        app.copy_all()
        self.assertEqual(app.clipboard, expected)
        self.assertTrue(app.clipboard.endswith("내역 204"))
        self.assertEqual(app.logs, [f"내역 {i}" for i in range(205)])


if __name__ == "__main__":
    unittest.main()
