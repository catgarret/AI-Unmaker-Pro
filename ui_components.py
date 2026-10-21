"""화면 공통 규격과 설명 문단. 판정·이미지 처리 기능과 독립적입니다."""

UI_COLORS = {"bg": "#F5F6F8", "surface": "#FFFFFF", "line": "#E2E6EC",
             "ink": "#172033", "sub": "#626E80", "soft": "#F3F5F8",
             "green": "#157C61", "amber": "#995A0A", "red": "#B13E43"}
UI_LAYOUT = {"outer": 24, "panel_width": 360, "panel_padding": 20,
             "card_padding": 14, "card_gap": 10, "control_height": 40,
             "compact_control_height": 28, "paragraph_line_gap": 3}


def paragraph_widget_class(ctk, font_name):
    """명시적으로 나눈 설명 줄 사이에 일정한 여백을 주는 공통 위젯입니다.

    CTkLabel 한 개에 긴 문자열을 넣으면 행간을 지정할 수 없습니다.
    문장별 라벨을 쌓아 행간을 유지하고, 긴 문장은 지정한 폭에서 줄바꿈합니다.
    configure(text=...)를 제공하여 상태 갱신 코드가 표시 구현을 알 필요가 없습니다.
    """
    class Paragraph(ctk.CTkFrame):
        def __init__(self, parent, text="", size=12, color=None, wraplength=300,
                     justify="left", **kwargs):
            super().__init__(parent, fg_color="transparent", corner_radius=0,
                             width=wraplength, height=0, **kwargs)
            self._paragraph_text = None
            self._paragraph_color = color or UI_COLORS["sub"]
            self._paragraph_size = size
            self._paragraph_width = wraplength
            self._paragraph_justify = justify
            self._paragraph_rows = []
            self.configure(text=text)

        def configure(self, **kwargs):
            text = kwargs.pop("text", None)
            color = kwargs.pop("text_color", None)
            if kwargs:
                super().configure(**kwargs)
            if color is not None:
                self._paragraph_color = color
                for row in self._paragraph_rows:
                    row.configure(text_color=color)
            if text is None or text == self._paragraph_text:
                return
            self._paragraph_text = text
            for row in self._paragraph_rows:
                row.destroy()
            self._paragraph_rows.clear()
            for index, line in enumerate(str(text).splitlines()):
                row = ctk.CTkLabel(self, text=line, height=round(self._paragraph_size * 1.5),
                    font=(font_name, self._paragraph_size), text_color=self._paragraph_color,
                    wraplength=self._paragraph_width, justify=self._paragraph_justify,
                    anchor="center" if self._paragraph_justify == "center" else "w")
                row.pack(fill="x", pady=(UI_LAYOUT["paragraph_line_gap"] if index else 0, 0))
                self._paragraph_rows.append(row)
    return Paragraph
