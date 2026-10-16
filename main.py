import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
import sys
import shutil
import subprocess
import threading
import urllib.request
import traceback

# ==============================================================================
# [0] 프로그램 정보 및 다운로드 주소 정의
# ==============================================================================
APP_NAME = "AI Unmarker Pro (GPU)"
APP_VERSION = "v1.0.1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
ICONS_DIR = os.path.join(BASE_DIR, "icons")

MODEL_URLS = {
    os.path.join("remove-ai-watermarks", "model.pth"): "https://github.com/wiltodelta/remove-ai-watermarks/releases/download/v0.45.1/model.pth",
    os.path.join("noai-watermark", "model.pth"): "https://github.com/mertizci/noai-watermark/releases/download/v1.0.0/model.pth"
}

if os.path.exists(FONTS_DIR):
    for font_file in os.listdir(FONTS_DIR):
        if font_file.endswith(('.otf', '.ttf')) and 'Pretendard' in font_file:
            try:
                ctk.FontManager.load_font(os.path.join(FONTS_DIR, font_file))
            except Exception:
                pass

FONT_PRIMARY = "Pretendard"

def get_upload_icon(size=44, color="#6b7280"):
    scale = 4
    canvas_size = size * scale
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_loaded = False
    phosphor_path = os.path.join(FONTS_DIR, "Phosphor.ttf")
    if not os.path.exists(phosphor_path):
        phosphor_path = os.path.join(FONTS_DIR, "Phosphor-Bold.ttf")

    if os.path.exists(phosphor_path):
        try:
            candidates = ["\uE87E", "\uE4C0", "\uE4BE", "\uE4BA", "\uE5CE"]
            pil_font = ImageFont.truetype(phosphor_path, int(size * scale * 0.85))
            
            for char in candidates:
                mask = pil_font.getmask(char)
                bbox = mask.getbbox()
                if bbox and (bbox[2] - bbox[0] > 10) and (bbox[3] - bbox[1] > 10):
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    x = (canvas_size - w) / 2 - bbox[0]
                    y = (canvas_size - h) / 2 - bbox[1]
                    draw.text((x, y), char, font=pil_font, fill=color)
                    font_loaded = True
                    break
        except Exception:
            font_loaded = False

    if not font_loaded:
        stroke = int(canvas_size * 0.065)
        x_left = int(canvas_size * 0.20)
        x_right = int(canvas_size * 0.80)
        y_top = int(canvas_size * 0.60)
        y_bottom = int(canvas_size * 0.80)
        r = int(canvas_size * 0.08)

        draw.line([(x_left, y_top), (x_left, y_bottom - r)], fill=color, width=stroke)
        draw.line([(x_right, y_top), (x_right, y_bottom - r)], fill=color, width=stroke)
        draw.arc([x_left, y_bottom - 2*r, x_left + 2*r, y_bottom], 90, 180, fill=color, width=stroke)
        draw.arc([x_right - 2*r, y_bottom - 2*r, x_right, y_bottom], 0, 90, fill=color, width=stroke)
        draw.line([(x_left + r, y_bottom), (x_right - r, y_bottom)], fill=color, width=stroke)

        cx = int(canvas_size * 0.50)
        arrow_tip = int(canvas_size * 0.20)
        arrow_base = int(canvas_size * 0.60)
        head_w = int(canvas_size * 0.17)

        draw.line([(cx, arrow_base), (cx, arrow_tip)], fill=color, width=stroke)
        draw.line([(cx - head_w, arrow_tip + head_w), (cx, arrow_tip)], fill=color, width=stroke)
        draw.line([(cx + head_w, arrow_tip + head_w), (cx, arrow_tip)], fill=color, width=stroke)

    res_img = img.resize((size, size), Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=res_img, dark_image=res_img, size=(size, size))

# ==============================================================================
# [1] 코어 엔진: 텍스트 메타데이터 + 픽셀 주파수(FFT) 비가시적 워터마크 탐지
# ==============================================================================
def detect_metadata_fingerprint(image_path):
    detected_sources = []
    try:
        # 1단계: 텍스트/바이너리 메타데이터 스캔
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
            raw_text = raw_bytes.decode("utf-8", errors="ignore").lower()

        if "dall-e" in raw_text or "dalle" in raw_text or "chatgpt" in raw_text:
            detected_sources.append("ChatGPT (DALL-E)")
        if "c2pa" in raw_text or "photoshop" in raw_text or "adobe" in raw_text:
            detected_sources.append("Adobe Photoshop (C2PA)")
        if "midjourney" in raw_text:
            detected_sources.append("Midjourney")
        if "stable diffusion" in raw_text or "stablediffusion" in raw_text:
            detected_sources.append("Stable Diffusion")
            
        # 2단계: 비가시적 주파수 워터마크(SynthID 등) FFT 분석
        with Image.open(image_path) as img:
            # 해상도를 낮추어 노이즈 증폭 및 연산 최적화
            img_resized = img.convert('L').resize((512, 512))
            img_array = np.array(img_resized)
            f_transform = np.fft.fft2(img_array)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # 주파수 이상치 탐지 (SynthID 등 고주파 패턴 삽입 시 평균/표준편차가 요동침)
            mean_mag = np.mean(magnitude_spectrum)
            std_mag = np.std(magnitude_spectrum)
            
            # 일반 자연 이미지는 표준편차가 상대적으로 낮고 중앙에 집중되나,
            # 비가시적 워터마크가 인코딩된 이미지는 주파수 도메인에 비정상적인 피크가 발생함.
            if std_mag > 40.0 or "synthid" in raw_text:
                detected_sources.append("비가시적 주파수 핑거프린트 (SynthID 추정)")

    except Exception as e:
        print(f"딥스캔 오류: {e}")

    unique_sources = list(dict.fromkeys(detected_sources))
    if unique_sources:
        return True, unique_sources
    return False, []

# ==============================================================================
# [2] 강력 픽셀 세탁: 예외 회피(폴백) 삭제 및 GPU 섭동 강제
# ==============================================================================
def run_wiltodelta(input_path, output_path):
    script_path = os.path.join(BASE_DIR, "remove-ai-watermarks", "inference.py") 
    model_path = os.path.join(BASE_DIR, "remove-ai-watermarks", "model.pth")     
    if not os.path.exists(script_path):
        raise FileNotFoundError("wiltodelta 추론 모듈이 누락되었습니다.")
    subprocess.run(["python", script_path, "--image", input_path, "--output", output_path, "--model", model_path], check=True)

def run_mertizci(input_path, output_path):
    script_path = os.path.join(BASE_DIR, "noai-watermark", "inference.py")
    model_path = os.path.join(BASE_DIR, "noai-watermark", "model.pth")
    if not os.path.exists(script_path):
        raise FileNotFoundError("noai-watermark 추론 모듈이 누락되었습니다.")
    subprocess.run(["python", script_path, "--input", input_path, "--output", output_path, "--weights", model_path], check=True)

def remove_all_fingerprints(input_path):
    user_home = os.path.expanduser('~')
    temp_stage1 = os.path.join(user_home, ".ai_unmaker_stage1.png")
    temp_final = os.path.join(user_home, ".ai_unmaker_final.png")
    
    # 순수 PNG로 변환하여 1차 메타데이터 소각
    with Image.open(input_path) as img:
        # EXIF, XMP, IPTC 등 헤더 정보를 강제 소실시킴
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(list(img.getdata()))
        clean_img = clean_img.convert('RGBA') if img.mode in ('RGBA', 'P') else clean_img.convert('RGB')
        
        # JPEG 압축 노이즈로 인한 SynthID 생존을 막기 위해 약간의 가우시안 블러/노이즈를 더할 수도 있으나,
        # GPU 모델이 이 역할을 수행하므로 파일만 덤프함.
        clean_img.save(temp_stage1, format="PNG")
        
    try:
        # 비가시적 워터마크 완전 파괴 파이프라인 (에러 시 원본 복사 안 함)
        run_wiltodelta(temp_stage1, temp_final)
        run_mertizci(temp_final, temp_final)
    except Exception as e:
        print(traceback.format_exc())
        raise RuntimeError(f"GPU 세탁 엔진 실행 실패: {e}\n환경 설정 및 가중치 파일을 확인하세요.")
        
    return temp_final

# ==============================================================================
# [3] VDS Suite 디자인 팔레트 및 GUI 애플리케이션
# ==============================================================================
VDS_COLORS = {
    "bg": "#f5f6f8",          
    "surface": "#ffffff",     
    "border": "#e5e7eb",      
    "text_main": "#111827",   
    "text_sub": "#6b7280",    
    "primary": "#171717",     
    "primary_hover": "#333333",
    "accent": "#ef4444",      
    "success": "#10b981",     
    "gauge_bg": "#f3f4f6"     
}

class DnDApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class AIUnmakerApp(DnDApp):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1040x680")
        self.configure(fg_color=VDS_COLORS["bg"])
        
        self.apply_platform_icon()
        
        self.original_image_path = None
        self.processed_image_path = None
        self.detected_source = []
        self.is_ai = False
        
        self.build_header()
        self.build_body()
        
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop)
        self.render_idle_panel()

    def apply_platform_icon(self):
        ico_path = os.path.join(ICONS_DIR, "icon.ico")
        png_path = os.path.join(ICONS_DIR, "icon.png")

        if sys.platform.startswith("win"):
            if os.path.exists(ico_path):
                try: self.iconbitmap(ico_path)
                except: pass
            if os.path.exists(png_path):
                try:
                    self._icon_photo = tk.PhotoImage(file=png_path)
                    self.iconphoto(False, self._icon_photo)
                except: pass
        else:
            if os.path.exists(png_path):
                try:
                    self._icon_photo = tk.PhotoImage(file=png_path)
                    self.iconphoto(False, self._icon_photo)
                except: pass

    def build_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=45, pady=(32, 12))
        
        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(anchor="w")
        
        ctk.CTkLabel(title_box, text=APP_NAME, font=(FONT_PRIMARY, 26, "bold"), text_color=VDS_COLORS["text_main"]).pack(side="left")
        version_badge = ctk.CTkLabel(title_box, text=APP_VERSION, font=(FONT_PRIMARY, 11, "bold"), text_color=VDS_COLORS["text_sub"], fg_color=VDS_COLORS["border"], corner_radius=4, padx=6, pady=2)
        version_badge.pack(side="left", padx=(10, 0), pady=(4, 0))
        
        ctk.CTkLabel(header_frame, text="로컬 GPU의 딥러닝 추론을 통해 이미지에 내재된 비가시적 AI 핑거프린트와 메타데이터를 완벽하게 탐지하고 파괴합니다.", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="w", pady=(4, 0))

    def build_body(self):
        body_frame = ctk.CTkFrame(self, fg_color="transparent")
        body_frame.pack(fill="both", expand=True, padx=45, pady=(0, 35))
        
        self.left_card = ctk.CTkFrame(body_frame, fg_color=VDS_COLORS["surface"], border_color=VDS_COLORS["border"], border_width=1, corner_radius=10)
        self.left_card.pack(side="left", fill="both", expand=True, padx=(0, 20))
        self.left_card.bind("<Button-1>", lambda e: self.upload_image())
        
        self.drop_container = ctk.CTkFrame(self.left_card, fg_color="transparent")
        self.drop_container.place(relx=0.5, rely=0.5, anchor="center")
        self.drop_container.bind("<Button-1>", lambda e: self.upload_image())

        self.icon_img = get_upload_icon(size=44, color=VDS_COLORS["text_sub"])
        self.drop_icon = ctk.CTkLabel(self.drop_container, text="", image=self.icon_img)
        self.drop_icon.image = self.icon_img
        self.drop_icon.pack(anchor="center", pady=(0, 10))
        self.drop_icon.bind("<Button-1>", lambda e: self.upload_image())
        
        self.drop_text = ctk.CTkLabel(self.drop_container, text="클릭하여 업로드하거나 드래그 앤 드롭하세요", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"])
        self.drop_text.pack(anchor="center")
        self.drop_text.bind("<Button-1>", lambda e: self.upload_image())
        
        self.preview_label = ctk.CTkLabel(self.left_card, text="")
        
        self.right_card = ctk.CTkFrame(body_frame, fg_color=VDS_COLORS["surface"], border_color=VDS_COLORS["border"], border_width=1, corner_radius=10, width=340)
        self.right_card.pack(side="right", fill="both", expand=False)
        self.right_card.pack_propagate(False) 
        
        self.bottom_bar = ctk.CTkFrame(self.right_card, fg_color="transparent")
        self.bottom_bar.pack(side="bottom", fill="x", padx=24, pady=(0, 24))
        
        self.content_container = ctk.CTkFrame(self.right_card, fg_color="transparent")
        self.content_container.pack(side="top", fill="both", expand=True, padx=24, pady=(24, 8))

    def clear_panels(self):
        for widget in self.content_container.winfo_children():
            widget.destroy()
        for widget in self.bottom_bar.winfo_children():
            widget.destroy()

    def draw_semi_circle_gauge(self, percentage, color, main_text, sub_text):
        canvas = tk.Canvas(self.content_container, width=190, height=95, bg=VDS_COLORS["surface"], highlightthickness=0)
        canvas.pack(pady=(2, 8))
        canvas.create_arc(15, 15, 175, 175, start=0, extent=180, style=tk.ARC, outline=VDS_COLORS["gauge_bg"], width=16)
        extent_angle = -(180 * (percentage / 100))
        canvas.create_arc(15, 15, 175, 175, start=180, extent=extent_angle, style=tk.ARC, outline=color, width=16)
        
        canvas.create_text(95, 58, text=main_text, font=(FONT_PRIMARY, 26, "bold"), fill=VDS_COLORS["text_main"])
        canvas.create_text(95, 82, text=sub_text, font=(FONT_PRIMARY, 12, "bold"), fill=VDS_COLORS["text_sub"])

    def render_idle_panel(self):
        self.clear_panels()
        
        title = ctk.CTkLabel(self.content_container, text="비가시적 주파수 워터마크까지\nGPU 딥러닝으로 파괴하세요.", font=(FONT_PRIMARY, 16, "bold"), text_color=VDS_COLORS["text_main"], justify="left")
        title.pack(anchor="w", pady=(0, 16))
        
        list_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        list_frame.pack(fill="x", anchor="w")
        
        features = ["메타데이터 및 C2PA 인증 마크 소각", "생성형 AI (Midjourney 등) 출처 식별", "픽셀 단위 미세 노이즈 강제 재배열", "비가시적 주파수(SynthID) 완벽 무효화"]
        for f in features:
            ctk.CTkLabel(list_frame, text=f"•  {f}", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"], justify="left").pack(anchor="w", pady=1)
            
        btn = ctk.CTkButton(self.bottom_bar, text="이미지 대기 중", font=(FONT_PRIMARY, 14, "bold"), fg_color=VDS_COLORS["gauge_bg"], text_color=VDS_COLORS["text_sub"], state="disabled", height=44, corner_radius=6)
        btn.pack(fill="x")

    def render_scanning_panel(self):
        self.clear_panels()
        ctk.CTkLabel(self.content_container, text="이미지 FFT 및 청크 분석 중...", font=(FONT_PRIMARY, 14, "bold"), text_color=VDS_COLORS["text_main"]).pack(expand=True)

    def render_result_panel(self):
        self.clear_panels()
        
        if self.is_ai:
            self.draw_semi_circle_gauge(100, VDS_COLORS["accent"], "100%", "AI/SynthID")
            ctk.CTkLabel(self.content_container, text="AI 생성 혹은 비가시적 워터마크가 감지되었습니다.", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="center", pady=(0, 10))
            
            info_frame = ctk.CTkFrame(self.content_container, fg_color="#f9fafb", corner_radius=8, border_width=1, border_color=VDS_COLORS["border"])
            info_frame.pack(fill="x", pady=2)
            
            ctk.CTkLabel(info_frame, text="식별된 생성 흔적", font=(FONT_PRIMARY, 11, "bold"), text_color=VDS_COLORS["text_sub"]).pack(anchor="w", padx=12, pady=(10, 4))
            
            for src in self.detected_source:
                row = ctk.CTkFrame(info_frame, fg_color="#ffffff", corner_radius=6, border_width=1, border_color=VDS_COLORS["border"])
                row.pack(fill="x", padx=10, pady=(2, 4))
                ctk.CTkLabel(row, text=f"•  {src}", font=(FONT_PRIMARY, 13, "bold"), text_color=VDS_COLORS["text_main"]).pack(anchor="w", padx=10, pady=6)
            
            ctk.CTkFrame(info_frame, fg_color="transparent", height=4).pack()
        else:
            self.draw_semi_circle_gauge(0, VDS_COLORS["success"], "0%", "No Signature")
            ctk.CTkLabel(self.content_container, text="확인된 AI 핑거프린트가 없습니다.", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="center", pady=(0, 12))
            ctk.CTkLabel(self.content_container, text="•  스캔되지 않은 숨겨진 노이즈가 있을 수 있습니다.\n    GPU 세탁을 통해 확실하게 픽셀을 재배열하세요.", font=(FONT_PRIMARY, 12, "normal"), text_color=VDS_COLORS["text_sub"], wraplength=280, justify="left").pack(anchor="w")
            
        btn = ctk.CTkButton(self.bottom_bar, text="강제 픽셀 세탁 실행 (GPU 파괴)", font=(FONT_PRIMARY, 14, "bold"), fg_color=VDS_COLORS["primary"], hover_color=VDS_COLORS["primary_hover"], height=44, corner_radius=6, command=self.process_removal)
        btn.pack(fill="x")

    def render_download_panel(self):
        self.clear_panels()
        self.dl_status_label = ctk.CTkLabel(self.content_container, text="최초 실행: GPU 파괴 모듈 다운로드 중...", font=(FONT_PRIMARY, 14, "bold"), text_color=VDS_COLORS["text_main"])
        self.dl_status_label.pack(anchor="center", pady=(40, 10))
        
        self.dl_progress_bar = ctk.CTkProgressBar(self.content_container, width=240, height=8, fg_color=VDS_COLORS["gauge_bg"], progress_color=VDS_COLORS["primary"])
        self.dl_progress_bar.set(0)
        self.dl_progress_bar.pack(anchor="center", pady=10)
        
        self.dl_percent_label = ctk.CTkLabel(self.content_container, text="0%", font=(FONT_PRIMARY, 12, "bold"), text_color=VDS_COLORS["text_sub"])
        self.dl_percent_label.pack(anchor="center")

    def render_done_panel(self):
        self.clear_panels()
        self.draw_semi_circle_gauge(100, VDS_COLORS["success"], "100%", "Destroyed")
        ctk.CTkLabel(self.content_container, text="메타데이터 및 주파수 핑거프린트가 파괴되었습니다.", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="center", pady=(0, 14))
        
        btn_save = ctk.CTkButton(self.bottom_bar, text="안전한 이미지 저장", font=(FONT_PRIMARY, 14, "bold"), fg_color=VDS_COLORS["primary"], hover_color=VDS_COLORS["primary_hover"], height=44, corner_radius=6, command=self.save_image)
        btn_save.pack(fill="x", pady=(0, 8))
        
        btn_reset = ctk.CTkButton(self.bottom_bar, text="다른 이미지 시도", font=(FONT_PRIMARY, 13, "bold"), fg_color=VDS_COLORS["surface"], text_color=VDS_COLORS["text_main"], border_width=1, border_color=VDS_COLORS["border"], hover_color=VDS_COLORS["gauge_bg"], height=44, corner_radius=6, command=self.reset_tool)
        btn_reset.pack(fill="x")

    def on_drop(self, event):
        file_path = event.data.strip('{}')
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            self.load_image_preview(file_path)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp")])
        if file_path:
            self.load_image_preview(file_path)

    def load_image_preview(self, file_path):
        self.original_image_path = file_path
        self.drop_container.place_forget()
        
        img = Image.open(file_path)
        img.thumbnail((560, 480))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        
        self.preview_label.configure(image=ctk_img)
        self.preview_label.image = ctk_img
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.render_scanning_panel()
        self.update()
        self.after(500, self.run_detection)

    def run_detection(self):
        self.is_ai, self.detected_source = detect_metadata_fingerprint(self.original_image_path)
        self.render_result_panel()

    def process_removal(self):
        models_to_download = []
        for path in MODEL_URLS.keys():
            full_path = os.path.join(BASE_DIR, path)
            if not os.path.exists(full_path):
                models_to_download.append((path, MODEL_URLS[path]))

        if models_to_download:
            self.render_download_panel()
            threading.Thread(target=self.download_models_thread, args=(models_to_download,), daemon=True).start()
        else:
            self.start_processing_ui()

    def download_models_thread(self, models_list):
        try:
            for i, (path, url) in enumerate(models_list):
                full_path = os.path.join(BASE_DIR, path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                folder_name = os.path.basename(os.path.dirname(full_path))

                def reporthook(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min((block_num * block_size) / total_size, 1.0)
                        self.after(0, self.update_dl_progress, folder_name, percent, i+1, len(models_list))

                try:
                    urllib.request.urlretrieve(url, full_path, reporthook)
                except Exception as e:
                    print(f"URL 다운로드 실패: {e}")
                    with open(full_path, 'wb') as f: f.write(b"")

            self.after(500, self.start_processing_ui)
        except Exception as e:
            print(f"스레드 오류: {e}")
            self.after(0, self.start_processing_ui)

    def update_dl_progress(self, folder_name, percent, current_idx, total_count):
        if hasattr(self, 'dl_status_label') and self.dl_status_label.winfo_exists():
            self.dl_status_label.configure(text=f"AI 가중치 다운로드 중... [{current_idx}/{total_count}]\n({folder_name})")
            self.dl_progress_bar.set(percent)
            self.dl_percent_label.configure(text=f"{int(percent * 100)}%")

    def start_processing_ui(self):
        self.clear_panels()
        ctk.CTkLabel(self.content_container, text="GPU 가동 및 주파수 핑거프린트 파괴 중...\n(잠시만 기다려주세요)", font=(FONT_PRIMARY, 14, "bold"), text_color=VDS_COLORS["primary"]).pack(expand=True)
        self.update()
        self.after(300, self.execute_removal_logic)

    def execute_removal_logic(self):
        try:
            self.processed_image_path = remove_all_fingerprints(self.original_image_path)
            
            img = Image.open(self.processed_image_path)
            img.thumbnail((560, 480))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.preview_label.configure(image=ctk_img)
            self.preview_label.image = ctk_img
            
            self.render_done_panel()
        except Exception as e:
            messagebox.showerror("추론 엔진 오류", f"GPU 섭동 엔진 실행 중 오류가 발생했습니다.\n{e}")
            self.render_result_panel()

    def save_image(self):
        directory = os.path.dirname(self.original_image_path)
        filename, ext = os.path.splitext(os.path.basename(self.original_image_path))
        new_filename = f"{filename}_cleaned.png"
        save_path = os.path.join(directory, new_filename)
        
        try:
            shutil.copy2(self.processed_image_path, save_path)
            messagebox.showinfo("다운로드 완료", f"저장 완료:\n{new_filename}")
        except:
            pass

    def reset_tool(self):
        self.original_image_path = None
        self.processed_image_path = None
        self.preview_label.place_forget()
        self.drop_container.place(relx=0.5, rely=0.5, anchor="center")
        self.render_idle_panel()

if __name__ == "__main__":
    ctk.set_appearance_mode("Light")
    app = AIUnmakerApp()
    app.mainloop()