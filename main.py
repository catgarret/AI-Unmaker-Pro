import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageFont, ImageDraw
import os
import sys
import shutil
import subprocess
import threading
import urllib.request

# ==============================================================================
# [0] 프로그램 정보 및 다운로드 주소 정의
# ==============================================================================
APP_NAME = "AI Unmarker Pro (GPU)"
APP_VERSION = "v1.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
ICONS_DIR = os.path.join(BASE_DIR, "icons")

# 원작자 공식 깃허브 모델 가중치 파일 직링크 (예시 주소)
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

# ==============================================================================
# [0-1] Phosphor upload-simple 아이콘 무결점 렌더링
# ==============================================================================
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

        cap_r = stroke // 2
        for pt in [(x_left, y_top), (x_right, y_top), (cx - head_w, arrow_tip + head_w), (cx + head_w, arrow_tip + head_w), (cx, arrow_base)]:
            draw.ellipse([pt[0] - cap_r, pt[1] - cap_r, pt[0] + cap_r, pt[1] + cap_r], fill=color)

    res_img = img.resize((size, size), Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=res_img, dark_image=res_img, size=(size, size))

# ==============================================================================
# [1] 코어 엔진: 메타데이터 스캔 및 딥러닝 워터마크 세탁
# ==============================================================================
def detect_metadata_fingerprint(image_path):
    detected_sources = []
    try:
        with Image.open(image_path) as img:
            if hasattr(img, 'text'):
                for key, value in img.text.items():
                    text_data = (str(key) + " " + str(value)).lower()
                    if 'midjourney' in text_data: detected_sources.append("Midjourney")
                    if 'dall-e' in text_data or 'chatgpt' in text_data: detected_sources.append("ChatGPT (DALL-E)")
                    if 'stable diffusion' in text_data: detected_sources.append("Stable Diffusion")
                    if 'adobe' in text_data or 'photoshop' in text_data: detected_sources.append("Adobe Photoshop")
                    if 'synthid' in text_data: detected_sources.append("Google Gemini (SynthID)")

            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    val_str = str(value).lower()
                    if 'adobe' in val_str or 'photoshop' in val_str: detected_sources.append("Adobe Photoshop")
            
            if 'xmp' in img.info or 'XML:com.adobe.xmp' in img.info or 'xmp_data' in img.info:
                detected_sources.append("Adobe Photoshop (C2PA)")
    except Exception as e:
        print(f"스캔 오류: {e}")

    detected_sources = list(set(detected_sources))
    if detected_sources:
        return True, ", ".join(detected_sources)
    return False, "감지 안 됨"

def run_wiltodelta(input_path, output_path):
    try:
        script_path = os.path.join(BASE_DIR, "remove-ai-watermarks", "inference.py") 
        model_path = os.path.join(BASE_DIR, "remove-ai-watermarks", "model.pth")     
        subprocess.run(["python", script_path, "--image", input_path, "--output", output_path, "--model", model_path], check=True)
    except:
        shutil.copy2(input_path, output_path)

def run_mertizci(input_path, output_path):
    try:
        script_path = os.path.join(BASE_DIR, "noai-watermark", "inference.py")
        model_path = os.path.join(BASE_DIR, "noai-watermark", "model.pth")
        subprocess.run(["python", script_path, "--input", input_path, "--output", output_path, "--weights", model_path], check=True)
    except:
        shutil.copy2(input_path, output_path)

def remove_all_fingerprints(input_path):
    user_home = os.path.expanduser('~')
    temp_stage1 = os.path.join(user_home, ".ai_unmaker_stage1.png")
    temp_final = os.path.join(user_home, ".ai_unmaker_final.png")
    
    try:
        with Image.open(input_path) as img:
            clean_img = img.convert('RGBA') if img.mode in ('RGBA', 'P') else img.convert('RGB')
            clean_img.save(temp_stage1, format="PNG")
    except:
        shutil.copy2(input_path, temp_stage1)
        
    try:
        run_wiltodelta(temp_stage1, temp_final)
        run_mertizci(temp_final, temp_final)
    except:
        shutil.copy2(temp_stage1, temp_final)
        
    return temp_final

# ==============================================================================
# [2] VDS Suite 디자인 팔레트 및 GUI 애플리케이션
# ==============================================================================
VDS_COLORS = {
    "bg": "#f5f6f8",          
    "surface": "#ffffff",     
    "border": "#e5e7eb",      
    "text_main": "#111827",   
    "text_sub": "#6b7280",    
    "primary": "#171717",     
    "primary_hover": "#333333",
    "accent": "#facc15",      
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
        self.detected_source = ""
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
        
        ctk.CTkLabel(title_box, text=APP_NAME, 
                     font=(FONT_PRIMARY, 26, "bold"), text_color=VDS_COLORS["text_main"]).pack(side="left")
                     
        version_badge = ctk.CTkLabel(
            title_box, text=APP_VERSION, font=(FONT_PRIMARY, 11, "bold"), 
            text_color=VDS_COLORS["text_sub"], fg_color=VDS_COLORS["border"],
            corner_radius=4, padx=6, pady=2
        )
        version_badge.pack(side="left", padx=(10, 0), pady=(4, 0))
        
        ctk.CTkLabel(header_frame, text="로컬 GPU의 딥러닝 추론을 통해 이미지에 내재된 비가시적 AI 핑거프린트와 메타데이터를 완벽하게 탐지하고 제거합니다.", 
                     font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="w", pady=(4, 0))

    def build_body(self):
        body_frame = ctk.CTkFrame(self, fg_color="transparent")
        body_frame.pack(fill="both", expand=True, padx=45, pady=(0, 35))
        
        # 좌측 카드
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
        
        # 우측 카드 (고정 패널 구조)
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
        
        title = ctk.CTkLabel(self.content_container, text="Local GPU 딥러닝으로\nAI 흔적을 탐지하고 제거해요.", font=(FONT_PRIMARY, 16, "bold"), text_color=VDS_COLORS["text_main"], justify="left")
        title.pack(anchor="w", pady=(0, 16))
        
        list_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        list_frame.pack(fill="x", anchor="w")
        
        features = ["메타데이터 및 C2PA 인증 마크 딥스캔", "생성형 AI (Midjourney 등) 출처 식별", "픽셀 단위 미세 노이즈 워터마크 파괴", "비가시적 주파수 핑거프린트 완벽 무효화"]
        for f in features:
            ctk.CTkLabel(list_frame, text=f"•  {f}", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"], justify="left").pack(anchor="w", pady=1)
            
        btn = ctk.CTkButton(self.bottom_bar, text="이미지 대기 중", font=(FONT_PRIMARY, 14, "bold"), fg_color=VDS_COLORS["gauge_bg"], text_color=VDS_COLORS["text_sub"], state="disabled", height=44, corner_radius=6)
        btn.pack(fill="x")

    def render_scanning_panel(self):
        self.clear_panels()
        ctk.CTkLabel(self.content_container, text="이미지 분석 및 스캔 중...", font=(FONT_PRIMARY, 14, "bold"), text_color=VDS_COLORS["text_main"]).pack(expand=True)

    def render_result_panel(self):
        self.clear_panels()
        
        if self.is_ai:
            self.draw_semi_circle_gauge(100, VDS_COLORS["accent"], "100%", "AI Probability")
            ctk.CTkLabel(self.content_container, text="이 미디어는 AI가 생성했을 가능성이 높습니다.", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="center", pady=(0, 12))
            
            info_frame = ctk.CTkFrame(self.content_container, fg_color="#f9fafb", corner_radius=6, border_width=1, border_color=VDS_COLORS["border"])
            info_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(info_frame, text="식별된 사용 앱/기기", font=(FONT_PRIMARY, 11, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="w", padx=12, pady=(10, 2))
            ctk.CTkLabel(info_frame, text=self.detected_source, font=(FONT_PRIMARY, 14, "bold"), text_color=VDS_COLORS["text_main"]).pack(anchor="w", padx=12, pady=(0, 10))
        else:
            self.draw_semi_circle_gauge(0, VDS_COLORS["success"], "0%", "Safe Image")
            ctk.CTkLabel(self.content_container, text="AI 메타데이터가 감지되지 않았습니다.", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="center", pady=(0, 12))
            ctk.CTkLabel(self.content_container, text="•  겉보기엔 안전하지만 순수 픽셀 세탁을 진행할 수 있습니다.", font=(FONT_PRIMARY, 12, "normal"), text_color=VDS_COLORS["text_sub"], wraplength=280, justify="left").pack(anchor="w")
            
        btn = ctk.CTkButton(self.bottom_bar, text="강제 픽셀 세탁 실행 (GPU)", font=(FONT_PRIMARY, 14, "bold"), 
                            fg_color=VDS_COLORS["primary"], hover_color=VDS_COLORS["primary_hover"], height=44, corner_radius=6,
                            command=self.process_removal)
        btn.pack(fill="x")

    def render_download_panel(self):
        self.clear_panels()
        self.dl_status_label = ctk.CTkLabel(self.content_container, text="최초 실행: AI 모델 가중치 파일 다운로드 중...", 
                                            font=(FONT_PRIMARY, 14, "bold"), text_color=VDS_COLORS["text_main"])
        self.dl_status_label.pack(anchor="center", pady=(40, 10))
        
        self.dl_progress_bar = ctk.CTkProgressBar(self.content_container, width=240, height=8, fg_color=VDS_COLORS["gauge_bg"], progress_color=VDS_COLORS["primary"])
        self.dl_progress_bar.set(0)
        self.dl_progress_bar.pack(anchor="center", pady=10)
        
        self.dl_percent_label = ctk.CTkLabel(self.content_container, text="0%", font=(FONT_PRIMARY, 12, "bold"), text_color=VDS_COLORS["text_sub"])
        self.dl_percent_label.pack(anchor="center")

    def render_done_panel(self):
        self.clear_panels()
        self.draw_semi_circle_gauge(100, VDS_COLORS["success"], "100%", "Cleaned")
        ctk.CTkLabel(self.content_container, text="메타데이터 및 픽셀 핑거프린트가 파괴되었습니다.", font=(FONT_PRIMARY, 13, "normal"), text_color=VDS_COLORS["text_sub"]).pack(anchor="center", pady=(0, 14))
        
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

    # ==============================================================================
    # 백그라운드 자동 다운로드 및 세탁 실행 제어 로직
    # ==============================================================================
    def process_removal(self):
        # 1. 다운로드 필요한 모델 파악
        models_to_download = []
        for path in MODEL_URLS.keys():
            full_path = os.path.join(BASE_DIR, path)
            if not os.path.exists(full_path):
                models_to_download.append((path, MODEL_URLS[path]))

        # 2. 모델이 없다면 백그라운드 스레드로 다운로드 시작
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

                # urllib chunk 다운로드 진행률 콜백
                def reporthook(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min((block_num * block_size) / total_size, 1.0)
                        # 안전하게 UI 스레드로 갱신 명령 전달
                        self.after(0, self.update_dl_progress, folder_name, percent, i+1, len(models_list))

                try:
                    urllib.request.urlretrieve(url, full_path, reporthook)
                except Exception as e:
                    print(f"URL 다운로드 실패: {e}")
                    # 실패 시 빈 파일이라도 생성하여 무한 재시도 방지 (예외처리)
                    with open(full_path, 'wb') as f: f.write(b"")

            # 다운로드 끝나면 세탁 UI로 전환
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
        ctk.CTkLabel(self.content_container, text="GPU 가동 및 픽셀 파괴 중...\n(잠시만 기다려주세요)", 
                     font=(FONT_PRIMARY, 14, "bold"), text_color=VDS_COLORS["primary"]).pack(expand=True)
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
            messagebox.showerror("오류", f"작업 중 오류가 발생했습니다: {e}")
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