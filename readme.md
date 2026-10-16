# AI Unmarker Pro (GPU)

이미지에 내재된 비가시적 AI 핑거프린트, 노이즈 워터마크, 메타데이터(C2PA, EXIF, PNG tEXt Chunks)를 스캔하고 완전히 제거하는 로컬 딥러닝 데스크톱 도구입니다.

---

## 📁 프로젝트 폴더 구조

```text
📁 AI-Unmaker-Pro/
 ├── 📄 main.py                   # 메인 GUI 실행 엔트리포인트
 ├── 📄 .gitignore                # 대용량 모델 및 캐시 배제 규칙
 ├── 📄 README.md                 # 프로젝트 문서
 ├── 📁 icons/                    # 앱 아이콘 리소스
 │    ├── icon.ico                # Windows 실행 파일용 아이콘
 │    └── icon.png                # macOS / Linux / GUI 렌더용 아이콘
 ├── 📁 fonts/                    # Pretendard & Phosphor 폰트 폴더
 │    ├── Pretendard-Bold.otf
 │    ├── Pretendard-SemiBold.otf
 │    ├── Pretendard-Medium.otf
 │    ├── Pretendard-Regular.otf
 │    ├── Phosphor.ttf
 │    └── Phosphor-Bold.ttf
 ├── 📁 remove-ai-watermarks/     # wiltodelta 추론 모듈
 ├── 📁 noai-watermark/           # mertizci 추론 모듈
 └── 📁 sample/                   # 검사 테스트용 이미지 폴더
```

---

## ⚙️ 주요 핵심 기능

1. **VDS Suite 디자인 시스템**: Pretendard 타이포그래피 hierarchy 적용, 최적 10px 간격 드롭존, 도넛 게이지 시각화 Canvas 탑재
2. **GPU 딥러닝 세탁 파이프라인**: `wiltodelta` + `mertizci` + `SynthID 압축 우회` 3중 필터링
3. **독립 프로세스 격리 (`subprocess`)**: 모델 연산 메모리를 분리하여 VRAM OOM(메모리 누수) 및 메인 GUI 크래시 원천 차단
4. **크로스 플랫폼 아이콘 지원**: Windows(`icon.ico`) 및 macOS(`icon.png`) 자동 환경 감지 및 타이틀바/독(Dock) 적용

---

## 🛠️ 실행 및 빌드 가이드

### 1. 저장소 클론 및 패키지 설치
```bash
git clone https://github.com/catgarret/AI-Unmaker-Pro.git
cd AI-Unmaker-Pro

# 필수 GUI 및 그래픽 라이브러리 설치
pip install customtkinter tkinterdnd2 Pillow

# PyTorch (GPU 가속 라이브러리)
pip install torch torchvision torchaudio
```

### 2. 로컬 실행
```bash
python main.py
```

### 3. PyInstaller 단일 바이너리 빌드

#### Windows OS (.exe 단일 파일)
```bash
pyinstaller --noconfirm --noconsole --onefile --icon=icons/icon.ico --name "AI Unmarker" --collect-all customtkinter --collect-all tkinterdnd2 main.py
```

#### macOS (.app 번들)
```bash
pyinstaller --noconfirm --noconsole --windowed --icon=icons/icon.png --name "AI Unmarker" --collect-all customtkinter --collect-all tkinterdnd2 main.py
```

> **배포 시 주의**: 빌드된 실행 파일은 `fonts/`, `icons/`, 그리고 딥러닝 모듈 폴더(`remove-ai-watermarks/`, `noai-watermark/`)와 동일한 디렉토리에 함께 배포되어야 가중치 파일 및 리소스를 정상 참조합니다.

---

## 📜 오픈소스 라이선스 (License & Notice)

본 프로젝트는 연구 및 개인 창작물 권리 보호 목적의 유틸리티 프로그램이며, 사용된 오픈소스 소프트웨어의 라이선스를 엄격히 준수합니다.

- **remove-ai-watermarks**: Copyright (c) wiltodelta (MIT License)
- **noai-watermark**: Copyright (c) mertizci (MIT License)
- **Pretendard Font**: Copyright (c) 2021 Kil Hyung-jin (SIL Open Font License 1.1)
- **Phosphor Icons**: Copyright (c) 2020 Phosphor Icons (MIT License)
- **가중치 파일 안내**: 사전 학습 가중치 파일(`.pth`, `.pt`)은 최초 실행 시 원작자 공식 깃허브에서 자동 다운로드됩니다.