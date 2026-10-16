# AI Unmarker Pro (GPU)

이미지에 숨겨진 AI 핑거프린트, 픽셀 워터마크, 메타데이터(C2PA, EXIF, PNG Text Chunks)를 탐지하고 완전히 제거하는 로컬 딥러닝 데스크톱 프로그램입니다.

---

## 📁 프로젝트 폴더 구조

```text
📁 AI-Unmarker-Pro-GPU/
 ├── 📄 main.py                   # 메인 GUI 실행 파일
 ├── 📄 .gitignore                # Git 제외 규칙
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
 ├── 📁 remove-ai-watermarks/     # wiltodelta GPU 연산 모듈
 ├── 📁 noai-watermark/           # mertizci GPU 연산 모듈
 └── 📁 sample/                   # 검사 테스트용 이미지 폴더
```

---

## ⚙️ 주요 핵심 기능

1. **VDS Suite 디자인 시스템**: Pretendard 타이포그래피 hierarchy 적용, 10px 최적 간격 드롭존, 반원형 게이지 그래프 Canvas 시각화
2. **GPU 딥러닝 파이프라인**: `wiltodelta` + `mertizci` + `SynthID 압축 우회` 3단계 연속 세탁
3. **독립 프로세스 실행 (`subprocess`)**: 연산 시 메모리를 분리하여 VRAM OOM(메모리 부족) 및 프로그램 비정상 종료 방지
4. **크로스 플랫폼 아이콘 감지**: Windows(`icon.ico`) 및 macOS(`icon.png`) 자동 환경 감지 및 윈도우 타이틀바/독(Dock) 적용

---

## 🛠️ 실행 및 빌드 방법

### 1. 가상환경 패키지 설치
```bash
# 기본 GUI 및 이미지 처리 라이브러리
pip install customtkinter tkinterdnd2 Pillow

# PyTorch (GPU 연산 라이브러리)
pip install torch torchvision torchaudio
```

### 2. 프로그램 로컬 실행
```bash
python main.py
```

### 3. PyInstaller 단일 실행 파일 빌드

#### Windows OS (.exe 단일 파일)
```bash
pyinstaller --noconfirm --noconsole --onefile --icon=icons/icon.ico --name "AI Unmarker" --collect-all customtkinter --collect-all tkinterdnd2 main.py
```

#### macOS (.app 번들)
```bash
pyinstaller --noconfirm --noconsole --windowed --icon=icons/icon.png --name "AI Unmarker" --collect-all customtkinter --collect-all tkinterdnd2 main.py
```

> **배포 시 주의사항**: 빌드된 실행 파일(`dist/AI Unmarker`)은 `fonts/`, `icons/`, 그리고 딥러닝 모듈 폴더(`remove-ai-watermarks/`, `noai-watermark/`)와 동일한 디렉토리에 위치해야 정상 가동됩니다.

---

## 📜 오픈소스 라이선스 및 크레딧 (License & Notice)

본 프로젝트는 연구 및 개인 창작물 보호 목적의 유틸리티 프로그램이며, 사용된 오픈소스 라이브러리와 원저작자의 권리를 준수합니다.

- **remove-ai-watermarks**: Copyright (c) wiltodelta (MIT License)
- **noai-watermark**: Copyright (c) mertizci (MIT License)
- **Pretendard Font**: Copyright (c) 2021 Kil Hyung-jin (SIL Open Font License 1.1)
- **Phosphor Icons**: Copyright (c) 2020 Phosphor Icons (MIT License)
- **주의**: 대용량 사전 학습 가중치 파일(`.pth`, `.pt`, `.bin`)은 저작권 및 용량 정책에 따라 원격 저장소 커밋 대상에서 제외됩니다(`.gitignore` 관리).