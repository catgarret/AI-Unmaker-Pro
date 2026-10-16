# AI Unmaker 빌드 및 업데이트 가이드 (Windows & Mac)

이 문서는 AI Unmaker 파이썬 코드를 실행하고 데스크톱 애플리케이션(`AI Unmaker.exe`, `AI Unmaker.app`)으로 패키징하는 방법과, 깃허브 오픈소스 도구들의 최신 버전을 업데이트하는 방법을 안내합니다.

---

## 1. 초기 파이썬 및 가상환경 세팅

Windows와 Mac 모두 터미널(또는 CMD)을 열고 프로젝트 폴더로 이동한 뒤 아래 명령어를 순서대로 실행하여 필수 라이브러리를 설치합니다.

### 1.1 Mac (macOS)
```bash
# 1. 프로젝트 전용 가상환경 생성
python3 -m venv venv

# 2. 가상환경 활성화
source venv/bin/activate

# 3. 필수 라이브러리 및 패키징 도구 설치
pip install --upgrade pip
pip install customtkinter pillow tkinterdnd2 pyinstaller
```

### 1.2 Windows
*(주의: Python 설치 시 반드시 "Add Python to PATH" 옵션을 체크해야 합니다.)*
```cmd
# 1. 프로젝트 전용 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
venv\Scripts\activate

# 3. 필수 라이브러리 및 패키징 도구 설치
python -m pip install --upgrade pip
pip install customtkinter pillow tkinterdnd2 pyinstaller
```

---

## 2. 애플리케이션 빌드하기 (수동)

실행 파일 이름을 `AI Unmaker`로 지정하고, UI/드래그 앤 드롭 라이브러리 누락으로 인한 튕김(Crash) 현상을 방지하기 위해 `--collect-all` 옵션을 반드시 포함해야 합니다. (가상환경이 활성화된 상태에서 진행)

### 2.1 Mac용 빌드 (`AI Unmaker.app`)
```bash
pyinstaller --noconfirm --noconsole --windowed --name "AI Unmaker" --collect-all customtkinter --collect-all tkinterdnd2 main.py
```
* **결과물:** `dist` 폴더 안에 `AI Unmaker.app` 생성 완료.

### 2.2 Windows용 빌드 (`AI Unmaker.exe`)
```cmd
pyinstaller --noconfirm --noconsole --onefile --name "AI Unmaker" --collect-all customtkinter --collect-all tkinterdnd2 main.py
```
* **결과물:** `dist` 폴더 안에 `AI Unmaker.exe` 생성 완료.

---

## 3. 원클릭 자동화 스크립트 (.sh, .bat)

매번 명령어를 치기 번거로울 때 사용하는 통합 스크립트입니다. (의존성 설치, 오픈소스 깃허브 업데이트, 빌드를 한 번에 처리합니다)

### 3.1 Mac용 전체 스크립트 (`build_mac.sh`)
```bash
#!/bin/bash
echo "1. 가상환경 세팅 및 의존성 설치..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install customtkinter pillow tkinterdnd2 pyinstaller

echo "2. AI 워터마크 도구 최신 버전 업데이트..."
# wiltodelta 도구 업데이트
if [ -d "remove-ai-watermarks" ]; then
    cd remove-ai-watermarks && git pull && cd ..
else
    git clone https://github.com/wiltodelta/remove-ai-watermarks.git
fi
# mertizci 도구 업데이트
if [ -d "noai-watermark" ]; then
    cd noai-watermark && git pull && cd ..
else
    git clone https://github.com/mertizci/noai-watermark.git
fi
pip install --upgrade google-cloud-aiplatform

echo "3. AI Unmaker Mac용 앱 빌드 시작..."
pyinstaller --noconfirm --noconsole --windowed --name "AI Unmaker" --collect-all customtkinter --collect-all tkinterdnd2 main.py

echo "완료! dist/ 폴더에서 AI Unmaker.app을 확인하세요."
```
*(사용법: 터미널에서 `chmod +x build_mac.sh`로 실행 권한 부여 후 `./build_mac.sh` 실행)*

### 3.2 Windows용 전체 스크립트 (`build_win.bat`)
```bat
@echo off
echo 1. 가상환경 세팅 및 의존성 설치...
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install customtkinter pillow tkinterdnd2 pyinstaller

echo 2. AI 워터마크 도구 최신 버전 업데이트...
IF EXIST "remove-ai-watermarks" (
    cd remove-ai-watermarks
    git pull
    cd ..
) ELSE (
    git clone https://github.com/wiltodelta/remove-ai-watermarks.git
)
IF EXIST "noai-watermark" (
    cd noai-watermark
    git pull
    cd ..
) ELSE (
    git clone https://github.com/mertizci/noai-watermark.git
)
python -m pip install --upgrade google-cloud-aiplatform

echo 3. AI Unmaker Windows용 앱 빌드 시작...
pyinstaller --noconfirm --noconsole --onefile --name "AI Unmaker" --collect-all customtkinter --collect-all tkinterdnd2 main.py

echo 완료! dist 폴더에서 AI Unmaker.exe를 확인하세요.
pause
```
*(사용법: 해당 배치 파일을 더블클릭하여 실행)*