@echo off
chcp 65001 > nul
set REPO_URL=https://github.com/catgarret/AI-Unmaker-Pro-GPU-.git

echo 🚀 AI Unmaker Pro Git 업로드를 시작합니다...

if not exist ".git" (
    echo 📦 Git 저장소를 초기화합니다...
    git init
    git branch -M main
)

git remote | findstr /C:"origin" > nul
if %errorlevel% equ 0 (
    echo 🔗 기존 origin URL을 업데이트합니다...
    git remote set-url origin %REPO_URL%
) else (
    echo 🔗 origin 원격 저장소를 연결합니다...
    git remote add origin %REPO_URL%
)

echo 📝 파일 변경 사항을 준비합니다...
git add .

git diff --cached --quiet
if %errorlevel% equ 0 (
    echo ⚠️ 커밋할 새로운 변경 사항이 없습니다.
) else (
    echo 💾 커밋 생성 중...
    git commit -m "feat: UI 및 딥러닝 모듈 연동 업데이트"
)

echo ⬆️ GitHub으로 푸시합니다 (main 브랜치)...
git push -u origin main

echo ✅ 성공적으로 업로드되었습니다! 레포지토리: %REPO_URL%
pause