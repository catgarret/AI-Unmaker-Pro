#!/usr/bin/env bash

# 스크립트 실행 중 에러 발생 시 중단
set -e

REPO_URL="https://github.com/catgarret/AI-Unmaker-Pro-GPU-.git"

echo "🚀 AI Unmaker Pro Git 업로드를 시작합니다..."

# 1. git 초기화 여부 확인
if [ ! -d ".git" ]; then
    echo "📦 Git 저장소를 초기화합니다..."
    git init
    git branch -M main
fi

# 2. 원격 저장소(remote) 설정 확인 및 변경
if git remote | grep -q "origin"; then
    echo "🔗 기존 origin URL을 업데이트합니다..."
    git remote set-url origin "$REPO_URL"
else
    echo "🔗 origin 원격 저장소를 연결합니다..."
    git remote add origin "$REPO_URL"
fi

# 3. 변경 사항 스태이징 및 커밋
echo "📝 파일 변경 사항을 준비합니다..."
git add .

# 변경 사항이 있는지 확인
if git diff --cached --quiet; then
    echo "⚠️ 커밋할 새로운 변경 사항이 없습니다."
else
    COMMIT_MSG="feat: UI 및 딥러닝 모듈 연동 업데이트 ($(date +'%Y-%m-%d %H:%M:%S'))"
    echo "💾 커밋 생성 중: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
fi

# 4. GitHub 원격 저장소로 푸시
echo "⬆️ GitHub으로 푸시합니다 (main 브랜치)..."
git push -u origin main

echo "✅ 성공적으로 업로드되었습니다! 레포지토리: $REPO_URL"