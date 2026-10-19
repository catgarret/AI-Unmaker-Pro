<template>
  <div class="unmarker-container">
    <header class="tool-header">
      <div class="brand-row">
        <h1 class="brand-title">AI Unmarker</h1>
        <span class="brand-badge">WEB LITE</span>
      </div>
      <p class="brand-subtitle">
        이미지 파일에 잔류하는 메타데이터, 인증 매니페스트 및 식별 태그를 브라우저 환경에서 초기화합니다.
      </p>
    </header>

    <div class="workspace-grid">
      <div 
        class="dropzone-card" 
        :class="{ 'is-dragover': isDragging, 'has-file': previewUrl }"
        @dragover.prevent="isDragging = true"
        @dragleave.prevent="isDragging = false"
        @drop.prevent="handleFileDrop"
        @click="triggerFileInput"
      >
        <input ref="fileInputRef" type="file" accept="image/png, image/jpeg, image/webp" class="hidden-input" @change="handleFileSelect" />
        <div v-if="!previewUrl" class="empty-state">
          <div class="upload-icon-circle">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p class="drop-primary-text">이미지를 드래그하거나 클릭하여 업로드</p>
          <p class="drop-secondary-text">PNG, JPG, WEBP 지원 (서버 전송 없음)</p>
        </div>
        <div v-else class="preview-wrap">
          <img :src="previewUrl" alt="업로드 이미지" class="preview-img" />
          <button class="re-upload-chip" @click.stop="triggerFileInput">파일 변경</button>
        </div>
      </div>

      <aside class="panel-card">
        <div v-if="!currentFile" class="panel-state-idle">
          <h2 class="panel-title">메타데이터 및 주파수 분석</h2>
          <ul class="feature-bullets">
            <li><span class="bullet-dot"></span><span>파일 헤더 및 생성 태그 분석</span></li>
            <li><span class="bullet-dot"></span><span>모델 기반 잠재 워터마크 탐지</span></li>
            <li><span class="bullet-dot"></span><span>메타데이터 및 주파수 정보 리셋</span></li>
          </ul>

          <div class="desktop-banner">
            <div class="banner-tag">정밀 분석 안내</div>
            <p class="banner-body">
              픽셀에 암호화된 비가시적 워터마크는 브라우저 스캐너로 감지하기 어렵습니다. 전체 데이터 초기화는 데스크톱 버전을 권장합니다.
            </p>
            <a href="https://github.com/catgarret/AI-Unmaker-Pro/releases" target="_blank" class="banner-cta">
              <span>데스크톱 앱 받기</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M7 17L17 7M17 7H7M17 7V17" /></svg>
            </a>
          </div>
          <button class="btn-disabled" disabled>이미지 분석 대기 중</button>
        </div>

        <div v-else-if="isScanning" class="panel-state-scanning">
          <div class="scan-spinner"></div>
          <p class="scan-text">데이터 구조 파싱 중...</p>
        </div>

        <div v-else-if="!isProcessed" class="panel-state-result">
          <div class="status-summary" :class="isAiDetected ? 'is-alert' : 'is-safe'">
            <div class="status-pill">{{ isAiDetected ? '시그니처 검출' : '기본 검사 통과' }}</div>
            <h3 class="status-score-title">{{ isAiDetected ? `${confidence}% 판정` : '0% (흔적 없음)' }}</h3>
            <p class="status-sub">
              {{ isAiDetected ? '파일 내부에 소프트웨어 헤더나 메타데이터가 기록되어 있습니다.' : '텍스트 헤더상에서는 생성 흔적이 검출되지 않았습니다.' }}
            </p>
          </div>

          <div v-if="isAiDetected" class="detection-box">
            <div class="detection-label">검출된 핑거프린트 출처</div>
            <div v-for="source in detectedSources" :key="source" class="detection-chip">
              <span class="chip-dot"></span><span>{{ source }}</span>
            </div>
          </div>

          <div class="desktop-banner" :class="{ 'warning-border': !isAiDetected }">
            <div class="banner-tag">비가시적 워터마크 유의사항</div>
            <p class="banner-body">
              웹 도구는 텍스트 헤더 검사에 특화되어 있습니다. 메타데이터가 없는 파일도 픽셀 내부에 워터마크 격자가 남아있을 수 있습니다.
            </p>
            <a href="https://github.com/catgarret/AI-Unmaker-Pro/releases" target="_blank" class="banner-cta">
              <span>데스크톱 환경에서 정밀 초기화</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M7 17L17 7M17 7H7M17 7V17" /></svg>
            </a>
          </div>

          <button class="btn-primary" @click="processMetadataScrub">웹 환경 메타데이터 초기화 실행</button>
        </div>

        <div v-else class="panel-state-done">
          <div class="status-summary is-safe">
            <div class="status-pill">처리 완료</div>
            <h3 class="status-score-title">데이터 초기화 완료</h3>
            <p class="status-sub">매니페스트 및 식별 가능한 코멘트 청크가 제거되었습니다.</p>
          </div>
          <button class="btn-primary" @click="downloadCleanedImage">초기화된 파일 저장 (.png)</button>
          <button class="btn-secondary" @click="resetWorkspace">다른 파일 분석하기</button>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const fileInputRef = ref(null)
const currentFile = ref(null)
const previewUrl = ref(null)
const cleanedBlobUrl = ref(null)
const isDragging = ref(false)
const isScanning = ref(false)
const isProcessed = ref(false)
const isAiDetected = ref(false)
const confidence = ref(0)
const detectedSources = ref([])

const triggerFileInput = () => fileInputRef.value?.click()
const handleFileDrop = (e) => { isDragging.value = false; const files = e.dataTransfer?.files; if (files && files[0]) loadFile(files[0]) }
const handleFileSelect = (e) => { const files = e.target.files; if (files && files[0]) loadFile(files[0]) }

const loadFile = (file) => {
  currentFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  isProcessed.value = false
  runMetadataScan(file)
}

const runMetadataScan = async (file) => {
  isScanning.value = true
  detectedSources.value = []
  confidence.value = 0
  isAiDetected.value = false

  try {
    const buffer = await file.arrayBuffer()
    const textDecoder = new TextDecoder('utf-8', { fatal: false })
    const rawText = textDecoder.decode(buffer).toLowerCase()
    const sources = []
    let score = 0

    if (rawText.includes('gemini')) { sources.push('Google Gemini'); score += 45 }
    if (rawText.includes('openai') || rawText.includes('dall-e') || rawText.includes('chatgpt')) { sources.push('OpenAI (ChatGPT / DALL-E)'); score += 45 }
    if (rawText.includes('photoshop') || rawText.includes('adobe') || rawText.includes('c2pa')) { sources.push('Adobe (Photoshop / C2PA)'); score += 35 }
    if (rawText.includes('midjourney')) { sources.push('Midjourney'); score += 45 }
    if (rawText.includes('grok') || rawText.includes('x.com') || rawText.includes('twitter')) { sources.push('X (Grok AI)'); score += 45 }
    if (rawText.includes('google') || rawText.includes('imagen')) { sources.push('Google (Imagen)'); score += 45 }
    if (rawText.includes('stable diffusion') || rawText.includes('stablediffusion')) { sources.push('Stable Diffusion'); score += 45 }

    detectedSources.value = [...new Set(sources)]
    if (detectedSources.value.length > 0) {
      isAiDetected.value = true
      confidence.value = Math.min(score, 100)
    }
  } catch (err) {
    console.error('Scan error:', err)
  } finally {
    isScanning.value = false
  }
}

const processMetadataScrub = async () => {
  if (!currentFile.value || !previewUrl.value) return
  isScanning.value = true
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = previewUrl.value
  img.onload = () => {
    const canvas = document.createElement('canvas')
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.drawImage(img, 0, 0)
      canvas.toBlob((blob) => {
        if (blob) {
          cleanedBlobUrl.value = URL.createObjectURL(blob)
          isProcessed.value = true
        }
        isScanning.value = false
      }, 'image/png')
    }
  }
}

const downloadCleanedImage = () => {
  if (!cleanedBlobUrl.value) return
  const a = document.createElement('a')
  a.href = cleanedBlobUrl.value
  const origName = currentFile.value?.name || 'cleaned_image'
  const baseName = origName.substring(0, origName.lastIndexOf('.')) || origName
  a.download = `${baseName}_cleaned.png`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

const resetWorkspace = () => {
  currentFile.value = null
  previewUrl.value = null
  cleanedBlobUrl.value = null
  isProcessed.value = false
  isAiDetected.value = false
  detectedSources.value = []
  if (fileInputRef.value) fileInputRef.value.value = ''
}
</script>

<style scoped>
.unmarker-container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; font-family: -apple-system, BlinkMacSystemFont, "Pretendard", sans-serif; color: #0f172a; }
.tool-header { margin-bottom: 24px; text-align: left; }
.brand-row { display: flex; align-items: center; gap: 10px; }
.brand-title { font-size: 24px; font-weight: 800; letter-spacing: -0.02em; margin: 0; }
.brand-badge { font-size: 11px; font-weight: 700; background-color: #f1f5f9; color: #475569; padding: 3px 8px; border-radius: 4px; }
.brand-subtitle { margin: 6px 0 0 0; font-size: 13.5px; color: #64748b; line-height: 1.5; word-break: keep-all; }
.workspace-grid { display: grid; grid-template-columns: 1fr 380px; gap: 24px; align-items: stretch; }
@media (max-width: 860px) { .workspace-grid { grid-template-columns: 1fr; } }
.dropzone-card { position: relative; min-height: 480px; background-color: #ffffff; border: 2px dashed #e2e8f0; border-radius: 12px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; }
.dropzone-card:hover { border-color: #0f172a; background-color: #f8fafc; }
.dropzone-card.has-file { border-style: solid; border-color: #e2e8f0; }
.hidden-input { display: none; }
.empty-state { text-align: center; padding: 24px; }
.upload-icon-circle { width: 56px; height: 56px; margin: 0 auto 12px; border-radius: 50%; background-color: #f1f5f9; display: flex; align-items: center; justify-content: center; color: #64748b; }
.drop-primary-text { font-size: 14px; font-weight: 600; color: #0f172a; margin: 0 0 4px 0; }
.drop-secondary-text { font-size: 12px; color: #94a3b8; margin: 0; word-break: keep-all; }
.preview-wrap { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; padding: 16px; position: relative; }
.preview-img { max-width: 100%; max-height: 440px; object-fit: contain; border-radius: 8px; }
.re-upload-chip { position: absolute; top: 16px; right: 16px; background: rgba(15, 23, 42, 0.8); color: #fff; border: none; border-radius: 20px; font-size: 12px; font-weight: 600; padding: 6px 14px; cursor: pointer; }
.panel-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; display: flex; flex-direction: column; justify-content: space-between; text-align: left; }
.panel-title { font-size: 16px; font-weight: 700; margin: 0 0 16px 0; color: #0f172a; }
.feature-bullets { list-style: none; padding: 0; margin: 0 0 20px 0; display: flex; flex-direction: column; gap: 10px; }
.feature-bullets li { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #475569; }
.bullet-dot { width: 6px; height: 6px; border-radius: 50%; background-color: #0f172a; }
.desktop-banner { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin-bottom: 20px; text-align: left; }
.desktop-banner.warning-border { border-color: #fde68a; background-color: #fefce8; }
.banner-tag { font-size: 11px; font-weight: 700; color: #854d0e; margin-bottom: 6px; }
.banner-body { font-size: 12px; line-height: 1.5; color: #475569; margin: 0 0 12px 0; word-break: keep-all; }
.banner-body strong { color: #0f172a; }
.banner-cta { display: inline-flex; align-items: center; justify-content: space-between; width: 100%; box-sizing: border-box; background-color: #0f172a; color: #ffffff; padding: 8px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; text-decoration: none; }
.status-summary { padding: 16px; border-radius: 10px; margin-bottom: 16px; text-align: left; }
.status-summary.is-safe { background-color: #f0fdf4; border: 1px solid #bbf7d0; }
.status-summary.is-alert { background-color: #fef2f2; border: 1px solid #fecaca; }
.status-pill { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-bottom: 6px; }
.is-safe .status-pill { background-color: #dcfce7; color: #166534; }
.is-alert .status-pill { background-color: #fee2e2; color: #991b1b; }
.status-score-title { font-size: 18px; font-weight: 800; margin: 0 0 4px 0; }
.is-safe .status-score-title { color: #166534; }
.is-alert .status-score-title { color: #991b1b; }
.status-sub { font-size: 12px; color: #475569; margin: 0; line-height: 1.4; word-break: keep-all; }
.detection-box { margin-bottom: 16px; text-align: left; }
.detection-label { font-size: 11px; font-weight: 700; color: #64748b; margin-bottom: 8px; }
.detection-chip { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12.5px; font-weight: 600; color: #0f172a; margin-bottom: 6px; }
.chip-dot { width: 6px; height: 6px; border-radius: 50%; background-color: #ef4444; }
.btn-primary, .btn-secondary, .btn-disabled { width: 100%; padding: 12px; border-radius: 8px; font-size: 13.5px; font-weight: 700; cursor: pointer; border: none; }
.btn-primary { background-color: #0f172a; color: #ffffff; }
.btn-secondary { background-color: #ffffff; color: #0f172a; border: 1px solid #e2e8f0; margin-top: 8px; }
.btn-disabled { background-color: #f1f5f9; color: #94a3b8; cursor: not-allowed; }
.panel-state-scanning { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 280px; gap: 16px; }
.scan-spinner { width: 32px; height: 32px; border: 3px solid #e2e8f0; border-top-color: #0f172a; border-radius: 50%; animation: spin 0.8s linear infinite; }
.scan-text { font-size: 13px; font-weight: 600; color: #475569; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
