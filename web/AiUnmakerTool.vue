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
          <div class="panel-body-content">
            <h2 class="panel-title">AI 메타데이터 및 태그 스캔</h2>
            <ul class="feature-bullets">
              <li><span class="bullet-dot"></span><span>주요 AI 플랫폼 핑거프린트 스캔</span></li>
              <li><span class="bullet-dot"></span><span>이미지 내 식별용 텍스트 청크 파싱</span></li>
              <li><span class="bullet-dot"></span><span>AI 메타데이터 탐지 및 제거</span></li>
            </ul>

            <div class="desktop-banner">
              <div class="banner-tag">정밀 분석 안내</div>
              <p class="banner-body">
                픽셀에 암호화된 비가시적 워터마크는 브라우저 스캐너로 감지하기 어렵습니다. 전체 데이터 초기화는 데스크톱 버전을 권장합니다.
              </p>
              <a href="?v=files" class="banner-cta">
              <span>설치형 파일(정밀 클리닝) 받기</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M7 17L17 7M17 7H7M17 7V17" /></svg>
              </a>
            </div>
          </div>
          <button class="btn-disabled" disabled>이미지 분석 대기 중</button>
        </div>

        <div v-else-if="isScanning" class="panel-state-scanning">
          <div class="scan-spinner"></div>
          <p class="scan-text">플랫폼 및 데이터 구조 파싱 중...</p>
        </div>

        <div v-else-if="!isProcessed" class="panel-state-result">
          <div class="panel-body-content">
            <div class="status-summary" :class="isAiDetected ? 'is-alert' : 'is-safe'">
              <div class="status-pill">{{ isAiDetected ? '시그니처 검출' : '기본 검사 통과' }}</div>
              <h3 class="status-score-title">{{ isAiDetected ? `${confidence}% 판정` : '0% (흔적 없음)' }}</h3>
              <p class="status-sub">
                {{ isAiDetected ? '파일 내부에서 AI 생성 플랫폼 흔적이 검출되었습니다.' : '텍스트 헤더상에서는 생성 흔적이 검출되지 않았습니다.' }}
              </p>
            </div>

            <!-- 1. 추정 생성 플랫폼 / 소프트웨어 출처 -->
            <div v-if="genPlatforms.length > 0" class="detection-box">
              <div class="detection-label">추정 생성 플랫폼 (소프트웨어)</div>
              <div v-for="plat in genPlatforms" :key="plat" class="detection-chip">
                <span class="chip-dot"></span><span>{{ plat }}</span>
              </div>
            </div>

            <!-- 2. 오픈소스 탐지 도구별 평가 스코어 -->
            <div v-if="Object.keys(detectorScores).length > 0" class="detection-box">
              <div class="detection-label">오픈소스 탐지 도구별 평가</div>
              <div v-for="(score, name) in detectorScores" :key="name" class="detection-chip-row">
                <span class="chip-name">• {{ name }}</span>
                <span class="chip-status" :class="score > 50 ? 'is-detected' : 'is-passed'">
                  {{ score > 50 ? `감지됨 (${score}%)` : `통과 (${score}%)` }}
                </span>
              </div>
            </div>

            <!-- 3. 검출 세부 항목 -->
            <div v-if="detectedSources.length > 0" class="detection-box">
              <div class="detection-label">검출 세부 항목</div>
              <div v-for="source in detectedSources" :key="source" class="detection-chip">
                <span class="chip-dot"></span><span>{{ source }}</span>
              </div>
            </div>

            <div class="desktop-banner">
              <div class="banner-tag">비가시적 워터마크 유의사항</div>
              <p class="banner-body">
                웹 도구는 텍스트 헤더 검사에 특화되어 있습니다. 메타데이터가 없는 파일도 픽셀 내부에 워터마크 격자가 남아있을 수 있습니다.
              </p>
              <a href="r?v=files" class="banner-cta">
                <span>데스크톱 환경에서 정밀 초기화</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M7 17L17 7M17 7H7M17 7V17" /></svg>
              </a>
            </div>
          </div>

          <button class="btn-primary" @click="processMetadataScrub">웹 환경 메타데이터 초기화 실행</button>
        </div>

        <div v-else class="panel-state-done">
          <div class="panel-body-content">
            <div class="status-summary is-safe">
              <div class="status-pill">처리 완료</div>
              <h3 class="status-score-title">데이터 초기화 완료</h3>
              <p class="status-sub">매니페스트 및 식별 가능한 코멘트 청크가 제거되었습니다.</p>
            </div>
          </div>
          <div class="panel-actions-bottom">
            <button class="btn-primary" @click="downloadCleanedImage">초기화된 파일 저장 (.png)</button>
            <button class="btn-secondary" @click="resetWorkspace">다른 파일 분석하기</button>
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { ref, onBeforeUnmount } from 'vue'

// Local, bounded container inspection. No pixel watermark decoder, model, or network call.
const LIMITS = Object.freeze({ metadata: 8 * 1024 * 1024, totalRead: 128 * 1024 * 1024,
  text: 4 * 1024 * 1024, entries: 4096, containers: 16384, depth: 20, canvasPixels: 32000000 })
const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10]
const C2PA_UUID = '6332706100110010800000aa00389b71'
const KEEP_PNG = new Set(['IHDR', 'PLTE', 'tRNS', 'IDAT', 'IEND', 'cHRM', 'gAMA', 'sRGB',
  'sBIT', 'iCCP', 'cICP', 'mDCV', 'cLLI', 'bKGD', 'acTL', 'fcTL', 'fdAT'])
const utf8 = new TextDecoder('utf-8', { fatal: true })
const latin1 = (bytes) => { let s = ''; for (let i = 0; i < bytes.length; i += 8192) s += String.fromCharCode(...bytes.subarray(i, i + 8192)); return s }
const ascii = (bytes, start = 0, end = bytes.length) => latin1(bytes.subarray(start, end))
const u16 = (b, p, little = false) => new DataView(b.buffer, b.byteOffset, b.byteLength).getUint16(p, little)
const u32 = (b, p, little = false) => new DataView(b.buffer, b.byteOffset, b.byteLength).getUint32(p, little)
const hex = (b) => Array.from(b, v => v.toString(16).padStart(2, '0')).join('')
const fail = (code) => { throw new Error(code) }
const checkAbort = (signal) => { if (signal?.aborted) { const e = new Error('scan_cancelled'); e.name = 'AbortError'; throw e } }
const requireRange = (bytes, offset, size) => { if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size < 0 || offset + size > bytes.length) fail('invalid_metadata_bounds') }
const concatBytes = (parts) => { const size = parts.reduce((n, b) => n + b.length, 0); if (size > LIMITS.metadata) fail('metadata_size_limit'); const out = new Uint8Array(size); let p = 0; for (const b of parts) { out.set(b, p); p += b.length } return out }
const crcTable = Array.from({ length: 256 }, (_, n) => { for (let k = 0; k < 8; k++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1; return n >>> 0 })
const crcUpdate = (crc, bytes) => { for (const b of bytes) crc = crcTable[(crc ^ b) & 255] ^ (crc >>> 8); return crc >>> 0 }

function blobReader(file, signal) {
  if (!file || typeof file.slice !== 'function' || !Number.isSafeInteger(file.size) || file.size < 12) fail('invalid_image_file')
  let total = 0
  let page = new Uint8Array(0)
  let pageStart = 0
  return async (offset, size) => {
    checkAbort(signal)
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size < 0 || offset + size > file.size || size > LIMITS.metadata) fail('invalid_image_bounds')
    if (size <= 4096 && offset >= pageStart && offset + size <= pageStart + page.length) return page.subarray(offset - pageStart, offset - pageStart + size)
    const count = size <= 4096 ? Math.min(65536, file.size - offset) : size
    total += count
    if (total > LIMITS.totalRead) fail('image_read_limit')
    const bytes = new Uint8Array(await file.slice(offset, offset + count).arrayBuffer())
    checkAbort(signal)
    if (bytes.length !== count) fail('image_read_failed')
    if (size <= 4096) { page = bytes; pageStart = offset }
    return bytes.subarray(0, size)
  }
}

async function inflateText(bytes, signal) {
  if (typeof DecompressionStream !== 'function') fail('deflate_not_supported')
  checkAbort(signal)
  const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate')).getReader()
  const parts = []
  let count = 0
  try {
    while (true) {
      checkAbort(signal)
      const { value, done } = await reader.read()
      if (done) break
      count += value.length
      if (count > LIMITS.text) fail('metadata_inflate_limit')
      parts.push(value)
    }
  } catch (e) { await reader.cancel().catch(() => {}); throw e }
  finally { reader.releaseLock() }
  return concatBytes(parts)
}

function newReport() {
  return { status: 'complete', format: '', width: 0, height: 0, bitDepth: 0, orientation: 1,
    metadataTextBytes: 0, animated: false,
    metadata: [], evidence: [], platforms: [], aiDeclared: false, chunks: [], warnings: [],
    c2pa: { present: false, manifestStoreRecognized: false, signatureVerified: false, labels: [] },
    capabilities: { synthidPixelDetection: false, c2paSignatureValidation: false } }
}
function addWarning(report, code) { report.status = 'incomplete'; if (!report.warnings.includes(code)) report.warnings.push(code) }
function addEvidence(report, kind, label, location, field = '') {
  if (!report.evidence.some(e => e.kind === kind && e.label === label && e.location === location && e.field === field)) report.evidence.push({ kind, label, location, field })
}
const PLATFORM_RULES = [
  [/\bmidjourney\b/i, 'Midjourney'], [/\b(?:openai|chatgpt|dall[-· ]?e)\b/i, 'OpenAI (ChatGPT / DALL-E)'],
  [/\b(?:gemini|imagen)\b/i, 'Google (Gemini / Imagen)'],
  [/\b(?:stable[ _-]?diffusion|comfyui|automatic1111)\b/i, 'Stable Diffusion'],
  [/\bfirefly\b/i, 'Adobe (Photoshop / C2PA)'], [/\bgrok\b/i, 'X (Grok AI)'],
  [/\bbing image creator\b/i, 'Microsoft Bing Image Creator']
]
const SOURCE_FIELD = /(?:^|[.:/])(?:software|creatortool|generator|claim_generator|claim_generator_info(?:\[\d+\])?\.name|softwareagent(?:\.name)?)$/i
function addMetadata(report, kind, field, value, location) {
  if (report.metadata.length >= LIMITS.entries) fail('metadata_entry_limit')
  const text = String(value).replace(/\0+$/g, '')
  report.metadataTextBytes += text.length + field.length
  if (report.metadataTextBytes > LIMITS.text) fail('metadata_text_limit')
  report.metadata.push({ kind, field, value: text, location })
  // These are literal metadata observations, never a SynthID pixel verdict.
  const content = `${field}\n${text}`
  if (/\bsynthid\b/i.test(content)) addEvidence(report, 'literal_metadata', 'SynthID 메타데이터', location, field)
  if (/\b(?:wiltodelta|remove-ai-watermarks)\b/i.test(content)) addEvidence(report, 'literal_metadata', 'wiltodelta 메타데이터', location, field)
  if (/\b(?:mertizci|noai(?:-watermark)?)\b/i.test(content)) addEvidence(report, 'literal_metadata', 'mertizci 메타데이터', location, field)
  // Only declared software/generator fields identify an app. Ordinary comments,
  // filenames, XML namespace URLs and compressed pixel bytes are not classifiers.
  if (SOURCE_FIELD.test(field)) {
    for (const [pattern, label] of PLATFORM_RULES) if (pattern.test(text) && !report.platforms.includes(label)) report.platforms.push(label)
  }
  if (kind === 'text' && field === 'parameters') {
    const settings = text.slice(text.lastIndexOf('\n') + 1)
    if (settings.length < 4096 && /^Steps:\s*\d+,/.test(settings) && /(?:^|, )Sampler: [^,]+/.test(settings) &&
      /(?:^|, )Seed: \d+/.test(settings) && /(?:^|, )Size: \d+x\d+/.test(settings) && !report.platforms.includes('Stable Diffusion')) report.platforms.push('Stable Diffusion')
  }
  if (kind === 'text' && ['prompt', 'workflow'].includes(field) && text.length < 1024 * 1024) {
    // Recognize the actual ComfyUI graph schema, not prose that mentions it.
    try {
      const graph = JSON.parse(text)
      const nodes = field === 'workflow' ? graph?.nodes : graph && typeof graph === 'object' && !Array.isArray(graph) ? Object.values(graph) : []
      if (Array.isArray(nodes) && nodes.length <= LIMITS.entries) {
        const types = nodes.filter(n => n && typeof n === 'object').map(n => field === 'workflow' ? n.type : n.class_type)
        if (types.some(t => ['KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced'].includes(t)) &&
          types.some(t => ['CheckpointLoaderSimple', 'UNETLoader', 'DiffusersLoader'].includes(t)) && !report.platforms.includes('ComfyUI')) report.platforms.push('ComfyUI')
      }
    } catch { /* A prompt can legitimately be plain text. */ }
  }
  if (/digitalSourceType$/i.test(field) && /\/(?:trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia)$/.test(text)) report.aiDeclared = true
}

function readXmp(bytes, report, location) {
  if (typeof DOMParser !== 'function') fail('xml_parser_not_supported')
  const text = utf8.decode(bytes)
  if (/<!DOCTYPE|<!ENTITY/i.test(text)) fail('xmp_entities_not_allowed')
  const doc = new DOMParser().parseFromString(text, 'application/xml')
  if (doc.getElementsByTagName('parsererror').length || !doc.documentElement) fail('invalid_xmp')
  const nodes = doc.getElementsByTagName('*')
  if (nodes.length > LIMITS.entries) fail('xmp_node_limit')
  const safeNs = new Set(['http://ns.adobe.com/xap/1.0/', 'http://ns.adobe.com/tiff/1.0/',
    'http://ns.adobe.com/exif/1.0/', 'http://purl.org/dc/elements/1.1/',
    'http://ns.adobe.com/xap/1.0/mm/', 'http://ns.adobe.com/photoshop/1.0/',
    'http://ns.useplus.org/ldf/xmp/1.0/', 'http://iptc.org/std/Iptc4xmpExt/2008-02-29/'])
  addEvidence(report, 'metadata_container', 'XMP', location)
  for (const node of Array.from(nodes)) {
    if (node.attributes) for (const a of Array.from(node.attributes)) {
      if (a.name === 'xmlns' || a.prefix === 'xmlns' || a.namespaceURI === 'http://www.w3.org/1999/02/22-rdf-syntax-ns#') continue
      const field = safeNs.has(a.namespaceURI) ? a.localName : a.name
      addMetadata(report, 'xmp', field, a.value, location)
    }
    const children = Array.from(node.childNodes || [])
    if (!children.some(c => c.nodeType === 1) && children.some(c => c.nodeType === 3 || c.nodeType === 4)) {
      const field = safeNs.has(node.namespaceURI) ? node.localName : node.nodeName
      addMetadata(report, 'xmp', field, node.textContent || '', location)
    }
  }
}

function readTiff(input, report, location) {
  const b = ascii(input, 0, 6) === 'Exif\0\0' ? input.subarray(6) : input
  requireRange(b, 0, 8)
  const marker = ascii(b, 0, 2)
  if (marker !== 'II' && marker !== 'MM') fail('invalid_tiff_byte_order')
  const little = marker === 'II'
  if (u16(b, 2, little) !== 42) fail('invalid_tiff_header')
  const visited = new Set()
  const sizes = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8 }
  const tags = { 0x010e: 'ImageDescription', 0x010f: 'Make', 0x0110: 'Model', 0x0131: 'Software', 0x013b: 'Artist', 0x8298: 'Copyright', 0x9286: 'UserComment', 0x9c9b: 'XPTitle', 0x9c9c: 'XPComment', 0x9c9d: 'XPAuthor', 0x9c9e: 'XPKeywords', 0x9c9f: 'XPSubject' }
  function ifd(offset, depth) {
    if (!offset) return
    if (depth > LIMITS.depth || visited.has(offset)) fail('invalid_tiff_ifd_cycle')
    visited.add(offset); requireRange(b, offset, 2)
    const count = u16(b, offset, little)
    if (count > LIMITS.entries) fail('tiff_entry_limit')
    requireRange(b, offset + 2, count * 12 + 4)
    for (let i = 0; i < count; i++) {
      const p = offset + 2 + i * 12, tag = u16(b, p, little), type = u16(b, p + 2, little), n = u32(b, p + 4, little)
      if (!sizes[type]) continue
      const size = n * sizes[type]
      if (size > LIMITS.metadata) fail('tiff_value_limit')
      const at = size <= 4 ? p + 8 : u32(b, p + 8, little)
      requireRange(b, at, size)
      if (tag === 0x0112 && depth === 0) {
        if (type !== 3 || n !== 1) fail('invalid_exif_orientation')
        const orientation = u16(b, at, little)
        if (orientation < 1 || orientation > 8) fail('invalid_exif_orientation')
        const previous = report.metadata.find(m => m.kind === 'exif' && m.field === 'Orientation')
        if (previous && Number(previous.value) !== orientation) fail('conflicting_exif_orientation')
        report.orientation = orientation
        addMetadata(report, 'exif', 'Orientation', orientation, location)
      }
      if ([0x8769, 0x8825, 0xa005].includes(tag) && type === 4 && n === 1) ifd(u32(b, at, little), depth + 1)
      if (tag === 0x014a && type === 4) for (let j = 0; j < n; j++) ifd(u32(b, at + j * 4, little), depth + 1)
      if (tag === 0x02bc) readXmp(b.subarray(at, at + size), report, `${location}:XMP`)
      if (!tags[tag]) continue
      let data = b.subarray(at, at + size), value = ''
      if (type === 2) value = latin1(data)
      else if (tag >= 0x9c9b && tag <= 0x9c9f) value = new TextDecoder('utf-16le', { fatal: true }).decode(data)
      else if (tag === 0x9286 && data.length >= 8) {
        const encoding = ascii(data, 0, 8); data = data.subarray(8)
        if (encoding === 'ASCII\0\0\0') value = latin1(data)
        else if (encoding === 'UNICODE\0') value = new TextDecoder(little ? 'utf-16le' : 'utf-16be', { fatal: true }).decode(data)
      }
      if (value) addMetadata(report, 'exif', tags[tag], value, location)
    }
    ifd(u32(b, offset + 2 + count * 12, little), depth + 1)
  }
  ifd(u32(b, 4, little), 0)
  addEvidence(report, 'metadata_container', 'EXIF', location)
}

function decodeCbor(bytes) {
  let p = 0, count = 0
  const take = n => { requireRange(bytes, p, n); const s = bytes.subarray(p, p + n); p += n; return s }
  const number = ai => {
    if (ai < 24) return ai
    if (ai === 24) return take(1)[0]
    if (ai === 25) return u16(take(2), 0)
    if (ai === 26) return u32(take(4), 0)
    if (ai === 27) { const b = take(8), n = u32(b, 0) * 4294967296 + u32(b, 4); if (!Number.isSafeInteger(n)) fail('cbor_integer_limit'); return n }
    fail('unsupported_cbor_length')
  }
  function value(depth = 0) {
    if (++count > LIMITS.entries || depth > LIMITS.depth) fail('cbor_structure_limit')
    const head = take(1)[0], major = head >> 5, ai = head & 31
    if (major === 7) {
      if (ai === 20) return false
      if (ai === 21) return true
      if (ai === 22 || ai === 23) return null
      if (ai === 25) { take(2); return null }
      if (ai === 26) { take(4); return null }
      if (ai === 27) { take(8); return null }
      fail('unsupported_cbor_simple')
    }
    if (ai === 31) {
      if (major !== 4 && major !== 5) fail('unsupported_cbor_indefinite')
      const out = major === 4 ? [] : Object.create(null)
      while (true) {
        requireRange(bytes, p, 1)
        if (bytes[p] === 255) { p++; return out }
        if (major === 4) out.push(value(depth + 1))
        else { const key = value(depth + 1); if (typeof key !== 'string' && typeof key !== 'number') fail('invalid_cbor_key'); if (Object.hasOwn(out, key)) fail('duplicate_cbor_key'); out[key] = value(depth + 1) }
      }
    }
    const n = number(ai)
    if (major === 0) return n
    if (major === 1) return -1 - n
    if (major === 2) { take(n); return null }
    if (major === 3) return utf8.decode(take(n))
    if (major === 6) return value(depth + 1)
    if (n > LIMITS.entries) fail('cbor_container_limit')
    if (major === 4) return Array.from({ length: n }, () => value(depth + 1))
    if (major === 5) {
      const out = Object.create(null)
      for (let i = 0; i < n; i++) { const key = value(depth + 1); if (typeof key !== 'string' && typeof key !== 'number') fail('invalid_cbor_key'); if (Object.hasOwn(out, key)) fail('duplicate_cbor_key'); out[key] = value(depth + 1) }
      return out
    }
    fail('invalid_cbor_major')
  }
  const result = value()
  if (p !== bytes.length) fail('cbor_trailing_bytes')
  return result
}
function collectStructured(value, report, location, path = '', depth = 0) {
  if (depth > LIMITS.depth) fail('metadata_structure_limit')
  if (typeof value === 'string') addMetadata(report, 'c2pa', path, value, location)
  else if (Array.isArray(value)) value.forEach((item, i) => collectStructured(item, report, location, `${path}[${i}]`, depth + 1))
  else if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) collectStructured(item, report, location, path ? `${path}.${key}` : key, depth + 1)
}
function readJumbf(bytes, report, location) {
  let boxes = 0
  function boxAt(p, end) {
    requireRange(bytes, p, 8)
    let size = u32(bytes, p), header = 8
    if (size === 1) { requireRange(bytes, p, 16); size = u32(bytes, p + 8) * 4294967296 + u32(bytes, p + 12); header = 16 }
    if (size === 0) size = end - p
    if (!Number.isSafeInteger(size) || size < header || p + size > end) fail('invalid_jumbf_box')
    if (++boxes > LIMITS.entries) fail('jumbf_box_limit')
    return { type: ascii(bytes, p + 4, p + 8), payload: p + header, end: p + size }
  }
  function descriptor(box) {
    if (box.type !== 'jumd' || box.end - box.payload < 17) fail('invalid_jumbf_description')
    const p = box.payload, flags = bytes[p + 16]
    let at = p + 17, label = ''
    if (flags & 2) { const zero = bytes.indexOf(0, at); if (zero < 0 || zero >= box.end) fail('invalid_jumbf_label'); label = utf8.decode(bytes.subarray(at, zero)); at = zero + 1 }
    if (flags & 4) at += 4
    if (flags & 8) at += 32
    if (at > box.end) fail('invalid_jumbf_optional_fields')
    return { uuid: hex(bytes.subarray(p, p + 16)), label }
  }
  function walk(start, end, depth, inStore = false) {
    if (depth > LIMITS.depth) fail('jumbf_depth_limit')
    let p = start
    while (p < end) {
      const box = boxAt(p, end)
      if (box.type === 'jumb') {
        const first = boxAt(box.payload, box.end), desc = descriptor(first)
        const root = desc.uuid === C2PA_UUID && desc.label === 'c2pa'
        if (root) {
          report.c2pa.present = true; report.c2pa.manifestStoreRecognized = true
          addEvidence(report, 'c2pa_manifest', 'C2PA 매니페스트', location)
        }
        if (inStore || root) {
          if (report.c2pa.labels.length < LIMITS.entries && !report.c2pa.labels.includes(desc.label)) report.c2pa.labels.push(desc.label)
          let childAt = first.end
          while (childAt < box.end) {
            const child = boxAt(childAt, box.end)
            // Claim/actions only. Thumbnails, arbitrary attachments and signature
            // byte strings never become source or watermark text evidence.
            if (/^c2pa\.(?:claim|actions)(?:\.|$)/.test(desc.label) && (child.type === 'cbor' || child.type === 'json')) {
              const payload = bytes.subarray(child.payload, child.end)
              const obj = child.type === 'cbor' ? decodeCbor(payload) : JSON.parse(utf8.decode(payload))
              collectStructured(obj, report, `${location}:${desc.label}`)
            }
            childAt = child.end
          }
        }
        walk(first.end, box.end, depth + 1, inStore || root)
      }
      p = box.end
    }
  }
  walk(0, bytes.length, 0)
}

async function inspectImageMetadata(file, { signal } = {}) {
  const report = newReport(), read = blobReader(file, signal), head = await read(0, 12)
  const safely = async (fn) => { try { await fn() } catch (e) { if (e.name === 'AbortError') throw e; addWarning(report, e.message || 'invalid_metadata') } }
  if (PNG_SIGNATURE.every((v, i) => head[i] === v)) {
    report.format = 'PNG'
    let p = 8, count = 0, idat = false, ended = false
    while (p < file.size) {
      if (++count > LIMITS.containers) fail('png_chunk_limit')
      const h = await read(p, 8), size = u32(h, 0), kind = ascii(h, 4, 8), end = p + size + 12
      if (!/^[A-Za-z]{4}$/.test(kind) || end > file.size || size > 0x7fffffff) fail('invalid_png_chunk')
      if (count === 1 && (kind !== 'IHDR' || size !== 13)) fail('invalid_png_header')
      if (kind === 'IHDR' && count !== 1) fail('duplicate_png_header')
      if (kind[0] === kind[0].toUpperCase() && !['IHDR', 'PLTE', 'IDAT', 'IEND'].includes(kind)) fail('unknown_png_critical_chunk')
      report.chunks.push({ type: kind, offset: p, size, end })
      if (kind === 'IDAT') idat = true
      if (kind === 'IHDR' || kind === 'acTL' || kind === 'fcTL' || ['tEXt', 'zTXt', 'iTXt', 'eXIf', 'caBX'].includes(kind)) await safely(async () => {
        if (size > LIMITS.metadata) fail('png_metadata_limit')
        const data = await read(p + 8, size), expected = u32(await read(p + 8 + size, 4), 0)
        if (((crcUpdate(crcUpdate(0xffffffff, h.subarray(4)), data) ^ 0xffffffff) >>> 0) !== expected) fail('png_metadata_crc_mismatch')
        const location = `PNG:${kind}@${p}`
        if (kind === 'IHDR') {
          report.width = u32(data, 0); report.height = u32(data, 4); report.bitDepth = data[8]
          const depths = { 0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16] }
          if (!report.width || !report.height || !depths[data[9]]?.includes(data[8]) || data[10] !== 0 || data[11] !== 0 || data[12] > 1) fail('invalid_png_ihdr')
        } else if (kind === 'acTL') { if (size !== 8 || u32(data, 0) === 0) fail('invalid_apng_control'); report.animated = true }
        else if (kind === 'fcTL') { if (size !== 26) fail('invalid_apng_frame') }
        else if (kind === 'eXIf') readTiff(data, report, location)
        else if (kind === 'caBX') { readJumbf(data, report, location); if (!report.c2pa.present) addEvidence(report, 'metadata_container', 'JUMBF', location) }
        else {
          const zero = data.indexOf(0)
          if (zero < 1 || zero > 79) fail('invalid_png_keyword')
          const field = latin1(data.subarray(0, zero))
          let textBytes = data.subarray(zero + 1), encoding = 'latin1'
          if (kind === 'zTXt') {
            if (!textBytes.length || textBytes[0] !== 0) fail('unsupported_png_compression')
            textBytes = await inflateText(textBytes.subarray(1), signal)
          } else if (kind === 'iTXt') {
            if (textBytes.length < 4 || textBytes[0] > 1 || textBytes[1] !== 0) fail('invalid_png_itxt')
            const compressed = textBytes[0], languageEnd = textBytes.indexOf(0, 2), translatedEnd = textBytes.indexOf(0, languageEnd + 1)
            if (languageEnd < 2 || translatedEnd < languageEnd + 1) fail('invalid_png_itxt_fields')
            textBytes = textBytes.subarray(translatedEnd + 1)
            if (compressed) textBytes = await inflateText(textBytes, signal)
            encoding = 'utf8'
          }
          if (field === 'XML:com.adobe.xmp') readXmp(textBytes, report, location)
          else addMetadata(report, 'text', field, encoding === 'utf8' ? utf8.decode(textBytes) : latin1(textBytes), location)
        }
      })
      if (kind === 'IEND') {
        if (size || !idat) fail('invalid_png_end')
        const expected = u32(await read(p + 8, 4), 0)
        if (((crcUpdate(0xffffffff, h.subarray(4)) ^ 0xffffffff) >>> 0) !== expected) fail('png_end_crc_mismatch')
        if (end !== file.size) addWarning(report, 'png_trailing_bytes')
        ended = true; break
      }
      p = end
    }
    if (!ended) fail('missing_png_end')
  } else if (head[0] === 255 && head[1] === 216) {
    report.format = 'JPEG'
    const fragments = new Map(), extendedXmp = new Map()
    let p = 2, count = 0, ended = false, entropy = false
    while (p < file.size) {
      if (++count > LIMITS.containers) fail('jpeg_segment_limit')
      if (entropy) {
        let found = false
        while (p < file.size && !found) {
          const data = await read(p, Math.min(65536, file.size - p))
          let i = 0
          for (; i < data.length; i++) if (data[i] === 255) {
            const at = p + i, next = (await read(at + 1, 1))[0]
            if (next === 0 || (next >= 208 && next <= 215)) { i++; continue }
            p = at; found = true; break
          }
          if (!found) p += data.length
        }
        entropy = false
        if (!found) fail('missing_jpeg_end')
      }
      if ((await read(p, 1))[0] !== 255) fail('invalid_jpeg_marker')
      do { p++ } while ((await read(p, 1))[0] === 255)
      const marker = (await read(p++, 1))[0]
      if (marker === 217) { ended = true; if (p !== file.size) addWarning(report, 'jpeg_trailing_bytes'); break }
      if (marker === 1 || (marker >= 208 && marker <= 215)) continue
      const size = u16(await read(p, 2), 0)
      if (size < 2 || p + size > file.size) fail('invalid_jpeg_segment')
      const start = p + 2, length = size - 2, location = `JPEG:APP${marker - 224}@${p - 2}`
      if (marker === 225 || marker === 235 || marker === 254) await safely(async () => {
        const data = await read(start, length)
        if (marker === 254) addMetadata(report, 'comment', 'Comment', latin1(data), location)
        else if (marker === 225 && ascii(data, 0, 6) === 'Exif\0\0') readTiff(data, report, location)
        else if (marker === 225 && ascii(data, 0, 29) === 'http://ns.adobe.com/xap/1.0/\0') readXmp(data.subarray(29), report, location)
        else if (marker === 225 && ascii(data, 0, 35) === 'http://ns.adobe.com/xmp/extension/\0') {
          requireRange(data, 35, 40)
          const guid = ascii(data, 35, 67), total = u32(data, 67), offset = u32(data, 71), part = data.slice(75)
          if (!/^[A-Fa-f0-9]{32}$/.test(guid) || total > LIMITS.metadata || offset + part.length > total) fail('invalid_extended_xmp')
          const group = extendedXmp.get(guid) || { total, parts: [], location }
          if (group.total !== total) fail('inconsistent_extended_xmp')
          group.parts.push({ offset, part }); extendedXmp.set(guid, group)
        } else if (marker === 235 && ascii(data, 0, 2) === 'JP') {
          requireRange(data, 0, 16)
          const en = u16(data, 2), sequence = u32(data, 4), total = u32(data, 8), type = ascii(data, 12, 16)
          if (!sequence || total < 8 || total > LIMITS.metadata || type !== 'jumb') fail('invalid_jpeg_jumbf_packet')
          const group = fragments.get(en) || { total, type, parts: new Map(), location }
          if (group.total !== total || group.type !== type || group.parts.has(sequence)) fail('inconsistent_jpeg_jumbf')
          group.parts.set(sequence, data.slice(16)); fragments.set(en, group)
        }
      })
      if ([192, 193, 194, 195, 197, 198, 199, 201, 202, 203, 205, 206, 207].includes(marker)) {
        const frame = await read(start, Math.min(length, 6)); requireRange(frame, 0, 5)
        report.height = u16(frame, 1); report.width = u16(frame, 3); report.bitDepth = frame[0]
      }
      p += size
      if (marker === 218) entropy = true
    }
    if (!ended) fail('missing_jpeg_end')
    for (const group of fragments.values()) await safely(async () => {
      const parts = [...group.parts].sort((a, b) => a[0] - b[0])
      if (parts.some(([sequence], i) => sequence !== i + 1)) fail('missing_jpeg_jumbf_fragment')
      const header = new Uint8Array(8); new DataView(header.buffer).setUint32(0, group.total); header.set([106, 117, 109, 98], 4)
      const bytes = concatBytes([header, ...parts.map(([, part]) => part)])
      if (bytes.length !== group.total) fail('incomplete_jpeg_jumbf')
      readJumbf(bytes, report, group.location)
    })
    for (const group of extendedXmp.values()) await safely(async () => {
      const parts = group.parts.sort((a, b) => a.offset - b.offset); let offset = 0
      for (const part of parts) { if (part.offset !== offset) fail('incomplete_extended_xmp'); offset += part.part.length }
      if (offset !== group.total) fail('incomplete_extended_xmp')
      readXmp(concatBytes(parts.map(part => part.part)), report, group.location)
    })
  } else if (ascii(head, 0, 4) === 'RIFF' && ascii(head, 8, 12) === 'WEBP') {
    report.format = 'WebP'
    const end = u32(head, 4, true) + 8
    if (end !== file.size) fail('invalid_webp_size')
    let p = 12, count = 0, image = false
    while (p < end) {
      if (++count > LIMITS.containers) fail('webp_chunk_limit')
      const h = await read(p, 8), kind = ascii(h, 0, 4), size = u32(h, 4, true), next = p + 8 + size + (size & 1)
      if (next > end) fail('invalid_webp_chunk')
      report.chunks.push({ type: kind, offset: p, size, end: next })
      if (['VP8 ', 'VP8L', 'ANMF'].includes(kind)) image = true
      if (kind === 'ANIM' || kind === 'ANMF') report.animated = true
      if (kind === 'VP8L') {
        const data = await read(p + 8, Math.min(size, 5)); requireRange(data, 0, 5)
        if (data[0] !== 47 || data[4] >> 5) fail('invalid_webp_lossless_header')
        report.width = 1 + data[1] + ((data[2] & 63) << 8)
        report.height = 1 + (data[2] >> 6) + (data[3] << 2) + ((data[4] & 15) << 10)
        report.bitDepth = 8
      } else if (kind === 'VP8 ') {
        const data = await read(p + 8, Math.min(size, 10)); requireRange(data, 0, 10)
        if ((data[0] & 1) || data[3] !== 157 || data[4] !== 1 || data[5] !== 42) fail('invalid_webp_lossy_header')
        report.width = u16(data, 6, true) & 16383; report.height = u16(data, 8, true) & 16383
        report.bitDepth = 8
      }
      if (['EXIF', 'XMP ', 'C2PA', 'VP8X'].includes(kind)) await safely(async () => {
        if (size > LIMITS.metadata) fail('webp_metadata_limit')
        const data = await read(p + 8, size), location = `WebP:${kind}@${p}`
        if (kind === 'EXIF') readTiff(data, report, location)
        if (kind === 'XMP ') readXmp(data, report, location)
        if (kind === 'C2PA') readJumbf(data, report, location)
        if (kind === 'VP8X') {
          if (size !== 10) fail('invalid_webp_vp8x')
          report.animated ||= !!(data[0] & 2)
          report.width = 1 + data[4] + (data[5] << 8) + (data[6] << 16)
          report.height = 1 + data[7] + (data[8] << 8) + (data[9] << 16)
        }
      })
      p = next
    }
    if (!image) fail('missing_webp_image')
  } else fail('unsupported_image_format')
  checkAbort(signal)
  return report
}

async function stripPngMetadata(file, { signal } = {}) {
  const report = await inspectImageMetadata(file, { signal })
  if (report.format !== 'PNG') fail('png_required')
  if (report.orientation !== 1) fail('orientation_requires_normalization')
  // Malformed provenance may still be removed, but structural/image errors are
  // never accepted. Every retained chunk must pass its actual CRC below.
  const read = blobReader(file, signal), parts = [file.slice(0, 8)]
  for (const chunk of report.chunks) {
    if (!KEEP_PNG.has(chunk.type)) continue
    let crc = 0xffffffff
    for (let p = chunk.offset + 4; p < chunk.end - 4; p += 65536) crc = crcUpdate(crc, await read(p, Math.min(65536, chunk.end - 4 - p)))
    if (((crc ^ 0xffffffff) >>> 0) !== u32(await read(chunk.end - 4, 4), 0)) fail('png_retained_crc_mismatch')
    parts.push(file.slice(chunk.offset, chunk.end))
  }
  const result = new Blob(parts, { type: 'image/png' })
  await verifyScrubbedPng(result, signal)
  return result
}
async function verifyScrubbedPng(blob, signal) {
  const report = await inspectImageMetadata(blob, { signal })
  if (report.status !== 'complete' || report.format !== 'PNG' || report.c2pa.present || report.metadata.length || report.evidence.length || report.chunks.some(c => !KEEP_PNG.has(c.type))) fail('metadata_removal_verification_failed')
  return report
}
function decodeLocalImage(file, signal) {
  checkAbort(signal)
  return new Promise((resolve, reject) => {
    const image = new Image(), url = URL.createObjectURL(file)
    const cleanup = () => { image.onload = null; image.onerror = null; signal?.removeEventListener('abort', abort); URL.revokeObjectURL(url) }
    const abort = () => { cleanup(); image.src = ''; const e = new Error('scan_cancelled'); e.name = 'AbortError'; reject(e) }
    image.onload = () => { cleanup(); resolve(image) }
    image.onerror = () => { cleanup(); reject(new Error('image_decode_failed')) }
    signal?.addEventListener('abort', abort, { once: true })
    image.src = url
  })
}
async function scrubMetadataToPng(file, { signal, sourceReport } = {}) {
  const report = sourceReport || await inspectImageMetadata(file, { signal })
  if (report.width && report.height && report.width * report.height > LIMITS.canvasPixels) fail('image_dimension_limit')
  if (report.format === 'PNG' && report.orientation === 1) {
    // Preserve the original encoded pixels, transparency and APNG frames.
    const image = await decodeLocalImage(file, signal)
    if (!image.naturalWidth || !image.naturalHeight) fail('invalid_image_dimensions')
    return stripPngMetadata(file, { signal })
  }
  if (report.animated) fail('animated_conversion_not_supported')
  if (report.bitDepth > 8) fail('high_depth_conversion_not_supported')
  const image = await decodeLocalImage(file, signal)
  if (!image.naturalWidth || !image.naturalHeight || image.naturalWidth * image.naturalHeight > LIMITS.canvasPixels) fail('image_dimension_limit')
  const canvas = document.createElement('canvas')
  try {
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) fail('canvas_2d_unavailable')
    context.drawImage(image, 0, 0)
    const output = await new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('png_encode_failed')), 'image/png'))
    checkAbort(signal)
    const result = await stripPngMetadata(output, { signal })
    await verifyScrubbedPng(result, signal)
    return result
  } finally { canvas.width = 0; canvas.height = 0 }
}

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
const genPlatforms = ref([])
const detectorScores = ref({})
const metadataScanReport = ref(null)
let generation = 0
let operation = null
let disposed = false
const revoke = (reference) => { if (reference.value) URL.revokeObjectURL(reference.value); reference.value = null }
const active = (file, token) => !disposed && token === generation && file === currentFile.value
const clearScan = () => { isAiDetected.value = false; confidence.value = 0; detectedSources.value = []; genPlatforms.value = []; detectorScores.value = {}; metadataScanReport.value = null }
const triggerFileInput = () => fileInputRef.value?.click()
const handleFileDrop = (e) => { isDragging.value = false; const file = e.dataTransfer?.files?.[0]; if (file) loadFile(file) }
const handleFileSelect = (e) => { const file = e.target.files?.[0]; if (file) loadFile(file); if (e.target) e.target.value = '' }

const resetWorkspace = () => {
  generation++; operation?.abort(); operation = null
  revoke(previewUrl); revoke(cleanedBlobUrl)
  currentFile.value = null; isProcessed.value = false; isScanning.value = false; isDragging.value = false
  clearScan()
  if (fileInputRef.value) fileInputRef.value.value = ''
}
const loadFile = (file) => {
  if (disposed || !file || typeof file.slice !== 'function') return
  resetWorkspace()
  const token = generation
  currentFile.value = file; previewUrl.value = URL.createObjectURL(file)
  operation = new AbortController()
  return runMetadataScan(file, token, operation.signal)
}
const runMetadataScan = async (file, token = generation, signal = operation?.signal) => {
  if (!active(file, token)) return
  isScanning.value = true; clearScan()
  try {
    const report = await inspectImageMetadata(file, { signal })
    if (!active(file, token)) return
    // An incomplete inspection must never fall into the existing 'passed' UI.
    if (report.status !== 'complete') fail(report.warnings.join(', ') || 'metadata_scan_incomplete')
    metadataScanReport.value = report
    genPlatforms.value = report.platforms
    detectedSources.value = [...new Set(report.evidence.map(e => e.label))]
    // No third-party tool was executed: do not create invented tool scores.
    detectorScores.value = {}
    isAiDetected.value = report.aiDeclared || report.platforms.length > 0 || report.evidence.some(e => e.kind === 'literal_metadata')
    // Compatibility with the unchanged template: binary observed-record state,
    // not an AI probability or a pixel-watermark confidence estimate.
    confidence.value = isAiDetected.value ? 100 : 0
  } catch (e) {
    if (active(file, token) && e.name !== 'AbortError') {
      resetWorkspace(); metadataScanReport.value = { status: 'error', error: e.message }
      console.error('Scan error:', e)
    }
  } finally { if (active(file, token)) isScanning.value = false }
}
const processMetadataScrub = async () => {
  if (!currentFile.value || !previewUrl.value || isScanning.value || disposed) return
  const file = currentFile.value, token = generation, sourceReport = metadataScanReport.value
  operation?.abort(); operation = new AbortController()
  const signal = operation.signal
  isScanning.value = true; isProcessed.value = false; revoke(cleanedBlobUrl)
  try {
    const blob = await scrubMetadataToPng(file, { signal, sourceReport })
    const verification = await verifyScrubbedPng(blob, signal)
    if (!active(file, token)) return
    cleanedBlobUrl.value = URL.createObjectURL(blob)
    metadataScanReport.value = { ...sourceReport, output: verification }
    isProcessed.value = true
  } catch (e) { if (active(file, token) && e.name !== 'AbortError') console.error('Scan error:', e) }
  finally { if (active(file, token)) isScanning.value = false }
}
const downloadCleanedImage = () => {
  if (!isProcessed.value || !cleanedBlobUrl.value) return
  const a = document.createElement('a')
  a.href = cleanedBlobUrl.value
  const origName = currentFile.value?.name || 'cleaned_image'
  const baseName = origName.substring(0, origName.lastIndexOf('.')) || origName
  a.download = `${baseName}_cleaned.png`
  document.body.appendChild(a); a.click(); document.body.removeChild(a)
}
onBeforeUnmount(() => { disposed = true; resetWorkspace() })
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

/* 패널 카드 및 내부 플렉스 확장 */
.panel-card {
  background-color: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  text-align: left;
}

/* 각 상태 컨테이너가 카드 전체 높이를 채우고 상하 분할되도록 설정 */
.panel-state-idle,
.panel-state-result,
.panel-state-done {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  flex: 1;
  height: 100%;
}

.panel-body-content {
  flex: 1;
}

.panel-title { font-size: 16px; font-weight: 700; margin: 0 0 16px 0; color: #0f172a; }
.feature-bullets { list-style: none; padding: 0; margin: 0 0 20px 0; display: flex; flex-direction: column; gap: 10px; text-align: left; }
.feature-bullets li { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #475569; }
.bullet-dot { width: 6px; height: 6px; border-radius: 50%; background-color: #0f172a; }

.desktop-banner {
  background-color: #fefce8;
  border: 1px solid #fde68a;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 20px;
  text-align: left;
}
.banner-tag { font-size: 11px; font-weight: 700; color: #854d0e; margin-bottom: 6px; }
.banner-body { font-size: 12px; line-height: 1.5; color: #475569; margin: 0 0 12px 0; word-break: keep-all; }
.banner-body strong { color: #0f172a; }

.banner-cta {
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  box-sizing: border-box;
  background-color: #fff8c2;
  color: #5e4c00;
  border: 1px solid #ffe74d;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
  transition: background-color 0.2s;
}
.banner-cta:hover {
  background-color: #fff085;
}

.status-summary { padding: 16px; border-radius: 10px; margin-bottom: 16px; text-align: left; }
.status-summary.is-safe { background-color: #f0fdf4; border: 1px solid #bbf7d0; }
.status-summary.is-alert { background-color: #fef2f2; border: 1px solid #fecaca; }
.status-pill { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-bottom: 6px; }
.is-safe .status-pill { background-color: #dcfce7; color: #166534; }
.is-alert .status-pill { background-color: #fee2e2; color: #991b1b; }
.status-score-title { font-size: 18px; font-weight: 800; margin: 0 0 4px 0; }
.is-safe .status-score-title { color: #166534; }
.is-alert .status-score-title { color: #991b1b; }
.status-sub { font-size: 12px; color: #475569; margin: 0; line-height: 1.4; word-break: keep-all; text-align: left; }

.detection-box { margin-bottom: 14px; text-align: left; }
.detection-label { font-size: 11px; font-weight: 700; color: #64748b; margin-bottom: 6px; text-align: left; }
.detection-chip { display: flex; align-items: center; gap: 5px; padding: 8px 12px; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12.5px; font-weight: 600; color: #0f172a; margin-bottom: 6px; text-align: left; }
.chip-dot { width: 6px; height: 6px; border-radius: 50%; background-color: #ef4444; }

.detection-chip-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 12px; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12px; margin-bottom: 4px; }
.chip-name { font-weight: 600; color: #0f172a; }
.chip-status { font-weight: 700; }
.chip-status.is-detected { color: #ef4444; }
.chip-status.is-passed { color: #10b981; }

.panel-actions-bottom {
  margin-top: auto;
}

/* 하단 버튼 스타일 */
.btn-primary, .btn-secondary, .btn-disabled {
  width: 100%;
  padding: 12px;
  border-radius: 8px;
  font-size: 13.5px;
  font-weight: 700;
  cursor: pointer;
  border: none;
}
.btn-primary { background-color: #0f172a; color: #ffffff; margin-top: auto; }
.btn-secondary { background-color: #ffffff; color: #0f172a; border: 1px solid #e2e8f0; margin-top: 8px; }
.btn-disabled { background-color: #f1f5f9; color: #94a3b8; cursor: not-allowed; margin-top: auto; }

.panel-state-scanning { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 280px; gap: 16px; flex: 1; }
.scan-spinner { width: 32px; height: 32px; border: 3px solid #e2e8f0; border-top-color: #0f172a; border-radius: 50%; animation: spin 0.8s linear infinite; }
.scan-text { font-size: 13px; font-weight: 600; color: #475569; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
