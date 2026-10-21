import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import vm from 'node:vm'
import { deflateSync } from 'node:zlib'
import test from 'node:test'

const directory = dirname(fileURLToPath(import.meta.url))
const source = await readFile(resolve(directory, '../AiUnmakerTool.vue'), 'utf8')
const require = createRequire(process.env.UNMARKER_WEB_TEST_MODULES
  ? resolve(process.env.UNMARKER_WEB_TEST_MODULES, '../package.json') : import.meta.url)
const { DOMParser: XmlParser } = require('@xmldom/xmldom')
const { parse, compileScript, compileTemplate, compileStyle } = require('@vue/compiler-sfc')
class DOMParser {
  parseFromString(text, type) {
    let malformed = false
    const parser = new XmlParser({ errorHandler: { warning() { malformed = true }, error() { malformed = true }, fatalError() { malformed = true } } })
    const document = parser.parseFromString(text, type)
    return malformed ? new XmlParser().parseFromString('<parsererror/>', type) : document
  }
}
const clean = value => JSON.parse(JSON.stringify(value))
function runtime() {
  const objectUrls = new Set(), errors = [], decoded = [], contexts = []
  let index = 0, unmount
  class TestImage {
    constructor() { this.naturalWidth = 2; this.naturalHeight = 2 }
    set src(value) { if (value) { decoded.push(value); queueMicrotask(() => this.onload?.()) } }
  }
  const context = vm.createContext({ Blob, DecompressionStream, TextDecoder, Uint8Array,
    DataView, AbortController, DOMParser, queueMicrotask, Image: TestImage,
    console: { error: (...args) => errors.push(args), log() {} },
    ref: value => ({ value }), onBeforeUnmount: fn => { unmount = fn },
    URL: { createObjectURL() { const url = `blob:test-${++index}`; objectUrls.add(url); return url }, revokeObjectURL(url) { objectUrls.delete(url) } },
    document: { createElement(kind) {
      if (kind !== 'canvas') throw new Error('unexpected_dom_operation')
      return { width: 0, height: 0,
        getContext(type, options) { contexts.push({ type, options }); return { drawImage() {} } },
        toBlob(callback, type) { callback(new Blob([png([])], { type })) } }
    } }
  })
  const script = source.split('<script setup>')[1].split('</script>')[0].replace(/^import .* from 'vue'\s*$/m, '')
  vm.runInContext(script + '\nglobalThis.api = {inspectImageMetadata,stripPngMetadata,verifyScrubbedPng,scrubMetadataToPng,decodeLocalImage,loadFile,resetWorkspace,processMetadataScrub,currentFile,previewUrl,cleanedBlobUrl,isScanning,isProcessed,isAiDetected,confidence,detectedSources,genPlatforms,detectorScores,metadataScanReport};', context)
  return { ...context.api, objectUrls, errors, decoded, contexts, unmount: () => unmount() }
}
const crcTable = Array.from({ length: 256 }, (_, n) => { for (let k = 0; k < 8; k++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1; return n >>> 0 })
const crc32 = b => { let crc = 0xffffffff; for (const v of b) crc = crcTable[(crc ^ v) & 255] ^ (crc >>> 8); return (crc ^ 0xffffffff) >>> 0 }
const be16 = n => { const b = Buffer.alloc(2); b.writeUInt16BE(n); return b }
const be32 = n => { const b = Buffer.alloc(4); b.writeUInt32BE(n); return b }
const le32 = n => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b }
function chunk(type, data = Buffer.alloc(0)) { const body = Buffer.concat([Buffer.from(type), data]); return Buffer.concat([be32(data.length), body, be32(crc32(body))]) }
const pixels = Buffer.from([0, 255, 0, 0, 0, 0, 255, 0, 128, 0, 0, 0, 255, 255, 255, 255, 255, 255])
const ihdr = Buffer.concat([be32(2), be32(2), Buffer.from([8, 6, 0, 0, 0])])
function png(metadata, options = {}) {
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), chunk('IHDR', ihdr), ...metadata,
    chunk('IDAT', options.idat || deflateSync(pixels)), ...(options.after || []), chunk('IEND')])
}
const file = (buffer, name = 'image.png') => Object.defineProperty(new Blob([buffer]), 'name', { value: name })
const text = (key, value) => chunk('tEXt', Buffer.from(`${key}\0${value}`, 'latin1'))
const ztext = (key, value) => chunk('zTXt', Buffer.concat([Buffer.from(`${key}\0\0`), deflateSync(Buffer.from(value))]))
const itext = (key, value, compressed = false) => chunk('iTXt', Buffer.concat([Buffer.from(`${key}\0`), Buffer.from([compressed ? 1 : 0, 0, 0, 0]), compressed ? deflateSync(Buffer.from(value)) : Buffer.from(value)]))
const xmp = (software = 'OpenAI GPT Image') => `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:CreatorTool="${software}"/></rdf:RDF></x:xmpmeta>`
function orientationTiff(orientation) {
  const bytes=Buffer.alloc(26);bytes.write('II');bytes.writeUInt16LE(42,2);bytes.writeUInt32LE(8,4)
  bytes.writeUInt16LE(1,8);bytes.writeUInt16LE(0x112,10);bytes.writeUInt16LE(3,12);bytes.writeUInt32LE(1,14);bytes.writeUInt16LE(orientation,18)
  return bytes
}
function tiff(software = 'Google Gemini') {
  const string = Buffer.from(`${software}\0`), b = Buffer.alloc(26 + string.length)
  b.write('II'); b.writeUInt16LE(42,2); b.writeUInt32LE(8,4); b.writeUInt16LE(1,8)
  b.writeUInt16LE(0x131,10); b.writeUInt16LE(2,12); b.writeUInt32LE(string.length,14); b.writeUInt32LE(26,18); string.copy(b,26)
  return b
}
const box = (type, bytes) => Buffer.concat([be32(bytes.length + 8), Buffer.from(type), bytes])
const uuid = Buffer.from('6332706100110010800000aa00389b71', 'hex')
const jumb = (label, children = [], id = uuid, toggle = 3) => box('jumb', Buffer.concat([box('jumd', Buffer.concat([id, Buffer.from([toggle]), Buffer.from(label + '\0'), toggle & 4 ? be32(42) : Buffer.alloc(0)])), ...children]))
function cbor(value) {
  const count = (major, n) => n < 24 ? Buffer.from([(major << 5) | n]) : n < 256 ? Buffer.from([(major << 5) | 24, n]) : Buffer.concat([Buffer.from([(major << 5) | 25]), be16(n)])
  if (typeof value === 'string') { const b = Buffer.from(value); return Buffer.concat([count(3, b.length), b]) }
  if (Array.isArray(value)) return Buffer.concat([count(4,value.length), ...value.map(cbor)])
  return Buffer.concat([count(5,Object.keys(value).length), ...Object.entries(value).flatMap(([k,v]) => [cbor(k),cbor(v)])])
}
function store({ source = 'OpenAI GPT Image', synthid = true } = {}) {
  return jumb('c2pa', [jumb('urn:uuid:sample', [jumb('c2pa.claim.v2', [box('cbor', cbor({ claim_generator: source }))]),
    jumb('c2pa.actions.v2', [box('cbor', cbor({ actions: [{ action: 'c2pa.watermarked', description: synthid ? 'SynthID applied' : 'Metadata recorded' }] }))])])])
}
const app = (marker, bytes) => Buffer.concat([Buffer.from([255,marker]), be16(bytes.length+2), bytes])
const jpeg = segments => Buffer.concat([Buffer.from([255,216]), ...segments, app(192,Buffer.from([8,0,2,0,2,1,1,0x11,0])), app(218,Buffer.from([1,1,0,0,63,0])), Buffer.from([13,3,42,255,0,13,255,217])])
function jpParts(bytes, cuts = [33,61]) {
  const payload = bytes.subarray(8), points = [0,...cuts,payload.length]
  return points.slice(0,-1).map((at,i) => app(235,Buffer.concat([Buffer.from('JP'),be16(1),be32(i+1),bytes.subarray(0,8),payload.subarray(at,points[i+1])])))
}
const webpChunk = (kind,b) => Buffer.concat([Buffer.from(kind),le32(b.length),b,b.length & 1 ? Buffer.from([0]) : Buffer.alloc(0)])
const webp = chunks => { const body=Buffer.concat([Buffer.from('WEBP'),...chunks,webpChunk('VP8L',Buffer.from([47,1,64,0,0,0]))]); return Buffer.concat([Buffer.from('RIFF'),le32(body.length),body]) }
const inspect = async b => runtime().inspectImageMetadata(file(b))

// Run with UNMARKER_WEB_TEST_MODULES=/path/to/node_modules node --test web/tests/detection.test.mjs.
test('Vue SFC script, template, and scoped styles compile', () => {
  const result = parse(source, { filename: 'AiUnmakerTool.vue' })
  assert.deepEqual(result.errors, [])
  const script = compileScript(result.descriptor, { id: 'unmarker-metadata' })
  assert.ok(script.content.includes('inspectImageMetadata'))
  assert.deepEqual(compileTemplate({ source: result.descriptor.template.content, filename: 'AiUnmakerTool.vue', id: 'unmarker-metadata', compilerOptions: { bindingMetadata: script.bindings } }).errors, [])
  assert.deepEqual(compileStyle({ source: result.descriptor.styles[0].content, filename: 'AiUnmakerTool.vue', id: 'unmarker-metadata', scoped: true }).errors, [])
})
test('original template and style bytes are preserved', () => {
  const hash = s => createHash('sha256').update(s).digest('hex')
  assert.equal(hash(source.slice(0, source.indexOf('<script setup>'))), '18b2ce5582c52c0bdce3dfa6e00e5b96cec9ea4fb0c07c2111d8cb33a8a9a736')
  assert.equal(hash(source.slice(source.indexOf('<style scoped>'))), 'a17ee489c0375bd05ad8c5a562aa703140c8dab6cba923c21d4bda0ca5383ff1')
})
test('PNG text and compressed Unicode fields are actually decoded', async () => {
  const report = await inspect(png([text('Software','Midjourney'), ztext('Comment','SynthID declaration'), itext('Description','한글 메모',true)]))
  assert.equal(report.status,'complete'); assert.equal(report.format,'PNG')
  assert.deepEqual(clean(report.platforms),['Midjourney'])
  assert.ok(report.metadata.some(m=>m.value==='한글 메모'))
  assert.ok(report.evidence.some(e=>e.label==='SynthID 메타데이터'))
  assert.equal(report.capabilities.synthidPixelDetection,false)
})
test('compressed XMP namespace and TIFF Software are read as fields', async () => {
  const report=await inspect(png([itext('XML:com.adobe.xmp',xmp(),true),chunk('eXIf',tiff())]))
  assert.equal(report.status,'complete')
  assert.deepEqual(clean(report.platforms),['OpenAI (ChatGPT / DALL-E)','Google (Gemini / Imagen)'])
})
test('C2PA root descriptor and bounded CBOR declarations are recognized', async () => {
  const report=await inspect(png([chunk('caBX',store())]))
  assert.equal(report.status,'complete');assert.equal(report.c2pa.present,true)
  assert.equal(report.c2pa.signatureVerified,false)
  assert.ok(report.evidence.some(e=>e.label==='SynthID 메타데이터'))
  assert.deepEqual(clean(report.platforms),['OpenAI (ChatGPT / DALL-E)'])
})
test('requestable toggle does not shift label; optional ID comes after label', async () => {
  for (const toggle of [2,3,7]) {
    const report=await inspect(png([chunk('caBX',jumb('c2pa',[],uuid,toggle))]))
    assert.equal(report.c2pa.present,true)
  }
})
test('generic C2PA never implies Adobe, an AI source, or a SynthID pixel verdict', async () => {
  const rt=runtime();await rt.loadFile(file(png([chunk('caBX',jumb('c2pa'))])))
  assert.equal(rt.metadataScanReport.value.c2pa.present,true)
  assert.deepEqual(clean(rt.genPlatforms.value),[]);assert.equal(rt.isAiDetected.value,false)
  assert.deepEqual(clean(rt.detectorScores.value),{})
})
test('JPEG EXIF, XMP, and repeated-header APP11 fragments are read', async () => {
  const report=await inspect(jpeg([app(225,Buffer.concat([Buffer.from('Exif\0\0'),tiff()])),app(225,Buffer.concat([Buffer.from('http://ns.adobe.com/xap/1.0/\0'),Buffer.from(xmp())])),...jpParts(store())]))
  assert.equal(report.status,'complete');assert.equal(report.c2pa.present,true)
  assert.ok(report.metadata.some(e=>e.field==='Software'&&e.value==='Google Gemini'))
  assert.equal(report.width,2);assert.equal(report.height,2)
})
test('missing, duplicate, or mismatched APP11 fragments never produce a clean absence', async () => {
  const parts=jpParts(store())
  for(const segments of [[parts[0],parts[2]],[parts[0],parts[0],parts[1],parts[2]]]){
    const report=await inspect(jpeg(segments));assert.equal(report.status,'incomplete')
  }
})
test('JPEG extended XMP is assembled by offsets, not by unrelated binary text', async () => {
  const bytes=Buffer.from(xmp('Midjourney')),guid=Buffer.from('A'.repeat(32)),cut=80
  const packets=[bytes.subarray(0,cut),bytes.subarray(cut)].map((b,i)=>app(225,Buffer.concat([Buffer.from('http://ns.adobe.com/xmp/extension/\0'),guid,be32(bytes.length),be32(i?cut:0),b])))
  const report=await inspect(jpeg(packets));assert.equal(report.status,'complete');assert.deepEqual(clean(report.platforms),['Midjourney'])
})
test('WebP EXIF, XMP and C2PA chunks are local structured evidence',async()=>{
  const report=await inspect(webp([webpChunk('EXIF',tiff()),webpChunk('XMP ',Buffer.from(xmp())),webpChunk('C2PA',store())]))
  assert.equal(report.status,'complete');assert.equal(report.c2pa.present,true);assert.ok(report.platforms.length===2)
})
test('pixel bytes, generic comments and XMP namespaces are not source classifiers',async()=>{
  const report=await inspect(png([text('Comment','a tutorial about OpenAI, adobe and c2pa')],{idat:deflateSync(Buffer.from('c2pa SynthID openai'),{level:0})}))
  assert.deepEqual(clean(report.platforms),[]);assert.equal(report.c2pa.present,false)
  assert.equal(report.evidence.length,0)
  const plainXmp=await inspect(png([itext('XML:com.adobe.xmp',xmp('ordinary camera application'))]))
  assert.deepEqual(clean(plainXmp.platforms),[])
})
test('malformed CRC, zlib, TIFF, XML and oversized decompression are incomplete',async()=>{
  const corrupt=text('Software','Midjourney');corrupt[corrupt.length-1]^=1
  const broken=[corrupt,chunk('zTXt',Buffer.from('Software\0\0invalid')),chunk('eXIf',Buffer.from('invalid')),itext('XML:com.adobe.xmp','<unclosed>'),ztext('Comment','a'.repeat(4*1024*1024+1))]
  for(const metadata of broken){const report=await inspect(png([metadata]));assert.equal(report.status,'incomplete');assert.deepEqual(clean(report.platforms),[])}
})
test('wrong C2PA label/UUID and non-JUMBF caBX cannot become a credential',async()=>{
  for(const data of [jumb('not-c2pa'),jumb('c2pa',[],Buffer.alloc(16)),Buffer.from('c2pa SynthID')]){
    const report=await inspect(png([chunk('caBX',data)]));assert.equal(report.c2pa.present,false)
  }
})
test('native PNG scrub preserves encoded pixels/alpha and strips all provenance chunks',async()=>{
  const rt=runtime(),input=png([text('Software','Midjourney'),ztext('Comment','SynthID'),itext('XML:com.adobe.xmp',xmp()),chunk('eXIf',tiff()),chunk('caBX',store()),chunk('zzZz',Buffer.from('unknown provenance'))])
  const cleaned=await rt.stripPngMetadata(file(input)),output=Buffer.from(await cleaned.arrayBuffer())
  assert.deepEqual(output,png([]))
  const report=await rt.verifyScrubbedPng(cleaned)
  assert.equal(report.c2pa.present,false);assert.equal(report.metadata.length,0)
  assert.deepEqual(clean(report.chunks.map(c=>c.type)),['IHDR','IDAT','IEND'])
})
test('native PNG scrub preserves APNG frame payloads',async()=>{
  const rt=runtime(), animation=[chunk('acTL',Buffer.concat([be32(2),be32(0)])),chunk('fcTL',Buffer.concat([be32(0),be32(2),be32(2),be32(0),be32(0),be16(1),be16(10),Buffer.from([0,0])]))]
  const after=[chunk('fcTL',Buffer.concat([be32(1),be32(2),be32(2),be32(0),be32(0),be16(1),be16(10),Buffer.from([0,0])])),chunk('fdAT',Buffer.concat([be32(2),deflateSync(pixels)]))]
  const input=png([...animation,text('Comment','SynthID')],{after}),expected=png(animation,{after})
  const cleaned=await rt.stripPngMetadata(file(input));assert.deepEqual(Buffer.from(await cleaned.arrayBuffer()),expected)
  assert.equal((await rt.inspectImageMetadata(cleaned)).animated,true)
})
test('palette colors and tRNS transparency are preserved byte for byte',async()=>{
  const rt=runtime(),header=Buffer.concat([be32(2),be32(2),Buffer.from([8,3,0,0,0])])
  const essentials=[chunk('IHDR',header),chunk('PLTE',Buffer.from([255,0,0,0,255,0])),chunk('tRNS',Buffer.from([0,128])),chunk('IDAT',deflateSync(Buffer.from([0,0,1,0,1,0]))),chunk('IEND')]
  const signature=Buffer.from([137,80,78,71,13,10,26,10])
  const input=Buffer.concat([signature,essentials[0],text('Comment','SynthID'),...essentials.slice(1)])
  const output=await rt.stripPngMetadata(file(input));assert.deepEqual(Buffer.from(await output.arrayBuffer()),Buffer.concat([signature,...essentials]))
})
test('EXIF orientation is recorded, normalized through Canvas2D, and not discarded raw',async()=>{
  const rt=runtime(),input=file(png([chunk('eXIf',orientationTiff(6))]))
  assert.equal((await rt.inspectImageMetadata(input)).orientation,6)
  await assert.rejects(rt.stripPngMetadata(input),/orientation_requires_normalization/)
  const output=await rt.scrubMetadataToPng(input)
  assert.equal(rt.contexts[0].type,'2d');assert.equal((await rt.inspectImageMetadata(output)).orientation,1)
  const high=Buffer.from(ihdr);high[8]=16
  const highFile=file(Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',high),chunk('eXIf',orientationTiff(6)),chunk('IDAT',deflateSync(pixels)),chunk('IEND')]))
  await assert.rejects(rt.scrubMetadataToPng(highFile),/high_depth_conversion_not_supported/)
})
test('oversized dimensions are rejected before browser decoding',async()=>{
  const rt=runtime(),header=Buffer.from(ihdr);header.writeUInt32BE(100000,0);header.writeUInt32BE(100000,4)
  const input=file(Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(pixels)),chunk('IEND')]))
  await assert.rejects(rt.scrubMetadataToPng(input),/image_dimension_limit/);assert.equal(rt.decoded.length,0)
})
test('SD parameters and ComfyUI graph require actual serialized schemas',async()=>{
  const a=await inspect(png([text('parameters','a tree\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123, Size: 512x512')]))
  assert.deepEqual(clean(a.platforms),['Stable Diffusion'])
  const b=await inspect(png([text('prompt',JSON.stringify({'1':{class_type:'CheckpointLoaderSimple',inputs:{}},'2':{class_type:'KSampler',inputs:{}}}))]))
  assert.deepEqual(clean(b.platforms),['ComfyUI'])
  const c=await inspect(png([text('prompt','a picture discussing Stable Diffusion and ComfyUI')]))
  assert.deepEqual(clean(c.platforms),[])
})
test('corrupt retained pixel chunk prevents completed output',async()=>{
  const rt=runtime(),input=png([]);input[input.length-13]^=1
  await assert.rejects(rt.stripPngMetadata(file(input)),/crc_mismatch/)
})
test('file replacement ignores slow stale scans and revokes old object URLs',async()=>{
  const rt=runtime(),slow=file(png([text('Software','Midjourney')])),fast=file(png([]))
  let release;const originalSlice=slow.slice.bind(slow)
  slow.slice=(...args)=>({arrayBuffer:async()=>{await new Promise(resolve=>{release=resolve});return originalSlice(...args).arrayBuffer()}})
  const stale=rt.loadFile(slow);await Promise.resolve();await rt.loadFile(fast);release();await stale
  assert.equal(rt.currentFile.value,fast);assert.equal(rt.isScanning.value,false);assert.equal(rt.isAiDetected.value,false)
  assert.deepEqual(clean(rt.detectorScores.value),{});assert.equal(rt.objectUrls.size,1)
  rt.resetWorkspace();assert.equal(rt.objectUrls.size,0)
})
test('invalid reads and cancelled scans never settle as passed or complete',async()=>{
  const rt=runtime();await rt.loadFile(file(Buffer.from('invalid_image_format_____')))
  assert.equal(rt.currentFile.value,null);assert.equal(rt.metadataScanReport.value.status,'error')
  assert.equal(rt.isProcessed.value,false);assert.equal(rt.objectUrls.size,0)
  const controller=new AbortController();controller.abort()
  await assert.rejects(rt.inspectImageMetadata(file(png([])),{signal:controller.signal}),{name:'AbortError'})
})
test('processing sets done only after saved PNG reinspection and releases resources',async()=>{
  const rt=runtime();await rt.loadFile(file(png([text('Software','Midjourney'),text('Comment','SynthID')])))
  await rt.processMetadataScrub();assert.equal(rt.isProcessed.value,true)
  assert.equal(rt.metadataScanReport.value.output.metadata.length,0)
  assert.equal(rt.objectUrls.size,2)
  rt.unmount();assert.equal(rt.objectUrls.size,0)
})
test('JPEG/WebP conversion uses only Canvas2D and rejects animated conversion',async()=>{
  const rt=runtime(),input=file(jpeg([]),'input.jpg')
  const output=await rt.scrubMetadataToPng(input)
  assert.equal(output.type,'image/png');assert.equal(rt.contexts[0].type,'2d')
  assert.equal(rt.contexts[0].options.willReadFrequently,true)
  const animated=file(webp([webpChunk('ANIM',Buffer.alloc(6))]),'animated.webp')
  await assert.rejects(rt.scrubMetadataToPng(animated),/animated_conversion_not_supported/)
})
