# Vue 웹 버전 — 파일 메타데이터 검사 및 제거

`AiUnmakerTool.vue`는 사용자가 제공한 Vue 파일의 `<template>`과 `<style scoped>`를 그대로 두고 `<script setup>`의 검사·제거 동작을 수정한 파일입니다. 기존 Vue 3 프로젝트에서 해당 컴포넌트를 교체하면 됩니다. 앱 이름, 문구, 아이콘, 간격, 색상과 화면 구성은 바꾸지 않았습니다.

## 이미지 입력 시 검사

파일을 넣으면 브라우저에서 파일 구조를 읽습니다. 서버 전송, API 키, 모델 다운로드, 딥러닝 추론을 사용하지 않습니다. 이미지의 압축된 픽셀 데이터를 문자열로 바꿔 검색하는 이전 방식은 사용하지 않습니다.

| 형식 | 검사하는 파일 구조 |
| --- | --- |
| PNG | `tEXt`, `zTXt`, `iTXt`의 실제 텍스트와 deflate 압축, `eXIf`의 TIFF 필드, `caBX`의 JUMBF |
| JPEG | APP1 EXIF/TIFF 및 XMP, 여러 APP1에 나뉜 확장 XMP, APP11 `JP` 패킷에 나뉜 JUMBF, 코멘트 |
| WebP | RIFF 안의 `EXIF`, `XMP `, `C2PA` 청크 |

C2PA는 올바른 JUMBF UUID와 `c2pa` 라벨을 가진 매니페스트 저장소 구조를 확인합니다. 제한된 크기의 CBOR/JSON claim 및 action을 읽어, 파일에 기록된 생성 도구와 워터마크 선언을 반환합니다. 일반 C2PA의 존재를 Adobe 또는 AI 생성의 증거로 바꾸지 않습니다. 이미지 픽셀의 SynthID를 검사하거나 C2PA 서명의 신뢰성을 검증하는 기능은 이 웹 컴포넌트에 포함되어 있지 않습니다.

기존 화면의 `SynthID 메타데이터` 항목은 실제로 읽은 메타데이터 문자열에서 해당 이름을 확인했다는 뜻입니다. 픽셀에서 SynthID를 검출했다는 뜻이 아닙니다. 생성 도구는 EXIF `Software`, XMP `CreatorTool`, C2PA의 generator/agent 필드에서 읽습니다. Stable Diffusion의 설정 행과 ComfyUI의 노드 그래프도 실제 직렬화 형식이 일치할 때만 식별합니다.

기존 마크업의 숫자 변수를 유지하되 임의의 85/90/95 점수 계산은 없앴습니다. 값은 해당 기록의 관측 여부를 나타내는 0/100이며, AI 생성 확률이나 워터마크 검출 확률이 아닙니다. 실행하지 않은 외부 도구에 점수를 붙이지 않으므로 도구별 평가 객체는 비어 있습니다.

## 메타데이터 제거

**보통의 PNG는 픽셀을 다시 그리지 않습니다.** 원래의 압축된 IDAT, 팔레트, 투명도와 APNG 프레임 데이터를 보존하고 메타데이터 청크를 제거합니다. 보존하는 모든 청크의 CRC를 확인합니다. EXIF/XMP/일반 텍스트/JUMBF/C2PA 및 알려지지 않은 부가 청크는 출력에 남기지 않습니다.

색상과 표시를 유지하기 위해 `PLTE`, `tRNS`, `cHRM`, `gAMA`, `sRGB`, `sBIT`, `iCCP`, `cICP`, `mDCV`, `cLLI`, `bKGD`를 유지합니다. `iCCP`는 ICC 색상 프로파일이므로 ICC 내부의 설명·저작자 필드까지 모두 삭제했다고 주장하지 않습니다. APNG의 `acTL`, `fcTL`, `fdAT`도 유지합니다.

EXIF 회전/반사 값이 있는 PNG는 해당 정보를 그냥 지우면 표시 방향이 달라집니다. 이런 파일은 8비트 정지 이미지일 때 Canvas2D로 방향을 정규화합니다. 방향 정규화가 필요한 16비트 이미지나 애니메이션은 조용히 변환하지 않고 중단합니다.

JPEG/WebP는 기존과 같이 PNG로 저장합니다. 브라우저의 2D 캔버스로 디코딩한 이미지를 PNG로 만든 뒤 부가 청크를 제거합니다. WebGL이나 GPU 모델은 호출하지 않으며, 캔버스는 `willReadFrequently: true`로 요청합니다. 애니메이션을 한 프레임으로 바꾸는 변환은 거부합니다. 브라우저의 이미지 디코더와 Canvas2D 구현은 운영체제에 따라 달라질 수 있습니다.

**완료 상태는 실제 출력 PNG를 다시 검사한 뒤에만 표시합니다.** 검사·인코딩·검증 실패가 완료로 표시되지 않으며, 파일 교체 또는 초기화 후 늦게 도착한 이전 결과도 적용하지 않습니다. 생성한 object URL은 파일 교체, 초기화와 컴포넌트 종료 때 해제합니다.

## 실행 제한과 오류

파서는 메타데이터 한 블록 8 MiB, 해제된 텍스트 합계 4 MiB, 읽기 합계 128 MiB, 중첩 깊이 20의 상한을 둡니다. Canvas2D 변환과 브라우저 이미지 디코딩은 3,200만 픽셀을 초과하면 시작하지 않습니다. PNG의 텍스트 CRC, TIFF 오프셋, XML 문법, JPEG 분할 패킷 순서와 RIFF 크기를 검사합니다.

잘못된 파일이나 지원하지 않는 압축을 정상적인 ‘미감지’로 처리하지 않습니다. 검사 실패는 입력 상태를 초기화하고 브라우저 콘솔의 `Scan error`에 원인을 기록합니다. 제거 실패는 완료 상태를 만들지 않습니다. 기존 디자인과 문구를 유지하기 위해 새 오류 패널이나 안내 문구는 추가하지 않았습니다.

## 테스트

테스트 의존성은 개발 검증에만 필요합니다. 웹 컴포넌트가 이미지 검사·제거를 실행할 때 이 패키지를 다운로드하지 않습니다.

```sh
cd web
npm install --ignore-scripts
npm test
```

테스트는 실제 PNG/JPEG/WebP 바이트, 압축 텍스트, EXIF/XMP, 분할된 C2PA, 잘못된 CRC와 누락 패킷, 메타데이터 폭탄, 음성 대조군, 원본 IDAT·알파·APNG 보존, 회전 정보, 오래된 비동기 결과와 URL 해제를 확인합니다. Vue SFC의 script/template/scoped-style을 컴파일하며 원본 template/style의 SHA-256도 대조합니다. Node의 Canvas2D 테스트 대역은 호출 순서와 완료 검증을 확인하기 위한 것이며, 실제 브라우저 렌더링 검증과는 구분합니다.

실제 브라우저에서 검사할 때는 파일 선택 → 결과 표시 → 메타데이터 제거 → 저장 순서로 진행하고, 내려받은 PNG를 다시 입력하여 남은 메타데이터를 확인할 수 있습니다.

## 구조 참고

- [PNG 텍스트·압축·CRC 규격](https://www.w3.org/TR/png-3/)
- [C2PA 2.4: JUMBF 매니페스트 및 파일 저장 형식](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html)
- [C2PA 공식 SDK의 JPEG APP11 구현](https://github.com/contentauth/c2pa-rs/blob/main/sdk/src/asset_handlers/jpeg_io.rs)

`inspectImageMetadata(file, { signal })`는 `format`, `metadata`, `evidence`, `platforms`, `c2pa`, `capabilities`, `warnings`를 반환합니다. `stripPngMetadata`는 픽셀을 재인코딩하지 않는 PNG 청크 제거, `scrubMetadataToPng`는 형식별 제거 및 방향 정규화, `verifyScrubbedPng`는 실제 출력 확인을 담당합니다. 모두 컴포넌트 내부 함수이며 추가 서버가 필요하지 않습니다.
