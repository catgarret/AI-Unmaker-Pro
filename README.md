# AI Unmarker Pro

이 문서부터 읽으시면 프로젝트를 이어서 수정하실 수 있습니다. Git 저장소와 작업 폴더 이름은 `AI-Unmaker-Pro`, 앱 표시 이름은 **AI Unmarker Pro**입니다. 두 이름 때문에 나뉘어 있던 프로젝트를 한 저장소로 통합했습니다.

## 현재 동작과 한계

- 이미지 처리는 Qwen-Image-2512 Q4_K_M, Lightning, DeSynth의 기본·윤곽 보존 디테일 복원과 선택적 선명화를 사용합니다.
- EXIF, XMP, C2PA와 등록된 공개 워터마크는 각각 검사합니다. 이 검사들은 SynthID 픽셀 검출기가 아닙니다.
- **로컬 SynthID 픽셀 검출기는 없습니다.** `SynthID 공식 검사`에서 Google과 OpenAI의 공식 검사 페이지를 각각 열 수 있습니다. OpenAI API 키를 설정한 경우 앱에서도 실제 C2PA·SynthID 결과를 받을 수 있습니다. 외부 전송은 파일 선택과 동의 후에만 진행됩니다.
- 처리 완료, 메타데이터 삭제, PSNR 수치는 SynthID 제거 성공을 의미하지 않습니다. 제거 여부를 검사하지 않은 결과는 계속 `not_tested`로 기록합니다.
- **2026-10-21 실제 검증에서 첫 샘플의 처리 후에도 SynthID가 검출됐습니다. 이 버전을 제거 성공이 검증된 버전으로 간주하지 마십시오.** 자세한 범위와 남은 작업은 [수정 기록](docs/REPAIR-2026-10-21.md)에 있습니다.

## 1.3.0 검사 표시

탐지된 항목은 반원과 제목에 **빨간색 · 탐지됨**으로 표시합니다. SynthID 결과가 없으면 **미검사**, 연결된 검사에서 신호가 없으면 **미검출**, 서비스 오류는 **검토**로 표시합니다. 처리 완료를 제거 성공으로 표시하지 않습니다.

공식 검사 팝업에서 현재 이미지와 해시가 같은 JSON 기록을 가져올 수 있습니다. 웹 관찰 기록은 웹 검사 기록으로 표시하고, OpenAI API 결과도 파일에 연결해 반원과 상세 카드에 반영합니다. 원본과 처리본의 판정은 독립적입니다. 새 로컬 SynthID 검출기를 추가한 것은 아닙니다. [검사 연결과 최신 공개 연구 검토](docs/DETECTION-RESEARCH-2026-10-21.md)에 데이터 형식·보안·후보별 판단을 정리했습니다.

## Google·OpenAI 공식 검사

| 서비스 | 앱에서 제공하는 기능 | 검사 범위 |
| --- | --- | --- |
| [Google SynthID Detector](https://synthid.com/) | 공식 페이지 열기, 사용자가 로그인·업로드 | Google과 지원 협력사의 SynthID |
| [OpenAI Verify](https://openai.com/ko-KR/research/verify/) | 공식 페이지 열기 또는 선택적 공식 API 검사 | OpenAI C2PA·SynthID |

Google 웹 페이지를 비공식 API처럼 호출하거나 로그인 정보를 수집하지 않습니다. Google 자동 검출 기능을 구현한 것은 아닙니다. Gemini 생성 샘플은 아직 없어 해당 생성 경로의 검증은 남아 있습니다. OpenAI 결과만으로 Google 생성 이미지의 신호 유무를 판단하지 마십시오.

OpenAI API를 이용하실 경우 실행 환경에 `OPENAI_API_KEY`를 안전하게 설정하십시오. 키를 코드, 명령 인수, 저장소, 채팅에 넣지 마십시오. 계정의 API 이용 조건과 접근 권한·요청 한도가 적용됩니다. [공식 API 문서](https://developers.openai.com/api/docs/guides/content-provenance)를 참고하십시오.

```bash
# 명시적으로 선택한 파일 한 개를 OpenAI에 업로드합니다.
python main.py --verify-openai input.png --allow-upload --result-file verification.json
```

응답에는 업로드한 파일의 SHA-256과 C2PA·SynthID의 개별 결과가 기록됩니다. 오류, 응답 누락, 권한 부족을 미검출로 표시하지 않습니다. 로컬 `--scan`과 `--clean`은 외부 검사를 자동 실행하지 않습니다.

## 실행

Python 3.11~3.14와 Tkinter가 필요합니다. 개발 환경에는 3.12를 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Windows에서는 가상 환경 활성화 명령을 `.venv\Scripts\activate`로 바꾸십시오. 소스와 별개인 앱 전용 패키지가 설치되어 있으면 실행 시 함께 검사합니다. 개발 환경만 사용하려면 `python main.py --ignore-overlay`로 실행하십시오.

처음 필요한 모델 전체 크기는 약 14.94 GB입니다. 기존 Q2 파일은 자동으로 삭제하지 않습니다. Q4를 준비하기 전에는 모델 크기와 저장 공간을 확인하십시오. 모델은 프로젝트나 Git에 포함하지 않고 운영체제의 사용자 캐시에 저장합니다. `UNMARK_MODEL_CACHE`로 위치를 바꾸실 수 있습니다.

```bash
# 메타데이터 및 공개 워터마크 검사
python main.py --scan input.png --result-file scan.json

# 기본 처리. 재현할 때만 시드를 지정합니다.
python main.py --clean input.png --output output.png --seed 42 --result-file report.json

# 복원 과정의 영향을 따로 비교할 때 사용합니다.
python main.py --clean input.png --output regenerated.png --no-restore --seed 42

# 픽셀을 유지하며 파일 메타데이터만 삭제합니다.
python main.py --clean input.png --output metadata-only.png --metadata-only
```

출력 파일은 기존 파일을 덮어쓰지 않습니다. 원본 크기, 투명도, 색상 프로필을 유지합니다. 이미지 재생성은 RGB 픽셀과 작은 글자·질감 등을 바꿀 수 있습니다.

## 윤곽 보존·약한 선명화

이미지를 첨부한 다음 **처리 옵션**을 누르십시오. 두 항목은 각각 또는 함께 선택할 수 있습니다. `설정 저장`은 다음 처리에 사용할 설정만 바꾸며, `적용 후 처리`는 현재 원본에서 다시 처리합니다. 이미 처리한 결과 위에 연속 적용하지 않습니다. 옵션은 현재 앱 실행 동안 유지됩니다.

| 옵션 | 동작 | 기본값 |
| --- | --- | --- |
| 윤곽 보존 | 원본 윤곽 강도에 따라 서로 다른 폭의 디테일 복원을 혼합합니다. | 끔 |
| 약한 선명화 | 복원 후 0.2 강도로 선명화하며 최종 픽셀 변환 전에 적용합니다. | 끔 |

두 옵션은 화질 조정입니다. 원본 픽셀과의 차이가 줄거나 SynthID 검출이 사라진다는 보장은 없습니다. DeSynth 기본값과 기존 결과를 유지하도록 기본 복원 모드와 선명화 끔을 유지했습니다. 비교 기록은 [옵션 검증](docs/DETAIL-OPTIONS-2026-10-21.md)을 확인하십시오.

```bash
# 윤곽 보존만 적용
python main.py --clean input.png --output edge.png --restore-mode edge --seed 42

# 약한 선명화만 적용
python main.py --clean input.png --output sharpened.png --unsharp 0.2 --seed 42

# 두 옵션 함께 적용
python main.py --clean input.png --output combined.png --restore-mode edge --unsharp 0.2 --seed 42
```

CLI의 `--unsharp`는 0~1을 지원합니다. `--no-restore`와 윤곽 보존·선명화를 함께 지정하면 무시하지 않고 오류로 알립니다. JSON의 `regeneration.parameters`와 `regeneration.detail_restore`에서 실제 적용 옵션을 확인하실 수 있습니다.

## 처리 흐름

1. 원본의 임시 사본을 만들고 파일이 읽는 중 바뀌지 않았는지 확인합니다.
2. 메타데이터와 공개 워터마크를 별도로 검사합니다.
3. 선택된 가시 표식을 처리하고 Qwen 이미지 재생성을 실행합니다.
4. 재생성 이미지와 원본의 디테일을 복원하고, 선택한 경우 윤곽 혼합과 약한 선명화를 순서대로 적용합니다.
5. 실제 저장된 PNG를 다시 열어 크기·투명도·메타데이터를 검사한 뒤 새 파일로 확정합니다.

CPU라고 자동으로 512px로 낮추거나 Q2 모델로 바꾸지 않습니다. `--max-working-side`를 지정하면 비율 유지 축소 모드가 되며 DeSynth 기본 전처리와 달라집니다. 기본값은 고정된 Diffusers 파이프라인의 입력 변환을 그대로 사용합니다.

## 수정할 파일

| 파일 | 역할 |
| --- | --- |
| `app_backend.py` | 검사와 처리 전체 흐름, CLI, 결과 JSON |
| `verification_records.py` | 파일 해시로 연결한 외부 검사 기록의 저장·가져오기·검증 |
| `provenance.py` | 공식 검사 서비스 명세, 동의가 필요한 OpenAI API 연결과 응답 해석 |
| `qwen_inference.py` | Qwen 로딩, 메모리 관리, 추론, 복원 연결 |
| `detail_restore.py` | 복원 정책의 단일 명세, 윤곽별 혼합, 선명화, 행 단위 메모리 제한 |
| `model_runtime.py` | 모델 명세, 고정 해시, HTTPS 다운로드·이어받기·캐시 검증 |
| `metadata_core.py` | PNG/JPEG/WebP 메타데이터·C2PA 읽기와 삭제 |
| `open_watermark_bridge.py`, `visible_bridge.py` | 공개 워터마크와 가시 표식 검사 연결 |
| `pixel_engine.py` | 선택적 픽셀 처리와 주파수 통계 |
| `desktop_ui.py`, `bootstrap_ui.py`, `ui_assets.py` | 데스크톱 화면과 준비 화면 |
| `worker_runtime.py`, `file_publish.py` | 별도 처리 프로세스와 덮어쓰기 없는 저장 |
| `runtime_setup.py` | 앱 전용 의존성 설치·검사 |
| `assemble_main.py` | 위 구성 파일을 배포용 `main.py`로 조립 |
| `build.py` | 운영체제별 앱 빌드, 구성·버전 검증 |
| `web/` | 브라우저용 메타데이터 처리 도구. Qwen이나 SynthID 검출 기능 없음 |
| `tests/` | 네트워크·모델 다운로드 없이 실행하는 회귀검사 |

`main.py`는 생성 파일이므로 직접 수정하지 마십시오. 해당 구성 파일을 수정한 다음 `python assemble_main.py`를 실행합니다. `assembly_header.txt`와 `build.py`의 버전·엔진·릴리스 ID는 일치해야 합니다. 새 구성 파일을 추가하실 때는 조립 순서와 필요한 import를 함께 점검하십시오.

```bash
python assemble_main.py
python -m pip install pytest
python -m pytest tests -q
python build.py --dry-run
python build.py
```

macOS 결과는 `dist/AI Unmarker Pro.app`에 생성됩니다. 가중치는 앱에 포함되지 않습니다. GitHub의 검사는 push와 PR에서 실행되며, 운영체제별 앱 빌드와 배포는 수동 실행 또는 `v*` 태그에서 수행합니다. CI에서 실제 Q4 추론이나 SynthID 제거 성공을 검증한다고 해석하지 마십시오.

## 보안과 인수인계

모델은 고정 리비전과 SHA-256으로 확인합니다. HTTPS 인증서·호스트명 검증을 유지하고 macOS/Windows 시스템 신뢰 저장소를 사용합니다. 공개 모델 다운로드에 API 키나 사용자 Hugging Face 토큰을 보내지 않습니다. 프롬프트 캐시는 고정 해시를 확인하고 `weights_only=True`로 읽습니다. C2PA 원격 참조는 자동 조회하지 않습니다.

OpenAI 검사는 고정된 `api.openai.com`에만 전송하며 리디렉션을 따르지 않습니다. 파일은 50 MiB, 응답은 1 MiB, 연결은 90초로 제한합니다. 키·원시 오류·이미지 본문은 결과나 로그에 기록하지 않습니다. 검사 결과로 자동 재처리하거나 반복 요청하지 않습니다.

개인 샘플, 로컬 검증 출력, 모델, 가상 환경, 로그는 Git에서 제외합니다. 병합 전 파일은 작업 폴더 바깥의 백업 압축 파일로 보존합니다. 상세 변경 이유와 검증 범위는 [수정 기록](docs/REPAIR-2026-10-21.md)을 확인하십시오.

참고 구현은 [DeSynth](https://github.com/0xROOTPLS/DeSynth/tree/0c2b14fa6b92fff3721877838fcb9de78fdec0ff)입니다. 해당 리비전에는 소스 라이선스 파일이 없으므로 전체 소스를 이 저장소에 복사하지 않았습니다. 모델·폰트와 라이브러리의 라이선스는 각각 별도로 적용됩니다.
