# AutoMeeting Agent 기획서 v3: Human-in-the-Loop 기반 다중 모달 회의 산출물 자동화 에이전트 시스템

---

## 1. 프로젝트 개요

* **프로젝트명**: AutoMeeting Agent (가칭)
* **목적**:
  * 회의 녹음을 구조화된 회의록으로 전환하고, 그중 구체화된 아이디어를 식별해 기획서를 작성한 뒤, 그 기획서(특히 장르)를 참고해 필요한 하위 산출물(설정집, 이미지, 비교대상 검색 보고서)을 능동적으로 판단·생성.
  * Human-in-the-Loop(2단계 게이트)과 Worker-Validator 자기 반성 루프를 결합해 산출물의 신뢰성 확보.
* **핵심 타깃 사용자**: PM, 기획자 등 회의 후 팔로업 문서 작업 리소스가 큰 실무자.
* **개발 단계**: POC — 파이프라인 구조·동적 라우팅·Self-Correction 검증이 목적. 모델 비용은 최소화하되, 정확도가 중요한 판단 지점은 예외적으로 상위 티어 사용.
* **사용 예시 도메인**: 게임 아이디어 회의 (게임이 아니어도 구조는 동일하게 작동).
* **v2 대비 변경**: 프로토타입 트랙 제거, 비교대상 검색 보고서 트랙 신설, 이미지 트랙에 Style Planner 신설.
* **이번 개정(순서 재설계)**: 회의록을 아이디어 단위로 태깅하고, 구체화된 아이디어 하나만 골라 기획서를 먼저 작성한 뒤, 그 기획서(장르)를 참고해 하위 산출물 필요 여부를 판단하는 순서로 변경. LangChain/LangGraph 역할 분담 명시.

---

## 2. 앱 사용 흐름

1. **녹음 화면**: 녹음 시작/종료 버튼. 실시간 자막(라이브 캡션) 없음 — 녹음 종료 후 전체 오디오를 한 번에 처리. **디버그용 텍스트 입력**: 녹음 대신 회의 내용을 직접 텍스트로 입력하는 옵션 제공 (1인 데모 시 실제 회의를 진행할 수 없으므로, 이 경로를 데모 기본 진행 방식으로 사용 예정). 텍스트 입력 시 STT+Diarization 단계는 건너뛰고 입력된 텍스트가 바로 Minutes Generator로 전달됨.
2. **1차 분석**: 녹음 종료 시점에 STT + Diarization을 오디오 전체에 대해 1회 실행.
3. **회의록 생성(태깅 포함)**: Minutes Generator가 회의록 초안을 작성하면서, 문단 단위로 `[논의 외]` / `[아이디어1]` / `[아이디어2]` ... 태그를 함께 붙임 (3.1 참고).
4. **Gate 0 — 회의록 확정**: 생성된 회의록(태그 포함)을 사람이 읽고 본문/태그를 필요한 만큼 수정. 문법적 어색함은 무시.
5. **아이디어 선정**: 확정된 태그를 기준으로 `[논의 외]`를 제외한 나머지 중 구체적으로 논의가 진전된 아이디어가 있는지 판정 (3.2 참고).
   * 없음 → 회의록만 출력, 파이프라인 종료.
   * 하나 확정됨 → 해당 아이디어 태그가 붙은 내용만 이후 단계로 전달.
6. **기획서 생성**: Spec Worker가 선정된 아이디어의 태그 내용만 입력받아 기획서 작성(3.3, 장르 포함) → Spec Validator.
7. **Task 분류**: 선정된 아이디어 태그 내용 + 완성된 기획서(장르 등)를 함께 입력으로, Task Classifier가 하위 산출물 3개 플래그와 브리핑을 판단 — 장르가 판단 임계값에 영향(3.4). `needs_image`가 True면 이어서 Image Style Planner가 공유 스타일 가이드 생성 (Gate 1 진입 전 완료).
8. **Gate 1 — 산출물 확정**: 완성된 기획서는 접힌 카드로 표시하고, 클릭하면 펼쳐지는 방식(줄글로 바로 노출하지 않음). 설정집/이미지 N개/비교대상 검색 보고서는 카드(체크박스+편집 가능한 브리핑)로 노출. 이미지는 개수가 가변이라 카드 목록도 동적으로 늘어남.
9. **Fan-out 실행**: 기획서는 이미 완료된 상태이므로 제외하고, 설정집/이미지 각각/비교대상 검색 보고서만 병렬 생성 + Validator 검증 루프. 트랙별 독립 진행 상태 표시.
10. **Gate 2 — 에스컬레이션**: 재시도 한도 초과 or 기술적 실패 항목만 별도 표시, 수동 확인/승인.
11. **최종 화면**: Aggregator 대시보드 + ZIP 다운로드(기획서 PDF, 설정집 PDF, 이미지 파일들, 비교대상 검색 보고서 PDF, 원본 회의록).

Gate 0: 회의록 자유 수정(본문+태그), 별도 하이라이트·이상탐지 없음. Gate 1: 기획서는 검토만(재생성 트리거 아님), 나머지 3개는 브리핑 수정 시 해당 Worker만 재실행.

---

## 3. 산출물 분류 로직

### 3.1. 회의록 태깅

Minutes Generator가 회의록을 생성하며 문단 단위로 다음 태그 중 하나를 붙임.

* `[논의 외]`: 게임 아이디어 논의 자체가 아닌 내용 (잡무 지시, 기존 제품 현황 보고 등).
* `[아이디어N]`: N번째로 제시된 아이디어에 대한 내용. 발언 시점과 무관하게 **어느 아이디어에 대한 내용인가**로 분류 — 예를 들어 아이디어를 처음 제시하는 발언 안에 이미 상세 메커니즘 설명이 포함돼 있으면, 그 부분도 해당 아이디어 태그로 들어감 (시점이 아니라 소속 기준).

### 3.2. 아이디어 선정

* `[논의 외]`를 제외한 태그된 내용을 보고, 그중 하나의 아이디어에 대해 논의가 구체적으로 진전됐는지(예: "이거 괜찮네, 자세히 말해봐" 같은 결정 신호 이후 상세 설명이 이어졌는가) 판단.
* 진전된 아이디어가 없음(여러 안이 제시만 되고 좁혀지지 않음) → 회의록만 출력.
* 하나로 좁혀짐 → 해당 아이디어 태그의 내용만 이후 전체 파이프라인의 입력이 됨.
* **2개 이상의 아이디어가 동시에 구체화되는 경우는 스코프 밖.** 구조상 여러 아이디어를 처리하지 못하게 막아두진 않지만(태그 체계 자체는 아이디어 개수에 열려 있음), 이번 POC 데모에서는 하나로 좁혀지는 케이스만 지원.

### 3.3. 기획서 — 선정된 아이디어 기준 작성

* 3.2에서 하나의 아이디어가 선정되면 **기획서는 무조건 생성 확정.**
* 입력은 선정된 아이디어 태그의 내용만 — `[논의 외]`와 다른 아이디어 태그 내용은 포함하지 않음.
* 장르를 포함한 `REQUIRED_SPEC_FIELDS`(4.9)를 채움. 이 장르 값은 3.4의 하위 산출물 판단에서 재사용됨(중복 추출 없음).

### 3.4. 하위 산출물 — 3개 독립 플래그

Task Classifier는 **선정된 아이디어 태그 내용 + 완성된 기획서**를 함께 입력받아 판단.

* `needs_setting_doc`: 캐릭터·장소·유물 등 개별 엔티티의 상세 속성이나 사건/관계 논의가 텍스트 요약만으로는 담기 부족할 만큼 나왔는가.
* `needs_image`: 아래 두 조건을 모두 충족해야 True. 대상은 장소/풍경/건축물 등 환경 요소로 한정 — 인물/캐릭터 초상은 후보에서 제외 (DALL-E 3는 동일 캐릭터를 여러 번 그려도 생김새 일관성을 유지할 수 없어, 인물 이미지는 확정 정보로서의 신뢰도가 낮음).
  1. 구체적 대상(장소/사물)이 최소 1개 이상 언급됨
  2. 그 대상에 대한 시각적 특징(형태·재질·색감·분위기 등 무엇이든) 묘사가 함께 언급됨 — 명시적 톤 단어("다크판타지" 등)가 아니어도, 대상 묘사 자체에 시각 정보가 담겨 있으면 충족(예: "돌담이 무너진 폐허")
  둘 중 하나만 있으면 False.
* `needs_reference_report`: 메커니즘/시스템에 대한 구체적 논의가 있었고, 3.5 기준으로 비교 조사할 가치가 있는가.

**장르에 따른 판단 조정**: 기획서의 장르 필드를 참고해 임계값 방향을 조정 — 예: RPG류면 캐릭터 관련 언급의 `needs_setting_doc` 판단 문턱을 상대적으로 낮게, SLG/전략류면 캐릭터보다 세력/자원 시스템 관련 논의를 우선 고려. (구체적 가중치 수치는 별도 확정 필요 — 현재는 방향성만 정의된 상태.)

세 플래그는 서로 독립이며, 하나만 True인 조합도 정상 케이스로 처리.

### 3.5. 비교대상 검색 보고서 — 논의된 시스템 목록과 스코프 필터

* `discussed_systems: List[str]`: 구체적으로 논의된 메커니즘/시스템 중, **게임마다 구현 방식이 실제로 갈리는 것만** 포함. WASD 이동처럼 업계에서 사실상 표준화되어 게임 간 차이가 거의 없는 요소는 구체적으로 논의·확정됐어도 제외.
* **최대 5개로 캡**: 스코프 필터를 통과한 시스템이 5개를 초과하면, 논의 중요도(회의에서 다뤄진 비중·상세도)를 기준으로 상위 5개만 남기고 나머지는 제외.
* 판단은 다른 분류 항목과 동일하게 Luna 티어(5.1). 오분류 가능성은 있으나 Gate 1에서 사람이 체크박스로 최종 확인.
* 리스트가 비어 있으면 `needs_reference_report`를 False로 처리하거나, Gate 1에서 사람이 직접 추가.
* 여러 시스템이 나와도 보고서는 1개로 유지(시스템별 섹션으로 구성, 4.9).

### 3.6. Gate 1에서의 사람 개입

기획서는 완성본을 그대로 검토(수정 시 재생성 아님). 설정집/이미지/비교대상 검색 보고서는 체크박스(켜기/끄기) + **추출된 브리핑 텍스트(편집 가능)** 로 노출. 이 브리핑이 실제로 Worker에게 전달되는 생성 지시문.

---

## 4. 핵심 모듈별 기능 상세 명세

### 4.1. 전체 파이프라인 순서

```
STT (Whisper) → Diarization (pyannote/WhisperX)   ※ 디버그: 텍스트 직접 입력 시 이 두 단계 건너뜀
       │
       ▼
Minutes Generator (회의록 생성 + [논의 외]/[아이디어N] 태깅)
       │
       ▼
Gate 0 (본문+태그 확정)
       │
       ▼
아이디어 선정 판정 (구체화된 아이디어 1개 식별, 없으면 회의록만 출력하고 종료)
       │
       ▼
Spec Worker (선정 아이디어 내용만 입력) → Spec Validator
       │
       ▼
Task Classifier (선정 아이디어 내용 + 완성된 기획서 입력, 장르 참고해 3개 플래그+브리핑 판단)
       │ (needs_image=True인 경우만)
       ▼
Image Style Planner
       │
       ▼
Gate 1 (기획서 검토 + 나머지 3개 카드 확정)
       │
       ▼
Fan-out: 설정집 / 이미지×N(동적) / 비교대상 검색 보고서 — Worker-Validator 병렬 실행
       │
       ▼
Gate 2 (에스컬레이션, 필요시) → Aggregator → ZIP
```

### 4.2. 전처리 및 분석 모듈

* **음성 전사 (STT)**: OpenAI Whisper API 또는 로컬 faster-whisper. 화자 분리 기능 없음 — STT는 텍스트 변환만 담당.
* **화자 분리 (Diarization)**: pyannote-audio 또는 WhisperX를 별도로 붙여 STT 결과와 타임스탬프 기준 병합.
* **사내 DB RAG 검색**: 회의 중 언급된 전문 용어, 기존 서비스 명세, 디자인 가이드라인을 사내 Vector DB에서 검색해 컨텍스트로 제공.
* **Minutes Generator**: STT+Diarization 결과로 회의록 초안(요약/결정사항/미결 안건) 생성 + 문단별 `[논의 외]`/`[아이디어N]` 태깅을 한 번에 처리.
* **Spec Worker**: 선정된 아이디어 태그 내용만 입력받아 기획서 작성 (3.3).
* **Task Classifier/Router**: 선정된 아이디어 태그 내용 + 완성된 기획서를 입력으로:
  * 3.4의 3개 독립 플래그(`needs_setting_doc`, `needs_image`, `needs_reference_report`) 판단 (장르 참고)
  * `needs_image == True`인 경우, 이미지 대상 개수와 각 대상을 리스트로 추출 (환경 요소로 한정)
  * 각 활성 산출물의 브리핑(`task_briefs`) 추출
  * `needs_reference_report == True`인 경우 `discussed_systems` 리스트 추출 (3.5 스코프 필터 적용, Luna 티어)

### 4.3. Gate 0 — 회의록 확정

* LangGraph `interrupt_before`로 아이디어 선정 판정 노드 진입 전 정지.
* UI: 회의록 본문 및 태그 자유 수정. 문법 교정 불필요, 사실 오류(숫자/고유명사 등)만 확인.

### 4.4. Gate 1 — 산출물 확정

* LangGraph `interrupt_before`로 Fan-out 진입 전 정지.
* UI: 완성된 기획서(접힌 카드, 클릭 시 펼쳐짐 — 줄글로 바로 노출하지 않음) + 설정집/이미지 N개/비교대상 검색 보고서 카드(조건부 노출, 체크박스 + 편집 가능한 브리핑). 이미지 카드에는 공유 스타일 가이드(4.6 참고)도 편집 가능하게 노출.

### 4.5. 산출물 생성 및 검증

| 산출물 | Worker | Validator | 형식 |
| --- | --- | --- | --- |
| **기획서 (Spec)** | 선정된 아이디어 내용 + RAG 컨텍스트로 고정 템플릿(`REQUIRED_SPEC_FIELDS`: 장르, 핵심 메커니즘, 진행 방식, 타겟 플랫폼, 핵심 게임플레이 루프, 타겟 유저층, 차별점·셀링포인트)을 채우는 방식. 세부 관계는 담지 않고 **요약 수준** 유지. Mermaid 흐름도 포함. | 필수 슬롯 누락 여부(언급 안 된 항목은 "TBD"로 명시돼야 함) / 핵심 요구사항 누락 여부 / Mermaid 문법 오류 | PDF, 1개 고정 |
| **설정집** | 상단: 언급된 캐릭터/장소/유물 등 엔티티별 **설정 시트**. 하단: 언급된 사건을 **사건 표**로 정리. | 설정 시트/사건 표 각각 언급된 내용이 누락 없이 포함됐는가 (완결성 체크. 영속 DB나 모순 검증은 스코프 아님) | PDF (설정 시트 + 하단 사건 표) |
| **이미지** | Image Style Planner가 만든 공유 스타일 가이드를 반영해, 대상별로 독립 프롬프트 작성 후 DALL-E 3 호출. 개수는 가변. | 별도 vision 지원 모델 호출: 금지 요소 포함 여부, 톤앤매너 일치 여부, 공유 스타일 가이드 반영 여부 | 이미지 파일, 대상 개수만큼 |
| **비교대상 검색 보고서** | `discussed_systems`(3.5 스코프 필터 적용됨) 각각을 웹 검색으로 조사해 유사 사례를 찾고, 시스템별 섹션으로 구성된 문서 1개 작성 | 각 섹션이 실제 검색 결과에 근거했는가 | PDF, 1개 고정 |

### 4.6. 이미지 전용 — Style Planner 서브노드

```
Task Classifier (분류+브리핑, needs_reference_report=True 시 discussed_systems 포함 — Luna)
       │ (needs_image=True인 경우에만)
       ▼
Image Style Planner (톤/설정 → 공유 스타일 가이드 문자열 — Luna)
       │
       ▼
Gate 1 (스타일 가이드 포함 전체 브리핑 편집)
       │
       ▼
Image Worker × 대상 개수 (동적 Fan-out) → Image Validator × 대상 개수
```

* DALL-E 3는 seed 고정이나 레퍼런스 이미지 컨디셔닝을 지원하지 않아, 완전한 시각적 일관성은 보장되지 않음. "프롬프트 레벨 일관성" 정도로 기대치 설정.
* Style Planner는 이미지 트랙에만 존재.

### 4.7. 이미지 — 동적 Fan-out

* LangGraph `Send` API로 런타임에 `image_tasks` 개수만큼 Worker-Validator 쌍 동적 생성 (기획서/설정집/비교대상 검색 보고서는 정적 처리).
* 이미지별로 재시도/에스컬레이션 독립 처리.

### 4.8. 자기 반성 루프, 에스컬레이션, 기술적 실패 안전망

* **내용 품질 실패**: Critique를 Worker에 주입해 재실행, `retry_count < MAX_RETRIES(예: 2회)`까지 반복.
* **한도 초과**: Gate 2로 에스컬레이션.
* **기술적 실패** (API 에러, rate limit, 타임아웃, 렌더링 실패, 웹 검색 실패 등): 각 노드를 예외 처리로 감싸, 실패 시 해당 트랙만 `technical_failure` 상태로 표시하고 나머지는 계속 진행. Aggregator도 동일하게 예외 처리.

### 4.9. 취합 및 최종 출력, 산출물 템플릿 정의

* 검증 완료/승인된 산출물을 대시보드로 렌더링.
* ZIP 다운로드: 기획서 PDF, 설정집 PDF, 이미지 파일들, 비교대상 검색 보고서 PDF, 원본 회의록(태그 포함).

**기획서 (`REQUIRED_SPEC_FIELDS`)**: 장르 / 핵심 메커니즘 / 진행 방식 / 타겟 플랫폼 / 핵심 게임플레이 루프 / 타겟 유저층 / 차별점·셀링포인트. 언급 안 된 항목은 "TBD"로 명시.

**설정집 — 설정 시트 컬럼**: 이름 / 유형 / 핵심 속성(자유 서술) / 언급 맥락. 강제 슬롯 없이 **언급된 필드만 표시**, 언급 안 된 속성은 행 자체를 생략.

**설정집 — 사건 표 컬럼**: 발생 시점 / 사건 / 관련 엔티티 / 결과·관계. 발생 시점은 회의에서 명시된 경우만 채우고, 대부분은 비어 있을 것으로 예상(빈칸 허용).

**비교대상 검색 보고서 — 시스템별 섹션**: 시스템명 / 논의 요약 / 유사 사례 표(게임명·유사점·차이점·출처) / 종합 코멘트(선택). 최대 5개 시스템(3.5 캡 적용).

---

## 5. 기술 스택

### 5.1. LLM 모델 티어 (POC 비용 최적화)

| 노드 | 티어 | 이유 |
| --- | --- | --- |
| Minutes Generator (태깅 포함) | GPT-5.6 Luna | 요약 + 태깅, 구조화된 작업 |
| Spec Worker/Validator | GPT-5.6 Luna | 고정 템플릿 채우기/체크리스트 검증 |
| Task Classifier — 분류/브리핑/`discussed_systems` | GPT-5.6 Luna | 구조화된 분류 수준 (신규성 판단 아님) |
| Image Style Planner | GPT-5.6 Luna | 이미 파악된 톤을 문구로 정리하는 좁은 작업 |
| 설정집/비교대상 검색 보고서 Worker·Validator | GPT-5.6 Luna | 구조화된 생성/체크리스트 검증 수준 |
| 이미지 Validator | GPT-5.6 Terra | vision 입력 필요 (Luna의 vision 지원 여부 불확실) |
| (Sol 미사용) | — | 코딩/추론이 매우 무거운 최상위 티어라 POC 범위 밖 |

### 5.2. 기타 스택

* **Language & Frameworks**: Python 3.11+, LangGraph, LangChain
* **STT/Diarization**: Whisper API 또는 faster-whisper + pyannote-audio/WhisperX
* **Image Generation**: DALL-E 3 (로컬 GPU 부재로 클라우드 API 방식 채택)
* **웹 검색**: 비교대상 검색 보고서용
* **Vector DB (RAG)**: ChromaDB 또는 FAISS (text-embedding-3-small)
* **UI**: Streamlit (Checkpointer 인터럽트 핸들링 연동)

### 5.3. LangGraph vs LangChain 역할 분담

**LangGraph — 전체 오케스트레이션(상태·분기·게이트)**
* 그래프 구조: 전체 노드(Minutes Generator, 아이디어 선정, Spec Worker/Validator, Task Classifier, Style Planner, 각 하위 Worker/Validator, Aggregator)와 그 사이 흐름
* 조건부 분기: `needs_image`/`needs_setting_doc`/`needs_reference_report` 판정에 따른 라우팅, 아이디어 미선정 시 조기 종료 분기
* `Send` API: 이미지 개수만큼 동적 Fan-out
* `interrupt_before`: Gate 0 / Gate 1 / Gate 2
* Checkpointer: 게이트 대기 중 상태 저장/재개
* `AgentMeetingState` TypedDict와 reducer(`operator.or_`, `operator.add`)
* 비교대상 검색 보고서의 검색 루프도 `ToolNode` + 조건부 엣지로 그래프 안에 명시적으로 포함 — LangChain의 `AgentExecutor` 같은 블랙박스 루프로 감싸지 않음. 그래야 검색 실패도 4.8의 기술적 실패 처리 방식과 일관되게 다뤄짐.

**LangChain — 개별 노드 내부 로직**
* 각 노드의 LLM 호출: `prompt | llm | output_parser` 형태의 Runnable 체인 (Task Classifier는 Pydantic 파서로 플래그/브리핑을 구조화 출력)
* RAG 체인: retriever(ChromaDB/FAISS) + prompt + LLM
* Embeddings/VectorStore 관리 (text-embedding-3-small)
* 비교대상 검색 보고서 Worker의 웹 검색 tool 바인딩(`bind_tools`)

### 5.4. LangGraph State 스키마

```python
from typing import Annotated, Dict, List, Optional, TypedDict
import operator

class TaskFeedback(TypedDict):
    is_valid: bool
    critique: str
    retry_count: int

class TaskBrief(TypedDict):
    enabled: bool
    brief: str

class ImageTask(TypedDict):
    subject: str
    enabled: bool
    brief: str
    image_url: Optional[str]
    validation: Optional[TaskFeedback]

class AgentMeetingState(TypedDict):
    # 1. 입력 및 전처리
    audio_path: str
    transcript: str
    rag_references: List[str]

    # 2. 회의록 (Gate 0) — 태그 포함
    meeting_minutes_draft: str  # [논의 외]/[아이디어N] 태그 포함
    meeting_minutes_confirmed: Optional[str]

    # 3. 아이디어 선정
    selected_idea_tag: Optional[str]  # 예: "아이디어2", 없으면 None (회의록만 출력)

    # 4. 기획서 (Task Classifier보다 먼저 생성)
    spec_document: Optional[str]

    # 5. 분류/브리핑 (Gate 1)
    discussed_systems: List[str]
    image_style_guide: Optional[str]
    task_briefs: Dict[str, TaskBrief]  # key: "setting_doc" | "reference_report"
    image_tasks: List[ImageTask]

    # 6. 하위 산출물 결과
    setting_doc: Optional[str]
    reference_report: Optional[str]

    # 7. 검증/에스컬레이션/기술적 실패
    validation_status: Annotated[Dict[str, TaskFeedback], operator.or_]
    escalated_tasks: Annotated[List[str], operator.add]
    technical_failures: Annotated[Dict[str, str], operator.or_]
```

---

## 6. 단계별 개발 일정

* **Phase 1**: STT+Diarization 파이프라인, Minutes Generator(태깅 포함), Gate 0 구현.
* **Phase 2**: 아이디어 선정 로직, Spec Worker-Validator 구현.
* **Phase 3**: Task Classifier(장르 참고 판단+`discussed_systems`), Image Style Planner, Gate 1, 설정집/이미지 동적 Fan-out(Send API)/비교대상 검색 보고서(웹 검색 ToolNode) Worker-Validator 구현.
* **Phase 4**: 기술적 실패 예외 처리, Gate 2 에스컬레이션 UI, Aggregator + ZIP 패키징.

---

## 7. 에이전트 4대 구조 및 안전망 자가 점검

| 항목 | 충족 여부 | 비고 |
| --- | --- | --- |
| 계획(Planning) | O | 아이디어 선정→기획서 우선 생성이라는 순서 설계 + Image Style Planner(이미지 트랙 한정) |
| 툴 사용 | O | Whisper, Diarization 모델, RAG 벡터 검색, DALL-E 3, 웹 검색(비교대상 검색 보고서) |
| 반성/자가 반추 | O | Worker-Validator Critique 기반 재시도 루프 |
| 멀티 에이전트 | O | Minutes Generator / Spec Worker-Validator / Task Classifier / Image Style Planner / 설정집·이미지·보고서 Worker-Validator / Aggregator |
| 내용 품질 실패 안전망 | O | 재시도 + 에스컬레이션(Gate 2) |
| 기술적 실패 안전망 | O | 노드별 예외 처리, 트랙별 격리, `technical_failures` 상태로 구분 표시 |

---

## 8. 기대 효과

1. **동적 라우팅**: 아이디어 태그 기반 선정 + 3개 독립 플래그 + 이미지 가변 개수 `Send` 동적 Fan-out으로, 회의 내용에 따라 파이프라인 진행 경로 자체가 달라짐.
2. **신뢰성 보장**: 내용 품질 실패(Self-Correction)와 기술적 실패(예외 격리)를 구분한 이중 안전망.
3. **비용 효율**: 대부분 Luna, 정확도가 중요한 지점(이미지 vision 검증)만 Terra로 선별 적용.
4. **맥락 오염 방지**: 회의록을 아이디어 단위로 태깅해, 브레인스토밍/잡담이 기획서·하위 산출물 판단에 섞여 들어가는 것을 원천 차단.
5. **실무형 산출물 연계**: 요약형 기획서(장르가 하위 판단 기준이 됨) + 설정집 + 근거 기반 비교대상 검색 보고서로, 텍스트 요약을 넘어선 실제 기획 라이프사이클 지원.

---

## 9. 향후 개선 방향

* **이미지 일관성**: 현재는 로컬 GPU가 없어 DALL-E 3(클라우드 API) + 프롬프트 레벨 일관성으로 타협. 추후 로컬/클라우드 GPU 환경이 갖춰지면 Stable Diffusion/Flux 계열 + IP-Adapter 등으로 전환해 시각적 일관성을 실질적으로 강화 가능.
* **다중 아이디어 처리**: 현재 스코프 밖인 "2개 이상 아이디어 동시 구체화" 케이스 — 태그 체계 자체는 열려 있으므로, 추후 여러 기획서/산출물 세트를 병렬로 처리하는 구조로 확장 가능.
* **장르별 임계값 수치화**: 3.4의 장르 기반 판단 조정은 현재 방향성만 정의된 상태 — 실제 가중치/임계값은 추후 확정 필요.
