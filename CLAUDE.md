# AutoMeeting Agent

LangGraph 기반 Human-in-the-Loop 다중 모달 회의 산출물 자동화 에이전트.
전체 스펙은 [AutoMeeting_Agent_기획서_v3.md](AutoMeeting_Agent_기획서_v3.md)이며, 항상 이 문서를 최우선 기준으로 따를 것.

## 현재 단계

Phase 1~3 구현됨: 녹음(STT) 또는 디버그 텍스트 입력 → RAG Retrieval(Phase 4,
아래 참고) → Minutes Generator → Gate 0(`interrupt_before`) → 아이디어
선정(3.2) → Spec Worker/Validator(3.3) → Task Classifier(3.4/3.5) →
(needs_image면) Image Style Planner(4.6) → Gate 1(`interrupt_before`) →
Fan-out: 설정집/비교대상 검색 보고서/이미지×N(Send 동적 생성) — 전부
spec과 동일하게 그래프 레벨 Worker→Validator→조건부 엣지로 재시도한다.
이미지는 Send 기반 N-way 병렬이라 재시도까지 포함한 모든 홉을 매번 새
`Send`로 명시해야 브랜치가 안 섞인다([src/nodes/image.py](src/nodes/image.py)
docstring 참고) - 한때는 이 문제를 피하려고 이미지별 전체 처리(생성+검증+
재시도)를 노드 하나 안에 캡슐화했었으나, 그러면 같은 슈퍼스텝에 파견된
설정집/검색 보고서 Validator가 이미지 "검증"까지 끝나야만 시작할 수 있어서
(LangGraph Pregel 슈퍼스텝 동기화) 불필요하게 오래 대기했다(사용자 실측
지적으로 발견) - 그래프 레벨로 다시 쪼개 이 대기를 없앴다(실측 확인:
분리 전에는 이미지 검증까지 다 끝나야 설정집 Validator가 시작했는데,
분리 후에는 이미지 생성만 끝나면 설정집 Validator가 동시에 시작함).
Diarization, Gate 2 에스컬레이션 UI, Aggregator/ZIP은 아직 없음(Phase 4
나머지 범위는 스펙 6장 참고).

**Phase 4 STT 구현됨(Diarization 제외)**: [src/stt.py](src/stt.py) — OpenAI
오디오 트랜스크립션 API(`STT_MODEL`, 기본 `whisper-1`) 직접 호출. UI 기본
화면이 `st.audio_input()`으로 브라우저 녹음을 받아 녹음 종료 시 자동으로
STT + 그래프 실행까지 진행한다([src/ui/app.py](src/ui/app.py)). 화면 우측
하단 "DEBUG" 버튼을 누르면 기존 텍스트 직접 입력 화면(`session_state`의
`debug_mode` 플래그)으로 전환된다. Diarization(화자 분리)은 의도적으로
스코프에서 뺐다 - Minutes Generator의 태깅 로직이 "발언 시점이 아니라
내용이 어느 아이디어에 속하는가"만 보므로 화자 식별이 결과에 영향을 주지
않고, pyannote 등 추가 모델 인증·정렬 로직 대비 실익이 낮다고 판단했다.

**Phase 4 RAG 구현됨**: [src/rag.py](src/rag.py) — ChromaDB(인메모리) +
`text-embedding-3-small`로 사내 코퍼스([sample_data/glossary.md](sample_data/glossary.md)
용어집, [sample_data/game_profiles.md](sample_data/game_profiles.md) 기존
게임 프로필)를 인덱싱. 각 파일을 "## " 섹션 단위로 쪼개 문서화하고 `type`
메타데이터(glossary/game_profile)를 붙인다. [src/nodes/rag_retrieval.py](src/nodes/rag_retrieval.py)
가 트랜스크립트 확보 직후(Minutes Generator보다 먼저) 트랜스크립트를 발화
단위로 쪼개 검색하고(`search_transcript` - 통째로 쿼리하면 관련도 신호가
희석되는 문제가 실측 확인됨), 의미 검색이 놓치는 짧은 고유명사 오탈자는
`difflib` 표면형 매칭으로 보완해(`RAG_FUZZY_MIN_RATIO`) 상위 `RAG_TOP_K`개
(기본 10, 관련도 `RAG_SCORE_THRESHOLD` 미만은 제외) 문서를 `rag_references`에
채운다.
Minutes Generator/Spec Worker/설정집 Worker/비교대상 검색 보고서 Worker
4곳이 `rag_references`를 프롬프트에 포함해 참고한다(Task Classifier/Image
Style Planner는 대상 아님) — Minutes Generator는 이걸로 STT 오인식 가능성이
있는 사내 고유명사를 용어집 표기로 교정하고(확신 없으면 원문 유지), 나머지
3곳은 용어 표기 일관성 참고용으로만 쓰고 새 사실을 채우는 근거로는 쓰지
않는다(각 노드 프롬프트에 명시). 테스트용 STT 오탐지 샘플:
[sample_data/sample_meeting_phase4.txt](sample_data/sample_meeting_phase4.txt)
("구스엔진"→"거스엔진" 등 사내 고유명사 위주로 오인식 섞음).

모델 티어는 현재 전 노드 GPT-5.6 Luna로 통일(스펙 5.1 문서는 미갱신 상태, 이미지
Validator 원안은 Terra였으나 지금은 Luna) — 노드별로 문제가 확인되면 해당 노드의
`*_MODEL` 환경변수만 개별적으로 상향할 것. 비교대상 검색 보고서/설정집
Validator는 한때 Terra로 올려봤으나(전달된 근거를 놓치거나, 같은 문서를 두고
재시도마다 스스로 모순되는 판정을 내리는 사례가 실측 확인됨) 근본 해결책이
아니었다 - 결국 두 Validator 모두 검증 범위를 "브리핑/목록 대비 완결성·형식"
같은 기계적 대조로 대폭 축소해서 판단 신뢰도 문제 자체를 없앴고, 그 뒤로는
Luna로도 충분해 원복했다. 각 노드 모듈 docstring 참고.

**중요**: GPT-5.6 계열(Luna/Terra/Sol)은 reasoning 모델이라 `temperature`를
지원하지 않고, 기본 reasoning_effort("medium")로는 구조화 출력/tool 바인딩을
`/v1/chat/completions`에서 쓸 수 없다(실제 API 에러로 확인: "Function tools
with reasoning_effort are not supported ... set reasoning_effort to 'none'").
그래서 모든 ChatOpenAI 인스턴스는 [src/llm.py](src/llm.py)의
`build_chat_model()`을 통해서만 생성하고(`reasoning_effort="none"` 고정,
`REASONING_EFFORT` 환경변수로 오버라이드 가능), 어디서도 `ChatOpenAI(...)`를
직접 호출하거나 `temperature=`를 넘기지 않는다.

**5.4 스키마에서 벗어난 부분 2건**:
1. `image_tasks`에 커스텀 reducer(`_merge_image_tasks`, subject 키 기준 병합)를
   추가함. 원안은 reducer 없는 plain `List[ImageTask]`지만, 4.7의 `Send` 기반
   동적 Fan-out은 이미지별 브랜치가 동시에 이 채널에 쓰기 때문에 reducer가
   없으면 LangGraph가 `InvalidUpdateError`를 낸다(실측 확인, 대안 없음).
2. `reference_report_messages`(검색 대화 기록, `add_messages` 리듀서) 필드를
   추가함. 비교대상 검색 보고서 Worker/Validator를 그래프 레벨 노드로 분리
   하면서(아래 참고), 재시도 사이에 검색 대화가 끊기지 않게 이어가려면 이
   기록이 그래프 노드 호출 사이에도 살아있어야 해서 필요해졌다.

자세한 내용은 [src/state.py](src/state.py) 주석 참고.

**5.2 스택에서 벗어난 부분 1건**: 이미지 생성 모델을 DALL-E 3에서 `gpt-image-2`
(GPT 이미지 모델)로 대체함. DALL-E 3가 API에서 폐지돼 `model 'dall-e-3' does
not exist` 에러가 실제로 발생해 확인했다. GPT 이미지 모델은 URL이 아니라
base64(`b64_json`)만 반환하므로, [src/nodes/image.py](src/nodes/image.py)가
`image_url` 필드에 data URI(`data:image/png;base64,...`)를 넣도록 바꿨고,
[src/ui/app.py](src/ui/app.py)의 `resolve_image_source()`가 렌더링 시 이를
다시 bytes로 디코드한다. `gpt-image-1`이 아니라 `gpt-image-2`를 기본값으로
고른 이유는 더 최신인데 공식 가격표 기준 더 저렴하기 때문(image input $8 vs
$10, output $30 vs $40 / 1M, 2026-09 기준). `IMAGE_MODEL` 환경변수로 다른
GPT 이미지 모델(`gpt-image-1`/`gpt-image-1-mini`/`gpt-image-1.5` 등)로 바꿀
수 있다.

## 환경

- Python 3.14 고정 (`.python-version`, `pyproject.toml`의 `requires-python` 참고). 스펙 5.2는 "3.11+"로 표기돼 있으나 이 프로젝트는 3.14로 고정.
- 모든 소스/설정/문서 파일은 UTF-8. 특히 한글 텍스트(회의록, 태그, 프롬프트)를 다루는 파일 입출력은 Windows 기본 인코딩(cp949)에 걸리지 않도록 `open(..., encoding="utf-8")`을 항상 명시할 것.

## 핵심 참고 위치

- 모델 티어: 스펙 5.1 (노드별 Luna/Terra 배정 원안). 현재 구현은 전부 Luna로 통일(위 "현재 단계" 참고).
- State 스키마: 스펙 5.4의 `AgentMeetingState` / `TaskFeedback` / `TaskBrief` / `ImageTask`를 [src/state.py](src/state.py)에 그대로 구현. 필드를 추가/변경할 경우 스펙 5.4와 동기화할 것.
- 회의록 태깅 규칙: 스펙 3.1 (`[논의 외]` / `[아이디어N]`, 발언 시점이 아닌 내용 소속 기준).
- 아이디어 선정: 스펙 3.2 (태그 소속 기준으로 소급 판단, 하나로 좁혀지는 케이스만 지원).
- 기획서: 스펙 3.3/4.9의 `REQUIRED_SPEC_FIELDS` 7개, TBD 명시 규칙, Mermaid 포함.
- 재시도 루프: 스펙 4.8 (critique 기반, `MAX_RETRIES`) — Worker/Validator 노드는 1회 실행만 담당하고, 루프 자체는 그래프의 조건부 엣지가 담당(스펙 5.3 역할 분담). 공용 로직은 [src/retry.py](src/retry.py).
- 하위 산출물 분류: 스펙 3.4/3.5 — needs_setting_doc/needs_reference_report는 `task_briefs[...].enabled`로, needs_image는 `image_tasks`가 비어있지 않은 것으로 표현(5.4 스키마에 별도 bool 필드가 없음). 장르 기반 임계값 조정은 [src/nodes/classifier.py](src/nodes/classifier.py) 프롬프트에 휴리스틱으로 반영(정확한 가중치는 9장 기준 미확정).
- 이미지 Fan-out: 스펙 4.7 — `Send`로 이미지별 병렬 파견, Worker/Validator는 spec과 동일하게 그래프 레벨 노드([src/nodes/image.py](src/nodes/image.py)). 재시도를 포함한 모든 홉(worker→validator, validator→worker 재시도)을 매번 새 `Send`로 명시해 브랜치 격리를 유지한다 - 자세한 이유(예전엔 서브그래프 캡슐화였다가 왜 다시 쪼갰는지)와 격리가 실제로 안전한지의 실측 검증은 image.py 모듈 docstring 참고.
- 웹 검색 루프: 스펙 5.3 — `AgentExecutor` 등 블랙박스 대신 `ToolNode` + 조건부 엣지로 그래프 안에 명시(agent ⇄ tools 반복, 비교대상 검색 보고서 Worker 노드 내부의 서브그래프). [src/nodes/reference_report.py](src/nodes/reference_report.py) — Worker/Validator 자체는 spec과 동일하게 그래프 레벨 노드(재시도 시 검색 대화는 `reference_report_messages`로 이어감). 검색 백엔드는 `ddgs` 패키지([src/tools.py](src/tools.py), `backend="duckduckgo,brave,google,startpage"`, API 키 불필요) — 원래 Tavily였으나 무료 티어 rate limit에 자주 걸려 교체함. `ddgs` 기본값("auto")은 grokipedia/wikipedia 등 이 용도에 안 맞거나 느린 엔진까지 다 시도해 쿼리당 5~6초씩 걸리고, `backend="duckduckgo"` 단일 엔진은 폴백이 없어 그 엔진이 막히면(실측 확인 - IP/세션 단위로 몇 번만에 막힘) 바로 예외를 던짐 - 그래서 일반 웹검색에 맞는 엔진 4개만 콤마로 나열해 명시적 폴백 목록으로 고정함(연속 20회 호출 실측 19/20 성공). 자세한 경위는 src/tools.py 모듈 docstring 참고. 다른 제공자로 바꾸려면 이 파일만 교체.

## 구조

- [src/state.py](src/state.py): LangGraph 상태 스키마 (+ image_tasks 커스텀 reducer, reference_report_messages)
- [src/llm.py](src/llm.py): ChatOpenAI 생성 공용 헬퍼 (`reasoning_effort="none"` 고정 - 위 "중요" 참고)
- [src/retry.py](src/retry.py): Worker-Validator 재시도 루프 공용 헬퍼 (spec/setting_doc/reference_report/이미지 전부 그래프 조건부 엣지에서 공유)
- [src/tagging.py](src/tagging.py): 태그별 문단 추출 유틸 (Spec Worker, Task Classifier가 공용)
- [src/tools.py](src/tools.py): 웹 검색 도구 (`ddgs` 패키지, duckduckgo/brave/google/startpage 폴백)
- [src/rag.py](src/rag.py): RAG 인덱싱/검색 (ChromaDB + text-embedding-3-small, 스펙 5.2)
- [src/nodes/rag_retrieval.py](src/nodes/rag_retrieval.py): RAG Retrieval 노드 - transcript -> rag_references (스펙 5.4)
- [src/stt.py](src/stt.py): STT (OpenAI 오디오 트랜스크립션 API, diarization 제외)
- [src/nodes/minutes.py](src/nodes/minutes.py): Minutes Generator 노드 (스펙 3.1, 4.2) - rag_references로 STT 용어 교정
- [src/nodes/idea_select.py](src/nodes/idea_select.py): 아이디어 선정 노드 (스펙 3.2)
- [src/nodes/spec.py](src/nodes/spec.py): Spec Worker/Validator 노드 (스펙 3.3, 4.9)
- [src/nodes/classifier.py](src/nodes/classifier.py): Task Classifier 노드 (스펙 3.4, 3.5)
- [src/nodes/style_planner.py](src/nodes/style_planner.py): Image Style Planner 노드 (스펙 4.6)
- [src/nodes/setting_doc.py](src/nodes/setting_doc.py): 설정집 Worker/Validator (스펙 4.5, 4.9)
- [src/nodes/reference_report.py](src/nodes/reference_report.py): 비교대상 검색 보고서 Worker(ToolNode 검색 서브그래프 내장)/Validator, 그래프 레벨 노드 (스펙 4.5, 4.9)
- [src/nodes/image.py](src/nodes/image.py): 이미지 Worker/Validator, Send 동적 Fan-out (스펙 4.6, 4.7)
- [src/graph.py](src/graph.py): 그래프 골격 + Gate 0/1 + 재시도 루프 + Fan-out 라우팅 (Phase 1~3 범위)
- [src/ui/app.py](src/ui/app.py): Streamlit — 녹음/디버그 텍스트 입력, Gate 0, Gate 1, 최종 결과 화면
