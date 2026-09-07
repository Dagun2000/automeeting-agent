# AutoMeeting Agent

LangGraph 기반 Human-in-the-Loop 다중 모달 회의 산출물 자동화 에이전트.
전체 스펙은 [AutoMeeting_Agent_기획서_v3.md](AutoMeeting_Agent_기획서_v3.md)이며, 항상 이 문서를 최우선 기준으로 따를 것.

## 현재 단계

Phase 1~3 구현됨: 디버그 텍스트 입력 → Minutes Generator → Gate 0(`interrupt_before`)
→ 아이디어 선정(3.2) → Spec Worker/Validator(3.3) → Task Classifier(3.4/3.5) →
(needs_image면) Image Style Planner(4.6) → Gate 1(`interrupt_before`) → Fan-out:
설정집/비교대상 검색 보고서/이미지×N(Send 동적 생성) — 각 트랙 독립 critique
재시도 루프. STT/Diarization, Gate 2 에스컬레이션 UI, Aggregator/ZIP은 아직 없음
(Phase 4 범위는 스펙 6장 참고). RAG(`rag_references`)도 Phase 4로 보류하기로 함.

모델 티어는 현재 전 노드 GPT-5.6 Luna로 통일(스펙 5.1 문서는 미갱신 상태, 이미지
Validator 원안은 Terra였으나 지금은 Luna) — 노드별로 문제가 확인되면 해당 노드의
`*_MODEL` 환경변수만 개별적으로 상향할 것.

**중요**: GPT-5.6 계열(Luna/Terra/Sol)은 reasoning 모델이라 `temperature`를
지원하지 않고, 기본 reasoning_effort("medium")로는 구조화 출력/tool 바인딩을
`/v1/chat/completions`에서 쓸 수 없다(실제 API 에러로 확인: "Function tools
with reasoning_effort are not supported ... set reasoning_effort to 'none'").
그래서 모든 ChatOpenAI 인스턴스는 [src/llm.py](src/llm.py)의
`build_chat_model()`을 통해서만 생성하고(`reasoning_effort="none"` 고정,
`REASONING_EFFORT` 환경변수로 오버라이드 가능), 어디서도 `ChatOpenAI(...)`를
직접 호출하거나 `temperature=`를 넘기지 않는다.

**5.4 스키마에서 벗어난 부분 1건**: `image_tasks`에 커스텀 reducer(`_merge_image_tasks`,
subject 키 기준 병합)를 추가함. 원안은 reducer 없는 plain `List[ImageTask]`지만,
4.7의 `Send` 기반 동적 Fan-out은 이미지별 브랜치가 동시에 이 채널에 쓰기 때문에
reducer가 없으면 LangGraph가 `InvalidUpdateError`를 낸다(실측 확인, 대안 없음).
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
- 이미지 Fan-out: 스펙 4.7 — `Send`로 이미지별 병렬 파견, 재시도까지 포함한 처리를 서브그래프 하나로 캡슐화([src/nodes/image.py](src/nodes/image.py)). 이유는 위 "5.4 스키마에서 벗어난 부분" 참고.
- 웹 검색 루프: 스펙 5.3 — `AgentExecutor` 등 블랙박스 대신 `ToolNode` + 조건부 엣지로 그래프 안에 명시(agent ⇄ tools 반복). [src/nodes/reference_report.py](src/nodes/reference_report.py). 검색 백엔드는 Tavily([src/tools.py](src/tools.py), `TAVILY_API_KEY`) — 다른 제공자로 바꾸려면 이 파일만 교체.

## 구조

- [src/state.py](src/state.py): LangGraph 상태 스키마 (+ image_tasks 커스텀 reducer)
- [src/llm.py](src/llm.py): ChatOpenAI 생성 공용 헬퍼 (`reasoning_effort="none"` 고정 - 위 "중요" 참고)
- [src/retry.py](src/retry.py): Worker-Validator 재시도 루프 공용 헬퍼 (spec/setting_doc/reference_report/이미지 서브파이프라인이 공유)
- [src/tagging.py](src/tagging.py): 태그별 문단 추출 유틸 (Spec Worker, Task Classifier가 공용)
- [src/tools.py](src/tools.py): 웹 검색 도구 (Tavily)
- [src/nodes/minutes.py](src/nodes/minutes.py): Minutes Generator 노드 (스펙 3.1, 4.2)
- [src/nodes/idea_select.py](src/nodes/idea_select.py): 아이디어 선정 노드 (스펙 3.2)
- [src/nodes/spec.py](src/nodes/spec.py): Spec Worker/Validator 노드 (스펙 3.3, 4.9)
- [src/nodes/classifier.py](src/nodes/classifier.py): Task Classifier 노드 (스펙 3.4, 3.5)
- [src/nodes/style_planner.py](src/nodes/style_planner.py): Image Style Planner 노드 (스펙 4.6)
- [src/nodes/setting_doc.py](src/nodes/setting_doc.py): 설정집 Worker/Validator (스펙 4.5, 4.9)
- [src/nodes/reference_report.py](src/nodes/reference_report.py): 비교대상 검색 보고서 Worker(ToolNode 검색 루프)/Validator (스펙 4.5, 4.9)
- [src/nodes/image.py](src/nodes/image.py): 이미지 Worker/Validator, Send 동적 Fan-out (스펙 4.6, 4.7)
- [src/graph.py](src/graph.py): 그래프 골격 + Gate 0/1 + 재시도 루프 + Fan-out 라우팅 (Phase 1~3 범위)
- [src/ui/app.py](src/ui/app.py): Streamlit — 텍스트 입력, Gate 0, Gate 1, 최종 결과 화면
