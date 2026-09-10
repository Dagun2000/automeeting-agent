# AutoMeeting Agent

[한국어](README.md) | [English](README.en.md)

LangGraph 기반 Human-in-the-Loop 다중 모달 회의 산출물 자동화 에이전트입니다.
회의 녹음(또는 텍스트)을 입력하면 회의록 → 기획서 → 설정집/비교대상 검색
보고서/컨셉 아트 이미지까지, 사람이 두 지점(Gate 0/1)에서 검토·확정하며
자동으로 생성합니다.

전체 기획 의도와 스펙은 [AutoMeeting_Agent_기획서_v3.md](AutoMeeting_Agent_기획서_v3.md)에,
구현 세부사항·아키텍처 결정 경위는 [CLAUDE.md](CLAUDE.md)에 있습니다.

## 파이프라인

```
녹음(STT) 또는 텍스트 입력
  → RAG Retrieval (사내 용어집/게임 프로필 검색)
  → Minutes Generator (태그 포함 회의록 생성)
  → Gate 0 (회의록 검토/수정)
  → 아이디어 선정 → Spec Worker/Validator (기획서 생성, critique 재시도)
  → Task Classifier (설정집/검색 보고서/이미지 필요 여부 판단)
  → Gate 1 (기획서 검토 + 하위 산출물 확정)
  → Fan-out: 설정집 / 비교대상 검색 보고서 / 이미지×N (병렬, 각자 재시도 루프)
  → 완료 화면: 확인이 필요한 항목(내용 품질 실패/기술적 실패 구분) + ZIP 다운로드
```

## 주요 특징

- **Human-in-the-Loop**: Gate 0(회의록), Gate 1(기획서+산출물 확정) 두 지점에서
  사람이 직접 검토·수정 후 진행합니다.
- **Critique 기반 자기 검증**: 각 산출물은 Worker가 생성하고 별도 Validator가
  검증해, 문제가 있으면 구체적 피드백과 함께 재시도합니다(한도 도달 시
  에스컬레이션).
- **기술적 실패 안전망**: API 에러 등 기술적 실패는 내용 품질 실패와 구분해서
  기록하고, 한 트랙이 실패해도 나머지는 계속 진행됩니다.
- **RAG 기반 용어 일관성**: 사내 용어집/게임 프로필을 검색해 STT 오인식
  가능성이 있는 고유명사를 교정하고, 산출물 전반의 용어 표기를 맞춥니다.
- **실시간 파이프라인 시각화**: 지금 어느 노드가 실행 중인지 그래프 형태로
  보여줍니다(병렬 실행 중인 트랙도 함께 표시).
- **부분 실패 허용 + ZIP 다운로드**: 일부 산출물이 실패해도 나머지는 정상
  담아서 ZIP으로 내려받을 수 있습니다(Markdown + 이미지 파일).

## 기술 스택

- **오케스트레이션**: [LangGraph](https://github.com/langchain-ai/langgraph)
  (`StateGraph`, `interrupt_before`, 동적 `Send` Fan-out)
- **LLM**: OpenAI GPT-5.6 계열 (`reasoning_effort="none"`), 이미지 생성은
  `gpt-image-2`
- **RAG**: ChromaDB(인메모리) + `text-embedding-3-small`
- **웹 검색**: [`ddgs`](https://pypi.org/project/ddgs/) (API 키 불필요)
- **STT**: OpenAI 오디오 트랜스크립션 API
- **UI**: Streamlit

## 시작하기

Python 3.14가 필요합니다(`.python-version` 참고).

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt

cp .env.example .env
# .env를 열어 OPENAI_API_KEY를 채워 넣으세요 (다른 값은 기본값으로 충분합니다)

streamlit run src/ui/app.py
```

## 프로젝트 구조

```
src/
  graph.py            # 그래프 골격 (Gate 0/1, Fan-out, 재시도/기술적 실패 라우팅)
  state.py            # LangGraph 상태 스키마
  aggregator.py        # ZIP 패키징 (완료 화면 다운로드 버튼이 호출)
  nodes/               # Minutes/Spec/설정집/검색 보고서/이미지 등 각 노드
  ui/app.py            # Streamlit UI
tests/
  test_full_pipeline.py  # 통합 테스트 (기본 구조 확인, --real로 전체 실행)
sample_data/            # 테스트용 샘플 회의록/사내 코퍼스
```

각 파일의 상세한 설계 의도(왜 이렇게 만들었는지, 어떤 실측 근거로 바꿨는지)는
[CLAUDE.md](CLAUDE.md)와 각 소스 파일의 모듈 docstring을 참고하세요.

## 테스트

```bash
python tests/test_full_pipeline.py          # 구조 확인만 (무료)
python tests/test_full_pipeline.py --real   # 실제 API로 전체 파이프라인 실행 (비용 발생)
```

## 라이선스

[MIT](LICENSE)
