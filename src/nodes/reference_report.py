"""비교대상 검색 보고서 Worker/Validator 노드. 기획서 4.5, 4.9, 3.5 참고.

웹 검색은 스펙 5.3대로 LangChain의 AgentExecutor 같은 블랙박스 루프가 아니라,
LangGraph ToolNode + 조건부 엣지로 그래프 안에 명시적으로 구현한다(아래
_build_search_subgraph: agent 노드 <-> tools 노드를 도구 호출이 없을 때까지
반복).

Worker(`reference_report_worker`)/Validator(`reference_report_validator`)는
spec/설정집과 동일하게 **그래프 레벨의 별도 노드**이고, 재시도도 그래프
조건부 엣지가 담당한다([src/graph.py](../graph.py)). 한때는 두 단계를
한 파이썬 함수 안에 캡슐화했었는데(아래 실측 기록 1~4번이 그 시절 기록),
그때 캡슐화했던 이유(Validator에 evidence를 state 필드 없이 넘기기 위함)가
이후 Validator 검증 범위를 대폭 축소하면서 사라졌다(evidence 자체를 안
씀). 그런데도 구조는 그대로 뒀던 걸 뒤늦게 알아차리고(5번) 다시 그래프
레벨 노드로 쪼갰다.

쪼개면서 새로 필요해진 것: 검색 대화(`messages`)가 재시도 사이에도
이어져야 하는데(3번 참고 - 안 이어가면 검색을 매번 처음부터 다시 해서
회귀가 났었다), 별도 그래프 노드는 매번 새 함수 호출이라 지역 변수로는
안 되고 [src/state.py](../state.py)의 `reference_report_messages`
(add_messages 리듀서)에 넣어 노드 호출 사이에도 살아있게 했다. 인용 ID
매칭 여부는 이제 evidence를 따로 안 넘기고, Worker가 만든 최종 보고서
텍스트에 매칭 안 된 `[id]` 패턴이 그대로 남아있는지 Validator가 직접
확인한다(`_CITE_ID_RE` 재사용, `_substitute_citations`가 매칭 실패한 것만
치환 안 하고 남겨두므로 - 아래 reference_report_validator 참고).

모델 티어: 스펙 5.1 - GPT-5.6 Luna (Worker/Validator 둘 다).

실측 기록(한 노드로 합쳐져 있던 시절 - 지금은 5번대로 다시 나뉨):
1. 스니펫 범위를 넘어서는 주장이 여러 실행에서 반복 재현되는 문제가
   있었다(예: inFAMOUS 2의 "Good/Evil 모두 완료 시 카르마 제한 능력
   해금"). Worker를 Terra로 올려도 실패율이 그대로였다 - 원복함(모델
   역량 문제가 아니라는 단서).
2. VALIDATOR_SYSTEM_PROMPT를 "모순/신규 사실 날조만 무효"로 완화했는데도
   실패가 반복돼 로그를 다시 대조해보니, **Validator 자신에게 전달된
   "실제로 인용된 검색 결과"에 분명히 포함돼 있던 출처를 "매칭 안 됨"으로
   잘못 판단한 사례를 실측으로 확인했다** - 근거가 눈앞에 있는데도 놓친
   것이므로 Worker나 프롬프트 문구의 문제가 아니라 Validator 자체의 판단
   신뢰도 문제로 보인다.
3. 재시도할 때마다 검색 대화를 완전히 새로 시작했더니(매번 새
   SystemMessage+HumanMessage), 두 가지 문제가 생겼다: (a) 이미 찾은
   좋은 근거까지 매번 새로 검색해서 쓸모없는 정보만 계속 쌓였고, (b)
   critique가 지적 안 한 행까지 매번 다시 쓰게 해서 멀쩡했던 내용에 새
   오류가 생겼는데, "그 행은 그대로 유지하라"고 지시해도 그 행이 인용한
   ID가 이번 라운드에 다시 검색되지 않으면 무효 처리됐다. 해결책: 검색
   대화(`messages`)를 재시도 간에도 끊지 않고 이어간다(`_run_search`가
   새로 시작하지 않고 기존 messages를 이어받아 계속함).
4. 그럼에도 Terra Validator가 실제로 매칭되는 근거를 "매칭 안 됨"으로
   잘못 판단하는 사례가 또 발생해(2번과 같은 유의 오류), 결국 "인용 ID가
   실제 검색 결과와 매칭되는가"라는 판단 자체를 LLM에서 코드로 옮겼다
   (순수 문자열 대조라 애초에 LLM이 필요 없는 문제였다). 더 나아가
   사용자 요청으로 Validator의 검증 범위 자체를 "브리핑에 나열된
   시스템이 섹션으로 다 있는가/범위 밖 시스템이 없는가/형식이 맞는가"
   까지로 대폭 축소했다 - 검증 범위가 기계적인 대조로 좁아지면서 Terra로
   올릴 필요가 없어져 Luna로 원복했다.
5. 4번 때 evidence를 이미 Validator에 안 넘기게 됐는데도, 애초에 한
   노드로 합쳤던 구조 자체는 관성적으로 그대로 뒀다(사용자 지적으로
   뒤늦게 발견). LangGraph를 쓰는 의미가 없어진다는 지적에 따라, 검색
   대화 유지 문제만 state 필드 추가로 풀고 spec/설정집과 같은 구조로
   다시 나눴다.
6. 조사 대상 시스템이 여러 개(5개)로 늘어난 실사용에서, 검색 서브그래프가
   MAX_SEARCH_STEPS(8) 한도에 도달했는데 마지막 메시지가 여전히
   tool_calls를 요청 중인 채로 끝나는 사례가 실측 확인됐다 - 이때
   content가 빈 문자열이라 Worker가 빈 보고서를 만들었고, 게다가 그
   tool_calls-미응답 메시지가 `reference_report_messages`에 그대로
   저장돼 다음 재시도에서 OpenAI가 400 Bad Request로 거부해(그래프
   실행 자체가 죽음) 시각화도 그 자리에서 멈췄다. `_build_search_subgraph`에
   `finalize` 노드를 추가해 한도 도달 시 tool을 바인딩하지 않은 LLM으로
   강제 텍스트 응답을 받도록 고쳤다(아래 `_finalize` 참고). MAX_SEARCH_STEPS도
   8 -> 16으로 올려 이 경로 자체가 덜 밟히게 여유를 늘렸다.
"""
import logging
import os
import re
from typing import Annotated, List, Optional, Set, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import RemoveMessage, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.rag import format_rag_context
from src.retry import next_feedback
from src.state import AgentMeetingState, TaskFeedback
from src.tools import web_search

REFERENCE_REPORT_WORKER_MODEL = os.getenv("REFERENCE_REPORT_WORKER_MODEL", "gpt-5.6-luna")
REFERENCE_REPORT_VALIDATOR_MODEL = os.getenv("REFERENCE_REPORT_VALIDATOR_MODEL", "gpt-5.6-luna")
# 재시도 한도(MAX_RETRIES)는 이 모듈에서 안 쓴다 - spec/설정집과 동일하게
# 그래프 레벨 조건부 엣지가 담당한다([src/graph.py](../graph.py)).

MAX_SEARCH_STEPS = 16  # 무한 tool-call 루프 방지용 안전장치

logger = logging.getLogger(__name__)

SEARCH_SYSTEM_PROMPT = """당신은 게임 비교대상 검색 보고서 작성 담당자입니다.
주어진 시스템(메커니즘) 목록 각각에 대해 web_search 도구로 실제 웹을 검색해,
유사하게 그 시스템을 구현한 게임 사례를 찾으세요. 검색 없이 알고 있는
지식만으로 사례를 지어내지 마세요 - 반드시 도구 호출 결과에 근거해야
합니다.

**주어진 시스템 목록에 정확히 있는 것만** 섹션으로 다루세요. 검색하다가
목록에 없는 다른 흥미로운 시스템(예: 다른 게임의 유사 메커니즘)을 발견해도,
그건 목록에 있는 시스템의 사례로만 활용하고 별도 섹션을 새로 만들지
마세요 - 목록에 없는 시스템 섹션을 추가하는 것 자체가 오류입니다.

시스템별로 최소 1~2회는 검색하세요. 모든 시스템에 대한 조사가 끝나면, 더
이상 도구를 호출하지 말고 아래 형식의 최종 보고서를 마크다운으로 작성해
답하세요.

검색 스니펫의 내용을 자연스럽게 요약·정리해서 쓰는 것은 괜찮습니다 -
문장을 그대로 베낄 필요는 없습니다. 다만 스니펫에 **전혀 근거가 없는
새로운 구체적 사실**(스니펫이 말하지 않은 수치, 이름 붙은 조건, 세부
규칙·메커니즘)을 지어내 붙이지 마세요. 예를 들어 스니펫이 "일부 능력이
특정 조건에 의존한다"고만 말했는데, 그 조건이 뭔지는 스니펫에 없으면서
"그 조건에 따라 엔딩이 갈린다"거나 "이탈/합류 조건이 -100이다"처럼
스니펫에 없는 구체적 규칙·수치를 만들어 붙이면 안 됩니다. 판단 기준은
"이 문장이 스니펫과 모순되거나, 스니펫에 전혀 없는 새 정보(수치·조건·
규칙)를 추가하는가"입니다 - 스니펫 내용을 다른 표현으로 풀어 쓰거나
요지를 정리하는 것 자체는 문제 삼지 않습니다.

표의 각 행에서 "유사점"/"차이점"에 적는 내용은 그 행의 "출처"로 쓸 검색
결과의 스니펫을 요약·정리한 것이어야 합니다. 다른 검색 결과나 그 게임에
대해 일반적으로 알려진 사실(상식)에서 가져온 구체적 사실을 그 출처가
말한 것처럼 붙이지 마세요.

## 출력 형식 (시스템별 섹션 반복)
### {시스템명}
- 논의 요약: (회의에서 어떻게 논의됐는지 1~2문장)
- 유사 사례 표 (컬럼: 게임명 / 유사점 / 차이점 / 출처)
- 종합 코멘트: (선택)

**출처 컬럼에는 URL을 절대 직접 타이핑하지 마세요.** 대신 그 검색 결과
앞에 표시된 8자리 인용 ID를 대괄호 그대로 적으세요(예: `[a1b2c3d4]`).
URL은 나중에 시스템이 그 ID를 보고 정확한 링크로 자동 치환합니다 - 직접
옮겨 적으면 오탈자가 생기니 옮겨 적지 말고 ID만 복사하세요. 실제로 검색
결과에 나온 ID만 쓸 수 있고, 존재하지 않는 ID를 지어내면 안 됩니다.
검색 결과에 없는 내용을 지어내지 마세요.
"""


class SearchState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    steps: int


_FORCE_FINAL_INSTRUCTION = (
    "검색 시도 횟수 한도에 도달했습니다. 더 이상 도구를 호출하지 말고,"
    " 지금까지 찾은 검색 결과만으로 최종 보고서를 지금 바로 형식에 맞게"
    " 작성하세요. 근거를 못 찾은 시스템이 있으면 그 시스템 섹션에 유사"
    " 사례를 찾지 못했다고 명시하고 표는 생략하세요."
)


def _route_search(state: SearchState) -> str:
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if not has_tool_calls:
        return "done"
    return "tools" if state["steps"] < MAX_SEARCH_STEPS else "finalize"


def _build_search_subgraph(model: str = REFERENCE_REPORT_WORKER_MODEL):
    llm = build_chat_model(model).bind_tools([web_search])
    # finalize 전용: tool을 바인딩하지 않은 별도 인스턴스 - 한도 도달 시
    # 강제로 텍스트만 답하게 만들어야 하므로(아래 _finalize 참고).
    llm_text_only = build_chat_model(model)

    def call_model(state: SearchState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "steps": state["steps"] + 1}

    def _finalize(state: SearchState) -> dict:
        """MAX_SEARCH_STEPS 한도에 도달했는데도 마지막 메시지가 아직
        tool_calls를 요청 중인 경우를 위한 안전한 종료 경로.

        실측 확인된 버그: 예전엔 한도 도달 시 그냥 라우팅만 "done"으로
        바꿔 END로 보냈는데, 그러면 (1) 그 tool_calls 요청은 끝내 실행되지
        않은 채로 응답의 content가 빈 문자열로 남아 Worker가 빈 보고서를
        만들었고(Validator가 "보고서가 비어있음"으로 잡아냄), (2) 더 심각하게
        tool_calls가 달린 채 이후 응답(ToolMessage)이 없는 AIMessage가
        `reference_report_messages`에 그대로 저장돼, 다음 재시도에서 이
        메시지 이력을 이어서 OpenAI에 보내면 API가 400 Bad Request로
        거부했다("assistant message with 'tool_calls' must be followed by
        tool messages") - 그래프 실행 전체가 죽어서 시각화도 그 자리에서
        멈췄다.

        해결: tool_calls가 매달린 마지막 메시지를 이력에서 빼고, tool을
        바인딩하지 않은 LLM으로 "그만 검색하고 지금까지 찾은 것만으로
        최종 보고서를 작성하라"고 지시한다 - tool을 바인딩 안 했으므로
        모델이 다시 tool_calls를 낼 수가 없어 반드시 텍스트로만 응답한다.

        `base`(마지막 메시지를 뺀 이력)는 이번 LLM 호출에만 쓰고 끝나는
        게 아니다 - `add_messages` 리듀서는 기본적으로 append만 하므로,
        여기서 `messages` 키에 새 메시지만 반환하면 그 tool_calls-매달린
        메시지가 SearchState에 그대로 남아있다가 Worker가 반환하는
        `reference_report_messages`(그래프 state)에도 그대로 들어가버린다
        (실측 확인 - 처음엔 이 부분을 놓쳐서 LLM 호출 자체는 고쳤는데도
        다음 재시도에서 여전히 400 에러가 났었다). 그래서 `RemoveMessage`로
        그 메시지를 state 이력에서 명시적으로 삭제해야 한다.
        """
        messages = state["messages"]
        dangling = messages[-1] if getattr(messages[-1], "tool_calls", None) else None
        base = messages[:-1] if dangling else messages
        instruction = HumanMessage(content=_FORCE_FINAL_INSTRUCTION)
        response = llm_text_only.invoke(base + [instruction])
        new_messages: List[AnyMessage] = [instruction, response]
        if dangling is not None:
            new_messages.insert(0, RemoveMessage(id=dangling.id))
        return {"messages": new_messages}

    graph = StateGraph(SearchState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([web_search]))
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_search, {"tools": "tools", "finalize": "finalize", "done": END})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


# web_search 도구가 반환하는 "- [ID] 제목 | URL\n  스니펫" 형식 파싱용.
_SEARCH_RESULT_RE = re.compile(
    r"^-\s*\[([0-9a-f]{8})\]\s*(.+?)\s*\|\s*(\S+)\s*\n\s*(.*)$", re.MULTILINE
)
_CITE_ID_RE = re.compile(r"\[([0-9a-f]{6,8})\]")


def _parse_search_results(messages: List[AnyMessage]) -> List[dict]:
    """검색 서브그래프 메시지 히스토리의 모든 ToolMessage에서 (ID, 제목, URL, 스니펫) 추출."""
    entries = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            for cid, title, url, snippet in _SEARCH_RESULT_RE.findall(content):
                entries.append(
                    {"id": cid, "title": title.strip(), "url": url.strip(), "snippet": snippet.strip()}
                )
    return entries


def _cited_ids(report_text: str) -> Set[str]:
    return set(_CITE_ID_RE.findall(report_text))


def _resolve_id(cid: str, id_to_result: dict) -> Optional[str]:
    """cid가 8자 정확히 일치하면 그대로 반환. 아니면(실측 확인 - 모델이
    8자리 ID를 옮겨 적다 끝을 잘라먹어 6~7자만 남는 경우가 있었다, 예:
    "3708d46a" -> "3708d46") 그 조각이 유일하게 접두사로 일치하는 실제
    ID를 찾아 반환한다. 여러 개와 동시에 일치하면(모호함) 실패로 처리한다
    - 검색 세션당 결과가 많아야 수십 개라 우연히 충돌할 확률은 매우 낮다.
    """
    if cid in id_to_result:
        return cid
    candidates = [full for full in id_to_result if full.startswith(cid)]
    return candidates[0] if len(candidates) == 1 else None


def _substitute_citations(report_text: str, messages: List[AnyMessage]) -> str:
    """보고서의 [ID] 인용을 실제 검색 결과의 마크다운 링크로 코드가 직접
    치환한다 - URL은 항상 검색 결과에서 그대로 가져오므로, LLM이 URL을
    다시 타이핑하다 생기는 오탈자/누락 가능성을 원천 차단한다. 매칭되지
    않는 ID(모델이 지어냈거나 접두사로도 특정 안 되는 것)는 일부러 그대로
    둬서 최종 결과물에서도 눈에 띄게 만든다 - Validator가 이미 무효로
    잡아냈어야 할 항목이다.
    """
    parsed = _parse_search_results(messages)
    id_to_result = {r["id"]: r for r in parsed}

    def _replace(match: re.Match) -> str:
        resolved = _resolve_id(match.group(1), id_to_result)
        r = id_to_result.get(resolved) if resolved else None
        return f"[{r['title']}]({r['url']})" if r else match.group(0)

    return _CITE_ID_RE.sub(_replace, report_text)


def _run_search(messages: List[AnyMessage]) -> List[AnyMessage]:
    """검색 서브그래프를 이어서 실행한다 - 매 라운드 새 대화로 초기화하지
    않고, 지금까지의 전체 메시지(이전 라운드의 검색 결과 + 직전 보고서
    포함)에 이어 붙여 계속한다.

    실측 확인된 회귀: 재시도마다 대화를 새로 시작하면 (1) 이미 찾은 좋은
    근거까지 매번 새로 검색해야 해서 쓸모없는 정보만 계속 쌓이고, (2) "지적
    안 된 행은 이전 그대로 유지"를 지시해도 그 행이 인용한 ID가 이번
    라운드에 다시 검색되지 않으면 무효 처리되는 문제가 있었다. 대화를
    이어가면 모델이 이미 검색한 근거를 그대로(같은 메시지 히스토리 안에서)
    보고, 검증 피드백이 지적한 부분에 대해서만 필요한 만큼 추가로
    web_search를 호출하면 된다 - 이미 아는 것을 다시 찾을 필요가 없다.
    """
    search_subgraph = _build_search_subgraph()
    result = search_subgraph.invoke({"messages": messages, "steps": 0})
    return result["messages"]


class ReferenceReportValidationResult(BaseModel):
    is_valid: bool = Field(
        description="조사 대상 시스템이 모두 섹션으로 포함되고, 목록에 없는 시스템이 섹션으로 추가되지 않았는가"
    )
    critique: str = Field(description="구체적인 지적사항 (통과 시에는 짧은 확인 코멘트)")


VALIDATOR_SYSTEM_PROMPT = """당신은 비교대상 검색 보고서 검증 담당자입니다.
"조사 대상 시스템 목록"과 "검증 대상 보고서"만 비교하세요 - 보고서 내용이
검색 근거와 사실적으로 부합하는지는 이 검증의 범위가 아닙니다(인용 ID가
실제 검색 결과와 매칭되는지는 코드가 별도로 확인하며, 실측 확인 - 그
이상의 세부 사실 대조에서 Validator 자신이 신뢰할 수 없는 판정을 반복
했습니다).

## 체크리스트
1. 조사 대상 시스템이 모두 별도 섹션으로 포함됐는가.
2. 목록에 없는 시스템이 섹션으로 추가되지 않았는가(다른 시스템 섹션 안에서
   비교 사례로 잠깐 언급되는 것은 괜찮습니다 - 목록에 없는 시스템 자체가
   섹션 제목이 된 경우만 지적하세요).
3. 형식(시스템명/논의 요약/유사 사례 표/종합 코멘트)이 지켜졌는가.

**이 세 가지 외에는 확인하지 마세요.** 표 안의 개별 주장이 검색 스니펫과
정확히 부합하는지, 출처가 신뢰할 만한지는 이 검증의 범위가 아닙니다 -
Worker의 판단을 그대로 신뢰하세요. 무효 사유가 하나라도 있으면
is_valid=false로 판정하고 critique에 정확히 무엇을 고쳐야 하는지
적으세요. 모두 통과하면 is_valid=true, critique는 짧은 확인 코멘트만
적으세요.
"""

_validator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VALIDATOR_SYSTEM_PROMPT),
        (
            "human",
            "## 조사 대상 시스템 목록\n{systems}\n\n"
            "## 검증 대상 보고서\n{reference_report}",
        ),
    ]
)


def _build_validator_chain(model: str = REFERENCE_REPORT_VALIDATOR_MODEL):
    llm = build_chat_model(model)
    return _validator_prompt | llm.with_structured_output(ReferenceReportValidationResult)


def _validate(systems: List[str], report: str) -> ReferenceReportValidationResult:
    chain = _build_validator_chain()
    return chain.invoke({"systems": ", ".join(systems), "reference_report": report})


def reference_report_worker(state: AgentMeetingState) -> dict:
    """비교대상 검색 보고서 Worker 노드: 검색 -> 초안 작성.

    첫 시도면 새 검색 대화를 시작하고, 재시도면 state에 저장된 기존 검색
    대화(`reference_report_messages`)에 critique만 새 turn으로 추가해
    이어간다(모듈 docstring 3번 참고 - 대화를 안 이어가면 검색을 매번
    처음부터 다시 해서 회귀가 났었다).
    """
    existing_messages = state.get("reference_report_messages") or []
    prev_feedback = state.get("validation_status", {}).get("reference_report")

    if not existing_messages:
        brief = state["task_briefs"]["reference_report"]["brief"]
        systems = state["discussed_systems"]
        rag_context = format_rag_context(state.get("rag_references") or [])
        human_content = (
            f"조사할 시스템 목록: {', '.join(systems)}\n\n"
            f"생성 지시문(브리핑): {brief}\n\n"
            f"참고 자료 (사내 용어집/게임 프로필 - RAG 검색 결과, 용어 표기 확인용"
            f" - 검색 근거로 쓰지 말 것):\n{rag_context}"
        )
        input_messages: List[AnyMessage] = [
            SystemMessage(content=SEARCH_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
    else:
        critique = prev_feedback["critique"] if prev_feedback else ""
        input_messages = list(existing_messages) + [
            HumanMessage(
                content=(
                    "검증 결과 다음 문제가 지적됐습니다 - 지적된 부분만 고쳐서"
                    " 보고서 전체를 다시 출력하세요:\n"
                    f"{critique}\n\n"
                    "지적되지 않은 시스템 섹션/표 행/문장은 방금 작성한 보고서"
                    " 그대로 유지하세요 - 문제 없다고 이미 확인된 내용을"
                    " 건드리면 거기서 새 오류가 생길 수 있습니다. 지적된 부분을"
                    " 고치는 데 필요하면 web_search를 추가로 호출해서 근거를"
                    " 보강하세요 - 이미 위에서 찾은 근거로 충분한 부분을 다시"
                    " 검색할 필요는 없습니다. 근거를 못 찾으면 그 부분만"
                    " 삭제하거나 실제로 뒷받침되는 수준으로 줄이세요. 검색이"
                    " 끝나면(또는 추가 검색이 필요 없으면) 보고서 전체를 다시"
                    " 출력하세요."
                )
            )
        ]

    new_messages = _run_search(input_messages)
    raw_draft = new_messages[-1].content
    draft = _substitute_citations(raw_draft, new_messages)

    logger.debug("reference_report_worker output:\n%s", draft)

    return {
        # add_messages 리듀서가 있으므로 새로 생긴 부분만 반환하면 이어붙는다.
        "reference_report_messages": new_messages[len(existing_messages):],
        "reference_report": draft,
    }


class ReferenceReportValidationResult(BaseModel):
    is_valid: bool = Field(
        description="조사 대상 시스템이 모두 섹션으로 포함되고, 목록에 없는 시스템이 섹션으로 추가되지 않았는가"
    )
    critique: str = Field(description="구체적인 지적사항 (통과 시에는 짧은 확인 코멘트)")


VALIDATOR_SYSTEM_PROMPT = """당신은 비교대상 검색 보고서 검증 담당자입니다.
"조사 대상 시스템 목록"과 "검증 대상 보고서"만 비교하세요 - 보고서 내용이
검색 근거와 사실적으로 부합하는지는 이 검증의 범위가 아닙니다(인용 ID가
실제 검색 결과와 매칭되는지는 코드가 별도로 확인하며, 실측 확인 - 그
이상의 세부 사실 대조에서 Validator 자신이 신뢰할 수 없는 판정을 반복
했습니다).

## 체크리스트
1. 조사 대상 시스템이 모두 별도 섹션으로 포함됐는가.
2. 목록에 없는 시스템이 섹션으로 추가되지 않았는가(다른 시스템 섹션 안에서
   비교 사례로 잠깐 언급되는 것은 괜찮습니다 - 목록에 없는 시스템 자체가
   섹션 제목이 된 경우만 지적하세요).
3. 형식(시스템명/논의 요약/유사 사례 표/종합 코멘트)이 지켜졌는가.

**이 세 가지 외에는 확인하지 마세요.** 표 안의 개별 주장이 검색 스니펫과
정확히 부합하는지, 출처가 신뢰할 만한지는 이 검증의 범위가 아닙니다 -
Worker의 판단을 그대로 신뢰하세요. 무효 사유가 하나라도 있으면
is_valid=false로 판정하고 critique에 정확히 무엇을 고쳐야 하는지
적으세요. 모두 통과하면 is_valid=true, critique는 짧은 확인 코멘트만
적으세요.
"""

_validator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VALIDATOR_SYSTEM_PROMPT),
        (
            "human",
            "## 조사 대상 시스템 목록\n{systems}\n\n"
            "## 검증 대상 보고서\n{reference_report}",
        ),
    ]
)


def _build_validator_chain(model: str = REFERENCE_REPORT_VALIDATOR_MODEL):
    llm = build_chat_model(model)
    return _validator_prompt | llm.with_structured_output(ReferenceReportValidationResult)


def _validate(systems: List[str], report: str) -> ReferenceReportValidationResult:
    chain = _build_validator_chain()
    return chain.invoke({"systems": ", ".join(systems), "reference_report": report})


def reference_report_validator(state: AgentMeetingState) -> dict:
    """비교대상 검색 보고서 Validator 노드: reference_report 검증 ->
    validation_status["reference_report"].

    인용 ID 매칭은 다시 검색하지 않고, Worker가 만든 최종 보고서에 매칭 안
    된 `[id]` 패턴이 그대로 남아있는지만 문자열로 확인한다(`_substitute_citations`가
    매칭 실패한 ID는 치환하지 않고 그대로 두므로 - 코드로 100% 정확히
    판단 가능해 LLM에게 다시 묻지 않는다, 모듈 docstring 참고).
    """
    systems = state["discussed_systems"]
    report = state["reference_report"]

    leftover_ids = _cited_ids(report)
    if leftover_ids:
        result = ReferenceReportValidationResult(
            is_valid=False,
            critique=(
                f"다음 인용 ID가 실제 검색 결과와 매칭되지 않습니다: "
                f"{sorted(leftover_ids)}. 이 ID를 인용한 사례/행을 삭제하거나,"
                " 실제로 검색해서 찾은 다른 출처의 ID로 바꾸세요(ID를 새로"
                " 만들어내지 말고, 검색 결과에 실제로 표시된 ID만 쓰세요)."
            ),
        )
    else:
        result = _validate(systems, report)

    prev_feedback = state.get("validation_status", {}).get("reference_report")
    feedback: TaskFeedback = next_feedback(prev_feedback, result.is_valid, result.critique)

    if not result.is_valid:
        logger.warning(
            "reference_report_validator FAILED (retry_count=%d) - critique: %s\n"
            "--- 검증 대상 보고서 ---\n%s",
            feedback["retry_count"],
            result.critique,
            report,
        )
    else:
        logger.info("reference_report_validator passed (retry_count=%d)", feedback["retry_count"])

    return {"validation_status": {"reference_report": feedback}}
