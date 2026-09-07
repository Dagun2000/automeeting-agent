"""비교대상 검색 보고서 Worker/Validator 노드. 기획서 4.5, 4.9, 3.5 참고.

웹 검색은 스펙 5.3대로 LangChain의 AgentExecutor 같은 블랙박스 루프가 아니라,
LangGraph ToolNode + 조건부 엣지로 그래프 안에 명시적으로 구현한다(아래
_build_search_subgraph: agent 노드 <-> tools 노드를 도구 호출이 없을 때까지
반복). 이 서브그래프는 reference_report_worker 노드 내부에서 한 번 실행되어
초안을 만들고, 전체 재시도 루프(critique 기반)는 spec.py/setting_doc.py와
동일하게 부모 그래프의 조건부 엣지가 담당한다.

모델 티어: 스펙 5.1 - GPT-5.6 Luna.
"""
import os
from typing import Annotated, List, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.retry import next_feedback
from src.state import AgentMeetingState, TaskFeedback
from src.tools import web_search

REFERENCE_REPORT_WORKER_MODEL = os.getenv("REFERENCE_REPORT_WORKER_MODEL", "gpt-5.6-luna")
REFERENCE_REPORT_VALIDATOR_MODEL = os.getenv("REFERENCE_REPORT_VALIDATOR_MODEL", "gpt-5.6-luna")

MAX_SEARCH_STEPS = 8  # 무한 tool-call 루프 방지용 안전장치

SEARCH_SYSTEM_PROMPT = """당신은 게임 비교대상 검색 보고서 작성 담당자입니다.
주어진 시스템(메커니즘) 목록 각각에 대해 web_search 도구로 실제 웹을 검색해,
유사하게 그 시스템을 구현한 게임 사례를 찾으세요. 검색 없이 알고 있는
지식만으로 사례를 지어내지 마세요 - 반드시 도구 호출 결과에 근거해야
합니다.

시스템별로 최소 1~2회는 검색하세요. 모든 시스템에 대한 조사가 끝나면, 더
이상 도구를 호출하지 말고 아래 형식의 최종 보고서를 마크다운으로 작성해
답하세요.

## 출력 형식 (시스템별 섹션 반복)
### {시스템명}
- 논의 요약: (회의에서 어떻게 논의됐는지 1~2문장)
- 유사 사례 표 (컬럼: 게임명 / 유사점 / 차이점 / 출처)
- 종합 코멘트: (선택)

출처 컬럼에는 실제 검색 결과에서 얻은 URL을 적으세요. 검색 결과에 없는
내용을 지어내지 마세요.
"""


class SearchState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    steps: int


def _route_search(state: SearchState) -> str:
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if has_tool_calls and state["steps"] < MAX_SEARCH_STEPS:
        return "tools"
    return "done"


def _build_search_subgraph(model: str = REFERENCE_REPORT_WORKER_MODEL):
    llm = build_chat_model(model).bind_tools([web_search])

    def call_model(state: SearchState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "steps": state["steps"] + 1}

    graph = StateGraph(SearchState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([web_search]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_search, {"tools": "tools", "done": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def reference_report_worker(state: AgentMeetingState) -> dict:
    """비교대상 검색 보고서 Worker 노드: discussed_systems -> reference_report(초안)."""
    brief = state["task_briefs"]["reference_report"]["brief"]
    systems = state["discussed_systems"]

    prev_feedback = state.get("validation_status", {}).get("reference_report")
    critique_block = ""
    if prev_feedback and not prev_feedback["is_valid"]:
        critique_block = (
            "\n\n이전 시도 검증 피드백(critique) - 반드시 반영해서 다시 조사/작성"
            f"하세요:\n{prev_feedback['critique']}\n\n"
            f"이전 초안:\n{state.get('reference_report') or '(없음)'}"
        )

    human_content = (
        f"조사할 시스템 목록: {', '.join(systems)}\n\n"
        f"생성 지시문(브리핑): {brief}"
        f"{critique_block}"
    )

    search_subgraph = _build_search_subgraph()
    result = search_subgraph.invoke(
        {
            "messages": [
                SystemMessage(content=SEARCH_SYSTEM_PROMPT),
                HumanMessage(content=human_content),
            ],
            "steps": 0,
        }
    )
    draft = result["messages"][-1].content
    return {"reference_report": draft}


class ReferenceReportValidationResult(BaseModel):
    is_valid: bool = Field(description="모든 섹션이 실제 검색 결과에 근거했는가")
    critique: str = Field(description="구체적인 지적사항 (통과 시에는 짧은 확인 코멘트)")


VALIDATOR_SYSTEM_PROMPT = """당신은 비교대상 검색 보고서 검증 담당자입니다.
아래 "조사 대상 시스템 목록"과 "검증 대상 보고서"를 비교해 다음을
확인하세요.

1. 조사 대상 시스템이 모두 섹션으로 포함됐는가(최대 5개, 목록에 없는 항목이
   추가됐다면 그것도 지적하세요).
2. 각 섹션의 유사 사례 표에 실제 출처(URL)가 채워져 있는가 - 출처가
   비어있거나, 내용이 구체적 근거 없이 일반적인 상식 수준으로만 쓰여
   있다면(실제 검색 없이 지어낸 것으로 의심되면) 무효로 판단하세요. 이
   항목을 최우선으로 확인하세요.
3. 형식(시스템명/논의 요약/유사 사례 표/종합 코멘트)이 지켜졌는가.

하나라도 위반하면 is_valid=false로 판정하고 critique에 무엇을 보완해야
하는지 구체적으로 적으세요(예: 어떤 시스템의 출처가 비어있는지). 모두
통과하면 is_valid=true, critique는 짧은 확인 코멘트만 적으세요.
"""

_validator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VALIDATOR_SYSTEM_PROMPT),
        (
            "human",
            "## 조사 대상 시스템 목록\n{systems}\n\n## 검증 대상 보고서\n{reference_report}",
        ),
    ]
)


def _build_validator_chain(model: str = REFERENCE_REPORT_VALIDATOR_MODEL):
    llm = build_chat_model(model)
    return _validator_prompt | llm.with_structured_output(ReferenceReportValidationResult)


def reference_report_validator(state: AgentMeetingState) -> dict:
    """비교대상 검색 보고서 Validator 노드."""
    chain = _build_validator_chain()
    result: ReferenceReportValidationResult = chain.invoke(
        {
            "systems": ", ".join(state["discussed_systems"]),
            "reference_report": state["reference_report"],
        }
    )

    prev_feedback = state.get("validation_status", {}).get("reference_report")
    feedback: TaskFeedback = next_feedback(prev_feedback, result.is_valid, result.critique)
    return {"validation_status": {"reference_report": feedback}}
