"""LangGraph 그래프 골격 - Phase 1~4 범위.

텍스트 입력 -> RAG Retrieval -> Minutes Generator -> Gate 0(interrupt_before)
   -> 아이디어 선정 -> (없으면 종료, 회의록만 출력)
   -> (있으면) Spec Worker -> Spec Validator -> 재시도 루프
   -> Task Classifier -> (needs_image면) Image Style Planner
   -> Gate 1(interrupt_before)
   -> Fan-out: 설정집 / 비교대상 검색 보고서 / 이미지×N(Send 동적 생성)
      각 트랙 독립 재시도 루프(critique 기반, MAX_RETRIES) -> 통과/한도초과 시 종료

설정집/비교대상 검색 보고서/이미지 전부 spec과 동일하게 Worker/Validator가
그래프 레벨 조건부 엣지로 재시도한다(이미지는 Send 기반 동적 N-way
브랜치라 재시도도 매 홉마다 `Send`로 명시적으로 다시 호출해야 브랜치가
안 섞인다 - 자세한 이유와 실측 검증은 [src/nodes/image.py](nodes/image.py)
docstring 참고). 이미지도 한때는 evidence 전달 문제로 한때 한 노드였던
비교대상 검색 보고서와 비슷하게, Worker+Validator+재시도 전체를 노드 하나
안에 캡슐화했었다 - 그래야 Send 병렬 브랜치가 안 섞인다고 판단했었지만,
실측 결과 `Send`를 매 홉마다 명시하면 그래프 레벨로 쪼개도 안전하다는 게
확인돼 다시 나눴다. 쪼갠 이유: 캡슐화된 상태에서는 이미지 "검증+재시도"까지
전부 끝나야 그 슈퍼스텝이 끝난 것으로 처리돼, 같은 슈퍼스텝에 파견된
설정집/검색 보고서 Validator가 자기 Worker는 진작 끝냈는데도 이미지 검증이
끝날 때까지 기다려야 했다(LangGraph의 Pregel 슈퍼스텝 동기화 - 사용자가
실측 지적으로 발견). 쪼갠 뒤로는 슈퍼스텝이 "이미지 생성"까지만 기다리면
되므로 이 대기가 사라진다. 자세한 경위는
[src/nodes/reference_report.py](nodes/reference_report.py)(비교대상 검색
보고서 분리)와 [src/nodes/image.py](nodes/image.py)(이미지 분리)의
docstring 참고.

기획서 트랙이 재시도 한도를 초과해도(escalate) 이후 파이프라인은 계속
진행한다 - Gate 2(에스컬레이션 확인)는 Fan-out 이후 한 번에 모아 보여주는
단계이므로(2번 섹션 10단계), 한 트랙의 품질 실패가 다른 트랙 진행을 막지
않는다(4.8, "나머지는 계속 진행"과 동일한 원칙을 품질 실패에도 적용).

RAG Retrieval(5.2, 5.4): 트랜스크립트 확보 직후 사내 코퍼스(용어집/게임
프로필)를 검색해 rag_references를 채운다([src/nodes/rag_retrieval.py](nodes/rag_retrieval.py)).
Minutes Generator/Spec Worker/설정집 Worker/비교대상 검색 보고서 Worker가
이 rag_references를 프롬프트에 포함해 참고한다(Task Classifier/Image Style
Planner는 대상 아님). Minutes Generator는 이걸로 STT 오인식 가능성이 있는
사내 고유명사를 용어집 표기로 교정한다(확신 없으면 원문 유지).

Aggregator/ZIP/Gate 2 UI는 아직 없음(Phase 4 나머지 범위).
"""
import logging
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes.classifier import classify_tasks
from src.nodes.idea_select import select_idea
from src.nodes.image import (
    dispatch_image_tasks,
    image_validator,
    image_worker,
    route_after_image_validate,
    route_worker_to_validator,
)
from src.nodes.minutes import generate_minutes
from src.nodes.rag_retrieval import retrieve_rag_references
from src.nodes.reference_report import reference_report_validator, reference_report_worker
from src.nodes.setting_doc import setting_doc_validator, setting_doc_worker
from src.nodes.spec import spec_validator, spec_worker
from src.nodes.style_planner import plan_image_style
from src.retry import route_decision
from src.state import AgentMeetingState

GATE_0_NODE = "gate_0"
GATE_1_NODE = "gate_1"

logger = logging.getLogger(__name__)

# 4.8: 내용 품질 실패 시 retry_count < MAX_RETRIES까지 critique 기반 재시도.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


def _gate_0_checkpoint(state: AgentMeetingState) -> dict:
    """Gate 0 체크포인트 노드.

    `interrupt_before`가 이 노드 진입 전에 그래프를 멈추므로 실제 로직은 없다.
    사람이 UI에서 회의록을 확정하면(meeting_minutes_confirmed를 채우면)
    그래프가 재개되어 이 노드를 통과한다.
    """
    return {}


def _gate_1_checkpoint(state: AgentMeetingState) -> dict:
    """Gate 1 체크포인트 노드.

    사람이 UI에서 기획서를 검토하고(재생성 트리거 아님), 설정집/이미지/
    비교대상 검색 보고서 카드의 체크박스+브리핑(이미지는 스타일 가이드도)을
    확정하면 그래프가 재개되어 [_dispatch_after_gate1][]로 Fan-out한다.
    """
    return {}


def _route_after_idea_select(state: AgentMeetingState) -> str:
    return "has_idea" if state.get("selected_idea_tag") else "no_idea"


def _route_after_classifier(state: AgentMeetingState) -> str:
    return "needs_image" if state.get("image_tasks") else "no_image"


def _dispatch_after_gate1(state: AgentMeetingState) -> list:
    """Gate 1 확정 후 활성화된 트랙만 병렬로 파견(플레인 노드 이름 + Send 혼합 반환)."""
    dests: list = []
    if state.get("task_briefs", {}).get("setting_doc", {}).get("enabled"):
        dests.append("setting_doc_worker")
    if state.get("task_briefs", {}).get("reference_report", {}).get("enabled"):
        dests.append("reference_report_worker")
    dests.extend(dispatch_image_tasks(state))
    return dests if dests else [END]


def _make_validation_router(track: str):
    """spec/setting_doc/reference_report 공용: validator 이후 done/retry/escalate 라우팅."""

    def _route(state: AgentMeetingState) -> str:
        feedback = state["validation_status"][track]
        return route_decision(feedback, MAX_RETRIES)

    return _route


def _make_escalate_node(track: str):
    """재시도 한도 초과 시 escalated_tasks에 기록. Gate 2 에스컬레이션 UI는 Phase 4."""

    def _escalate(state: AgentMeetingState) -> dict:
        feedback = state.get("validation_status", {}).get(track)
        logger.error(
            "%s ESCALATED after retry limit - critique: %s",
            track,
            feedback["critique"] if feedback else "(no feedback recorded)",
        )
        return {"escalated_tasks": [track]}

    return _escalate


def build_graph():
    graph = StateGraph(AgentMeetingState)

    graph.add_node("rag_retrieval", retrieve_rag_references)
    graph.add_node("minutes_generator", generate_minutes)
    graph.add_node(GATE_0_NODE, _gate_0_checkpoint)
    graph.add_node("idea_select", select_idea)
    graph.add_node("spec_worker", spec_worker)
    graph.add_node("spec_validator", spec_validator)
    graph.add_node("spec_escalate", _make_escalate_node("spec"))
    graph.add_node("task_classifier", classify_tasks)
    graph.add_node("style_planner", plan_image_style)
    graph.add_node(GATE_1_NODE, _gate_1_checkpoint)
    graph.add_node("setting_doc_worker", setting_doc_worker)
    graph.add_node("setting_doc_validator", setting_doc_validator)
    graph.add_node("setting_doc_escalate", _make_escalate_node("setting_doc"))
    graph.add_node("reference_report_worker", reference_report_worker)
    graph.add_node("reference_report_validator", reference_report_validator)
    graph.add_node("reference_report_escalate", _make_escalate_node("reference_report"))
    graph.add_node("image_worker", image_worker)
    graph.add_node("image_validator", image_validator)

    # Phase 1~2: 텍스트 입력 -> RAG Retrieval -> 회의록 -> Gate 0 -> 아이디어 선정 -> 기획서
    graph.add_edge(START, "rag_retrieval")
    graph.add_edge("rag_retrieval", "minutes_generator")
    graph.add_edge("minutes_generator", GATE_0_NODE)
    graph.add_edge(GATE_0_NODE, "idea_select")
    graph.add_conditional_edges(
        "idea_select",
        _route_after_idea_select,
        {"has_idea": "spec_worker", "no_idea": END},
    )
    graph.add_edge("spec_worker", "spec_validator")
    graph.add_conditional_edges(
        "spec_validator",
        _make_validation_router("spec"),
        {"done": "task_classifier", "escalate": "spec_escalate", "retry": "spec_worker"},
    )
    graph.add_edge("spec_escalate", "task_classifier")

    # Phase 3: 기획서 -> Task Classifier -> (필요 시) Style Planner -> Gate 1
    graph.add_conditional_edges(
        "task_classifier",
        _route_after_classifier,
        {"needs_image": "style_planner", "no_image": GATE_1_NODE},
    )
    graph.add_edge("style_planner", GATE_1_NODE)

    # Gate 1 이후 Fan-out: 설정집 / 비교대상 검색 보고서 / 이미지×N
    graph.add_conditional_edges(
        GATE_1_NODE,
        _dispatch_after_gate1,
        ["setting_doc_worker", "reference_report_worker", "image_worker", END],
    )

    graph.add_edge("setting_doc_worker", "setting_doc_validator")
    graph.add_conditional_edges(
        "setting_doc_validator",
        _make_validation_router("setting_doc"),
        {"done": END, "escalate": "setting_doc_escalate", "retry": "setting_doc_worker"},
    )
    graph.add_edge("setting_doc_escalate", END)

    graph.add_edge("reference_report_worker", "reference_report_validator")
    graph.add_conditional_edges(
        "reference_report_validator",
        _make_validation_router("reference_report"),
        {"done": END, "escalate": "reference_report_escalate", "retry": "reference_report_worker"},
    )
    graph.add_edge("reference_report_escalate", END)

    # 이미지: 매 홉을 Send로 명시해 브랜치 격리를 유지한다(image.py docstring 참고).
    graph.add_conditional_edges("image_worker", route_worker_to_validator, ["image_validator"])
    graph.add_conditional_edges("image_validator", route_after_image_validate, ["image_worker", END])

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=[GATE_0_NODE, GATE_1_NODE])
