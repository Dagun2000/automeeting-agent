"""LangGraph 그래프 골격 - Phase 1~3 범위.

텍스트 입력 -> Minutes Generator -> Gate 0(interrupt_before)
   -> 아이디어 선정 -> (없으면 종료, 회의록만 출력)
   -> (있으면) Spec Worker -> Spec Validator -> 재시도 루프
   -> Task Classifier -> (needs_image면) Image Style Planner
   -> Gate 1(interrupt_before)
   -> Fan-out: 설정집 / 비교대상 검색 보고서 / 이미지×N(Send 동적 생성)
      각 트랙 독립 재시도 루프(critique 기반, MAX_RETRIES) -> 통과/한도초과 시 종료

기획서 트랙이 재시도 한도를 초과해도(escalate) 이후 파이프라인은 계속
진행한다 - Gate 2(에스컬레이션 확인)는 Fan-out 이후 한 번에 모아 보여주는
단계이므로(2번 섹션 10단계), 한 트랙의 품질 실패가 다른 트랙 진행을 막지
않는다(4.8, "나머지는 계속 진행"과 동일한 원칙을 품질 실패에도 적용).

Aggregator/ZIP/Gate 2 UI는 아직 없음(Phase 4).
"""
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes.classifier import classify_tasks
from src.nodes.idea_select import select_idea
from src.nodes.image import dispatch_image_tasks, image_worker
from src.nodes.minutes import generate_minutes
from src.nodes.reference_report import reference_report_validator, reference_report_worker
from src.nodes.setting_doc import setting_doc_validator, setting_doc_worker
from src.nodes.spec import spec_validator, spec_worker
from src.nodes.style_planner import plan_image_style
from src.retry import route_decision
from src.state import AgentMeetingState

GATE_0_NODE = "gate_0"
GATE_1_NODE = "gate_1"

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
        return {"escalated_tasks": [track]}

    return _escalate


def build_graph():
    graph = StateGraph(AgentMeetingState)

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

    # Phase 1~2: 텍스트 입력 -> 회의록 -> Gate 0 -> 아이디어 선정 -> 기획서
    graph.add_edge(START, "minutes_generator")
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

    # 이미지는 image_worker 노드 내부에서 서브파이프라인이 재시도까지 전부 처리한다.
    graph.add_edge("image_worker", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=[GATE_0_NODE, GATE_1_NODE])
