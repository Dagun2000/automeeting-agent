"""LangGraph 상태 스키마. 기획서 5.4를 그대로 구현.

필드를 추가/변경할 경우 AutoMeeting_Agent_기획서_v3.md 5.4와 동기화할 것.
"""
from typing import Annotated, Dict, List, Optional, TypedDict
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


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


def _merge_image_tasks(current: List[ImageTask], update: List[ImageTask]) -> List[ImageTask]:
    """image_tasks 전용 reducer. 스펙 5.4의 원안은 reducer 없는 plain List[ImageTask]지만,
    4.7의 Send 기반 동적 Fan-out(이미지별 독립 Worker-Validator)이 동시에 이 채널에
    쓰기 때문에 reducer가 없으면 LangGraph가 InvalidUpdateError를 낸다(실측 확인).
    subject를 키로 기존 리스트의 해당 항목만 교체하는 방식으로 병합해, 각 이미지
    브랜치가 자기 항목 1개만 담은 리스트를 반환해도 충돌 없이 합쳐지게 한다.
    """
    merged: Dict[str, ImageTask] = {t["subject"]: t for t in current}
    for t in update:
        merged[t["subject"]] = t
    return list(merged.values())


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
    image_tasks: Annotated[List[ImageTask], _merge_image_tasks]

    # 6. 하위 산출물 결과
    setting_doc: Optional[str]
    reference_report: Optional[str]
    # 스펙 5.4 원안에는 없는 필드 - reference_report Worker/Validator를 그래프
    # 레벨 노드로 분리하면서 추가함(그전엔 한 노드 안의 지역 변수였음). 재시도
    # 때 검색 대화를 이어가려면(웹 검색을 매번 처음부터 다시 하지 않으려면)
    # 이 대화 기록이 노드 호출 사이에도 살아있어야 하는데, 별도 그래프 노드는
    # 매번 새 함수 호출이라 지역 변수로는 안 되고 state에 있어야 한다.
    # [src/nodes/reference_report.py](nodes/reference_report.py) 참고.
    reference_report_messages: Annotated[List[AnyMessage], add_messages]

    # 7. 검증/에스컬레이션/기술적 실패
    validation_status: Annotated[Dict[str, TaskFeedback], operator.or_]
    escalated_tasks: Annotated[List[str], operator.add]
    technical_failures: Annotated[Dict[str, str], operator.or_]
