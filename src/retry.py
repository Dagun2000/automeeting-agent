"""Worker-Validator 재시도 루프 공용 헬퍼. 기획서 4.8 참고.

기획서(Spec)/설정집/비교대상 검색 보고서, 이미지 서브파이프라인이 모두 동일한
critique 기반 재시도 정책(MAX_RETRIES)을 쓰므로 여기 모아 공유한다. 다음에
어느 노드로 갈지 결정하는 라우팅 자체는 각 그래프의 조건부 엣지가 담당하고
(5.3: 루프는 LangGraph 오케스트레이션 책임), 이 모듈은 그 판단에 필요한
TaskFeedback 계산만 담당한다.
"""
from typing import Optional

from src.state import TaskFeedback


def next_feedback(prev: Optional[TaskFeedback], is_valid: bool, critique: str) -> TaskFeedback:
    """검증 결과로부터 다음 TaskFeedback(재시도 횟수 갱신 포함)을 계산."""
    prev_retry_count = prev["retry_count"] if prev else 0
    retry_count = prev_retry_count if is_valid else prev_retry_count + 1
    return {"is_valid": is_valid, "critique": critique, "retry_count": retry_count}


def route_decision(feedback: TaskFeedback, max_retries: int) -> str:
    """"done" / "retry" / "escalate" 중 하나를 반환.

    retry_count는 "지금까지 실패한 시도 횟수"이므로, 1차 시도가 실패한
    시점에 이미 1이 된다. 그래서 `retry_count >= max_retries`로 비교하면
    max_retries=2일 때 재시도가 실제로는 1번만 일어나고 에스컬레이션되는
    오프바이원 버그가 있었다(실측 확인). "MAX_RETRIES번까지 재시도"라는
    이름 그대로(총 MAX_RETRIES+1번 시도) 동작하도록 `>`로 비교한다.
    """
    if feedback["is_valid"]:
        return "done"
    if feedback["retry_count"] > max_retries:
        return "escalate"
    return "retry"
