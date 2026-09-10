"""Worker-Validator 재시도 루프 공용 헬퍼. 기획서 4.8 참고.

기획서(Spec)/설정집/비교대상 검색 보고서, 이미지 서브파이프라인이 모두 동일한
critique 기반 재시도 정책(MAX_RETRIES)을 쓰므로 여기 모아 공유한다. 다음에
어느 노드로 갈지 결정하는 라우팅 자체는 각 그래프의 조건부 엣지가 담당하고
(5.3: 루프는 LangGraph 오케스트레이션 책임), 이 모듈은 그 판단에 필요한
TaskFeedback 계산만 담당한다.

**내용 품질 실패(critique 기반 재시도)와 기술적 실패(API 에러 등)는 서로
다른 안전망이다**(4.8) - 전자는 여기 `next_feedback`/`route_decision`이,
후자는 각 Worker/Validator 노드 함수 안의 try/except가 `technical_failures`
state 필드에 직접 기록한다(`has_technical_failure` 참고). 기술적 실패는
재시도하지 않고 그 자리에서 멈춘다(API 에러가 한 번 났다고 곧바로 다시
호출해도 성공할 거라는 보장이 없고, 스펙 4.8도 "실패 시 해당 트랙만
technical_failure 상태로 표시하고 나머지는 계속 진행"이라고만 하지 자동
재시도를 요구하지 않는다) - 그래서 각 트랙의 라우팅 함수는 항상
`has_technical_failure`부터 확인하고, 있으면 `route_decision` 없이 바로
멈추는 쪽으로 라우팅한다.
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


def has_technical_failure(state: dict, track: str) -> bool:
    """이 트랙이 이미 기술적 실패로 기록됐는지 확인한다. Worker가 실패를
    기록하면 뒤이은 Validator는 이 함수로 확인하고 자기 검증 로직(LLM 호출
    포함)을 건너뛴다 - 이미 없는/부실한 산출물을 검증해봐야 의미가 없고,
    쓸데없는 API 호출만 늘어난다. 그래프 라우팅 함수도 이걸로 재시도 여부를
    가른다(모듈 docstring 참고)."""
    return bool((state.get("technical_failures") or {}).get(track))
