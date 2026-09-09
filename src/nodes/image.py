"""이미지 Worker/Validator 노드. 기획서 4.6, 4.7 참고.

image_tasks 중 enabled인 항목 개수만큼 LangGraph `Send` API로 동적 Fan-out한다
(dispatch_image_tasks, [src/graph.py](../graph.py)의 Gate 1 이후 조건부
엣지에서 호출). image_worker/image_validator는 spec/설정집/비교대상 검색
보고서와 동일하게 **그래프 레벨의 별도 노드**다 - 재시도(worker<->validator)도
내부 서브그래프가 아니라 그래프 조건부 엣지가 매번 새 `Send`로 담당한다.

한때는 이미지별 재시도 루프 전체(Worker+Validator+재시도)를 이미지 하나마다
독립된 서브그래프로 캡슐화했었다 - Send로 병렬 실행되는 여러 브랜치가 같은
채널(image_tasks)에 동시에 쓰기 때문에, 그래프 레벨로 쪼개면 validator가
"어느 이미지를 검증해야 하는지" 알 수 없어(같은 이름의 여러 브랜치가 합쳐진
전역 state를 보게 됨) 충돌이 날 거라고 판단했었다. 그런데 실측 확인 결과
이 판단이 틀렸다 - `Send`로 파견된 브랜치가 조건부 엣지에서 *다시* `Send`로
다음 노드를 명시적으로 호출하면(예: `Send("image_validator", {...전체
페이로드...})`), 그 브랜치는 자기 자신의 로컬 state만 보고 다른 형제
브랜치와 섞이지 않는다(합쳐진 전역 state를 보는 건 일반 `add_edge`로 연결된
경우뿐). 재시도(validator -> worker)도 같은 방식(다시 `Send`)으로 하면
동일하게 안전하다 - 별도 합성 테스트로 검증(빠른 브랜치와 2번 재시도가
필요한 느린 브랜치를 동시에 돌려도 서로 섞이지 않고 각자 정확한 재시도
횟수로 수렴함을 확인).

그래프 레벨로 쪼갠 이유: image_worker가 통째로(생성+검증+재시도) 하나의
불투명한 노드였을 때는, 이 노드가 완전히 끝나야만(내부 검증까지 전부 끝나야)
LangGraph의 Pregel 슈퍼스텝이 다음 단계로 넘어갔다 - 그래서 Gate 1에서 같이
파견된 설정집/비교대상 검색 보고서 Validator가 자기 Worker는 훨씬 전에 끝냈는데도
이미지 "검증"까지 다 끝날 때까지 기다려야 했다(사용자가 실측 지적으로 발견).
쪼개고 나면 슈퍼스텝은 "이미지 생성(worker)"까지만 기다리면 되고, 그 다음
슈퍼스텝에서 이미지 validator와 설정집/검색 보고서 validator가 동시에
스케줄된다 - 이미지 자체 검증/재시도에 걸리는 시간은 더 이상 다른 트랙을
붙잡지 않는다.

이 분리 이후로는 이미지도 [src/ui/app.py](../ui/app.py)의 최상위 이벤트
스트림에 바로 잡히므로(예전처럼 `subgraphs=True`로 별도 namespace를 봐야
하는 중첩 서브그래프가 아님), 이미지별 Worker/Validator 진행 상황은
`payload["input"]["subject"]`로 바로 구분한다.

이미지 생성은 채팅 모델 호출이 아니라 openai 이미지 API를 직접 호출한다.
스펙 5.2의 원안은 DALL-E 3였지만, 그 모델은 API에서 폐지돼(`model 'dall-e-3'
does not exist`, 실제 API 에러로 확인) GPT 이미지 모델(`gpt-image-1` 등)로
대체했다 - 이 모델군은 DALL-E와 달리 URL이 아니라 base64(`b64_json`)만
반환하므로, `image_url` 필드에는 data URI(`data:image/png;base64,...`)를
그대로 넣는다(브라우저/Streamlit img 태그, vision 입력 모두 data URI를
직접 지원). 프롬프트 자체는 스타일 가이드+대상+묘사를 그대로 이어붙이는
결정적 조합이라 별도 LLM 호출 없이 구성한다(4.5: "대상별로 독립 프롬프트
작성"). Validator는 vision 입력이 필요해 이미지를 직접 검토한다(5.1 원안은
Terra였으나, 지금은 전부 GPT-5.6 Luna로 통일 - 문제 생기면
IMAGE_VALIDATOR_MODEL만 개별 상향).
"""
import logging
import os
from typing import Optional

from langgraph.types import Send
from openai import OpenAI
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.retry import next_feedback, route_decision
from src.state import AgentMeetingState, ImageTask

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2")
IMAGE_VALIDATOR_MODEL = os.getenv("IMAGE_VALIDATOR_MODEL", "gpt-5.6-luna")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

logger = logging.getLogger(__name__)


def _build_image_prompt(subject: str, brief: str, style_guide: str, critique: Optional[str]) -> str:
    parts = [style_guide, f"대상: {subject}", f"세부 묘사: {brief}"]
    if critique:
        parts.append(f"이전 시도에 대한 피드백을 반영: {critique}")
    parts.append("인물이나 캐릭터는 포함하지 말 것 - 환경/풍경/사물만 그릴 것.")
    return "\n".join(p for p in parts if p)


def _generate_image(prompt: str) -> str:
    """이미지를 생성하고 data URI(data:image/png;base64,...) 문자열로 반환."""
    client = OpenAI()
    response = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", n=1)
    b64_json = response.data[0].b64_json
    if b64_json:
        return f"data:image/png;base64,{b64_json}"
    return response.data[0].url


class ImageValidationResult(BaseModel):
    is_valid: bool = Field(
        description="금지 요소 없음 + 톤앤매너 일치 + 스타일 가이드 반영 여부가 모두 통과했는가"
    )
    critique: str = Field(description="구체적인 지적사항 (통과 시에는 짧은 확인 코멘트)")


_VALIDATOR_SYSTEM_PROMPT = """당신은 게임 컨셉 아트 이미지 검증 담당자입니다.
주어진 이미지를 아래 기준으로 검토하세요.

1. 금지 요소: 인물/캐릭터가 그려져 있으면 무효(대상은 환경 요소로
   한정됩니다). 선정성/폭력성 등 부적절한 요소가 있어도 무효.
2. 톤앤매너 일치: 공유 스타일 가이드 및 대상 설명과 이미지의 분위기가
   맞는가.
3. 스타일 가이드 반영: 색감/렌더링 스타일 등이 공유 스타일 가이드를
   따르는가.

하나라도 위반하면 is_valid=false로 판정하고, critique에 무엇이 문제인지
구체적으로 적으세요(재생성 시 프롬프트에 반영됩니다). 모두 통과하면
is_valid=true, critique는 짧은 확인 코멘트만 적으세요.
"""


def _build_validator(model: str = IMAGE_VALIDATOR_MODEL):
    llm = build_chat_model(model)
    return llm.with_structured_output(ImageValidationResult)


def _validate_image(subject: str, brief: str, style_guide: str, image_url: str) -> ImageValidationResult:
    validator = _build_validator()
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": _VALIDATOR_SYSTEM_PROMPT},
            {
                "type": "text",
                "text": (
                    f"대상: {subject}\n세부 묘사: {brief}\n공유 스타일 가이드: {style_guide}\n"
                    "위 기준에 따라 아래 이미지를 검토하세요."
                ),
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
    return validator.invoke([message])


def dispatch_image_tasks(state: AgentMeetingState) -> list:
    """image_tasks 중 enabled인 것만 Send로 병렬 파견. 그래프 노드가 아니라
    Gate 1 이후 조건부 엣지 함수([src/graph.py](../graph.py))에서 호출한다."""
    return [
        Send(
            "image_worker",
            {
                "subject": t["subject"],
                "brief": t["brief"],
                "style_guide": state.get("image_style_guide") or "",
                "feedback": None,
            },
        )
        for t in state["image_tasks"]
        if t["enabled"]
    ]


def image_worker(payload: dict) -> dict:
    """이미지 Worker 노드: 이미지 하나를 생성한다(재시도 시 이전 critique
    반영). subject/brief/style_guide/feedback을 그대로 반환해 다음 홉
    (route_worker_to_validator)이 이어받게 한다 - 이 값들은 이 브랜치의
    로컬 state일 뿐 전역에 병합되는 채널이 아니므로(모듈 docstring 참고),
    다음 노드에 명시적으로 다시 넘겨줘야 한다.
    """
    prev = payload.get("feedback")
    critique = prev["critique"] if prev and not prev["is_valid"] else None
    prompt = _build_image_prompt(payload["subject"], payload["brief"], payload["style_guide"], critique)
    image_url = _generate_image(prompt)
    return {
        "subject": payload["subject"],
        "brief": payload["brief"],
        "style_guide": payload["style_guide"],
        "image_url": image_url,
        "feedback": prev,
    }


def route_worker_to_validator(state: dict) -> list:
    """image_worker 다음 홉 - 반드시 `Send`로 명시해야 이 브랜치의 subject가
    다른 이미지 브랜치와 안 섞인다(모듈 docstring 참고, 실측 확인)."""
    return [
        Send(
            "image_validator",
            {
                "subject": state["subject"],
                "brief": state["brief"],
                "style_guide": state["style_guide"],
                "image_url": state["image_url"],
                "feedback": state.get("feedback"),
            },
        )
    ]


def image_validator(state: dict) -> dict:
    """이미지 Validator 노드: 이미지를 검증하고, 매 시도마다 최신 결과를
    `image_tasks`/`validation_status`에 기록한다(재시도 중이어도 항상 최신
    시도 결과로 덮어써서, 어느 시점에 봐도 "가장 최근 시도"가 반영되게 한다
    - image_tasks 커스텀 reducer가 subject 키 기준으로 병합하므로 여러 번
    써도 안전).
    """
    subject = state["subject"]
    result = _validate_image(subject, state["brief"], state["style_guide"], state["image_url"])
    feedback = next_feedback(state.get("feedback"), result.is_valid, result.critique)

    updated_task: ImageTask = {
        "subject": subject,
        "enabled": True,
        "brief": state["brief"],
        "image_url": state["image_url"],
        "validation": feedback,
    }
    update: dict = {
        "subject": subject,
        "brief": state["brief"],
        "style_guide": state["style_guide"],
        "image_url": state["image_url"],
        "feedback": feedback,
        "image_tasks": [updated_task],
        "validation_status": {f"image_{subject}": feedback},
    }

    if not feedback["is_valid"] and feedback["retry_count"] > MAX_RETRIES:
        update["escalated_tasks"] = [f"image_{subject}"]
        logger.error(
            "image_worker(%s) ESCALATED after retry limit - critique: %s", subject, feedback["critique"]
        )
    elif not feedback["is_valid"]:
        logger.warning(
            "image_worker(%s) validation FAILED (retry_count=%d) - critique: %s",
            subject,
            feedback["retry_count"],
            feedback["critique"],
        )
    return update


def route_after_image_validate(state: dict) -> list:
    """image_validator 다음 홉 - 재시도면 `Send`로 image_worker를 다시
    호출(모듈 docstring 참고), 아니면 빈 리스트를 반환해 이 브랜치를
    자연스럽게 끝낸다(다른 브랜치에 영향 없음, 실측 확인)."""
    decision = route_decision(state["feedback"], MAX_RETRIES)
    if decision != "retry":
        return []
    return [
        Send(
            "image_worker",
            {
                "subject": state["subject"],
                "brief": state["brief"],
                "style_guide": state["style_guide"],
                "feedback": state["feedback"],
            },
        )
    ]
