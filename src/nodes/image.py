"""이미지 Worker/Validator 노드. 기획서 4.6, 4.7 참고.

image_tasks 중 enabled인 항목 개수만큼 LangGraph `Send` API로 동적 Fan-out한다
(dispatch_image_tasks, [src/graph.py](../graph.py)의 Gate 1 이후 조건부
엣지에서 호출). 이미지별 재시도 루프(critique 기반)는 이미지 하나마다 독립된
서브그래프(_build_image_pipeline)로 실행한다 - Send로 병렬 실행되는 여러
브랜치가 같은 채널(image_tasks)에 동시에 쓰기 때문에, 재시도까지 포함한 한
이미지의 전체 처리를 서브그래프 하나의 단일 호출로 캡슐화해 부모 그래프에는
결과 1건만 반환한다(충돌 없는 병합은 [src/state.py](../state.py)의
image_tasks 커스텀 reducer가 담당 - 실측으로 필요성 확인).

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
import os
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import OpenAI
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.retry import next_feedback, route_decision
from src.state import AgentMeetingState, ImageTask, TaskFeedback

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2")
IMAGE_VALIDATOR_MODEL = os.getenv("IMAGE_VALIDATOR_MODEL", "gpt-5.6-luna")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


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


class ImagePipelineState(TypedDict):
    subject: str
    brief: str
    style_guide: str
    image_url: Optional[str]
    feedback: Optional[TaskFeedback]


def _pipeline_worker(state: ImagePipelineState) -> dict:
    prev = state.get("feedback")
    critique = prev["critique"] if prev and not prev["is_valid"] else None
    prompt = _build_image_prompt(state["subject"], state["brief"], state["style_guide"], critique)
    image_url = _generate_image(prompt)
    return {"image_url": image_url}


def _pipeline_validator(state: ImagePipelineState) -> dict:
    result = _validate_image(state["subject"], state["brief"], state["style_guide"], state["image_url"])
    feedback = next_feedback(state.get("feedback"), result.is_valid, result.critique)
    return {"feedback": feedback}


def _pipeline_route(state: ImagePipelineState) -> str:
    decision = route_decision(state["feedback"], MAX_RETRIES)
    return "retry" if decision == "retry" else "stop"


def _build_image_pipeline():
    graph = StateGraph(ImagePipelineState)
    graph.add_node("worker", _pipeline_worker)
    graph.add_node("validator", _pipeline_validator)
    graph.add_edge(START, "worker")
    graph.add_edge("worker", "validator")
    graph.add_conditional_edges("validator", _pipeline_route, {"retry": "worker", "stop": END})
    return graph.compile()


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
            },
        )
        for t in state["image_tasks"]
        if t["enabled"]
    ]


def image_worker(payload: dict) -> dict:
    """부모 그래프의 이미지 노드: Send로 받은 이미지 하나를 서브파이프라인
    (재시도 포함)으로 끝까지 처리하고, 결과 1건만 부모 상태에 반환한다."""
    pipeline = _build_image_pipeline()
    result = pipeline.invoke(
        {
            "subject": payload["subject"],
            "brief": payload["brief"],
            "style_guide": payload["style_guide"],
            "image_url": None,
            "feedback": None,
        }
    )

    feedback = result["feedback"]
    updated_task: ImageTask = {
        "subject": payload["subject"],
        "enabled": True,
        "brief": payload["brief"],
        "image_url": result["image_url"],
        "validation": feedback,
    }

    update: dict = {
        "image_tasks": [updated_task],
        "validation_status": {f"image_{payload['subject']}": feedback},
    }
    if not feedback["is_valid"] and feedback["retry_count"] >= MAX_RETRIES:
        update["escalated_tasks"] = [f"image_{payload['subject']}"]
    return update
