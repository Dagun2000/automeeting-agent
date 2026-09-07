"""Image Style Planner 노드. 기획서 4.6 참고.

needs_image가 True(= image_tasks가 비어있지 않음)인 경우에만 그래프에서
호출된다(라우팅은 [src/graph.py](../graph.py)). 기획서(톤/장르/설정)와 이미지
대상 목록을 종합해, 모든 이미지 Worker가 공유할 스타일 가이드 문자열 하나를
만든다.

모델 티어: 스펙 5.1 - GPT-5.6 Luna.
"""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import build_chat_model
from src.state import AgentMeetingState

STYLE_PLANNER_MODEL = os.getenv("STYLE_PLANNER_MODEL", "gpt-5.6-luna")

SYSTEM_PROMPT = """당신은 게임 컨셉 아트의 아트 디렉터(Image Style Planner)입니다.
아래 기획서와 이미지로 그릴 대상 목록을 보고, 모든 이미지에 공통으로 적용할
"공유 스타일 가이드"를 한 단락으로 작성하세요.

## 포함할 내용
- 전체적인 톤앤매너/분위기
- 색감/조명
- 렌더링 스타일(예: 사실적 컨셉 아트, 반사실적 페인팅 등 - 기획서/대상
  목록에서 유추 가능한 범위 내에서)
- 대상들 사이에 공통으로 지켜야 할 시각적 모티프(있다면)

## 주의
- 이미지 생성 API는 seed 고정이나 레퍼런스 이미지 컨디셔닝을 지원하지
  않으므로, 이 가이드는 "완전한 일관성"이 아니라 "프롬프트 레벨 일관성"을
  목표로 합니다. 각 이미지 프롬프트에 그대로 삽입해도 자연스러운 문장으로
  작성하세요.
- 인물/캐릭터에 대한 지시는 넣지 마세요(이미지 대상은 환경 요소로
  한정됩니다).
- 회의/기획서에 없는 내용을 과도하게 지어내지 말고, 있는 톤 정보를 정리하는
  수준으로 작성하세요.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "## 기획서\n{spec_document}\n\n## 이미지 대상 목록\n{image_subjects}",
        ),
    ]
)


def _build_chain(model: str = STYLE_PLANNER_MODEL):
    llm = build_chat_model(model)
    return _prompt | llm | StrOutputParser()


def plan_image_style(state: AgentMeetingState) -> dict:
    """Image Style Planner 노드: spec_document + image_tasks -> image_style_guide."""
    image_subjects = "\n".join(f"- {t['subject']}: {t['brief']}" for t in state["image_tasks"])

    chain = _build_chain()
    guide = chain.invoke({"spec_document": state["spec_document"], "image_subjects": image_subjects})
    return {"image_style_guide": guide}
