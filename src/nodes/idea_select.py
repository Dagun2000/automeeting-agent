"""아이디어 선정 노드. 기획서 3.2 참고.

Gate 0에서 확정된 회의록(태그 포함)을 입력받아, [논의 외]를 제외한 아이디어
태그들 중 구체적으로 논의가 진전된 아이디어가 하나 있는지 판단한다.

판단은 발언 시점이 아니라 태그 소속 기준(3.1)을 그대로 따른다 — 어떤 아이디어에
대한 초기 pitch 발언이라도 이미 해당 아이디어 태그로 묶여 있다면(3.1의 소속 기준
태깅 규칙), 그 문단도 해당 아이디어의 논의 근거로 소급 포함해서 판단한다. 즉 회의록
안에서 태그별 문단들의 등장 순서는 무시하고, 같은 태그의 문단 전체를 하나로 놓고
"이 아이디어가 구체적으로 진전됐는가"를 판단한다.

모델 티어: 스펙 5.1 - GPT-5.6 Luna.
"""
import os
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.state import AgentMeetingState

IDEA_SELECT_MODEL = os.getenv("IDEA_SELECT_MODEL", "gpt-5.6-luna")

SYSTEM_PROMPT = """당신은 게임 아이디어 브레인스토밍 회의록에서, 하나의 아이디어로
논의가 좁혀졌는지 판단하는 담당자입니다.

## 입력
문단(불릿) 단위로 [논의 외] 또는 [아이디어N] 태그가 붙은 회의록입니다. 태그는
발언 시점이 아니라 내용의 소속 기준으로 이미 붙어 있습니다 — 즉 어떤 아이디어에
대한 초기 제안 발언 안에 이미 상세 설명이 포함돼 있었다면, 그 발언도 해당 아이디어
태그로 이미 함께 묶여 있습니다. 같은 태그를 가진 문단들은 회의록 안에서의 등장
순서와 무관하게 모두 같은 아이디어에 대한 근거로 취급하세요(시점이 아니라 태그
소급 적용 기준).

## 판단 기준
- [논의 외] 문단은 판단에서 제외합니다.
- 남은 아이디어 태그별로, 그 태그에 속한 모든 문단을 한데 모아 놓고 볼 때 논의가
  구체적으로 진전됐는지 판단하세요. 예: "이거 괜찮네, 자세히 말해봐" 같은 결정
  신호 이후 상세 설명(메커니즘, 세계관, 시스템 등)이 이어졌는가.
- 여러 아이디어가 제시만 되고 하나로 좁혀지지 않았다면(구체화된 아이디어 없음)
  -> selected_idea_tag = null.
- 정확히 하나의 아이디어만 구체적으로 진전됐다면 -> 그 태그 이름을 그대로
  (예: "아이디어2") selected_idea_tag에 반환하세요.
- 두 개 이상의 아이디어가 동시에 비슷한 수준으로 구체화된 경우는 이 시스템의
  지원 범위 밖입니다. 이 경우에도 selected_idea_tag = null로 반환하세요(하나로
  좁혀지지 않은 것과 동일하게 처리).
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "다음은 확정된 회의록입니다.\n\n{minutes}\n\n위 기준에 따라 판단하세요."),
    ]
)


class IdeaSelectionResult(BaseModel):
    selected_idea_tag: Optional[str] = Field(
        default=None,
        description='구체적으로 논의가 진전된 유일한 아이디어 태그 (예: "아이디어2"). 없거나 애매하면 null.',
    )


def _build_chain(model: str = IDEA_SELECT_MODEL):
    llm = build_chat_model(model)
    return _prompt | llm.with_structured_output(IdeaSelectionResult)


def select_idea(state: AgentMeetingState) -> dict:
    """아이디어 선정 노드: meeting_minutes_confirmed -> selected_idea_tag."""
    chain = _build_chain()
    result: IdeaSelectionResult = chain.invoke({"minutes": state["meeting_minutes_confirmed"]})
    return {"selected_idea_tag": result.selected_idea_tag}
