"""Minutes Generator 노드. 기획서 3.1, 4.2 참고.

디버그 경로(스펙 2번 섹션): STT/Diarization 없이 텍스트가 바로 이 노드로 들어온다.
입력 transcript로부터 회의록 초안(요약/결정사항/미결 안건)을 생성하면서,
문단 단위로 [논의 외]/[아이디어N] 태그를 함께 붙인다.

모델 티어: 스펙 5.1 — GPT-5.6 Luna. MINUTES_MODEL 환경변수로 오버라이드 가능.
"""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import build_chat_model
from src.state import AgentMeetingState

MINUTES_MODEL = os.getenv("MINUTES_MODEL", "gpt-5.6-luna")

SYSTEM_PROMPT = """당신은 게임 아이디어 브레인스토밍 회의의 회의록 작성 담당자입니다.
주어진 회의 내용을 바탕으로, 문단(불릿) 단위로 정리된 회의록 초안을 작성하세요.

## 출력 형식
- 회의록은 "## 요약", "## 결정사항", "## 미결 안건" 세 섹션으로 구성합니다. 해당 내용이 없는 섹션은 생략해도 됩니다.
- 각 섹션 안의 모든 문단(불릿) 앞에는 반드시 다음 두 태그 중 하나를 붙입니다.
  - `[논의 외]`: 게임 아이디어 논의 자체가 아닌 내용 (잡무 지시, 기존 제품 현황 보고, 인사말 등)
  - `[아이디어N]`: N번째로 제시된 아이디어에 관한 내용 (N은 1부터 시작, 회의에서 새 아이디어가 처음 제시된 순서대로 번호를 매김)
- 태그는 발언 시점이 아니라 **내용이 어느 아이디어에 소속되는가**로 결정합니다. 예를 들어 아이디어를 처음 제시하는 발언 안에 이미 상세 메커니즘 설명이 포함돼 있다면, 그 부분도 같은 아이디어 태그로 묶습니다.
- 이미 회의에서 다뤄졌던 아이디어가 뒤에서 다시 언급되면 같은 번호(N)를 재사용합니다.
- 문법을 다듬거나 문어체로 매끄럽게 고칠 필요는 없습니다. 실제로 오간 내용을 정리하는 데 집중하세요.
- 각 문단은 "- [태그] 내용" 형식의 한 줄(또는 짧은 여러 문장)로 작성합니다.
- 태그나 섹션 제목 외에 다른 메타 설명(예: "다음은 회의록입니다")은 출력하지 마세요.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "다음은 회의 내용입니다.\n\n{transcript}\n\n"
            "위 형식/태깅 규칙에 맞춰 회의록 초안을 작성하세요.",
        ),
    ]
)


def _build_chain(model: str = MINUTES_MODEL):
    llm = build_chat_model(model)
    return _prompt | llm | StrOutputParser()


def generate_minutes(state: AgentMeetingState) -> dict:
    """Minutes Generator 노드: transcript -> meeting_minutes_draft (태그 포함)."""
    chain = _build_chain()
    draft = chain.invoke({"transcript": state["transcript"]})
    return {"meeting_minutes_draft": draft}
