"""Spec Worker / Spec Validator 노드. 기획서 3.3, 4.5(기획서 행), 4.9 참고.

입력은 선정된 아이디어 태그([아이디어N])의 내용만 - 다른 아이디어/[논의 외]
내용은 src.tagging.extract_tagged_content로 걸러내 섞이지 않게 한다.

재시도 루프(4.8, MAX_RETRIES 기반 critique 재주입)는 그래프 레벨의 조건부
엣지로 구현한다([src/graph.py](../graph.py)) - 5.3의 역할 분담대로 재시도
루프 자체는 LangGraph(오케스트레이션) 책임이고, 이 모듈의 노드 함수는 1회
실행 로직만 담당한다.

모델 티어: 스펙 5.1 - GPT-5.6 Luna.
"""
import logging
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.rag import format_rag_context
from src.retry import has_technical_failure, next_feedback
from src.state import AgentMeetingState, TaskFeedback
from src.tagging import extract_tagged_content

SPEC_WORKER_MODEL = os.getenv("SPEC_WORKER_MODEL", "gpt-5.6-luna")
SPEC_VALIDATOR_MODEL = os.getenv("SPEC_VALIDATOR_MODEL", "gpt-5.6-luna")

REQUIRED_SPEC_FIELDS = [
    "장르",
    "핵심 메커니즘",
    "진행 방식",
    "타겟 플랫폼",
    "핵심 게임플레이 루프",
    "타겟 유저층",
    "차별점·셀링포인트",
]

_FIELD_LIST_TEXT = "\n".join(f"{i + 1}. {field}" for i, field in enumerate(REQUIRED_SPEC_FIELDS))

logger = logging.getLogger(__name__)

WORKER_SYSTEM_PROMPT = f"""당신은 게임 기획서 작성 담당자(Spec Worker)입니다. 아래에
주어지는, 회의록에서 선정된 하나의 아이디어에 관한 내용만을 근거로 기획서를
작성하세요.

## 출력 형식
다음 {len(REQUIRED_SPEC_FIELDS)}개 필드를 이 순서대로 포함하는 기획서를 마크다운으로
작성하세요:
{_FIELD_LIST_TEXT}

## 규칙
- 회의 내용에서 명시적으로 언급되지 않은 필드는 절대 지어내지 말고, 값을 정확히
  "TBD"라고 표시하세요.
- 세부 설정/관계는 담지 말고 요약 수준으로 작성하세요(설정집이 아닙니다).
- "핵심 게임플레이 루프" 섹션 아래에는 그 루프를 나타내는 Mermaid 순서도를 반드시
  포함하세요(```mermaid 코드 블록, `flowchart TD` 형식). 회의 내용에서 루프를
  유추할 최소한의 근거조차 없다면 Mermaid 블록 없이 "TBD"라고만 적으세요.
- 이전 시도에 대한 검증 피드백(critique)이 주어지면, **critique가 지적한 필드만
  정확히 고치세요 - 전체를 다시 쓰는 게 아닙니다.** critique에 언급되지 않은
  필드는 이전 기획서 그대로 유지하세요. 문제 없다고 이미 확인된 필드까지 다시
  쓰면 거기서 새 오류가 생길 수 있습니다.

## 참고 자료 활용
사내 용어집/기존 게임 프로필이 "참고 자료"로 주어질 수 있습니다. 이건 용어
표기를 맞추거나 맥락을 이해하는 데만 쓰고, 회의에서 언급되지 않은 내용을
참고 자료에서 가져와 기획서 필드를 채우지 마세요 - 여전히 "회의에서
명시적으로 언급됐는가"만이 근거입니다.
"""

_worker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WORKER_SYSTEM_PROMPT),
        (
            "human",
            "## 선정된 아이디어 관련 회의 내용\n{idea_content}\n\n"
            "## 참고 자료 (사내 용어집/게임 프로필 - RAG 검색 결과)\n{rag_context}\n\n"
            "## 이전 시도 / 검증 피드백\n{feedback_block}\n\n"
            "위 내용을 반영해 기획서를 작성하세요.",
        ),
    ]
)


def _build_worker_chain(model: str = SPEC_WORKER_MODEL):
    llm = build_chat_model(model)
    return _worker_prompt | llm | StrOutputParser()


def spec_worker(state: AgentMeetingState) -> dict:
    """Spec Worker 노드: 선정된 아이디어 내용 -> spec_document.

    기술적 실패(API 에러/타임아웃 등, 4.8)는 통째로 잡아서 technical_failures에
    기록하고 예외를 밖으로 내보내지 않는다 - 그래야 이 트랙만 멈추고 다른
    트랙(설정집/검색 보고서/이미지)은 계속 진행된다(그래프 실행 전체가 죽지
    않음). 내용 품질 실패(critique 재시도)와는 별개의 안전망이라 별도 필드에
    기록한다([src/retry.py](../retry.py) 모듈 docstring 참고).
    """
    try:
        idea_content = extract_tagged_content(
            state["meeting_minutes_confirmed"], state["selected_idea_tag"]
        )

        prev_feedback = state.get("validation_status", {}).get("spec")
        if prev_feedback and not prev_feedback["is_valid"]:
            feedback_block = (
                f"이전 기획서:\n{state.get('spec_document') or '(없음)'}\n\n"
                f"검증 피드백(critique):\n{prev_feedback['critique']}\n\n"
                "critique가 지적한 필드만 고치세요 - 지적되지 않은 필드는 이전"
                " 기획서 그대로 유지하고 다시 쓰지 마세요."
            )
        else:
            feedback_block = "(초기 생성 - 이전 시도 없음)"

        chain = _build_worker_chain()
        spec_document = chain.invoke(
            {
                "idea_content": idea_content,
                "rag_context": format_rag_context(state.get("rag_references") or []),
                "feedback_block": feedback_block,
            }
        )
    except Exception as e:
        logger.error("spec_worker technical failure: %s", e, exc_info=True)
        return {"technical_failures": {"spec": str(e)}}
    return {"spec_document": spec_document}


class SpecValidationResult(BaseModel):
    is_valid: bool = Field(description="필수 슬롯/요구사항/Mermaid 문법이 모두 통과했는가")
    critique: str = Field(description="구체적인 지적사항(통과 시에는 짧은 확인 코멘트)")


VALIDATOR_SYSTEM_PROMPT = f"""당신은 게임 기획서 검증 담당자(Spec Validator)입니다.
"선정된 아이디어 관련 회의 내용"과 "검증 대상 기획서"를 비교해 다음을 확인하세요.

## 체크리스트
1. 필수 슬롯 {len(REQUIRED_SPEC_FIELDS)}개({", ".join(REQUIRED_SPEC_FIELDS)})가
   모두 존재하는가. 회의에서 언급되지 않은 필드는 "TBD"로 명시돼 있어야 하며,
   필드 자체가 빠져 있으면 무효 처리하세요.
2. 회의 내용에서 명확히 언급된 핵심 요구사항(장르, 메커니즘 등)이 기획서에서
   누락되지 않았는가.
3. "핵심 게임플레이 루프" 아래 Mermaid 코드 블록의 문법이 올바른가(`flowchart`/
   `graph` 선언, 노드-화살표 문법, 괄호 짝 등). 근거 부족으로 Mermaid 블록 자체를
   생략한 경우는 문법 오류로 보지 않습니다.
4. 회의에서 언급되지 않은 내용을 지어내지 않았는가(근거 없는 수치·고유명사 등).

하나라도 위반하면 is_valid=false로 판정하고, critique에 무엇을 어떻게 고쳐야
하는지 구체적으로 적으세요. 모두 통과하면 is_valid=true, critique에는 짧은 확인
코멘트만 적으세요.
"""

_validator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VALIDATOR_SYSTEM_PROMPT),
        (
            "human",
            "## 선정된 아이디어 관련 회의 내용\n{idea_content}\n\n"
            "## 검증 대상 기획서\n{spec_document}",
        ),
    ]
)


def _build_validator_chain(model: str = SPEC_VALIDATOR_MODEL):
    llm = build_chat_model(model)
    return _validator_prompt | llm.with_structured_output(SpecValidationResult)


def spec_validator(state: AgentMeetingState) -> dict:
    """Spec Validator 노드: spec_document 검증 -> validation_status["spec"].

    Worker가 이미 기술적 실패를 기록했으면 검증 자체를 건너뛴다(없는/부실한
    산출물을 검증해봐야 의미가 없고, API 호출만 낭비). Validator 자신의
    LLM 호출이 실패하는 경우도 별도로 잡아 technical_failures에 기록한다.
    """
    if has_technical_failure(state, "spec"):
        return {}

    try:
        idea_content = extract_tagged_content(
            state["meeting_minutes_confirmed"], state["selected_idea_tag"]
        )

        chain = _build_validator_chain()
        result: SpecValidationResult = chain.invoke(
            {"idea_content": idea_content, "spec_document": state["spec_document"]}
        )
    except Exception as e:
        logger.error("spec_validator technical failure: %s", e, exc_info=True)
        return {"technical_failures": {"spec": str(e)}}

    prev_feedback = state.get("validation_status", {}).get("spec")
    feedback: TaskFeedback = next_feedback(prev_feedback, result.is_valid, result.critique)

    if not result.is_valid:
        logger.warning(
            "spec_validator FAILED (retry_count=%d) - critique: %s\n"
            "--- 검증 대상 기획서 ---\n%s",
            feedback["retry_count"],
            result.critique,
            state["spec_document"],
        )
    else:
        logger.info("spec_validator passed (retry_count=%d)", feedback["retry_count"])

    return {"validation_status": {"spec": feedback}}
