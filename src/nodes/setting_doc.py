"""설정집 Worker/Validator 노드. 기획서 4.5(설정집 행), 4.9 참고.

파이프라인 순서: 기획서 생성 -> Task Classifier(무엇이 필요한지 판단 +
브리핑 작성) -> 설정집 Worker(브리핑대로 채움) -> Validator(완결성 체크).
"기획서 대비 새 정보인가"는 Task Classifier가 브리핑을 쓸 때 이미 판단한다
([src/nodes/classifier.py](classifier.py)) - 여기서는 다시 판단하지 않는다.

재시도 루프는 [src/nodes/spec.py](spec.py)와 동일하게 그래프 레벨 조건부
엣지가 담당한다([src/graph.py](../graph.py), [src/retry.py](../retry.py) 공용
헬퍼 사용).

모델 티어: 스펙 5.1 - GPT-5.6 Luna (Worker/Validator 둘 다).

경위(실측 기록): Validator가 한때 사건 표 관련 세부 판단(날조/누락/형식)
까지 맡았을 때, 같은 설정집을 두고 재시도 3회 동안 "사건 없음, 빈 표
허용" -> "여전히 사건 없음, 표 형식만 문제" -> "'왕국의 몰락'이라는
사건이 빠졌음"으로 스스로 모순되는 판정을 내렸다(원본 회의록 대조 결과
마지막 판정 자체가 틀렸음 - "몰락한 왕국의 마지막 기사"는 형용사적 배경
묘사일 뿐 서술된 사건이 아니었다). Validator를 Terra로 올려봐도 이런
유의 판단은 여전히 불안정했다. 근본 해결책은 "사건이 실제로 있었는가"
판단 자체를 Validator/Worker에서 걷어내 Task Classifier가 브리핑을 쓸 때
한 번만 하도록 옮기는 것이었다(setting_doc_brief에 "사건 목록" 추가,
[src/nodes/classifier.py](classifier.py) 참고) - 엔티티 목록과 완전히
같은 패턴이다. 그 결과 Validator는 이제 "브리핑 항목(엔티티+사건) 완결성"
과 "유형 카테고리 준수" 두 가지 기계적 대조만 하면 되므로, Terra로 올릴
필요 없이 Luna로 충분하다(원복함).
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

SETTING_DOC_WORKER_MODEL = os.getenv("SETTING_DOC_WORKER_MODEL", "gpt-5.6-luna")
SETTING_DOC_VALIDATOR_MODEL = os.getenv("SETTING_DOC_VALIDATOR_MODEL", "gpt-5.6-luna")

logger = logging.getLogger(__name__)

WORKER_SYSTEM_PROMPT = """당신은 게임 설정집 작성 담당자(Setting Doc Worker)입니다.
"생성 지시문(브리핑)"에 무엇을 포함할지 이미 확정돼 있습니다 - 그 목록을
"선정된 아이디어 관련 회의 내용"에서 찾아 옮기는 게 이 작업입니다. 브리핑에
없는 엔티티/사실을 임의로 추가하거나, 브리핑에 있는 항목을 스스로 판단해서
빼지 마세요.

**이건 추출 작업이지 창작 작업이 아닙니다.** 회의에서 짧게만 언급된 내용은
짧은 그대로 적으세요. 회의가 말하지 않은 디테일(정확한 수치·과정·동기·배경
사연 등)을 만들어서 채우지 마세요 - 항목이 짧다고 부족한 게 아닙니다.

## 출력 형식
### 1. 설정 시트 (엔티티별 표, 컬럼: 이름 / 유형 / 핵심 속성 / 언급 맥락)
- 유형은 반드시 **인물 / 장소 / 물건(유물) / 세력 / 종족** 중 하나여야
  합니다(닫힌 목록). "동료 NPC"·"동료들"·"시스템"·"집단" 같은 통칭 행을
  만들지 마세요 - 여러 인물을 묶어 부르는 표현이 브리핑에 있어도, 실제로는
  이름이 있는 각 인물 항목에 관련 내용을 나눠 넣으세요.
- 브리핑에 나열된 엔티티를 행으로 포함하고, 회의 내용에서 브리핑이 가리키는
  부분을 찾아 그 내용을 "핵심 속성"에 반영하세요. 회의는 대화체라 그대로
  베낄 문장이 없을 수 있습니다 - 문장을 그대로 베끼라는 뜻이 아니라, 회의에
  없는 내용을 새로 추가하지 말라는 뜻입니다.
- 강제 슬롯은 없습니다 - 언급된 필드만 표시하고, 언급 안 된 속성은 억지로
  채우지 마세요.

### 2. 사건 표 (컬럼: 발생 시점 / 사건 / 관련 엔티티 / 결과·관계)
브리핑의 "사건 목록"에 무엇을 넣을지 이미 확정돼 있습니다 - 설정 시트와
마찬가지로 그 목록을 회의 내용에서 찾아 옮기는 게 이 작업입니다. 사건이
있는지 없는지 스스로 다시 판단하지 마세요: 브리핑이 "사건 목록: 없음"이면
사건 표는 빈 채로 두고, 목록이 있으면 그 사건들을 빠짐없이 표에 옮기세요.

- 각 사건의 내용은 회의에서 실제로 언급된 것만 반영하세요(문장을 그대로
  베낄 필요는 없습니다) - "시작됐다", "대립했다"처럼 회의에 없던 인과·
  서사 연결어를 새로 붙이지 마세요.
- 발생 시점은 회의에서 명시적으로 언급된 경우에만 채우고, 그렇지 않으면
  빈칸으로 두세요.

## 표 작성 규칙 (형식)
설정 시트/사건 표 모두, 헤더 행 / 구분선 행(`---`) / 데이터 행의 **컬럼 수가
반드시 서로 같아야 합니다.** 각 표를 다 쓴 뒤 헤더의 `|` 개수와 모든 행의
`|` 개수가 일치하는지 스스로 다시 확인하세요.

## 규칙
- 회의에서 언급되지 않은 내용을 지어내지 마세요.
- 세부 관계의 영속적 일관성(DB화)이나 모순 검증은 이 작업의 범위가
  아닙니다.
- 이전 시도에 대한 검증 피드백(critique)이 주어지면, **critique가 지적한
  행/부분만 정확히 고치세요 - 전체를 다시 쓰는 게 아닙니다.** critique에
  언급되지 않은 행·항목은 이전 설정집 그대로 유지하세요. 문제 없다고
  이미 확인된 내용까지 다시 쓰면 거기서 새 오류(형식 붕괴, 새로운 날조 등)가
  생길 수 있습니다.
- "참고 자료"(사내 용어집/게임 프로필)는 용어 표기를 맞추는 데만 쓰고,
  회의에 없는 내용을 참고 자료에서 가져와 채우지 마세요.
"""

_worker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WORKER_SYSTEM_PROMPT),
        (
            "human",
            "## 생성 지시문(브리핑) - 포함할 항목이 이미 확정돼 있습니다\n{brief}\n\n"
            "## 선정된 아이디어 관련 회의 내용\n{idea_content}\n\n"
            "## 참고 자료 (사내 용어집/게임 프로필 - RAG 검색 결과)\n{rag_context}\n\n"
            "## 이전 시도 / 검증 피드백\n{feedback_block}\n\n"
            "위 브리핑을 그대로 실행해 설정집을 작성하세요.",
        ),
    ]
)


def _build_worker_chain(model: str = SETTING_DOC_WORKER_MODEL):
    llm = build_chat_model(model)
    return _worker_prompt | llm | StrOutputParser()


def setting_doc_worker(state: AgentMeetingState) -> dict:
    """설정집 Worker 노드: 브리핑 + 아이디어 내용 -> setting_doc.

    기술적 실패(4.8)는 통째로 잡아 technical_failures에 기록하고 예외를
    밖으로 내보내지 않는다 - 이 트랙만 멈추고 다른 트랙은 계속 진행된다.
    """
    try:
        idea_content = extract_tagged_content(
            state["meeting_minutes_confirmed"], state["selected_idea_tag"]
        )
        brief = state["task_briefs"]["setting_doc"]["brief"]

        prev_feedback = state.get("validation_status", {}).get("setting_doc")
        if prev_feedback and not prev_feedback["is_valid"]:
            feedback_block = (
                f"이전 설정집:\n{state.get('setting_doc') or '(없음)'}\n\n"
                f"검증 피드백(critique):\n{prev_feedback['critique']}\n\n"
                "critique가 지적한 부분만 고치세요 - 지적되지 않은 행/항목은"
                " 이전 설정집 그대로 유지하고 다시 쓰지 마세요."
            )
            logger.info(
                "setting_doc_worker retry (attempt=%d) - previous critique: %s",
                prev_feedback["retry_count"],
                prev_feedback["critique"],
            )
        else:
            feedback_block = "(초기 생성 - 이전 시도 없음)"

        chain = _build_worker_chain()
        setting_doc = chain.invoke(
            {
                "brief": brief,
                "idea_content": idea_content,
                "rag_context": format_rag_context(state.get("rag_references") or []),
                "feedback_block": feedback_block,
            }
        )
    except Exception as e:
        logger.error("setting_doc_worker technical failure: %s", e, exc_info=True)
        return {"technical_failures": {"setting_doc": str(e)}}
    logger.debug("setting_doc_worker output:\n%s", setting_doc)
    return {"setting_doc": setting_doc}


class SettingDocValidationResult(BaseModel):
    is_valid: bool = Field(
        description="브리핑에 나열된 항목이 누락 없이 포함됐고, 유형이 닫힌 목록(인물/장소/물건/세력/종족)을 따르는가"
    )
    critique: str = Field(description="구체적인 지적사항 (통과 시에는 짧은 확인 코멘트)")


VALIDATOR_SYSTEM_PROMPT = """당신은 게임 설정집 검증 담당자(Setting Doc
Validator)입니다. "생성 지시문(브리핑)"과 "검증 대상 설정집"만 비교하세요 -
회의 원문과 대조하는 세부 사실 검증은 이 검증의 범위가 아닙니다(실측 확인 -
그런 세부 판단에서 Validator 자신이 반복적으로 스스로 모순되는 판정을
내렸습니다).

## 체크리스트
1. **완결성**: 브리핑에 나열된 엔티티가 설정 시트에, 브리핑의 "사건
   목록"이 사건 표에 빠짐없이 포함됐는가(브리핑이 "사건 목록: 없음"이면
   사건 표가 비어 있는 게 정답입니다). 빠진 게 있으면 무효 처리하고,
   critique에 브리핑의 어떤 항목이 빠졌는지 정확히 적으세요.
2. **유형 체크**: 설정 시트의 "유형"이 인물/장소/물건(유물)/세력/종족
   다섯 카테고리 중 하나가 아니면(예: "동료 NPC"/"집단"/"시스템" 같은
   통칭) 무효 처리하세요.

**이 두 가지 외에는 확인하지 마세요.** 브리핑에 없는 사건이 실제로
있었는지, 핵심 속성의 세부 사실 정확성은 이 검증의 범위가 아닙니다(그건
브리핑을 쓸 때 Classifier가 이미 판단했습니다) - Worker의 판단을 그대로
신뢰하세요. 무효 사유가 하나라도 있으면 is_valid=false로 판정하고,
critique에 정확히 무엇을 고쳐야 하는지 적으세요. 모두 통과하면
is_valid=true, critique에는 짧은 확인 코멘트만 적으세요.
"""

_validator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VALIDATOR_SYSTEM_PROMPT),
        (
            "human",
            "## 생성 지시문(브리핑)\n{brief}\n\n"
            "## 검증 대상 설정집\n{setting_doc}",
        ),
    ]
)


def _build_validator_chain(model: str = SETTING_DOC_VALIDATOR_MODEL):
    llm = build_chat_model(model)
    return _validator_prompt | llm.with_structured_output(SettingDocValidationResult)


def setting_doc_validator(state: AgentMeetingState) -> dict:
    """설정집 Validator 노드: setting_doc 검증 -> validation_status["setting_doc"].

    브리핑 대비 완결성/카테고리만 확인한다(회의 원문과는 대조하지 않음) -
    VALIDATOR_SYSTEM_PROMPT 상단 설명 참고. Worker가 이미 기술적 실패를
    기록했으면 검증 자체를 건너뛴다.
    """
    if has_technical_failure(state, "setting_doc"):
        return {}

    try:
        brief = state["task_briefs"]["setting_doc"]["brief"]

        chain = _build_validator_chain()
        result: SettingDocValidationResult = chain.invoke(
            {"brief": brief, "setting_doc": state["setting_doc"]}
        )
    except Exception as e:
        logger.error("setting_doc_validator technical failure: %s", e, exc_info=True)
        return {"technical_failures": {"setting_doc": str(e)}}

    prev_feedback = state.get("validation_status", {}).get("setting_doc")
    feedback: TaskFeedback = next_feedback(prev_feedback, result.is_valid, result.critique)

    if not result.is_valid:
        logger.warning(
            "setting_doc_validator FAILED (retry_count=%d) - critique: %s\n"
            "--- 검증 대상 설정집 ---\n%s",
            feedback["retry_count"],
            result.critique,
            state["setting_doc"],
        )
    else:
        logger.info("setting_doc_validator passed (retry_count=%d)", feedback["retry_count"])

    return {"validation_status": {"setting_doc": feedback}}
