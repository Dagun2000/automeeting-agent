"""설정집 Worker/Validator 노드. 기획서 4.5(설정집 행), 4.9 참고.

파이프라인 순서: 기획서 생성 -> Task Classifier(무엇이 필요한지 판단 +
브리핑 작성) -> 설정집 Worker(브리핑대로 채움) -> Validator(완결성 체크).
"기획서 대비 새 정보인가"는 Task Classifier가 브리핑을 쓸 때 이미 판단한다
([src/nodes/classifier.py](classifier.py)) - 여기서는 다시 판단하지 않는다.

재시도 루프는 [src/nodes/spec.py](spec.py)와 동일하게 그래프 레벨 조건부
엣지가 담당한다([src/graph.py](../graph.py), [src/retry.py](../retry.py) 공용
헬퍼 사용).

모델 티어: 스펙 5.1 - GPT-5.6 Luna.
"""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.retry import next_feedback
from src.state import AgentMeetingState, TaskFeedback
from src.tagging import extract_tagged_content

SETTING_DOC_WORKER_MODEL = os.getenv("SETTING_DOC_WORKER_MODEL", "gpt-5.6-luna")
SETTING_DOC_VALIDATOR_MODEL = os.getenv("SETTING_DOC_VALIDATOR_MODEL", "gpt-5.6-luna")

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
**위 설정 시트의 "다섯 카테고리(인물/장소/물건/세력/종족)" 제한은 설정
시트의 "유형" 칸에만 적용됩니다. 사건 표는 완전히 다른 규칙을 따르는
별도 구조이니, 사건이 그 다섯 카테고리 중 하나가 아니라는 이유로 사건
표를 비우지 마세요.**

- "사건"은 회의에서 명시적으로 서술된 구체적 1회성 발생만을 의미합니다.
  회의에서 실제로 언급된 내용만 반영하세요(문장을 그대로 베낄 필요는
  없습니다) - "시작됐다", "대립했다"처럼 회의에 없던 인과·서사 연결어를
  새로 붙이지 마세요.
- **이 규칙은 사건을 지어내지 말라는 뜻이지, 사건을 뽑지 말라는 뜻이
  아닙니다.** 회의가 과거에 실제로 일어난 일로 서술한 내용이 있다면
  반드시 사건 표에 넣으세요. 참고 예시(다른 시나리오, 형식만 참고):
  "폐업한 다방 '별빛'이 화재로 문을 닫았다", "형사 노아가 3년 전 그
  사건에서 손을 뗐다"처럼 회의에서 이미 벌어진 일로 분명히 언급된 것은
  포함 대상입니다. 이런 걸 넣지 않고 사건 표를 비워두는 것도 오답입니다.
- 조건으로만 언급되고 실제 발생 여부가 불확실하면(예: "~하면 ~한다"는
  조건문 자체) 사건으로 적지 마세요. 플레이어가 매번 반복하는 게임플레이
  행동(탐험/전투/선택 등)도 1회성 사건이 아니므로 사건 표에 넣지 마세요.
- 발생 시점은 회의에서 명시적으로 언급된 경우에만 채우고, 그렇지 않으면
  빈칸으로 두세요.

## 규칙
- 회의에서 언급되지 않은 내용을 지어내지 마세요.
- 세부 관계의 영속적 일관성(DB화)이나 모순 검증은 이 작업의 범위가
  아닙니다.
- 이전 시도에 대한 검증 피드백(critique)이 주어지면 반드시 반영해 다시
  작성하세요.
"""

_worker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WORKER_SYSTEM_PROMPT),
        (
            "human",
            "## 생성 지시문(브리핑) - 포함할 항목이 이미 확정돼 있습니다\n{brief}\n\n"
            "## 선정된 아이디어 관련 회의 내용\n{idea_content}\n\n"
            "## 이전 시도 / 검증 피드백\n{feedback_block}\n\n"
            "위 브리핑을 그대로 실행해 설정집을 작성하세요.",
        ),
    ]
)


def _build_worker_chain(model: str = SETTING_DOC_WORKER_MODEL):
    llm = build_chat_model(model)
    return _worker_prompt | llm | StrOutputParser()


def setting_doc_worker(state: AgentMeetingState) -> dict:
    """설정집 Worker 노드: 브리핑 + 아이디어 내용 -> setting_doc."""
    idea_content = extract_tagged_content(
        state["meeting_minutes_confirmed"], state["selected_idea_tag"]
    )
    brief = state["task_briefs"]["setting_doc"]["brief"]

    prev_feedback = state.get("validation_status", {}).get("setting_doc")
    if prev_feedback and not prev_feedback["is_valid"]:
        feedback_block = (
            f"이전 설정집:\n{state.get('setting_doc') or '(없음)'}\n\n"
            f"검증 피드백(critique):\n{prev_feedback['critique']}\n\n"
            "위 피드백을 반드시 반영해 수정하세요."
        )
    else:
        feedback_block = "(초기 생성 - 이전 시도 없음)"

    chain = _build_worker_chain()
    setting_doc = chain.invoke(
        {"brief": brief, "idea_content": idea_content, "feedback_block": feedback_block}
    )
    return {"setting_doc": setting_doc}


class SettingDocValidationResult(BaseModel):
    is_valid: bool = Field(
        description="브리핑 항목이 누락 없이 포함됐고, 유형이 닫힌 목록(인물/장소/물건/세력/종족)을 따르며, 회의에 실제로 있는 사건이 빠짐없이 반영됐고, 날조가 없는가"
    )
    critique: str = Field(description="구체적인 지적사항 (통과 시에는 짧은 확인 코멘트)")


VALIDATOR_SYSTEM_PROMPT = """당신은 게임 설정집 검증 담당자(Setting Doc
Validator)입니다. "무엇을 포함할지"는 이미 브리핑에서 확정됐으므로, 여기서
다시 판단하지 마세요. 확인할 것은 오직 "브리핑대로 실행됐는가"입니다.

**이건 추출 검증이지 창작 검증이 아닙니다.** "선정된 아이디어 관련 회의
내용"에 실제로 적힌 문장보다 더 자세한 정보(정확한 수치, 과정, 동기, 배경
사연 등)가 빠졌다고 지적하지 마세요 - 회의가 말하지 않았다면 그건 "없는
정보"이지 "누락"이 아닙니다.

## 체크리스트
1. **완결성**: "생성 지시문(브리핑)"에 나열된 엔티티/사실이 "검증 대상
   설정집"에 빠짐없이 반영됐는지 확인하세요. 회의 문장을 인용하듯 정확히
   짚을 수 있는 누락만 지적하세요. 빠진 게 있으면 무효 처리하고,
   critique에 브리핑의 어떤 항목이 빠졌는지 정확히 적으세요.
2. **유형 체크(설정 시트만 해당)**: 설정 시트의 "유형"이 인물/장소/
   물건(유물)/세력/종족 중 하나가 아니면(예: "동료 NPC"/"집단"/"시스템"
   같은 통칭) 무효 처리하세요. critique에 그 행을 지우고 내용을 어느
   인물 항목으로 옮겨야 하는지 적으세요. **이 다섯 카테고리 제한은
   설정 시트에만 적용되며 사건 표에는 적용되지 않습니다** - 사건이 이
   카테고리에 안 맞는다는 이유로 사건 표를 비우거나 사건을 빼는 것 자체가
   오류입니다.
3. **사건 누락 체크**: "선정된 아이디어 관련 회의 내용"에서 과거에 실제로
   일어난 일로 명시적으로 서술된 1회성 사건이 있는데 "사건 표"가 비어
   있거나 그 사건이 빠져 있으면 무효 처리하세요. "날조하지 않았다"는
   이유로 정당한 빈 사건 표를 통과시키지 마세요 - 회의에 실제 사건이
   있다면 반드시 반영돼야 합니다.
4. **날조 체크**: "선정된 아이디어 관련 회의 내용"에 명시적으로 없는
   사건/사실을 지어낸 행이 있으면 무효 처리하세요. 회의에서 함축만 되고
   명시적으로 서술되지 않은 인과관계(예: "A와 B의 대립이 시작됐다")를
   단정한 경우, 조건으로만 언급되고 실제 발생이 확정되지 않은 내용을
   사건 표에 확정된 사건처럼 적은 경우도 포함됩니다.
5. **사건 표 형식**: 반복되는 게임플레이 행동(탐험/전투/선택 등)이 1회성
   "사건"인 것처럼 나열돼 있으면 무효 처리하세요.
6. 표 형식(설정 시트/사건 표 컬럼 구성, 헤더와 구분선의 컬럼 수 일치 등)이
   지켜지지 않았다면 그것도 지적하세요.

브리핑에 없는 내용이 설정집에 추가로 더 있다고 해서 그 자체로 무효 처리하지
마세요(브리핑 범위를 벗어난 날조가 아닌 이상 문제 삼지 않습니다). 무효
사유가 하나라도 있으면 is_valid=false로 판정하고, critique에 정확히 무엇을
고쳐야 하는지 적으세요. 모두 통과하면 is_valid=true, critique에는 짧은 확인
코멘트만 적으세요.
"""

_validator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VALIDATOR_SYSTEM_PROMPT),
        (
            "human",
            "## 생성 지시문(브리핑)\n{brief}\n\n"
            "## 선정된 아이디어 관련 회의 내용\n{idea_content}\n\n"
            "## 검증 대상 설정집\n{setting_doc}",
        ),
    ]
)


def _build_validator_chain(model: str = SETTING_DOC_VALIDATOR_MODEL):
    llm = build_chat_model(model)
    return _validator_prompt | llm.with_structured_output(SettingDocValidationResult)


def setting_doc_validator(state: AgentMeetingState) -> dict:
    """설정집 Validator 노드: setting_doc 검증 -> validation_status["setting_doc"]."""
    idea_content = extract_tagged_content(
        state["meeting_minutes_confirmed"], state["selected_idea_tag"]
    )
    brief = state["task_briefs"]["setting_doc"]["brief"]

    chain = _build_validator_chain()
    result: SettingDocValidationResult = chain.invoke(
        {"brief": brief, "idea_content": idea_content, "setting_doc": state["setting_doc"]}
    )

    prev_feedback = state.get("validation_status", {}).get("setting_doc")
    feedback: TaskFeedback = next_feedback(prev_feedback, result.is_valid, result.critique)
    return {"validation_status": {"setting_doc": feedback}}
