"""Task Classifier 노드. 기획서 3.4, 3.5, 4.2 참고.

입력은 선정된 아이디어 태그 내용 + 완성된 기획서(장르 포함). 출력은 상태
스키마(5.4)에 이미 있는 필드로만 표현한다 - needs_setting_doc/
needs_reference_report는 별도 bool 필드가 아니라 task_briefs의
enabled(3.6: "체크박스(켜기/끄기) + 브리핑"과 그대로 대응)로, needs_image는
image_tasks가 비어있지 않은 것으로 표현한다.

장르 기반 임계값 조정(3.4)은 정확한 가중치가 별도 확정 필요 상태(9장)이므로,
방향성만 시스템 프롬프트에 휴리스틱으로 반영했다:
- RPG/어드벤처: 캐릭터 언급이 짧아도 needs_setting_doc 문턱을 낮춤(캐릭터
  정보가 핵심 자산이라는 판단).
- SLG/전략/경영 시뮬레이션: 캐릭터보다 세력·자원·시스템 관계 논의를 우선
  근거로 삼음.
- 그 외 장르: 엔티티 상세도 기반 기본 기준 그대로 적용.

모델 티어: 스펙 5.1 - GPT-5.6 Luna.
"""
import os
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm import build_chat_model
from src.state import AgentMeetingState, ImageTask, TaskBrief
from src.tagging import extract_tagged_content

CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-5.6-luna")

MAX_DISCUSSED_SYSTEMS = 5

SYSTEM_PROMPT = """당신은 게임 기획 하위 산출물 분류 담당자(Task Classifier)입니다.
"선정된 아이디어 관련 회의 내용"과 "완성된 기획서"를 함께 보고, 아래 세 가지를
독립적으로 판단하세요.

## 1. needs_setting_doc (설정집 필요 여부)
캐릭터·장소·유물 등 개별 엔티티의 상세 속성이나 사건/관계 논의가, 기획서의
요약 수준 서술만으로는 담기 부족할 만큼 나왔는지 판단하세요.

장르에 따라 문턱을 다음과 같이 조정하세요(기획서의 장르 필드를 확인):
- RPG/어드벤처 계열: 캐릭터 이름과 간단한 역할/배경 한두 줄 정도만 언급됐어도
  needs_setting_doc 쪽으로 문턱을 낮춰 판단하세요. RPG는 캐릭터 정보가 핵심
  자산이라, 요약 수준 기획서만으로는 부족하다고 보는 경우가 많습니다.
- SLG/전략/경영 시뮬레이션 계열: 개별 캐릭터보다 세력·자원·시스템 관계(세력
  간 관계, 자원 종류와 흐름, 조직 구조 등) 논의를 needs_setting_doc 판단의
  우선 근거로 삼으세요. 캐릭터 언급은 상대적으로 낮은 비중으로 취급합니다.
- 그 외 장르: 기본 기준(엔티티의 상세 속성/사건/관계가 요약만으로 부족한
  수준으로 나왔는가)을 그대로 적용하세요.

## 2. needs_image (이미지 필요 여부)
대상은 장소·풍경·건축물 등 **환경 요소로만 한정**합니다. **인물/캐릭터
초상은 절대 후보에서 제외**하세요(캐릭터가 아무리 자세히 논의됐어도 이미지
대상이 될 수 없습니다). 아래 두 조건을 **모두** 충족하는 대상만
image_tasks에 포함하세요:
1. 구체적 대상(장소/사물)이 최소 1개 이상 언급됨
2. 그 대상에 대한 시각적 특징(형태·재질·색감·분위기 등 무엇이든) 묘사가
   함께 언급됨 - 명시적 톤 단어("다크판타지" 등)가 아니어도, 대상 묘사
   자체에 시각 정보가 담겨 있으면 충족(예: "돌담이 무너진 폐허")
둘 중 하나만 있으면 그 대상은 제외하세요. 조건을 만족하는 대상이 하나도
없으면 image_tasks는 빈 리스트로 반환하세요(needs_image=False와 동일한
의미입니다).

## 3. discussed_systems (비교대상 검색 보고서 대상)
구체적으로 논의된 메커니즘/시스템 중, **게임마다 구현 방식이 실제로 갈리는
것만** 포함하세요. WASD 이동처럼 업계에서 사실상 표준화되어 게임 간 차이가
거의 없는 요소는 구체적으로 논의·확정됐어도 제외합니다. 스코프를 통과한
시스템이 5개를 넘으면, 회의에서 다뤄진 비중·상세도(논의 중요도) 기준으로
상위 5개만 남기세요. 논의된 시스템이 없거나 조사할 가치가 없으면 빈
리스트를 반환하고 needs_reference_report도 False로 판단하세요.

## 브리핑(task_briefs)
needs_setting_doc / needs_reference_report가 True인 항목에는, 각 Worker에게
그대로 전달될 생성 지시문(brief)을 작성하세요. **이 brief가 Worker가 받는
최종 지시문입니다. Worker는 brief를 실행만 할 뿐, 무엇을 포함할지는 다시
판단하지 않습니다 - 그 판단은 여기서 끝내야 합니다.** False인 항목의
brief는 빈 문자열로 두세요.

### setting_doc_brief
brief의 항목은 회의에서 언급된 **인물 / 장소 / 물건(유물) / 세력 / 종족**
다섯 카테고리만 대상으로 나열하세요. **이 다섯 가지는 닫힌 목록입니다 -
비슷해 보여도 다섯 카테고리 중 정확히 하나에 속하지 않으면 항목으로 만들지
마세요.** 게임 시스템·메커니즘·규칙 자체, "동료 NPC들"·"주민들" 같은
집단/그룹 통칭도 다섯 카테고리에 속하지 않으므로 항목이 될 수 없습니다 -
그런 내용은 만들지 말고, 그게 영향을 주는 **특정 이름이 있는 인물** 항목의
핵심 내용에 넣으세요(예: "동료들이 특정 조건에 반응해 이탈한다"가 아니라,
실제로 이탈하는 그 인물 개개인의 항목에 나눠 넣을 것).

각 항목의 핵심 내용은 회의에서 실제로 언급된 내용만 반영하도록 지시하세요.
**문장을 그대로 베끼라는 뜻이 아닙니다** - 회의는 대화체라 그대로 옮길
문장이 없을 수 있으니, 무엇을 다뤄야 하는지 요지만 짚어주면 됩니다. 대신
"~을 구체화할 것", "~을 확정할 것"처럼 회의에 없는 내용을 채우거나
결정하라는 지시는 절대 쓰지 마세요 - brief는 회의에서 다뤄진 요지를
가리키는 지시이지, 앞으로 더 만들어야 할 것을 지시하는 게 아닙니다.

기획서에 이미 있는 내용(요약 수준이라도 실질적으로 같은 정보)은 제외하세요.
단, 인물/장소/물건/세력/종족의 이름이 기획서에 스치듯 언급된 것만으로
제외하지 마세요 - 그 이상의 설명이 기획서에 없다면 포함 대상입니다.

아래는 실제 회의와 무관한 형식 참고용 예시입니다(다른 시나리오의 예시일
뿐 - 실제로는 지금 다루는 회의 내용에서 이름과 사실을 뽑아야 합니다):
"은퇴 형사 '노아 브릭스'(인물) - 회의에서 언급된 역할(의뢰인을 캐묻는
성격이라는 내용)을 반영할 것", "폐업한 다방 '별빛'(장소) - 회의에서
언급된 분위기 묘사(먼지 쌓인 테이블, 깨진 네온사인)를 반영할 것".

### reference_report_brief
회의 내용에서 discussed_systems 각 시스템이 어떤 맥락으로 논의됐는지, Worker가
검색할 때 참고할 관점을 적으세요.

### 이미지 brief
이미지 대상별 brief에는 시각적 특징 묘사를 포함해 이미지 생성 지시문 형태로
작성하세요.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "## 선정된 아이디어 관련 회의 내용\n{idea_content}\n\n"
            "## 완성된 기획서\n{spec_document}",
        ),
    ]
)


class ImageTaskDraft(BaseModel):
    subject: str = Field(description="이미지 대상 (장소/사물 등 환경 요소만, 인물/캐릭터 제외)")
    brief: str = Field(description="이 대상에 대한 이미지 생성 지시문 (시각적 특징 포함)")


class TaskClassification(BaseModel):
    needs_setting_doc: bool
    setting_doc_brief: str = Field(
        description="설정집 Worker에게 전달할 최종 생성 지시문 - 기획서와 대조해 걸러진, 포함할 엔티티/사실의 확정 목록 (불필요 시 빈 문자열)"
    )
    needs_reference_report: bool
    reference_report_brief: str = Field(
        description="비교대상 검색 보고서 Worker에게 전달할 생성 지시문 (불필요 시 빈 문자열)"
    )
    discussed_systems: List[str] = Field(
        default_factory=list, description="최대 5개, 3.5 스코프 필터 적용"
    )
    image_tasks: List[ImageTaskDraft] = Field(
        default_factory=list, description="3.4의 두 조건을 모두 충족하는 환경 요소만"
    )


def _build_chain(model: str = CLASSIFIER_MODEL):
    llm = build_chat_model(model)
    return _prompt | llm.with_structured_output(TaskClassification)


def classify_tasks(state: AgentMeetingState) -> dict:
    """Task Classifier 노드: 아이디어 내용 + 기획서 -> task_briefs/discussed_systems/image_tasks."""
    idea_content = extract_tagged_content(
        state["meeting_minutes_confirmed"], state["selected_idea_tag"]
    )

    chain = _build_chain()
    result: TaskClassification = chain.invoke(
        {"idea_content": idea_content, "spec_document": state["spec_document"]}
    )

    task_briefs: Dict[str, TaskBrief] = {
        "setting_doc": {"enabled": result.needs_setting_doc, "brief": result.setting_doc_brief},
        "reference_report": {
            "enabled": result.needs_reference_report,
            "brief": result.reference_report_brief,
        },
    }

    discussed_systems = (
        result.discussed_systems[:MAX_DISCUSSED_SYSTEMS] if result.needs_reference_report else []
    )

    image_tasks: List[ImageTask] = [
        {
            "subject": t.subject,
            "enabled": True,
            "brief": t.brief,
            "image_url": None,
            "validation": None,
        }
        for t in result.image_tasks
    ]

    return {
        "task_briefs": task_briefs,
        "discussed_systems": discussed_systems,
        "image_tasks": image_tasks,
    }
