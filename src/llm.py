"""ChatOpenAI 생성 공용 헬퍼. 기획서 5.1 참고.

GPT-5.6 계열(Luna/Terra/Sol)은 reasoning 모델이라 `temperature`를 지원하지
않고(모델 프로필 `"temperature": False`), 함수 도구(구조화 출력/tool 바인딩)를
쓰려면 reasoning_effort를 "none"으로 낮춰야 한다 - 기본값인 "medium"인 채로
function tools를 쓰면 `/v1/chat/completions`에서 다음 에러가 난다(실제 API
응답으로 확인):

    Function tools with reasoning_effort are not supported for gpt-5.6-luna
    in /v1/chat/completions. To use function tools, use /v1/responses or
    set reasoning_effort to 'none'.

이 프로젝트의 모든 노드는 분류/체크리스트/요약 수준 작업(5.1 원안이 Luna
티어를 고른 이유 자체가 "신규성 판단이 아닌 구조화된 작업")이라 깊은
reasoning이 필요 없으므로, 도구 사용 여부와 무관하게 항상
reasoning_effort="none"으로 고정한다. 특정 노드에서 품질 문제로 더 깊은
추론이 필요해지면 그 노드의 build 함수에서만 개별적으로
`reasoning_effort=...`를 넘겨 오버라이드하면 된다.
"""
import os

from langchain_openai import ChatOpenAI

REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")


def build_chat_model(model: str, **kwargs) -> ChatOpenAI:
    kwargs.setdefault("reasoning_effort", REASONING_EFFORT)
    return ChatOpenAI(model=model, **kwargs)
