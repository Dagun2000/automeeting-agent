"""LangGraph 그래프에 바인딩되는 외부 도구. 현재는 비교대상 검색 보고서용
웹 검색 하나뿐이다([src/nodes/reference_report.py](nodes/reference_report.py)의
ToolNode에서 사용).

검색 백엔드는 `ddgs` 패키지(API 키 불필요)를 사용한다 - 다른 제공자로
바꾸려면 이 파일의 web_search 함수만 교체하면 된다. 원래는 Tavily(에이전트용
검색 API)를 썼으나, 실사용 중 무료 티어 rate limit에 자주 걸려 `ddgs`로
교체했다(API 키 없이도 동작해 이 문제 자체가 없어짐). `ddgs`의 결과
딕셔너리는 키가 title/href/body라 Tavily의 title/url/content와 이름만
다르고 구조는 동일 - 아래에서 매핑만 맞춰준다.

`ddgs`의 `_search_sync` 내부 동작을 뜯어봐야 하는 이유가 있었다(실측/코드
확인, `.venv/Lib/site-packages/ddgs/ddgs.py`) - `backend`를 안 넘기면
기본값 "auto"가 grokipedia/wikipedia부터 시작해 8개 엔진 후보를 순서대로
최대 2개씩 동시에 시도하다 결과가 max_results(5)를 채우면 멈추는 구조라,
개별 엔진 실패는 내부에서 로그만 남기고 다음 엔진으로 자동 폴백한다 -
그래서 어지간해선 예외를 던지지 않지만, grokipedia/wikipedia처럼 일반
웹검색이 아니거나 느린 엔진이 먼저 걸리면 쿼리 하나에 5~6초씩 걸리고
로그가 429/403/ConnectError로 지저분해진다.

반대로 `backend="duckduckgo"`처럼 엔진 하나만 고정하면 폴백이 아예 없어서,
그 엔진이 막히는 순간 바로 `DDGSException`을 던진다 - 실측 확인 결과
DuckDuckGo 자체의 html 스크래핑 엔드포인트(html.duckduckgo.com)가 짧은
시간에 연속 요청 몇 번(2~3회)만으로 막히기 시작했고, 8초 백오프 재시도로도
회복이 안 됐다(IP/세션 단위로 한동안 막히는 것으로 보임 - 비교대상 검색
보고서 하나를 만드는 데 필요한 연속 검색 호출 횟수를 못 버팀). 참고로
`backend="bing"`은 이 `ddgs` 버전에 아예 등록돼 있지 않은 이름이라(로그에
"backends do not exist" 경고 후 "auto"로 조용히 폴백됨 - 처음엔 이걸
몰라서 "bing 고정"이 잘 되는 줄 착각했었다), 실제로는 유효한 엔진 이름만
써야 한다(`Available: brave, duckduckgo, google, grokipedia, mojeek,
startpage, wikipedia, yahoo`).

그래서 절충으로 `backend="duckduckgo,brave,google,startpage"`처럼 일반
웹검색에 맞고 비교적 빠른 엔진 4개만 콤마로 나열해 명시적 폴백 목록을
구성했다 - grokipedia(전용 타입어헤드라 일반 쿼리에 안 맞음)/wikipedia
(백과사전 검색이라 일반 웹검색과 다름)/yahoo/mojeek(실측상 응답이 느리거나
쉽게 막힘)는 제외했다. 연속 20회 호출 실측에서 19/20 성공(쿼리당 1~3초,
실패 1건도 개별 엔진 ConnectError였을 뿐 전체 재시도 없이 다음 호출은
정상 성공)이라 이걸로 고정했다. 그래도 남는 실패는 `DDGSException`을
잡아 검색 실패 메시지로 대체한다(Validator가 "출처 없음"으로 정상
처리하도록).

각 검색 결과에 URL의 해시로 만든 8자리 인용 ID(예: [a1b2c3d4])를 붙여
반환한다. LLM이 보고서에 URL을 직접 타이핑하게 하면 긴 슬러그를 옮겨
적다가 잘라먹거나 구분자를 바꿔 쓰는 등 오탈자가 실측으로 자주 발생했다 -
정보를 LLM이 다시 받아적을 이유가 없으므로, LLM에게는 이 ID만 인용하게
하고 실제 URL 치환은 코드가 담당한다([src/nodes/reference_report.py]의
`_substitute_citations`). ID는 URL의 해시라 같은 URL은 항상 같은 ID를
얻고, 전역 카운터 없이도(동시 호출에도 안전) 결정적으로 계산된다.
"""
import hashlib
import logging

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def cite_id(url: str) -> str:
    """URL로부터 결정적인 8자리 인용 ID를 계산한다(reference_report.py와 공유)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


@tool
def web_search(query: str) -> str:
    """주어진 검색어로 웹을 검색해 관련 결과(인용 ID/제목/URL/스니펫)를
    반환한다. 최종 보고서에 출처를 적을 때는 각 결과 앞의 [ID]만 그대로
    사용하고, URL은 직접 타이핑하지 마세요."""
    try:
        results = DDGS().text(query, max_results=5, backend="duckduckgo,brave,google,startpage")
    except DDGSException:
        logger.warning("web_search failed for query=%r", query, exc_info=True)
        return f'"{query}" 검색 중 오류가 발생했습니다. 다른 검색어로 다시 시도해 주세요.'

    if not results:
        return f'"{query}"에 대한 검색 결과가 없습니다.'

    lines = [
        # content 내 개행을 제거해 한 줄로 만든다 - reference_report.py의
        # 파싱 정규식(_SEARCH_RESULT_RE)이 "[ID] 제목 | URL" 줄과 스니펫
        # 줄의 경계를 개행 하나로 구분하므로, content에 개행이 섞이면 다음
        # 검색 결과와 경계가 뒤섞여 그 항목이 인용 매칭에서 누락될 수 있다.
        f"- [{cite_id(r.get('href', ''))}] {r.get('title', '')} | {r.get('href', '')}\n"
        f"  {' '.join(r.get('body', '')[:300].split())}"
        for r in results
    ]
    return "\n".join(lines)
