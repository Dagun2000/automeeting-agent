"""LangGraph 그래프에 바인딩되는 외부 도구. 현재는 비교대상 검색 보고서용
웹 검색 하나뿐이다([src/nodes/reference_report.py](nodes/reference_report.py)의
ToolNode에서 사용).

검색 백엔드는 Tavily(에이전트용 검색 API, 무료 티어 있음)를 기본값으로
선택했다 - 다른 제공자로 바꾸려면 이 파일의 web_search 함수만 교체하면 된다.
"""
import os

import requests
from langchain_core.tools import tool

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"


@tool
def web_search(query: str) -> str:
    """주어진 검색어로 웹을 검색해 관련 결과(제목/URL/스니펫)를 반환한다."""
    if not TAVILY_API_KEY:
        return "웹 검색 실패: TAVILY_API_KEY가 설정되지 않았습니다."

    response = requests.post(
        TAVILY_ENDPOINT,
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 5,
            "search_depth": "basic",
        },
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("results", [])

    if not results:
        return f'"{query}"에 대한 검색 결과가 없습니다.'

    lines = [
        f"- {r.get('title', '')} | {r.get('url', '')}\n  {r.get('content', '')[:300]}"
        for r in results
    ]
    return "\n".join(lines)
