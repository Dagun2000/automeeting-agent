"""RAG Retrieval 노드. 기획서 5.2/5.4 참고.

파이프라인 순서: 트랜스크립트 확보 직후, Minutes Generator 이전에 실행한다
([src/graph.py](../graph.py)) - Minutes Generator가 STT 오인식으로 보이는
사내 고유명사를 용어집 기준으로 교정하려면, 그 전에 rag_references가 채워져
있어야 하기 때문이다.

원본 트랜스크립트를 발화 단위로 쪼개 각각을 쿼리로 사내 코퍼스(용어집/게임
프로필)에서 검색하고, 문서별 최고 점수 기준 상위 최대 RAG_TOP_K개를
rag_references에 채운다([src/rag.py](../rag.py)의 `search_transcript` -
트랜스크립트를 통째로 쿼리하면 관련도 점수가 희석되는 문제가 실측
확인돼 발화 단위로 쪼갠다). 관련도 점수가 낮은 문서는 RAG_TOP_K개를 다
못 채우더라도 버린다(`SCORE_THRESHOLD`) - 회의에서 직접 언급/연상되지
않는 문서를 억지로 끼워 넣지 않기 위함이다. 이 노드 자체는 LLM 호출이
없다 - 임베딩 기반 벡터 검색만 수행한다(text-embedding-3-small).
"""
import os

from src.rag import search_transcript
from src.state import AgentMeetingState

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))


def retrieve_rag_references(state: AgentMeetingState) -> dict:
    """RAG Retrieval 노드: transcript -> rag_references."""
    references = search_transcript(state["transcript"], k=RAG_TOP_K)
    return {"rag_references": references}
