"""RAG 인덱싱/조회 셋업. 기획서 5.2(Vector DB: ChromaDB, text-embedding-3-small),
5.4(rag_references) 참고.

Phase 4 최초 구현 - 목업 코퍼스([sample_data/glossary.md](../sample_data/glossary.md),
[sample_data/game_profiles.md](../sample_data/game_profiles.md))를 ChromaDB에
인덱싱하고, 트랜스크립트를 쿼리로 상위 K개 문서를 검색해 rag_references에
채운다([src/nodes/rag_retrieval.py](nodes/rag_retrieval.py) 참고).

문서 단위: 각 소스 파일을 "## " 헤더 기준으로 쪼개 항목(용어 하나/게임 하나)
단위 문서로 인덱싱한다 - 파일 하나를 통째로 한 문서로 검색하면 관련 없는
다른 용어까지 같이 딸려 오므로, 청크를 검색 가능한 최소 단위로 맞췄다.
문서마다 type 메타데이터(glossary/game_profile)를 붙여 어느 코퍼스에서
왔는지 구분한다.

벡터스토어는 인메모리 Chroma를 프로세스당 한 번만 만들어 재사용한다
(get_vector_store, @lru_cache) - 매 검색마다 재인덱싱하면 임베딩 API 호출이
낭비된다.

하이브리드 검색: 의미 임베딩 검색만으로는 STT가 살짝 틀리게 받아적은 짧은
고유명사(예: "크림즌 프로토콜" vs 정답 "크림슨 프로토콜")를 잘 못 잡는다
(실측 확인 - 의미상으론 멀쩡히 관련 있어도, 고유명사 자체는 의미 정보가
거의 없어서 임베딩 유사도 신호가 약하다). 반대로 이런 오탈자는 철자
자체는 원문과 거의 같으므로(편집 거리 1~2), `difflib` 기반 표면형(문자열)
유사도로 보완한다 - `_fuzzy_candidates`가 트랜스크립트의 1~3단어 n-gram과
각 문서 제목을 직접 비교해, 의미 검색이 놓친 오탈자 용어를 강제로
포함시킨다.
"""
import difflib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
# 코사인 관련도 점수 컷오프 - 이 밑으로는 상위 k 안에 들어도 버린다.
# 0.0(예전 기본값)은 사실상 필터링이 없는 것과 같았다 - 실측 확인(스크립트로
# 코퍼스와 아예 무관한 회의록, sample_meeting_phase3.txt로 재현): 코퍼스
# 용어를 하나도 언급하지 않은 트랜스크립트에서도 24개 문서 중 18개가 점수
# 0.0 이상으로 나왔고(최고 0.2542, 전부 오탐), RAG_TOP_K(10)만큼 그대로
# 채워져 관련 없는 노이즈가 프롬프트에 섞여 들어갔다. 반면 실제로 언급된
# 용어(sample_meeting_phase4.txt)는 점수가 0.12~0.36 범위였다 - 두 실측
# 분포 사이에 낮게라도 여유가 있는 0.3으로 올렸다(무관 트랜스크립트의
# 최고 노이즈 0.2542보다 위). 이러면 의미 검색만으로 걸리던 일부 실제
# 언급(0.3 미만, 예: "구스엔진" 0.21, "크림슨 프로토콜" 0.12)을 놓칠 수
# 있지만, 이 코퍼스는 애초에 "고유명사가 실제로 등장했는가"를 찾는
# 용도라 아래 표면형(fuzzy) 매칭이 그 역할을 이미 맡고 있다(실측 확인 -
# phase4 샘플의 진짜 언급 7개 중 5개를 fuzzy가 잡아냄, 나머지 2개는
# "QA봇"→"큐에이봇"처럼 알파벳 약어를 한글 발음대로 풀어써 표면형조차
# 안 맞는 케이스로 이미 알려진 한계). 그래서 의미 검색 임계값은 "확실히
# 강한 매치만 통과"로 보수적으로 잡고, 짧은 고유명사 탐지는 fuzzy가
# 전담하는 구조로 정리했다.
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
# difflib.SequenceMatcher.ratio() 컷오프(0~1) - 이 이상이면 "같은 용어의
# 오탈자"로 간주한다. 실측(예: "거스엔진"↔"구스엔진" 0.75, "크림즌
# 프로토콜"↔"크림슨 프로토콜" 0.86) 기준으로 오탐 없이 이 두 사례를 모두
# 잡는 값으로 보정했다.
FUZZY_MIN_RATIO = float(os.getenv("RAG_FUZZY_MIN_RATIO", "0.72"))

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_FILES = {
    "glossary": _ROOT / "sample_data" / "glossary.md",
    "game_profile": _ROOT / "sample_data" / "game_profiles.md",
}


def _split_into_documents(path: Path, doc_type: str) -> List[Document]:
    """"# 제목" 한 줄 + "## " 섹션 여러 개로 이뤄진 마크다운 파일을,
    섹션(항목) 하나당 문서 하나로 쪼갠다."""
    text = path.read_text(encoding="utf-8")
    _, *sections = text.split("\n## ")
    documents = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        title = section.splitlines()[0].strip()
        documents.append(
            Document(
                page_content=f"## {section}",
                metadata={"type": doc_type, "title": title, "source": path.name},
            )
        )
    return documents


def _load_corpus() -> List[Document]:
    documents: List[Document] = []
    for doc_type, path in _CORPUS_FILES.items():
        if path.exists():
            documents.extend(_split_into_documents(path, doc_type))
    return documents


_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WHITESPACE_RE.sub("", s).lower()


def _title_names(title: str) -> List[str]:
    """"구스엔진 (GooseEngine)" -> ["구스엔진", "GooseEngine"] (괄호 안 영문
    표기도 별도로 비교 대상에 넣는다)."""
    m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", title)
    return [m.group(1), m.group(2)] if m else [title]


def _transcript_ngrams(transcript: str) -> List[str]:
    """트랜스크립트를 단어 단위로 쪼갠 뒤 1~3단어 연속 조합을 후보로 만든다
    (용어집 항목이 "나이트폴 SDK"처럼 여러 단어일 수 있어서)."""
    words = _WORD_RE.findall(transcript)
    ngrams = set()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            ngrams.add(" ".join(words[i : i + n]))
    return list(ngrams)


def _fuzzy_candidates(transcript: str, min_ratio: float = FUZZY_MIN_RATIO) -> List[Document]:
    """의미 검색이 놓칠 수 있는, 철자만 살짝 다른(STT 오인식) 문서를
    표면형 유사도로 찾는다. 모듈 docstring 참고."""
    candidates = {_normalize(g) for g in _transcript_ngrams(transcript)}
    if not candidates:
        return []

    matched = []
    for doc in _load_corpus():
        names = [_normalize(n) for n in _title_names(doc.metadata.get("title", ""))]
        best_ratio = max(
            (
                difflib.SequenceMatcher(None, name, cand).ratio()
                for name in names
                for cand in candidates
                if name
            ),
            default=0.0,
        )
        if best_ratio >= min_ratio:
            matched.append(doc)
    return matched


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """인메모리 Chroma 벡터스토어를 만들고 목업 코퍼스를 인덱싱해 반환한다
    (프로세스당 1회만 실행됨 - lru_cache)."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(collection_name="automeeting_rag", embedding_function=embeddings)
    documents = _load_corpus()
    if documents:
        store.add_documents(documents)
    return store


def search(query: str, k: int = 5, score_threshold: float = SCORE_THRESHOLD) -> List[str]:
    """쿼리 하나로 상위 k개 문서를 검색하되, 관련도 점수가 score_threshold
    미만인 문서는 버리고 본문 텍스트 리스트로 반환한다."""
    store = get_vector_store()
    results = store.similarity_search_with_relevance_scores(
        query, k=k, score_threshold=score_threshold
    )
    return [doc.page_content for doc, _score in results]


def search_transcript(
    transcript: str,
    k: int = 5,
    score_threshold: float = SCORE_THRESHOLD,
    fuzzy_min_ratio: float = FUZZY_MIN_RATIO,
) -> List[str]:
    """회의록 트랜스크립트 전체가 아니라, 빈 줄로 구분된 발화(문단) 단위로
    쪼개 각각 의미 검색한 뒤, 표면형(오탈자) 매칭 결과와 합쳐 문서별 최고
    점수 기준 상위 k개를 반환한다.

    실측 확인: 트랜스크립트 전체를 하나의 벡터로 임베딩해서 검색하면(옛
    `search(transcript, k=...)` 방식), 회의 전체가 다양한 주제의 발화
    수십 줄로 이뤄져 있어 특정 용어 언급 신호가 희석된다 - 실제로 여러 번
    언급된 "구스엔진" 같은 용어조차 관련도 점수가 음수로 나왔다. 발화
    단위로 쪼개 쿼리하면 그 발화 하나의 의미에 훨씬 가깝게 매칭되어 점수가
    크게(수 배) 개선된다.

    그래도 순수 의미 유사도만으로는 "이름만 짧게 언급된" 고유명사(특히
    오탈자가 섞인 경우)를 놓칠 수 있어(실측 확인 - "크림슨 프로토콜"이
    분명히 언급됐는데도 관련도 0.018로 매우 낮게 나온 사례), `_fuzzy_candidates`
    표면형 매칭 결과를 최고 점수(1.0)로 강제 포함시켜 보완한다.
    """
    chunks = [c.strip() for c in transcript.split("\n\n") if c.strip()]

    store = get_vector_store()
    best: dict = {}
    for chunk in chunks:
        for doc, score in store.similarity_search_with_relevance_scores(
            chunk, k=k, score_threshold=score_threshold
        ):
            key = doc.metadata.get("title") or doc.page_content
            if key not in best or score > best[key][0]:
                best[key] = (score, doc.page_content)

    for doc in _fuzzy_candidates(transcript, min_ratio=fuzzy_min_ratio):
        key = doc.metadata.get("title") or doc.page_content
        best[key] = (1.0, doc.page_content)

    ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
    return [content for _, content in ranked[:k]]


def format_rag_context(references: List[str]) -> str:
    """state["rag_references"]를 프롬프트에 넣기 좋은 텍스트로 합친다
    (minutes/spec/setting_doc/reference_report Worker가 공용으로 사용)."""
    return "\n\n---\n\n".join(references) if references else "(참고 자료 없음)"
