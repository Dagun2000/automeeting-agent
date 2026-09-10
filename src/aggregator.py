"""ZIP 패키징 (기획서 4.9). 그래프 노드가 아니라 완료 화면의 다운로드
버튼이 직접 호출하는 평범한 함수다 - 그래프 노드로 만들지 않은 이유는
[src/graph.py](graph.py) 모듈 docstring 참고(대시보드 렌더링 없이 다운로드만
필요하다는 스코프 결정).

PDF 대신 .md 그대로 담는다. 원래 스펙 4.9는 PDF였지만, Markdown에 포함된
Mermaid 다이어그램(기획서의 "핵심 게임플레이 루프")을 이미지로 렌더링하려면
헤드리스 브라우저(mermaid-cli)나 시스템 의존성이 있는 PDF 라이브러리
(WeasyPrint -> Pango 등)가 필요해 복잡도가 크게 늘어난다 - 반면 .md 그대로
두면 GitHub/VSCode/Obsidian 등 대부분의 마크다운 뷰어가 ```mermaid``` 펜스를
그대로 렌더링해주므로 별다른 처리 없이도 다이어그램이 보인다(사용자 확인
후 PDF 변환 자체를 스코프에서 뺌).

기술적 실패(technical_failures)로 일부 산출물이 없어도 ZIP은 정상 생성된다
- 없는 파일은 그냥 빼고 있는 것만 담는다(4.8, "부분 실패 허용"과 같은
원칙을 최종 패키징에도 적용). 모든 트랙이 실패해도 최소한 원본 회의록
(태그 포함)은 항상 담는다(비어 있는 ZIP을 만들지 않기 위한 극단적 케이스
안전망).
"""
import base64
import io
import zipfile
from typing import Optional


def _decode_image_bytes(image_url: Optional[str]) -> Optional[bytes]:
    """image_url이 data URI(현재 이미지 모델이 항상 이 형태로 반환 -
    [src/nodes/image.py](nodes/image.py) 참고)면 PNG bytes로 디코드한다.
    일반 http(s) URL이거나 없으면(생성 실패) None을 반환해 ZIP에서 뺀다 -
    네트워크 호출로 다운받아오지는 않는다."""
    if not image_url or not image_url.startswith("data:") or "base64," not in image_url:
        return None
    return base64.b64decode(image_url.split("base64,", 1)[1])


def _safe_filename(name: str) -> str:
    """ZIP 항목 이름으로 쓰기 위험한 경로 구분자만 치환한다."""
    return name.replace("/", "_").replace("\\", "_").strip() or "이름없음"


def build_zip(values: dict) -> bytes:
    """그래프 최종 state(`graph.get_state(config).values`)로부터 ZIP
    바이트를 만들어 반환한다 - [src/ui/app.py](ui/app.py)의 다운로드
    버튼이 그대로 `st.download_button`에 넘긴다.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        minutes = values.get("meeting_minutes_confirmed") or values.get("meeting_minutes_draft")
        if minutes:
            zf.writestr("회의록.md", minutes)

        spec_document = values.get("spec_document")
        if spec_document:
            zf.writestr("기획서.md", spec_document)

        task_briefs = values.get("task_briefs", {})
        if task_briefs.get("setting_doc", {}).get("enabled") and values.get("setting_doc"):
            zf.writestr("설정집.md", values["setting_doc"])

        if task_briefs.get("reference_report", {}).get("enabled") and values.get("reference_report"):
            zf.writestr("비교대상_검색_보고서.md", values["reference_report"])

        for task in values.get("image_tasks", []):
            if not task.get("enabled"):
                continue
            image_bytes = _decode_image_bytes(task.get("image_url"))
            if image_bytes:
                zf.writestr(f"이미지/{_safe_filename(task['subject'])}.png", image_bytes)

    return buf.getvalue()
