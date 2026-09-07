"""회의록 태그([논의 외]/[아이디어N]) 파싱 유틸리티. 기획서 3.1 참고.

Minutes Generator가 붙인 태그를 기준으로 특정 태그에 속한 문단만 추출한다.
Spec Worker(3.3)가 "선정된 아이디어 태그 내용만" 입력받을 때 사용하며,
Task Classifier(3.4, Phase 3)도 동일한 방식으로 재사용할 예정.
"""
import re

_TAG_LINE_RE = re.compile(r"^\s*[-*]?\s*\[(논의 외|아이디어\d+)\]\s*(.*)$")


def extract_tagged_content(meeting_minutes: str, tag: str) -> str:
    """meeting_minutes에서 주어진 tag(예: "아이디어2")에 속한 문단만 이어붙여 반환.

    태그는 각 문단(불릿)의 맨 앞에 붙는다고 가정한다(3.1). 태그가 없는 줄(섹션
    제목, 빈 줄, 사람이 자유롭게 덧붙인 줄바꿈 등)은 직전 문단의 연속으로 간주해
    같은 태그에 속할 때만 포함한다.
    """
    lines: list[str] = []
    current_tag: str | None = None

    for raw_line in meeting_minutes.splitlines():
        match = _TAG_LINE_RE.match(raw_line)
        if match:
            current_tag, rest = match.group(1), match.group(2)
            if current_tag == tag:
                lines.append(rest.strip())
        elif current_tag == tag and raw_line.strip():
            lines.append(raw_line.strip())

    return "\n".join(lines)
