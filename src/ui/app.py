"""Streamlit UI — 녹음(STT)/디버그 텍스트 입력, Gate 0, Gate 1, 최종 결과 확인.

기본 화면은 녹음 화면이다(`st.audio_input`으로 브라우저 녹음 -> Whisper API
STT, [src/stt.py](../stt.py) - diarization은 구현하지 않음, 그 파일 docstring
참고). 화면 우측 하단 "DEBUG" 버튼을 누르면 오디오 없이 텍스트를 바로
Minutes Generator로 보내는 디버그 입력 화면으로 전환된다(session_state의
"debug_mode" 플래그로 전환, 새로고침 전까지 유지됨).

오른쪽 사이드 패널에 그래프 구조를 시각화한다(HTML/CSS만 사용 - Graphviz 등
시스템 의존성 없음). 현재 실행 중인 노드는 `graph.invoke()` 대신
`graph.stream(..., stream_mode="debug")`로 돌려서, 노드가 "시작"하는
즉시 그 노드를 강조 표시한다(run_with_visualization). 처음엔
stream_mode="updates"로 만들었는데, 그건 노드가 "완료"된 뒤에야 이벤트가
나와서 실제로는 결과가 이미 나온 다음에야 불이 들어왔다(사용자 실측
지적으로 발견 - 실행 중인 동안은 화면이 그대로였음). "debug" 모드는
시작/종료를 별개 이벤트로 주므로 시작 시점에 바로 강조할 수 있다(실측
확인). Gate 0/1처럼 멈춰 있는 화면에서는 정적으로 해당 체크포인트 노드를
강조한다. 이미지 Worker/Validator는 [src/nodes/image.py](../nodes/image.py)가
그래프 레벨 노드로 분리돼 있어(재시도도 매 홉 Send로 명시 - 그 파일
docstring 참고) 최상위 이벤트에 바로 잡힌다 - 예전엔 이미지 하나의 전체
재시도 루프가 서브그래프 하나로 캡슐화돼 있어서 `subgraphs=True`로 별도
namespace 이벤트를 따로 봐야 했지만, 지금은 그럴 필요가 없다(`Send`
페이로드의 `input.subject`로 어느 이미지인지 바로 구분됨).

실행: streamlit run src/ui/app.py  (프로젝트 루트에서 실행)
"""
import base64
import logging
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

# 노드별 Worker/Validator 실패(critique, 재시도, 에스컬레이션)를 콘솔과
# 파일에 남긴다 - Gate 2 UI가 아직 없어(Phase 4) 에스컬레이션된 트랙을
# 재현 없이 진단하려면 이 로그가 유일한 단서다.
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
# Windows 콘솔의 기본 인코딩(cp949)에서는 로그의 한글(critique 등)이
# 깨지므로, 콘솔 핸들러도 stdout을 UTF-8로 강제 재설정해서 사용한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "automeeting.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

from src.aggregator import build_zip  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.nodes.image import image_validator, image_worker, route_worker_to_validator  # noqa: E402
from src.nodes.reference_report import reference_report_validator, reference_report_worker  # noqa: E402
from src.nodes.setting_doc import setting_doc_validator, setting_doc_worker  # noqa: E402
from src.nodes.spec import spec_validator, spec_worker  # noqa: E402
from src.stt import transcribe  # noqa: E402

SAMPLE_TRANSCRIPT_PATH = ROOT / "sample_data" / "sample_meeting.txt"
SAMPLE_TRANSCRIPT_PHASE2_PATH = ROOT / "sample_data" / "sample_meeting_phase2.txt"
SAMPLE_TRANSCRIPT_PHASE3_PATH = ROOT / "sample_data" / "sample_meeting_phase3.txt"
SAMPLE_TRANSCRIPT_PHASE4_PATH = ROOT / "sample_data" / "sample_meeting_phase4.txt"

# 그래프 시각화 패널에 그릴 노드 목록 - src/graph.py의 add_node/add_edge와
# 그대로 대응시킨다(체인 내부 구조는 안 그림 - 노드 단위로만 표시). 이미지는
# 개수가 가변이라(Send 동적 Fan-out) 여기 고정 목록이 아니라 실행 시점에
# 실제 image_tasks 개수만큼 동적으로 그린다(render_graph_html의 image_states
# 참고).
_GRAPH_LINEAR_NODES = [
    ("rag_retrieval", "RAG Retrieval"),
    ("minutes_generator", "Minutes Generator"),
    ("gate_0", "Gate 0"),
    ("idea_select", "아이디어 선정"),
    ("spec_worker", "Spec Worker"),
    ("spec_validator", "Spec Validator"),
    ("task_classifier", "Task Classifier"),
    ("style_planner", "Style Planner"),
    ("gate_1", "Gate 1"),
]
_GRAPH_FANOUT_TEXT_BRANCHES = [
    [("setting_doc_worker", "설정집 Worker"), ("setting_doc_validator", "설정집 Validator")],
    [("reference_report_worker", "검색 보고서 Worker"), ("reference_report_validator", "검색 보고서 Validator")],
]

_GRAPH_PANEL_CSS = """
<style>
.am-graph-node {
  padding: 6px 8px;
  margin-bottom: 4px;
  border-radius: 6px;
  background: rgba(120,120,120,0.12);
  border: 1px solid rgba(120,120,120,0.25);
  font-size: 12.5px;
  line-height: 1.3;
  text-align: center;
  transition: all 0.25s ease;
}
.am-graph-node.active {
  background: #ffd54a;
  border: 2px solid #f5a623;
  box-shadow: 0 0 10px rgba(245,166,35,0.9);
  font-weight: 700;
  color: #1a1a1a;
}
.am-graph-node.done {
  background: #66bb6a;
  border: 2px solid #43a047;
  color: #ffffff;
  font-weight: 600;
}
.am-graph-arrow {
  text-align: center;
  font-size: 11px;
  color: rgba(120,120,120,0.7);
  margin-bottom: 4px;
}
.am-graph-fanout {
  display: flex;
  gap: 4px;
  margin-bottom: 4px;
}
.am-graph-branch {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.am-graph-image-row {
  margin-bottom: 4px;
}
.am-graph-image-label {
  font-size: 11px;
  color: rgba(120,120,120,0.9);
  margin-bottom: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.am-graph-image-chips {
  display: flex;
  gap: 3px;
}
.am-graph-image-chips .am-graph-node {
  flex: 1;
  padding: 4px 2px;
  font-size: 11px;
  margin-bottom: 0;
}
</style>
"""


def _node_chip(node_id: str, label: str, active: set, done: set = frozenset()) -> str:
    if node_id in active:
        cls = "am-graph-node active"
    elif node_id in done:
        cls = "am-graph-node done"
    else:
        cls = "am-graph-node"
    return f'<div class="{cls}">{label}</div>'


def render_graph_html(active: set, image_states: dict | None = None, done: set | None = None) -> str:
    """현재 active(노드 id 집합)를 강조한 파이프라인 다이어그램 HTML을 만든다.

    done(노드 id 집합)에 들어있는 노드는 초록색으로 "완료해서 대기 중"임을
    표시한다 - Gate 1 이후 fan-out된 트랙(설정집/검색 보고서)은 자기 할
    일을 먼저 끝내도, 같은 슈퍼스텝의 다른 브랜치(느린 이미지 생성 등)가
    끝날 때까지 다음 단계(Validator)를 시작 못 하고 기다린다(LangGraph의
    Pregel 슈퍼스텝 동기화 - 실측 확인, 자세한 내용은 run_with_visualization
    docstring 참고). 이 대기 구간에서 그냥 꺼뜨리면(활성 아님) "멈췄나?"로
    오인하기 쉬워서(사용자 실측 지적으로 발견), 대기 중에는 초록으로
    구분해서 "이미 끝냈고 대기 중"임을 보여준다.

    image_states는 {subject: "image_worker" | "image_validator"} - 이미지별로
    지금 어느 단계가 최근에 지나갔는지(None이면 아직 시작 전. 그래프 레벨
    노드 이름을 그대로 씀 - [src/nodes/image.py](../nodes/image.py) 참고,
    이 함수 안에서는 각각 "worker"/"validator" 미니 칩으로 매핑). 이미지는
    개수가 가변이고 Send로 병렬 처리되므로, 고정된 하나의 "이미지 ×N" 칩
    대신 실제 대상별로 Worker/Validator 두 칸짜리 행을 이미지 개수만큼
    그린다. 이미지 서브패널은 done 표시를 적용하지 않는다 - worker/validator
    미니 칩은 "지금 어느 단계인지"만 표시하는 replace 방식이라 done 개념이
    따로 필요 없다(run_with_visualization docstring 참고).
    """
    image_states = image_states or {}
    done = done or set()
    parts = [_GRAPH_PANEL_CSS]
    for node_id, label in _GRAPH_LINEAR_NODES:
        parts.append(_node_chip(node_id, label, active, done))
        parts.append('<div class="am-graph-arrow">↓</div>')

    parts.append('<div class="am-graph-fanout">')
    for branch in _GRAPH_FANOUT_TEXT_BRANCHES:
        parts.append('<div class="am-graph-branch">')
        for node_id, label in branch:
            parts.append(_node_chip(node_id, label, active, done))
        parts.append("</div>")

    parts.append('<div class="am-graph-branch">')
    if not image_states:
        parts.append(_node_chip("image_worker", "이미지 (없음)", active, done))
    else:
        for subject, step in image_states.items():
            worker_active = {"worker"} if step == "image_worker" else set()
            validator_active = {"validator"} if step == "image_validator" else set()
            parts.append('<div class="am-graph-image-row">')
            parts.append(f'<div class="am-graph-image-label">{subject}</div>')
            parts.append('<div class="am-graph-image-chips">')
            parts.append(_node_chip("worker", "Worker", worker_active))
            parts.append(_node_chip("validator", "Validator", validator_active))
            parts.append("</div></div>")
    parts.append("</div>")

    parts.append("</div>")
    parts.append('<div class="am-graph-arrow">↓</div>')
    parts.append(_node_chip("__end__", "완료", active, done))
    return "".join(parts)


def _static_active_nodes(stage: str) -> set:
    """스트리밍 중이 아닐 때(정적 화면) 강조할 노드."""
    if stage == "gate_0":
        return {"gate_0"}
    if stage == "gate_1":
        return {"gate_1"}
    if stage == "done":
        return {"__end__"}
    return set()


_IMAGE_NODE_NAMES = {"image_worker", "image_validator"}


def _validation_passed(payload: dict) -> bool | None:
    """task_result 이벤트의 payload에서 이 노드가 검증(validation_status)
    결과를 반환했다면 그 is_valid 값을, 검증 노드가 아니면 None을 반환한다.

    spec/setting_doc/reference_report Validator는 항상
    `{"validation_status": {<track>: feedback}}` 형태로 반환한다(각 노드
    모듈 참고) - "debug" 스트림의 task_result 이벤트는 payload["result"]에
    그 노드 함수가 실제로 반환한 값을 그대로 담고 있어(실측 확인), 여기서
    바로 pass/fail을 읽을 수 있다. Worker 노드는 validation_status를 안
    반환하므로 항상 None(구분 불필요 - 아래 사용처에서 None을 "초록으로
    표시"로 취급).
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    validation_status = result.get("validation_status")
    if not isinstance(validation_status, dict) or not validation_status:
        return None
    feedback = next(iter(validation_status.values()), None)
    if not isinstance(feedback, dict) or "is_valid" not in feedback:
        return None
    return feedback["is_valid"]


def run_with_visualization(graph, stream_input, config, placeholder, image_subjects=None) -> None:
    """graph.invoke() 대신 graph.stream(..., stream_mode="debug")로 돌려서,
    노드가 "시작"할 때 placeholder의 다이어그램을 갱신한다(진짜 실행 중인
    노드를 실시간으로 강조 표시하기 위함).

    stream_mode="updates"가 아니라 "debug"를 쓰는 이유(실측 확인): "updates"는
    노드가 "완료"된 뒤에야 그 결과가 이벤트로 나온다 - 즉 결과가 이미 나온
    다음에야 불이 들어오는 것이라, 실제로 그 노드가 도는 동안은 화면이
    그대로였다(사용자 실측 지적으로 발견). "debug" 모드는 각 노드 실행을
    {"type": "task", ...}(시작)와 {"type": "task_result", ...}(종료) 이벤트
    쌍으로 나눠서 주므로, "task" 이벤트가 오는 즉시(그 노드가 막 시작한
    시점) 강조할 수 있다 - Send 기반 병렬 fan-out도 각 브랜치가 동시에
    시작하면 "task" 이벤트도 동시에 여러 개 온다(실측 확인).

    이미지 Worker/Validator는 [src/nodes/image.py](../nodes/image.py)가
    그래프 레벨 노드로 분리돼 있어(재시도도 매 홉 `Send`로 명시 - 그 파일
    docstring 참고) `subgraphs=True` 없이도 최상위 이벤트에 바로 잡힌다 -
    "task" 이벤트의 payload["input"]에 그 Send 페이로드(subject 포함)가
    그대로 담겨 있어(실측 확인), 이미지별 진행 상황을 subject로 바로
    구분한다. 예전엔 이미지 하나의 전체 재시도 루프가 서브그래프 하나로
    캡슐화돼 있어서 namespace가 붙은 별도 이벤트를 봐야 했지만, 지금은
    그럴 필요가 없다.

    최상위 강조는 "몇 개의 task가 지금 이 노드 이름으로 도는 중인가"를
    카운트해서, 카운트 > 0인 동안만 켠다(task 시작 시 +1, task_result 종료
    시 -1). 예전엔 "Gate 1 이후 병렬 구간은 누적(꺼뜨리지 않음)/선형 구간은
    교체" 식으로 구간을 나눠 특별 취급했는데, 그러면 (1) 같은 트랙 안의
    순차 단계(예: setting_doc_worker -> setting_doc_validator)도 "누적"
    묶음에 걸려 있어서 validator가 시작해도 worker 칸이 안 꺼졌고(같은
    트랙인데 이미지 서브패널의 worker/validator 전환처럼 자연스럽게 안
    꺼졌음), (2) 병렬 구간 진입 직후의 "누적"이 이전 상태(Gate 1)를 지우지
    않고 그 위에 더하기만 해서 Gate 1 칸이 계속 켜진 채로 남았다(둘 다
    사용자가 실측으로 지적해 발견). 카운트 기반으로 바꾸면 두 문제가 한 번에
    풀린다 - 같은 이름의 노드가 다시 시작하면 그 이름만 새로 켜지고(순차
    구간은 자연히 하나씩만 켜짐), 여러 Send 브랜치가 같은 이름을 동시에
    쓰는 경우(image_worker 등)에도 다 끝나야(카운트 0) 꺼지므로 "구간별
    특별 취급"이 아예 필요 없어진다(실측 확인).

    카운트가 0이 된(완전히 끝난) 노드는 그냥 꺼뜨리지 않고 done 집합에
    넣어 초록으로 표시한다. 실측 확인된 이유: 설정집/검색 보고서 Worker는
    보통 수 초면 끝나는데, 같은 슈퍼스텝에 같이 파견된 이미지 브랜치는
    이미지 생성 API + 자체 재시도 루프까지 있어 훨씬 오래 걸린다 -
    LangGraph의 Pregel 모델은 같은 슈퍼스텝에 함께 스케줄된 태스크가 모두
    끝나야 다음 태스크(Validator)를 스케줄하므로, 먼저 끝난 트랙은 다른
    브랜치를 기다리는 동안 아무 이벤트도 안 온다. 이 대기 구간을 그냥
    꺼뜨리면(비활성) 사용자가 "멈췄나?"로 오인했다(실측 지적으로 발견) -
    초록으로 "끝냈고 대기 중"임을 구분해서 보여준다. 노드가 다시 시작하면
    (재시도 등) done에서 빼고 다시 active로 전환한다.

    단, Validator가 검증에 실패해서(is_valid=False) 재시도/에스컬레이션으로
    이어지는 경우는 초록으로 안 켠다(`_validation_passed` 참고) - 실패한
    시도를 "성공적으로 끝남"과 같은 색으로 보여주면 헷갈린다는 지적으로
    추가함. 이 경우는 done에도 안 넣고 그냥 꺼진 채로 둔다(카운트만 0으로
    내려가고 done_nodes.add를 건너뜀) - 곧이어 Worker가 다시 시작하면
    active로 바로 전환된다.
    """
    active_counts: dict = {}
    done_nodes: set = set()
    image_states: dict = {}
    if image_subjects:
        image_states = {subject: None for subject in image_subjects}

    # subgraphs=True를 안 쓰므로(더 이상 중첩 서브그래프가 없음 - 위 docstring
    # 참고) graph.stream()은 (namespace, event) 튜플이 아니라 event 딕셔너리를
    # 바로 준다(실측 확인 - subgraphs=True일 때만 튜플 형태였다).
    for event in graph.stream(stream_input, config, stream_mode="debug"):
        etype = event.get("type")
        if etype not in ("task", "task_result"):
            continue
        payload = event.get("payload", {})
        node_name = payload.get("name")
        if not node_name:
            continue

        if node_name in _IMAGE_NODE_NAMES:
            # 이미지별 진행 상황은 별도 행으로 그리므로(render_graph_html
            # 참고) 최상위 카운트 집계에는 넣지 않는다 - 시작 시점의 subject만
            # 필요하고, 종료 이벤트는 다음 단계 시작이 자연히 대체하므로 안 봐도 된다.
            if etype != "task":
                continue
            subject = (payload.get("input") or {}).get("subject")
            if not subject:
                continue
            image_states[subject] = node_name
        else:
            # 최상위 그래프 노드 시작/종료 이벤트 - 카운트로 켜짐/꺼짐을 추적.
            if etype == "task":
                active_counts[node_name] = active_counts.get(node_name, 0) + 1
                done_nodes.discard(node_name)
            else:
                active_counts[node_name] = max(0, active_counts.get(node_name, 0) - 1)
                if active_counts[node_name] == 0 and _validation_passed(payload) is not False:
                    done_nodes.add(node_name)

        active = {name for name, count in active_counts.items() if count > 0}
        placeholder.markdown(render_graph_html(active, image_states, done_nodes), unsafe_allow_html=True)


@st.cache_resource
def get_graph():
    return build_graph()


def load_sample_transcript(path: Path) -> str:
    # Windows 기본 인코딩(cp949)이 아닌 UTF-8로 명시해서 읽어야 한글이 깨지지 않는다.
    return path.read_text(encoding="utf-8")


def get_config() -> dict:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def reset_session() -> None:
    st.session_state.pop("thread_id", None)
    st.session_state.pop("manual_image_tasks", None)
    st.session_state.pop("stt_transcript", None)
    st.session_state.pop("dismissed_reviews", None)
    # debug_mode는 일부러 안 지운다 - 디버그로 들어온 사용자가 "새 회의로
    # 다시 시작"을 눌러도 녹음 화면으로 튕기지 않고 디버그 화면에 남는다.


def resolve_image_source(image_url: str):
    """image_url이 data URI(이미지 API가 base64로 반환)면 bytes로 디코드해
    st.image가 확실히 렌더링하게 한다. 일반 http(s) URL이면 그대로 넘긴다."""
    if image_url.startswith("data:") and "base64," in image_url:
        return base64.b64decode(image_url.split("base64,", 1)[1])
    return image_url


def render_validation(feedback: dict | None) -> None:
    """검증 상태(통과/미통과 + critique)만 보여준다 - 에스컬레이션/기술적
    실패는 화면 상단의 "확인이 필요한 항목" 요약 섹션이 따로 담당한다
    (아래 render_review_section 참고, 그쪽이 재시도/무시 버튼까지 제공)."""
    if feedback:
        status = "통과" if feedback["is_valid"] else "미통과"
        st.write(f"검증 상태: {status} (retry_count={feedback['retry_count']})")
        if not feedback["is_valid"]:
            st.write(f"critique: {feedback['critique']}")


_TRACK_LABELS = {
    "spec": "기획서",
    "setting_doc": "설정집",
    "reference_report": "비교대상 검색 보고서",
}


def _track_label(track: str) -> str:
    if track in _TRACK_LABELS:
        return _TRACK_LABELS[track]
    if track.startswith("image_"):
        return f"이미지 · {track[len('image_'):]}"
    return track


def _needs_attention(values: dict, track: str) -> str | None:
    """이 트랙이 지금도 확인이 필요한 상태인지 판단해 "escalated"/"technical"/None
    중 하나를 반환한다. escalated_tasks(append-only reducer)/technical_failures
    (dict-union reducer, 재시도 성공 시 값만 None으로 덮어씀 - 아래 _retry_track
    참고)는 리스트/딕셔너리에서 항목을 지울 수 없으므로, 과거 이력이 아니라
    "지금" validation_status/technical_failures가 실제로 실패를 나타내는지로
    다시 확인한다(재시도가 성공했으면 더 이상 표시하지 않기 위함)."""
    if (values.get("technical_failures") or {}).get(track):
        return "technical"
    if track in (values.get("escalated_tasks") or []):
        feedback = values.get("validation_status", {}).get(track)
        if not feedback or not feedback.get("is_valid"):
            return "escalated"
    return None


_TRACK_WORKER_VALIDATOR = {
    "spec": (spec_worker, spec_validator),
    "setting_doc": (setting_doc_worker, setting_doc_validator),
    "reference_report": (reference_report_worker, reference_report_validator),
}


def _retry_track(graph, config, values: dict, track: str, extra_context: str) -> None:
    """트랙 하나(spec/setting_doc/reference_report/image_<subject>)를 Worker
    -> Validator 직접 호출로 재실행하고, 결과를 `graph.update_state`로 반영한다
    ("확인이 필요한 항목" 섹션의 "재시도" 버튼이 호출).

    그래프 스케줄러를 다시 태우지 않고 두 노드 함수를 직접 호출하는 이유:
    텍스트 트랙(spec/설정집/검색 보고서)은 `update_state(..., as_node=...)`로
    기존 조건부 엣지(validator -> retry -> worker)를 재활용하는 방법도 합성
    테스트로 동작을 확인했지만, 이미지 트랙은 Worker/Validator가 `Send`로
    파견되는 로컬 스코프 필드(subject/brief/style_guide/feedback)를 쓰는데
    이 필드들은 전역 AgentMeetingState에 없어서 같은 방식이 성립하지 않는다
    (image_worker/image_validator의 라우팅 함수가 기대하는 입력을 전역
    state에서 못 채움). 두 경우 다 결과적으로 같은 전역 채널
    (image_tasks/validation_status/technical_failures)만 갱신하면 그래프
    입장에서는 구분이 안 되므로, 모든 트랙을 직접 호출 방식으로 통일했다.

    기술적 실패였던 트랙도 이 함수로 재시도할 수 있다 - Worker/Validator를
    부르기 전에 이 트랙의 technical_failures만 지운 로컬 뷰를 넘기고(안 지우면
    Validator가 has_technical_failure를 보고 검증 자체를 건너뛴다), 이번
    시도가 기술적 실패 없이 끝나면 실제 state에도 그 키를 None으로 덮어써
    화면에서 사라지게 한다(technical_failures는 dict-union reducer라 키
    자체를 지울 수는 없지만, 값을 None으로 덮으면 `_needs_attention`이
    falsy로 처리한다 - escalated_tasks도 같은 이유로 append-only라 못
    지우고, 대신 `_needs_attention`이 매번 validation_status를 다시 확인해
    최신 성공 여부를 판단한다).
    """
    reset_feedback = {
        "is_valid": False,
        "critique": extra_context.strip() if extra_context and extra_context.strip() else "(추가 코멘트 없이 재시도)",
        "retry_count": 0,
    }

    if track.startswith("image_"):
        subject = track[len("image_") :]
        task = next((t for t in values.get("image_tasks", []) if t["subject"] == subject), None)
        if task is None:
            return
        payload = {
            "subject": subject,
            "brief": task.get("brief", ""),
            "style_guide": values.get("image_style_guide") or "",
            "feedback": reset_feedback,
        }
        worker_result = image_worker(payload)
        validator_payload = route_worker_to_validator(worker_result)[0].arg
        result = image_validator(validator_payload)
        update = {
            k: v
            for k, v in result.items()
            if k in ("image_tasks", "validation_status", "technical_failures", "escalated_tasks")
        }
    else:
        worker_fn, validator_fn = _TRACK_WORKER_VALIDATOR[track]
        local_state = dict(values)
        local_state["validation_status"] = {**values.get("validation_status", {}), track: reset_feedback}
        local_state["technical_failures"] = {
            k: v for k, v in (values.get("technical_failures") or {}).items() if k != track
        }
        worker_result = worker_fn(local_state)
        merged = {**local_state, **worker_result}
        validator_result = validator_fn(merged)
        update = {**worker_result, **validator_result}

    if "technical_failures" not in update:
        # 이번 시도는 기술적 실패 없이 끝남 - 예전 기록이 있었다면 지운다.
        update["technical_failures"] = {track: None}

    graph.update_state(config, update)


st.set_page_config(page_title="AutoMeeting Agent", layout="wide")

graph = get_graph()
config = get_config()
st.session_state.setdefault("debug_mode", False)

st.title("AutoMeeting Agent")

snapshot = graph.get_state(config)
if not snapshot.values:
    stage = "input" if st.session_state["debug_mode"] else "record"
elif snapshot.next == ("gate_0",):
    stage = "gate_0"
elif snapshot.next == ("gate_1",):
    stage = "gate_1"
else:
    stage = "done"

main_col, graph_col = st.columns([3, 1])

with graph_col:
    st.markdown("#### 파이프라인")
    graph_placeholder = st.empty()
    graph_placeholder.markdown(render_graph_html(_static_active_nodes(stage)), unsafe_allow_html=True)

with main_col:
    # ---------------------------------------------------------------------------
    # 1. 회의 녹음 (기본 화면)
    # ---------------------------------------------------------------------------
    if stage == "record":
        st.subheader("1. 회의 녹음")

        if "stt_transcript" not in st.session_state:
            st.caption("녹음 버튼을 눌러 회의를 녹음하세요. 녹음을 마치면 STT 결과를 보여드립니다.")

            audio_value = st.audio_input("녹음")

            if audio_value is not None:
                with st.spinner("STT 실행 중..."):
                    st.session_state["stt_transcript"] = transcribe(audio_value)
                st.rerun()

            st.write("")
            _, col_debug = st.columns([6, 1])
            with col_debug:
                if st.button("DEBUG"):
                    st.session_state["debug_mode"] = True
                    st.rerun()
        else:
            st.caption("STT 결과입니다. 잘못 인식된 부분이 있으면 직접 수정한 뒤 진행하세요.")

            edited_transcript = st.text_area(
                "STT 결과 (수정 가능)",
                value=st.session_state["stt_transcript"],
                height=320,
                key="stt_transcript_edit",
            )

            col1, col2 = st.columns(2)
            with col1:
                confirm_stt = st.button(
                    "회의록 생성", type="primary", disabled=not edited_transcript.strip()
                )
            with col2:
                re_record = st.button("다시 녹음")

            if confirm_stt:
                with st.spinner("Minutes Generator 실행 중..."):
                    run_with_visualization(
                        graph,
                        {"transcript": edited_transcript, "audio_path": "", "rag_references": []},
                        config,
                        graph_placeholder,
                    )
                st.session_state.pop("stt_transcript", None)
                st.rerun()

            if re_record:
                st.session_state.pop("stt_transcript", None)
                st.rerun()

    # ---------------------------------------------------------------------------
    # 1b. 회의 내용 입력 (디버그용 텍스트 입력)
    # ---------------------------------------------------------------------------
    elif stage == "input":
        st.subheader("1. 회의 내용 입력 (디버그용 텍스트 입력)")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            if st.button("샘플 (아이디어 1개, Phase 1용)"):
                st.session_state["transcript_input"] = load_sample_transcript(SAMPLE_TRANSCRIPT_PATH)
                st.rerun()
        with col_s2:
            if st.button("샘플 (아이디어 3개→1개, Phase 2용)"):
                st.session_state["transcript_input"] = load_sample_transcript(
                    SAMPLE_TRANSCRIPT_PHASE2_PATH
                )
                st.rerun()
        with col_s3:
            if st.button("샘플 (설정집+이미지+보고서 모두 트리거, Phase 3용)"):
                st.session_state["transcript_input"] = load_sample_transcript(
                    SAMPLE_TRANSCRIPT_PHASE3_PATH
                )
                st.rerun()
        with col_s4:
            if st.button("샘플 (사내 용어 STT 오인식, Phase 4용)"):
                st.session_state["transcript_input"] = load_sample_transcript(
                    SAMPLE_TRANSCRIPT_PHASE4_PATH
                )
                st.rerun()

        transcript = st.text_area(
            "회의 내용",
            height=320,
            placeholder="회의에서 오간 내용을 텍스트로 붙여넣거나 직접 입력하세요...",
            key="transcript_input",
        )

        if st.button("회의록 생성", type="primary", disabled=not transcript.strip()):
            with st.spinner("Minutes Generator 실행 중..."):
                run_with_visualization(
                    graph,
                    {"transcript": transcript, "audio_path": "", "rag_references": []},
                    config,
                    graph_placeholder,
                )
            st.rerun()

    # ---------------------------------------------------------------------------
    # 2. Gate 0 — 회의록 확인/수정
    # ---------------------------------------------------------------------------
    elif stage == "gate_0":
        st.subheader("2. Gate 0 — 회의록 확인/수정")
        st.caption(
            "본문/태그를 자유롭게 수정하세요. 문법적 어색함은 무시해도 됩니다. "
            "숫자·고유명사 등 사실 오류만 확인하세요."
        )

        rag_references = snapshot.values.get("rag_references") or []
        with st.expander(f"RAG 검색 결과 - Minutes Generator가 참고한 자료 ({len(rag_references)}개)"):
            if rag_references:
                for ref in rag_references:
                    st.markdown(ref)
                    st.divider()
            else:
                st.caption("검색된 참고 자료가 없습니다.")

        draft = snapshot.values.get("meeting_minutes_draft", "")
        edited = st.text_area("회의록 초안 (태그 포함)", value=draft, height=420, key="minutes_edit")

        col1, col2 = st.columns(2)
        with col1:
            confirm = st.button("확정", type="primary")
        with col2:
            restart = st.button("새 회의로 다시 시작")

        if confirm:
            with st.spinner(
                "아이디어 선정 → 기획서 생성 → 하위 산출물 분류 중... "
                "(재시도 루프 포함, 시간이 걸릴 수 있습니다)"
            ):
                graph.update_state(config, {"meeting_minutes_confirmed": edited})
                run_with_visualization(graph, None, config, graph_placeholder)
            st.rerun()

        if restart:
            reset_session()
            st.rerun()

    # ---------------------------------------------------------------------------
    # 3. Gate 1 — 산출물 확정
    # ---------------------------------------------------------------------------
    elif stage == "gate_1":
        st.subheader("3. Gate 1 — 산출물 확정")

        values = snapshot.values
        selected_tag = values.get("selected_idea_tag")
        st.write(f"선정된 아이디어: **{selected_tag}**")

        with st.expander("기획서 (클릭해서 펼치기)"):
            st.markdown(values.get("spec_document") or "(없음)")

        task_briefs = values.get("task_briefs", {})

        st.markdown("#### 설정집")
        setting_enabled = st.checkbox(
            "생성",
            value=task_briefs.get("setting_doc", {}).get("enabled", False),
            key="setting_doc_enabled",
        )
        setting_brief = st.text_area(
            "브리핑 (Worker에게 그대로 전달됩니다)",
            value=task_briefs.get("setting_doc", {}).get("brief", ""),
            key="setting_doc_brief",
            height=120,
        )

        st.markdown("#### 비교대상 검색 보고서")
        discussed_systems = values.get("discussed_systems", [])
        if discussed_systems:
            st.caption("조사 대상 시스템: " + ", ".join(discussed_systems))
        reference_enabled = st.checkbox(
            "생성",
            value=task_briefs.get("reference_report", {}).get("enabled", False),
            key="reference_report_enabled",
        )
        reference_brief = st.text_area(
            "브리핑 (Worker에게 그대로 전달됩니다)",
            value=task_briefs.get("reference_report", {}).get("brief", ""),
            key="reference_report_brief",
            height=120,
        )

        image_tasks = values.get("image_tasks", [])
        manual_image_tasks = st.session_state.setdefault("manual_image_tasks", [])

        st.markdown(f"#### 이미지 ({len(image_tasks) + len(manual_image_tasks)}개)")
        if image_tasks or manual_image_tasks:
            style_guide = st.text_area(
                "공유 스타일 가이드 (모든 이미지 공통 적용)",
                value=values.get("image_style_guide") or "",
                key="image_style_guide_edit",
                height=100,
            )
        else:
            st.caption("이미지로 그릴 만한 환경 요소(장소/사물 + 시각적 특징)가 없어 AI가 제안한 이미지 대상이 없습니다. 아래에서 직접 추가할 수 있습니다.")
            style_guide = values.get("image_style_guide") or ""

        image_enabled_flags = []
        image_brief_edits = []
        for i, task in enumerate(image_tasks):
            st.markdown(f"**{i + 1}. {task['subject']}**")
            enabled = st.checkbox("생성", value=task.get("enabled", True), key=f"image_enabled_{i}")
            brief = st.text_area(
                "브리핑", value=task.get("brief", ""), key=f"image_brief_{i}", height=80
            )
            image_enabled_flags.append(enabled)
            image_brief_edits.append(brief)

        manual_subjects = []
        manual_briefs = []
        manual_enabled_flags = []
        remove_manual_index = None
        for i, task in enumerate(manual_image_tasks):
            st.markdown(f"**직접 추가 {i + 1}**")
            col_name, col_del = st.columns([5, 1])
            with col_name:
                subject = st.text_input(
                    "이미지 대상 이름", value=task.get("subject", ""), key=f"manual_image_subject_{i}"
                )
            with col_del:
                st.write("")
                if st.button("삭제", key=f"manual_image_remove_{i}"):
                    remove_manual_index = i
            enabled = st.checkbox("생성", value=task.get("enabled", True), key=f"manual_image_enabled_{i}")
            brief = st.text_area(
                "브리핑 (시각적 특징을 포함해 적어주세요)",
                value=task.get("brief", ""),
                key=f"manual_image_brief_{i}",
                height=80,
            )
            manual_subjects.append(subject)
            manual_briefs.append(brief)
            manual_enabled_flags.append(enabled)

        if remove_manual_index is not None:
            manual_image_tasks.pop(remove_manual_index)
            st.rerun()

        if st.button("+ 이미지 추가"):
            manual_image_tasks.append({"subject": "", "brief": "", "enabled": True})
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            confirm_gate1 = st.button("확정 - 산출물 생성 시작", type="primary")
        with col2:
            restart = st.button("새 회의로 다시 시작", key="restart_gate1")

        if confirm_gate1:
            updated_task_briefs = {
                "setting_doc": {"enabled": setting_enabled, "brief": setting_brief},
                "reference_report": {"enabled": reference_enabled, "brief": reference_brief},
            }
            updated_image_tasks = [
                {**task, "enabled": image_enabled_flags[i], "brief": image_brief_edits[i]}
                for i, task in enumerate(image_tasks)
            ]
            updated_image_tasks += [
                {
                    "subject": manual_subjects[i].strip(),
                    "enabled": manual_enabled_flags[i],
                    "brief": manual_briefs[i],
                    "image_url": None,
                    "validation": None,
                }
                for i in range(len(manual_image_tasks))
                if manual_subjects[i].strip()
            ]
            st.session_state.pop("manual_image_tasks", None)

            with st.spinner(
                "설정집/비교대상 검색 보고서/이미지 생성 중... (트랙별 재시도 루프, 웹 검색, "
                "이미지 API 호출 포함 - 시간이 오래 걸릴 수 있습니다)"
            ):
                graph.update_state(
                    config,
                    {
                        "task_briefs": updated_task_briefs,
                        "image_style_guide": style_guide,
                        "image_tasks": updated_image_tasks,
                    },
                )
                enabled_image_subjects = [
                    t["subject"] for t in updated_image_tasks if t["enabled"]
                ]
                run_with_visualization(
                    graph, None, config, graph_placeholder, image_subjects=enabled_image_subjects
                )
            st.rerun()

        if restart:
            reset_session()
            st.rerun()

    # ---------------------------------------------------------------------------
    # 4. 최종 결과
    # ---------------------------------------------------------------------------
    else:
        st.subheader("4. 결과")

        values = snapshot.values
        selected_tag = values.get("selected_idea_tag")

        confirmed = values.get("meeting_minutes_confirmed")
        if confirmed:
            with st.expander("확정된 회의록"):
                st.text(confirmed)

        # --- 확인이 필요한 항목 (에스컬레이션 + 기술적 실패, 구분 표시) ---
        # 그래프에 별도 Gate 2 interrupt 노드를 두지 않고 완료 화면에 요약
        # 섹션으로 넣기로 했다([src/graph.py](../graph.py) 모듈 docstring
        # 참고) - 재시도는 여기서 바로 처리한다(_retry_track).
        review_tracks: list[str] = []
        if selected_tag:
            review_tracks.append("spec")
            task_briefs = values.get("task_briefs", {})
            if task_briefs.get("setting_doc", {}).get("enabled"):
                review_tracks.append("setting_doc")
            if task_briefs.get("reference_report", {}).get("enabled"):
                review_tracks.append("reference_report")
            for t in values.get("image_tasks", []):
                if t.get("enabled"):
                    review_tracks.append(f"image_{t['subject']}")

        dismissed = st.session_state.setdefault("dismissed_reviews", set())
        needs_review = [
            (track, kind)
            for track in review_tracks
            if (kind := _needs_attention(values, track)) and track not in dismissed
        ]

        if needs_review:
            st.markdown("### 확인이 필요한 항목")
            st.caption(
                "내용 품질 실패(재시도 한도 초과)와 기술적 실패(API 에러 등)를 구분해서 보여줍니다."
            )
            for track, kind in needs_review:
                label = _track_label(track)
                with st.container(border=True):
                    if kind == "technical":
                        st.error(f"**{label}** — 기술적 실패")
                        st.write(f"오류: {(values.get('technical_failures') or {}).get(track)}")
                    else:
                        st.warning(f"**{label}** — 재시도 한도 초과(내용 품질)")
                        feedback = values.get("validation_status", {}).get(track)
                        if feedback:
                            st.write(f"critique: {feedback.get('critique')}")

                    extra_context = st.text_area(
                        "추가 컨텍스트(선택) — 재시도 시 Worker에게 그대로 전달됩니다",
                        key=f"retry_context_{track}",
                        height=80,
                    )
                    col_retry, col_dismiss = st.columns(2)
                    with col_retry:
                        if st.button("재시도", key=f"retry_btn_{track}"):
                            with st.spinner(f"{label} 재시도 중..."):
                                _retry_track(graph, config, values, track, extra_context)
                            st.rerun()
                    with col_dismiss:
                        if st.button("무시하고 넘어가기", key=f"dismiss_btn_{track}"):
                            dismissed.add(track)
                            st.rerun()
            st.divider()

        if not selected_tag:
            st.info("구체적으로 진전된 아이디어가 없어, 회의록만 출력하고 파이프라인을 종료했습니다.")
        else:
            st.write(f"선정된 아이디어: **{selected_tag}**")

            st.markdown("### 기획서")
            render_validation(values.get("validation_status", {}).get("spec"))
            with st.expander("기획서 펼치기", expanded=False):
                st.markdown(values.get("spec_document") or "(없음)")

            task_briefs = values.get("task_briefs", {})

            if task_briefs.get("setting_doc", {}).get("enabled"):
                st.markdown("### 설정집")
                render_validation(values.get("validation_status", {}).get("setting_doc"))
                st.markdown(values.get("setting_doc") or "(생성 중 문제가 발생했습니다)")

            if task_briefs.get("reference_report", {}).get("enabled"):
                st.markdown("### 비교대상 검색 보고서")
                render_validation(values.get("validation_status", {}).get("reference_report"))
                st.markdown(values.get("reference_report") or "(생성 중 문제가 발생했습니다)")

            image_tasks = [t for t in values.get("image_tasks", []) if t.get("enabled")]
            if image_tasks:
                st.markdown(f"### 이미지 ({len(image_tasks)}개)")
                for task in image_tasks:
                    st.markdown(f"**{task['subject']}**")
                    render_validation(task.get("validation"))
                    if task.get("image_url"):
                        st.image(resolve_image_source(task["image_url"]), caption=task["subject"])
                    else:
                        st.caption("이미지 URL이 없습니다 (생성 실패 가능).")

        st.markdown("### 다운로드")
        st.download_button(
            "결과 ZIP 다운로드",
            data=build_zip(values),
            file_name="automeeting_결과.zip",
            mime="application/zip",
        )

        if st.button("새 회의로 다시 시작", key="restart_done"):
            reset_session()
            st.rerun()
