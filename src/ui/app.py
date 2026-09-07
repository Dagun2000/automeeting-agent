"""Streamlit UI — Phase 1~3: 디버그 텍스트 입력, Gate 0, Gate 1, 최종 결과 확인.

실행: streamlit run src/ui/app.py  (프로젝트 루트에서 실행)
"""
import base64
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from src.graph import build_graph  # noqa: E402

SAMPLE_TRANSCRIPT_PATH = ROOT / "sample_data" / "sample_meeting.txt"
SAMPLE_TRANSCRIPT_PHASE2_PATH = ROOT / "sample_data" / "sample_meeting_phase2.txt"
SAMPLE_TRANSCRIPT_PHASE3_PATH = ROOT / "sample_data" / "sample_meeting_phase3.txt"


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


def resolve_image_source(image_url: str):
    """image_url이 data URI(이미지 API가 base64로 반환)면 bytes로 디코드해
    st.image가 확실히 렌더링하게 한다. 일반 http(s) URL이면 그대로 넘긴다."""
    if image_url.startswith("data:") and "base64," in image_url:
        return base64.b64decode(image_url.split("base64,", 1)[1])
    return image_url


def render_validation(feedback: dict | None, track_key: str, escalated_tasks: list) -> None:
    if feedback:
        status = "통과" if feedback["is_valid"] else "미통과"
        st.write(f"검증 상태: {status} (retry_count={feedback['retry_count']})")
        if not feedback["is_valid"]:
            st.write(f"critique: {feedback['critique']}")
    if track_key in (escalated_tasks or []):
        st.warning(
            "재시도 한도(MAX_RETRIES)를 초과해 Gate 2 에스컬레이션 대상으로 "
            "기록됐습니다 (Gate 2 UI는 Phase 4에서 구현 예정)."
        )


st.set_page_config(page_title="AutoMeeting Agent (Phase 1~3)", layout="centered")

graph = get_graph()
config = get_config()

st.title("AutoMeeting Agent — Phase 1~3")
st.caption("디버그 경로: 오디오 없이 텍스트를 바로 Minutes Generator로 전달합니다 (STT/Diarization 생략).")

snapshot = graph.get_state(config)
if not snapshot.values:
    stage = "input"
elif snapshot.next == ("gate_0",):
    stage = "gate_0"
elif snapshot.next == ("gate_1",):
    stage = "gate_1"
else:
    stage = "done"

# ---------------------------------------------------------------------------
# 1. 회의 내용 입력
# ---------------------------------------------------------------------------
if stage == "input":
    st.subheader("1. 회의 내용 입력 (디버그용 텍스트 입력)")

    col_s1, col_s2, col_s3 = st.columns(3)
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

    transcript = st.text_area(
        "회의 내용",
        height=320,
        placeholder="회의에서 오간 내용을 텍스트로 붙여넣거나 직접 입력하세요...",
        key="transcript_input",
    )

    if st.button("회의록 생성", type="primary", disabled=not transcript.strip()):
        with st.spinner("Minutes Generator 실행 중..."):
            graph.invoke(
                {"transcript": transcript, "audio_path": "", "rag_references": []},
                config,
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
            graph.invoke(None, config)
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
    st.markdown(f"#### 이미지 ({len(image_tasks)}개)")
    if image_tasks:
        style_guide = st.text_area(
            "공유 스타일 가이드 (모든 이미지 공통 적용)",
            value=values.get("image_style_guide") or "",
            key="image_style_guide_edit",
            height=100,
        )
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
    else:
        st.caption("이미지로 그릴 만한 환경 요소(장소/사물 + 시각적 특징)가 없어 이미지 대상이 없습니다.")
        style_guide = values.get("image_style_guide") or ""
        image_enabled_flags = []
        image_brief_edits = []

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
            graph.invoke(None, config)
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
    escalated_tasks = values.get("escalated_tasks", [])

    confirmed = values.get("meeting_minutes_confirmed")
    if confirmed:
        with st.expander("확정된 회의록"):
            st.text(confirmed)

    selected_tag = values.get("selected_idea_tag")
    if not selected_tag:
        st.info("구체적으로 진전된 아이디어가 없어, 회의록만 출력하고 파이프라인을 종료했습니다.")
    else:
        st.write(f"선정된 아이디어: **{selected_tag}**")

        st.markdown("### 기획서")
        render_validation(values.get("validation_status", {}).get("spec"), "spec", escalated_tasks)
        with st.expander("기획서 펼치기", expanded=False):
            st.markdown(values.get("spec_document") or "(없음)")

        task_briefs = values.get("task_briefs", {})

        if task_briefs.get("setting_doc", {}).get("enabled"):
            st.markdown("### 설정집")
            render_validation(
                values.get("validation_status", {}).get("setting_doc"), "setting_doc", escalated_tasks
            )
            st.markdown(values.get("setting_doc") or "(생성 중 문제가 발생했습니다)")

        if task_briefs.get("reference_report", {}).get("enabled"):
            st.markdown("### 비교대상 검색 보고서")
            render_validation(
                values.get("validation_status", {}).get("reference_report"),
                "reference_report",
                escalated_tasks,
            )
            st.markdown(values.get("reference_report") or "(생성 중 문제가 발생했습니다)")

        image_tasks = [t for t in values.get("image_tasks", []) if t.get("enabled")]
        if image_tasks:
            st.markdown(f"### 이미지 ({len(image_tasks)}개)")
            for task in image_tasks:
                st.markdown(f"**{task['subject']}**")
                render_validation(task.get("validation"), f"image_{task['subject']}", escalated_tasks)
                if task.get("image_url"):
                    st.image(resolve_image_source(task["image_url"]), caption=task["subject"])
                else:
                    st.caption("이미지 URL이 없습니다 (생성 실패 가능).")

    if st.button("새 회의로 다시 시작", key="restart_done"):
        reset_session()
        st.rerun()
