"""Phase 1~5 전체 파이프라인 통합 테스트 (디버그 텍스트 입력 기준).

pytest 등 테스트 프레임워크 없이(이 프로젝트에 아직 도입 안 됨) 직접 실행하는
스크립트다: `python tests/test_full_pipeline.py` (프로젝트 루트에서).

기본은 **구조 확인만**(무료, API 호출 없음) - 그래프가 정상적으로 빌드되고
노드/엣지가 기대한 대로 연결됐는지만 확인한다. 실제 파이프라인을 텍스트
입력부터 끝까지(RAG -> 회의록 -> 기획서 -> 설정집/검색 보고서/이미지 -> ZIP)
전부 진짜로 돌려보려면 `--real` 플래그를 추가한다:

    python tests/test_full_pipeline.py --real

`--real`은 실제 OpenAI API를 여러 번 호출한다(LLM 호출 다수 + 이미지 생성 1회
+ 웹 검색) - 비용이 발생한다(대략 이미지 생성 1회 $0.04~ + LLM 호출 10회+
+ 임베딩 호출). 구조 확인만으로 충분하면 플래그 없이 실행할 것.
"""
import sys
import warnings
import zipfile
from io import BytesIO
from pathlib import Path

# Windows 콘솔 기본 인코딩(cp949)에서는 한글 출력이 깨지므로 UTF-8로 재설정한다
# (다른 소스 파일들과 동일한 관례 - CLAUDE.md "환경" 섹션 참고).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def check_structure() -> None:
    """그래프가 빌드되고 노드/엣지가 기대한 대로 연결됐는지만 확인(API 호출 없음)."""
    from src.graph import build_graph

    graph = build_graph()
    g = graph.get_graph()
    node_names = set(g.nodes.keys())

    expected_nodes = {
        "rag_retrieval", "minutes_generator", "gate_0", "idea_select",
        "spec_worker", "spec_validator", "spec_escalate",
        "task_classifier", "style_planner", "gate_1",
        "setting_doc_worker", "setting_doc_validator", "setting_doc_escalate",
        "reference_report_worker", "reference_report_validator", "reference_report_escalate",
        "image_worker", "image_validator",
    }
    missing = expected_nodes - node_names
    assert not missing, f"그래프에 없는 노드: {missing}"
    print(f"[OK] 그래프 노드 {len(node_names)}개 확인 (기대한 {len(expected_nodes)}개 전부 포함)")

    # technical_failures/escalated_tasks가 state 스키마에 있는지 확인.
    from src.state import AgentMeetingState

    annotations = AgentMeetingState.__annotations__
    assert "technical_failures" in annotations, "state.py에 technical_failures 필드가 없음"
    assert "escalated_tasks" in annotations, "state.py에 escalated_tasks 필드가 없음"
    print("[OK] state.py에 technical_failures/escalated_tasks 필드 확인")

    # has_technical_failure/_make_validation_router가 기술적 실패를 올바르게 처리하는지
    # (재시도 없이 done으로 바로 보내는지) - API 호출 없이 순수 라우팅 로직만 확인.
    from src.graph import _make_validation_router

    state_with_failure = {"technical_failures": {"spec": "simulated"}, "validation_status": {}}
    decision = _make_validation_router("spec")(state_with_failure)
    assert decision == "done", f"기술적 실패가 있으면 'done'으로 바로 가야 하는데 '{decision}'을 반환함"
    print("[OK] 기술적 실패 라우팅(재시도 없이 종료) 확인")

    # aggregator.build_zip이 빈 state에서도(극단적 케이스) 죽지 않고 최소 회의록만 담는지.
    from src.aggregator import build_zip

    minimal_values = {"meeting_minutes_confirmed": "[아이디어1] 최소 회의록", "image_tasks": [], "task_briefs": {}}
    zip_bytes = build_zip(minimal_values)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
    assert names == ["회의록.md"], f"극단적 케이스(전부 실패)에서 ZIP이 예상과 다름: {names}"
    print("[OK] 극단적 케이스(전부 기술적 실패) - ZIP에 최소 회의록만 담기는 것 확인")

    empty_values = {"image_tasks": [], "task_briefs": {}}
    empty_zip = build_zip(empty_values)
    with zipfile.ZipFile(BytesIO(empty_zip)) as zf:
        assert zf.namelist() == [], "회의록조차 없을 때 빈 ZIP이 아니라 뭔가 들어있음"
    print("[OK] 회의록조차 없는 완전 극단 케이스도 죽지 않고 빈 ZIP 생성 확인")

    print("\n구조 확인 전부 통과.")


def run_real_pipeline() -> None:
    """디버그 텍스트 입력 -> Gate 0 -> Gate 1(설정집+검색 보고서+이미지 1개 활성화)
    -> 완료까지 실제 API로 전부 돌린다. 비용 발생(모듈 docstring 참고)."""
    import logging

    from dotenv import load_dotenv

    load_dotenv()
    logging.disable(logging.CRITICAL)

    from src.aggregator import build_zip
    from src.graph import build_graph

    graph = build_graph()
    config = {"configurable": {"thread_id": "integration-test-real-run"}}

    transcript_path = ROOT / "sample_data" / "sample_meeting_phase3.txt"
    transcript = transcript_path.read_text(encoding="utf-8")

    print("1/5 RAG Retrieval -> Minutes Generator -> Gate 0 ...")
    graph.invoke({"transcript": transcript}, config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("gate_0",), f"Gate 0에서 멈춰야 하는데: {snapshot.next}"
    draft = snapshot.values.get("meeting_minutes_draft", "")
    assert draft, "회의록 초안이 비어있음"
    print(f"   [OK] 회의록 초안 {len(draft)}자 생성, Gate 0에서 정지 확인")

    print("2/5 Gate 0 확정 -> 아이디어 선정 -> 기획서 -> Task Classifier -> Gate 1 ...")
    graph.update_state(config, {"meeting_minutes_confirmed": draft})
    graph.invoke(None, config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("gate_1",), f"Gate 1에서 멈춰야 하는데: {snapshot.next}"
    values = snapshot.values
    assert values.get("selected_idea_tag"), "아이디어가 선정되지 않음"
    assert values.get("spec_document"), "기획서가 생성되지 않음"
    print(f"   [OK] 아이디어 선정: {values['selected_idea_tag']}, 기획서 {len(values['spec_document'])}자, Gate 1에서 정지 확인")

    print("3/5 Gate 1 확정(설정집+검색 보고서+이미지 1개 활성화) -> Fan-out 실행 중 ...")
    image_tasks = values.get("image_tasks", [])
    updated_image_tasks = [
        {**t, "enabled": i == 0} for i, t in enumerate(image_tasks)
    ]
    task_briefs = values.get("task_briefs", {})
    graph.update_state(
        config,
        {
            "task_briefs": {
                "setting_doc": {"enabled": True, "brief": task_briefs.get("setting_doc", {}).get("brief", "")},
                "reference_report": {
                    "enabled": True,
                    "brief": task_briefs.get("reference_report", {}).get("brief", ""),
                },
            },
            "image_tasks": updated_image_tasks,
        },
    )
    graph.invoke(None, config)
    snapshot = graph.get_state(config)
    assert snapshot.next == (), f"Fan-out 이후 END에 도달해야 하는데: {snapshot.next}"
    values = snapshot.values
    print("   [OK] Fan-out 완료, 그래프가 END에 도달함")

    print("4/5 결과 검증 ...")
    assert values.get("setting_doc"), "설정집이 생성되지 않음"
    assert values.get("reference_report"), "비교대상 검색 보고서가 생성되지 않음"
    enabled_images = [t for t in values.get("image_tasks", []) if t.get("enabled")]
    assert enabled_images, "활성화된 이미지가 없음"
    print(f"   [OK] 설정집 {len(values['setting_doc'])}자, 검색 보고서 {len(values['reference_report'])}자, 이미지 {len(enabled_images)}개")
    print(f"   escalated_tasks: {values.get('escalated_tasks', [])}")
    print(f"   technical_failures: {values.get('technical_failures', {})}")

    print("5/5 ZIP 패키징 ...")
    zip_bytes = build_zip(values)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
    print(f"   [OK] ZIP 항목: {names}")
    assert "회의록.md" in names
    assert "기획서.md" in names
    assert "설정집.md" in names
    assert "비교대상_검색_보고서.md" in names

    print("\n실제 API 통합 실행 전부 통과.")


if __name__ == "__main__":
    check_structure()
    if "--real" in sys.argv:
        print("\n--- --real 플래그 감지: 실제 API 호출로 전체 파이프라인 실행 ---\n")
        run_real_pipeline()
    else:
        print("\n(--real 플래그 없이 실행됨 - 구조 확인만 완료. 실제 API로 전체 파이프라인을")
        print(" 돌려보려면 `python tests/test_full_pipeline.py --real` - 비용 발생함, 스크립트")
        print(" 상단 docstring 참고.)")
