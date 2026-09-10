# AutoMeeting Agent

[한국어](README.md) | [English](README.en.md)

A LangGraph-based, Human-in-the-Loop, multi-modal meeting-output automation
agent. Feed it a meeting recording (or transcript text), and it walks through
minutes → spec document → setting doc / comparable-games research report /
concept art images, with a human reviewing and confirming at two checkpoints
(Gate 0/1) along the way.

The full product spec and design intent live in
[AutoMeeting_Agent_기획서_v3.md](AutoMeeting_Agent_기획서_v3.md) (Korean).
Implementation details and the history behind architecture decisions live in
[CLAUDE.md](CLAUDE.md).

## Pipeline

```
Recording (STT) or text input
  → RAG Retrieval (search internal glossary/game-profile corpus)
  → Minutes Generator (tagged meeting minutes)
  → Gate 0 (review/edit minutes)
  → Idea selection → Spec Worker/Validator (spec doc, critique-based retry)
  → Task Classifier (decide which downstream outputs are needed)
  → Gate 1 (review spec + confirm downstream outputs)
  → Fan-out: setting doc / comparable-games report / images×N
    (parallel, each with its own retry loop)
  → Done screen: items needing attention (content failure vs. technical
    failure, shown separately) + ZIP download
```

## Key Features

- **Human-in-the-Loop**: two checkpoints (Gate 0 for minutes, Gate 1 for the
  spec + downstream outputs) where a person reviews and edits before the
  pipeline continues.
- **Critique-based self-validation**: each output is produced by a Worker and
  checked by a separate Validator; failures trigger a retry with specific
  feedback (escalating once the retry limit is hit).
- **Technical-failure safety net**: API errors and other technical failures
  are recorded separately from content-quality failures, and a failure in one
  track never blocks the others.
- **RAG-based terminology consistency**: searches an internal glossary/game
  profile corpus to correct likely STT misrecognitions of proper nouns and
  keep terminology consistent across outputs.
- **Live pipeline visualization**: shows which node is currently running as a
  graph, including multiple tracks running in parallel.
- **Partial-failure-tolerant ZIP export**: even if some outputs fail, the rest
  are still packaged into a downloadable ZIP (Markdown + image files).

## Tech Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
  (`StateGraph`, `interrupt_before`, dynamic `Send`-based fan-out)
- **LLM**: OpenAI GPT-5.6 family (`reasoning_effort="none"`); image
  generation via `gpt-image-2`
- **RAG**: ChromaDB (in-memory) + `text-embedding-3-small`
- **Web search**: [`ddgs`](https://pypi.org/project/ddgs/) (no API key
  required)
- **STT**: OpenAI audio transcription API
- **UI**: Streamlit

## Getting Started

Requires Python 3.14 (see `.python-version`).

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt

cp .env.example .env
# Open .env and fill in OPENAI_API_KEY (the other defaults are fine as-is)

streamlit run src/ui/app.py
```

## Project Structure

```
src/
  graph.py            # Graph skeleton (Gate 0/1, fan-out, retry/technical-failure routing)
  state.py            # LangGraph state schema
  aggregator.py        # ZIP packaging (called by the download button on the done screen)
  nodes/               # Minutes/Spec/setting-doc/reference-report/image nodes, etc.
  ui/app.py            # Streamlit UI
tests/
  test_full_pipeline.py  # Integration test (structural check by default, --real for a full run)
sample_data/            # Sample transcripts and internal-corpus test data
```

For the detailed design rationale behind each file (why it was built this
way, what real-world testing led to each change), see
[CLAUDE.md](CLAUDE.md) and the module docstring at the top of each source
file.

## Testing

```bash
python tests/test_full_pipeline.py          # structural check only (free)
python tests/test_full_pipeline.py --real   # runs the full pipeline against real APIs (costs money)
```

## License

[MIT](LICENSE)
