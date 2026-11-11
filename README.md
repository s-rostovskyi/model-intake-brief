# Model Intake Brief

An offline-first LangGraph CLI with a lightweight Streamlit page for previewing saved reports and running the existing workflow only after an explicit click.

## Scope
- CLI report generation for Hugging Face IDs and hosted provider + model IDs.
- Live source collection with Tavily Search, Jina Reader, Hugging Face Hub, OpenAI assessment, and regulatory retrieval.
- Saved sample reports in `samples/` for offline preview.
- Streamlit interface for opening saved reports and launching the same graph manually.

## Verified locally in this checkout
The interpreter, tests, and saved-report preview were verified here with the existing Miniconda interpreter.

1. Confirm the active interpreter:
   - `C:\Users\serge\miniconda3\python.exe -c "import sys; print(sys.executable)"`
2. Run the offline and mocked verification suite:
   - `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m pytest -q`
3. Verify the Streamlit helpers load both saved reports without calling the graph.

## Fresh-clone checks
Repeat these after a clean clone if you need to confirm the environment from scratch.

- Run the editable install command in the setup section if the Miniconda environment does not already have the package installed.
- Re-run the test suite command above.
- Launch the Streamlit page with the commands in the run section below and confirm that opening the page and switching samples does not trigger any live call.

## Run commands
Use the existing Miniconda interpreter and `PYTHONPATH=src`.

### Offline CLI
- Hugging Face checkpoint: `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake run --model Qwen/Qwen3-8B`
- Hosted checkpoint: `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake run --provider openai --model gpt-5.2-2025-12-11`
- Invalid input: `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake run --model invalid`

### Live research and indexing
- Hugging Face live draft: `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake run --live --model Qwen/Qwen3-8B`
- Hosted live draft: `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake run --live --provider openai --model gpt-5.2-2025-12-11`
- Stability check, same analysis several times, then a table of which checks keep their status: `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake stability --live --model Qwen/Qwen3-8B -n 5 --out stability.md`. Every run is a real live run (Tavily, Jina Reader, Pinecone, OpenAI credits). Runs go one after another with `--pause` seconds between them (default 2). Without `--live` it only compares fixtures and says so.
- Regulatory index command: `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake index-regulatory`

### Streamlit page
- `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m streamlit run app.py`

## Environment variables
The live commands read `.env` only when you run a live or indexing command.

- `TAVILY_API_KEY`
- `INTAKE_PROVIDER_DOMAINS`
- `INTAKE_TAVILY_TIMEOUT_SECONDS`
- `INTAKE_JINA_TIMEOUT_SECONDS`
- `INTAKE_HF_TIMEOUT_SECONDS`
- `INTAKE_ASSESSMENT_TIMEOUT_SECONDS`
- `INTAKE_RAG_TIMEOUT_SECONDS`
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_NAMESPACE`
- `PINECONE_API_VERSION`

## Architecture
```mermaid
flowchart LR
    UI[CLI or Streamlit page] --> GRAPH[LangGraph workflow]
    GRAPH --> TARGET[Validate target]
    GRAPH --> SOURCES[Collect sources]
    SOURCES --> TOOLS[Tavily, Jina, Hugging Face]
    GRAPH --> RAG[Regulatory retrieval]
    GRAPH --> ASSESS[OpenAI structured assessment]
    GRAPH --> RENDER[Markdown renderer]
    RAG --> PINECONE[Pinecone corpus]
    UI --> SAMPLES[Saved reports in samples/]
```

## Pipeline view
The Streamlit page draws the workflow in a collapsible section under the form in **Run a new analysis**.

- The nodes and edges come from `APP.get_graph()`, so the diagram follows the code.
- Nodes and the tool calls inside them emit small dict events with `get_stream_writer()`. `iter_run()` reads them with `stream_mode=['updates', 'custom']`. Outside a streaming run the events are dropped, so `invoke()` and the CLI behave as before.
- `validate_and_normalize_target` has a conditional edge: on `error` the run goes straight to `render_report`.
- `collect_sources` and `retrieve_regulatory_contexts` run in parallel after `load_checklist`; `assess_evidence` waits for both and skips itself if collection failed. Inside `collect_sources` the two Tavily searches and the Hub lookup run together, then the Jina reads. Pinecone queries and link checks also run concurrently. Results keep a fixed order, so source IDs and the report do not depend on timing.
- Worker threads run in a copy of the LangGraph context (`_parallel()` in `graph.py`), which is what keeps `get_stream_writer()` working in them.
- Events are stored in `st.session_state['last_trace']`, which powers **Replay last run** without API calls.
- **Demo pace** only delays the redraw of an offline run. `LayerGate` also holds a node's events until its parallel sibling has started, so side-by-side nodes are drawn running together. It is not part of the graph and is excluded from the reported run time.
- `st.html` removes inline SVG, so the diagram uses HTML and CSS only.

## Tools
- `Tavily Search` discovers candidate URLs for live research.
- `Jina Reader` extracts text from public pages after HTTPS and host validation.
- `Hugging Face Hub` fetches public metadata for Hugging Face model IDs.
- `OpenAI Responses` performs the structured evidence assessment with a 4,000-output-token cap.
- `Pinecone` stores and retrieves the small regulatory corpus.

## Indexing and retrieval
- Index the regulatory corpus with `PYTHONPATH=src C:\Users\serge\miniconda3\python.exe -m intake index-regulatory`.
- Retrieval happens inside the graph after the checklist is loaded.
- The configured namespace stays separate from other Pinecone data.
- The checked corpus covers the verified AI Act passages plus the verified GPAI Code of Practice transparency measures 1.1 and 1.2.

## File map
- `src/intake/cli.py`: command-line entry point and report output.
- `src/intake/graph.py`: LangGraph state machine, source routing, and the pipeline events the nodes emit.
- `src/intake/runner.py`: `iter_run()` streams a run as events and returns the same final state as `invoke()`.
- `src/intake/pipeline_view.py`: graph topology, per-run state reducer, and HTML for the Pipeline view.
- `assets/pipeline.css`: styles for the Pipeline view.
- `src/intake/adapters.py`: Tavily, Jina, and Hugging Face adapters.
- `src/intake/assessment.py`: OpenAI assessment, validation, and Markdown rendering.
- `src/intake/regulatory.py`: corpus loading, indexing, and retrieval helpers.
- `app.py`: Streamlit page for previewing saved reports and running the graph.
- `samples/qwen3-8b.md`: saved live Qwen report.
- `samples/gpt-5.2.md`: saved live hosted-model report.
- `tests/`: regression coverage for offline, live-mocked, and UI helpers.

## Saved reports
- [Qwen/Qwen3-8B sample](samples/qwen3-8b.md)
- [GPT-5.2 sample](samples/gpt-5.2.md)

## Current behavior
- Offline mode does not read `.env` and keeps the deterministic fixture draft.
- The Streamlit page loads saved reports without calling the graph until the user clicks `Run graph`.
- Live mode reads `.env` only for the configured API keys, timeout values, provider domain allowlist, embeddings key, assessment timeout, and Pinecone configuration.
- Live research drafts use Tavily Search, Jina Reader, and Hugging Face Hub for Hugging Face IDs, and skip Hugging Face for hosted provider IDs.
- The Day 3 report uses `gpt-5.4-mini` for structured evidence assessment, keeps model-provider attribution separate from the hosting platform, writes the final Markdown to stdout, and renders `## Full report` for a successful live run while preserving partial and offline labels when needed.
- The hosted GPT-5.2 demo target also seeds the provider-published training-content summary at `https://cdn.openai.com/pdf/gpt-5-2-eu-ai-act-public-summary-of-training-content.pdf` as a preferred Jina candidate source, while keeping the official model documentation and redirect allowlist in place.
- The regulatory corpus includes verified AI Act passages and verified GPAI Code transparency measures 1.1 and 1.2, extracted from the readable mirror when the official PDF could not be read directly in this environment.
- The report overview is computed by `src/intake/dashboard.py` from the report Markdown (findings, sources, regulatory section), so it is the same for a fresh run and a saved report. The trust tiles are derived values, not new data: quotes found in source text, link-check outcomes from the `Link check` lines, provider-domain match by host name (a heuristic).
