# model-intake-brief
An offline-first LangGraph CLI skeleton that can also run live research and a small regulatory RAG checkpoint.

## Setup

Use the existing Miniconda interpreter and leave `.venv` untouched.

1. Confirm the active interpreter: `C:\Users\serge\miniconda3\python.exe -c "import sys; print(sys.executable)"`
2. Set `PYTHONPATH=src` in the terminal you use for verification.
3. If editable install works in that environment, you may use `C:\Users\serge\miniconda3\python.exe -m pip install --no-build-isolation -e .`; otherwise keep using `PYTHONPATH=src`.

## Verification

### Offline CLI

- Hugging Face checkpoint: `C:\Users\serge\miniconda3\python.exe -m intake run --model Qwen/Qwen3-8B`
- Hosted checkpoint: `C:\Users\serge\miniconda3\python.exe -m intake run --provider openai --model gpt-5.2-2025-12-11`
- Invalid input: `C:\Users\serge\miniconda3\python.exe -m intake run --model invalid`

### Live research

- Hugging Face live draft: `C:\Users\serge\miniconda3\python.exe -m intake run --live --model Qwen/Qwen3-8B`
- Hosted live draft: `C:\Users\serge\miniconda3\python.exe -m intake run --live --provider openai --model gpt-5.2-2025-12-11`

### Regulatory RAG

- Index the small regulatory corpus: `C:\Users\serge\miniconda3\python.exe -m intake index-regulatory`

Redirect stdout to save any draft outside `samples/`.

## Environment

The live commands read `.env` only when you run a live or indexing command.

- `TAVILY_API_KEY`
- `INTAKE_PROVIDER_DOMAINS`
- `INTAKE_TAVILY_TIMEOUT_SECONDS`
- `INTAKE_JINA_TIMEOUT_SECONDS`
- `INTAKE_HF_TIMEOUT_SECONDS`
- `INTAKE_RAG_TIMEOUT_SECONDS`
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_NAMESPACE`
- `PINECONE_API_VERSION`

## Current behavior

- Offline mode does not read `.env` and keeps the deterministic fixture draft.
- Live mode reads `.env` only for the configured API key, timeout values, provider domain allowlist, embeddings key, and Pinecone configuration.
- Live research drafts use Tavily Search, Jina Reader, and Hugging Face Hub for Hugging Face IDs, and skip Hugging Face for hosted provider IDs.
- The live graph attaches separate regulatory context from the corpus when configuration is available.
- The regulatory corpus includes verified AI Act passages and verified GPAI Code Transparency measures 1.1 and 1.2, extracted from the readable mirror when the official PDF could not be read directly in this environment.
