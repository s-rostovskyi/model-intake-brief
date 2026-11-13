# Model Intake Brief — User Guide

Model Intake Brief helps AI governance, PMO, and delivery teams review public documentation for a third-party general-purpose AI model before integration. Give it a model identifier and it produces an English Markdown brief showing which checklist items have public evidence, which need more investigation, and what to ask the provider.

**Information, not legal advice.** Use the brief to prepare a human review. Evidence coverage does not determine legal applicability, compliance, model quality, safety, or approval to deploy.

## Contents

- [Reviewer guide](#reviewer-guide)
- [Use the web interface](#use-the-web-interface)
- [Understand the inputs and research process](#understand-the-inputs-and-research-process)
- [Read and use the report](#read-and-use-the-report)
- [Setup and CLI](#setup-and-cli)
- [Use the command line](#use-the-command-line)
- [Troubleshoot](#troubleshoot)

## Reviewer guide

### Try a report first

1. Leave **Workspace** on **Saved reports**.
2. Select **Qwen/Qwen3-8B** or **gpt-5.2-2025-12-11**.
3. Scan **Report overview** and expand **Run details and data sources**.
4. Filter or search **Findings**. Expand an item and compare its excerpt with the linked source.
5. Use **Download Markdown** to keep the report. Use **Copy questions** or **Download questions** for provider follow-ups.

Viewing, switching, and downloading saved reports requires no application API calls or keys. The samples contain results of earlier live analyses.

## Use the web interface

### Workspace and appearance

| Control | What it does |
|---|---|
| **Workspace → Saved reports** | Opens one of the two bundled reports. Select a different sample from the **Report** list to switch. Select a status in **Evidence coverage** to open the matching findings. |
| **Workspace → Run a new analysis** | Opens the analysis form and shows the most recent result from this browser session, if one exists. |
| **User guide** | Opens the reviewer guide at the bottom of either workspace. Use **Download full user guide** for the setup and CLI instructions. |
| **Main menu (⋮) → System, Light or Dark** | Selects the light or dark appearance. The header palette follows the theme. |

### Analysis form

| Control | How to use it |
|---|---|
| **Model input format** | Choose **Hugging Face ORG/MODEL** for a public Hugging Face repository. Choose **Hosted provider + exact model ID** for a model documented by a hosted service. |
| **Provider** | Available for hosted models. Enter the provider name, such as `openai`. Its official documentation domains must be configured for a live run. |
| **Model** | Enter the exact identifier. Use `Qwen/Qwen3-8B` for a Hugging Face target and `gpt-5.2-2025-12-11` for the hosted example. |
| **Use example** | Fills the fields with the sample for the selected input format. |
| **Recent models** | Reuses a previously submitted model for the current input format. The app keeps up to five entries in this session. |
| **Live run** | Off by default. With it off, the graph renders a deterministic offline preview. Turn it on to search public sources and assess the model using configured external services. A live run can consume service credits. |
| **Demo pace** | Offline only, on by default. Holds each pipeline step on screen for a moment so the run can be followed. The run time in the report ignores these pauses. Turn it off to see the real speed. |
| **Run offline preview** or **Run live analysis** | Starts the analysis. Typing in a field or switching workspace modes does not start a run. While a run is in progress, the form and the workspace switch are locked, because changing them would stop the run. |

**Offline preview** is a way to check the form, report layout, and Markdown download. The report is labelled **Offline demo — fixture data**, and the overview says the run was not assessed instead of showing evidence coverage.

**Live analysis** collects and assesses current public sources. When the page shows **Analysis in progress**, wait for the result. The indicator shows that work is underway. Evidence counts appear when the report is ready. If the run produces a partial report, read the visible failure message and **Methods and limitations** before using its findings.

### Pipeline view

Under the form, **Pipeline view** shows the workflow as a diagram that is drawn from the actual graph. Select its header to collapse or expand it.

| What you see | Meaning |
|---|---|
| Node cards 1 to 6 | The stages of the workflow. A card is blue while it runs, green when done, red when it failed, and dashed grey when it was skipped. |
| Steps under a card | The calls made inside that stage as they happen, for example each Tavily search, Jina page, Pinecone retrieval and the OpenAI assessment. They show the HTTP status or token counts when known, and the time each took. A warning mark means the call finished but gave no usable result. |
| **Passed on** | A short summary of what the stage added to the shared state, such as item counts. Source text and keys are never shown. |
| Moving arrow | Data is handed from the finished stage to the next one. |
| **‖** and the **parallel** arcs | **Collect sources** and **Regulatory context** run at the same time. Both start after the checklist is loaded, and **Assess evidence** waits for both. Both cards are blue while they run. Searches and page reads inside **Collect sources** also overlap. |
| **Parallel branches saved** | Shown after a run when the overlap saved at least half a second: the sum of the time each stage ran minus the real run time. |
| Red arc, **on error** | Appears when target validation fails and the graph goes directly to the renderer to report the error. If source collection fails, its node is marked failed, **Assess evidence** is skipped, and the renderer reports the error on the existing route. |
| **Replay last run** | Redraws the recorded events of the latest run with no API calls. Replay shows the same steps, with long waits shortened. |

In offline mode with **Demo pace**, parallel stages are drawn running together even though the offline data finishes instantly. The view is a picture of one run. It does not change the result, and the CLI works without it.

The **latest run result** stays available when you switch workspace modes in the same session. It remains associated with the model and mode of the completed run, even if you have since changed the form. Use **Download Markdown** to keep a copy before ending the session. A new analysis does not add an entry to the bundled **Saved reports** list.

### Report controls and metrics

| Element | How to read or use it |
|---|---|
| **Target model** | The exact identifier assessed. **Provider** identifies the model provider where available. **Platform** identifies where the model is hosted. For example, Qwen publishes the Qwen3‑8B model on Hugging Face. |
| **Evidence coverage** | The headline: how many of the 15 core checks are fully evidenced, with a bar split by status. The three training-resource checks (CT-01 to CT-03) are context only and are listed on a separate line, not counted. In a saved report, select a status to open the matching findings. That list also includes the context checks, so its counts can be higher. |
| **Where the gaps are** | One row per checklist group (Identity, Distribution and licence, Technical properties, Intended use, Training data) with one square per check. Hover a square for the check ID and status. |
| **Ask the provider** | The three most important provider questions: **Not found** before **Partial**. The full list is under **Provider questions** in the report details. |
| **How far to trust this report** | Four checks on the report itself. **Quotes verified in source text**: quotes that were found in the extracted page text, out of all quotes the model proposed (unsupported ones are dropped). **Cited links reachable**: cited pages that answered the link check (HTTP 2xx or 3xx), with the reason when one did not. A page that answered but sits outside the approved domains still counts as reachable and is called out separately. **Cited from provider domains**: cited pages on the provider's own domain, matched by host name, so treat it as a heuristic. **Evidence about the exact model**: flags findings that cited a different model or variant. |
| **Open gaps and AI Act provisions** | Provisions retrieved for the open gaps by similarity search. It is a pointer for the reviewer, not a compliance finding. |
| **Run time** and **Est. assessment cost** | A small line at the bottom of the overview. Run time is for a run started from the web interface. The bundled samples show **not recorded**. Cost is a USD estimate from recorded assessment tokens at the application's standard uncached rates. It excludes embeddings, search, extraction, and storage and may differ from the provider's bill. Offline preview shows **$0**. |
| **Run details and data sources** | Expands the assessment model, run date, checklist version, token counts when available, and recorded source-tool outcomes. |
| **Download Markdown** | Downloads the report currently displayed, including details and regulatory context. A partial report retains its partial label. |
| **Report details** | The full report, including evidence excerpts, links, questions for the provider, and limitations. |

## Understand the inputs and research process

### Choose the correct model identifier

The application takes one model per analysis. Use the identifier that matches the version you plan to integrate.

| Target type | Example input | Research route |
|---|---|---|
| Hugging Face repository | Model `Qwen/Qwen3-8B` | Looks up model metadata in Hugging Face Hub and collects public model-card and related source text. |
| Hosted model | Provider `openai`, model `gpt-5.2-2025-12-11` | Searches the configured official provider domains and collects public documentation. For this exact demo model, a known provider-published training-content summary is prioritised as a candidate. |

Enter an identifier in **Model**, not a documentation URL. A similar model name, a different snapshot, or a base model can have different evidence. The hosted route researches documentation about the target model. The separate assessment model used by the application is `gpt-5.4-mini`.

### What happens after you submit

1. **Identify the target.** The workflow validates the input format, resolves the route, and loads the fixed checklist.
2. **Collect public sources.** Tavily Search discovers candidates. Jina Reader extracts text from eligible public pages or PDFs. The Hugging Face route also calls the Hub API for repository metadata. Search snippets help locate sources.
3. **Retrieve regulatory context.** This runs at the same time as step 2, because it only needs the checklist. OpenAI's `text-embedding-3-small` represents search queries as vectors. Pinecone uses them to retrieve relevant passages from the indexed corpus. The corpus contains selected EU AI Act Article 53 and Annex XI/XII passages and the GPAI Code of Practice Transparency chapter.
4. **Assess and verify.** Starts when steps 2 and 3 have both finished. The assessment model uses the retrieved passages to interpret the associated checklist criteria. Only passages for a provision that the checklist item itself cites are passed on. Items whose provisions are not in the corpus are assessed as written. It uses public model documentation to propose a status and supporting evidence for each item. The workflow checks excerpts against retrieved source material, checks whether the evidence applies to the target, and records public link-check outcomes for selected URLs.
5. **Render the brief.** Python assembles the Markdown, including gaps and questions for the provider. Regulatory references appear as context for the checks. Findings about the target rely on the provider evidence collected for that run.

Search, extraction, and link checks run within request limits. Gated pages, rate limits, incomplete extraction, and documents published under a different version can affect the result. A link check establishes the reported outcome of accessing a URL at run time. Read the excerpt and its surrounding context to judge what it supports.

## Read and use the report

### Evidence statuses

| Status | Meaning | Reviewer action |
|---|---|---|
| **Evidence found** | The run identified attributable evidence that meets the item criterion and applies to the target. | Check the excerpt and its context before relying on it. |
| **Partial** | Available evidence covers part of the criterion or could only be verified in part. | Read **Gap** and decide whether to request more detail. |
| **Not found** | The configured search did not establish supporting evidence for the item in this run. | Check which sources were reviewed and follow up if the gap matters. |
| **Not checked** | The workflow could not complete the check, for example because of a failure or offline mode. | Review the failure reason before treating the item as open or rerunning it. |

These statuses describe coverage of the public sources reviewed. Compare evidence against your organisation's intake criteria before making a decision.

### Sections in the downloaded brief

| Section | What it contains |
|---|---|
| **Report header** | Report type, exact model ID, provider, hosting platform, checklist version, run date, legal notice, and assessment usage where available. |
| **Summary** | Counts by status and major gaps. Use this to decide which findings to review first. |
| **Source collection** | The services contacted, URLs or source candidates, available HTTP statuses, and indicators of usable results. |
| **Findings** | The checklist ID, criterion, **Evidence sufficient**, status, and available evidence. Depending on the result, a finding can include a quote, source URL or ID, link-check result, and explanation of the gap. |
| **Prioritised provider questions** | Follow-up questions derived from unresolved items. Review and adapt the questions before sharing them. Context-only items are separate from routine provider requests. |
| **Methods and limitations** | Scope of the run and limitations or failure information. |
| **Regulatory context** | Retrieved provisions, document versions, links, and match scores where available. A relevance score ranks retrieved text for a query. It is not an evidence confidence or compliance score. |

The 18 checks cover identity and release details, distribution and licensing, technical properties, intended use and integration, training-data disclosures, and selected training-resource context. The [evidence checklist](model-intake-checklist.md) defines the acceptance criteria for each item.

### A practical review workflow

1. Confirm the target version, provider, and run date.
2. Start with the items that matter most for the integration decision.
3. Open the source for each important positive finding and confirm that the cited text applies to the exact target or a clearly documented applicable model family.
4. Read the source outcomes before interpreting **Not found** or **Not checked**.
5. Convert material gaps into review actions or provider questions. Record the final decision in your organisation's governance process.
6. Download the Markdown and retain it with the review record. Run a new analysis when a model version or its documentation changes.

Two examples illustrate the distinction between provider evidence and a search gap. In the Qwen sample, **TP-02** links to the model card and cites 8.2B parameters. In the GPT‑5.2 sample, **DT-03** cites the provider's public training-content summary. The GPT‑5.2 sample records parameter size as **Not found** for that run. [Qwen sample](samples/qwen3-8b.md) · [GPT‑5.2 sample](samples/gpt-5.2.md)

## Setup and CLI

### Open the application

Open a PowerShell terminal in the project folder containing `app.py`, `src/`, and `samples/`. Use the existing Python environment. The package requires Python 3.11 or later. The interface was verified with Streamlit 1.66.1.

If you are setting up a fresh clone and the dependencies are missing, run the following in the intended Python environment:

```powershell
python -m pip install -e . "streamlit==1.66.1"
```

Start the application:

```powershell
$env:PYTHONPATH = 'src'
python -m streamlit run app.py
```

If several environments are active, use the full path to the intended interpreter.

Open the **Local URL** printed in the terminal, normally `http://localhost:8501`. Leave the terminal running while you use the page. Press **Ctrl+C** in the terminal to stop the server.

### Configure live analysis

### Set up the local configuration

Create a local `.env` in the project root using [.env.example](.env.example) as a template. Fill in your own values in `.env` and keep that file out of Git. Run commands from the project root so relative paths resolve correctly. For overlapping settings, process environment variables can take precedence over the file.

| Variable | Purpose |
|---|---|
| `TAVILY_API_KEY` | Authenticates Tavily Search. Required for live runs. |
| `OPENAI_API_KEY` | Authenticates the assessment model and embedding requests. |
| `PINECONE_API_KEY` | Authenticates access to the Pinecone index. |
| `PINECONE_INDEX_NAME` | Name of an existing, ready index compatible with the Vectors API, dense vectors of dimension `1536`, and cosine similarity. |
| `PINECONE_NAMESPACE` | Namespace for the regulatory corpus. Defaults to `model-intake-brief-regulatory`. Use the same index and namespace when indexing and running analyses. |
| `PINECONE_API_VERSION` | Pinecone API version. The configured default is `2025-10`. |
| `INTAKE_PROVIDER_DOMAINS` | Comma-separated official host names for hosted research. Use host names without `https://` or paths. Hugging Face targets ignore it and always use `huggingface.co`. |
| `INTAKE_TAVILY_TIMEOUT_SECONDS`, `INTAKE_JINA_TIMEOUT_SECONDS`, `INTAKE_HF_TIMEOUT_SECONDS` | Request timeouts in seconds for the research services. Each defaults to `15`. |
| `INTAKE_ASSESSMENT_TIMEOUT_SECONDS` | OpenAI assessment request timeout in seconds. Defaults to `90`. |
| `INTAKE_RAG_TIMEOUT_SECONDS` | Regulatory retrieval timeout in seconds. Defaults to `15`. |

For the OpenAI hosted example, use the following official domains in the local configuration:

```dotenv
INTAKE_PROVIDER_DOMAINS=openai.com,platform.openai.com,developers.openai.com,help.openai.com,cdn.openai.com
```

The current Jina Reader and Hugging Face Hub adapters use public endpoints without separate credentials. Sources that require sign-in, access approval, or a private repository are outside the documented workflow.

### Load the regulatory corpus

If the selected index and namespace already contain the current project corpus, reuse them. For a new namespace or an updated corpus, use **index-regulatory** from the project root after configuring the keys and index:

```powershell
$env:PYTHONPATH = 'src'
python -m intake index-regulatory
```

This command sends the corpus for embedding and writes its vectors to the configured namespace. It reports the corpus version, index name, namespace, indexed count, and blockers. It requires an existing index and consumes embedding and storage resources. Indexing is a setup action. The **Run live analysis** button retrieves existing context during a live analysis.

## Use the command line

The CLI and web interface run the same workflow. In PowerShell, set the source path from the project root if the package is not installed in the active environment:

```powershell
$env:PYTHONPATH = 'src'
```

| Task | Command |
|---|---|
| Offline Hugging Face preview | `python -m intake run --model Qwen/Qwen3-8B` |
| Offline hosted preview | `python -m intake run --provider openai --model gpt-5.2-2025-12-11` |
| Live Hugging Face analysis | `python -m intake run --live --model Qwen/Qwen3-8B` |
| Live hosted analysis | `python -m intake run --live --provider openai --model gpt-5.2-2025-12-11` |
| Index regulatory material | `python -m intake index-regulatory` |

The CLI prints Markdown to standard output and sends error messages to standard error. Exit code `0` indicates that the CLI returned a report without a workflow error. Exit code `2` indicates an input or workflow error and can accompany a partial report. Exit code `1` means no report was returned and no more specific error was available.

For direct UTF-8 capture on Windows, run the following from the project root. Choose an unused filename because redirection overwrites a file with the same name. Review the exit code and report label before treating the file as a completed analysis. Omit `--live` for a test preview.

```powershell
cmd /d /c "python -X utf8 -m intake run --live --provider openai --model gpt-5.2-2025-12-11 > intake-run.md"
$LASTEXITCODE
```

## Troubleshoot

| What you see | What to check |
|---|---|
| The browser does not open automatically | Open the **Local URL** printed by Streamlit. Use the port printed in the terminal if it differs from `8501`. |
| `No module named streamlit` or `No module named intake` | Confirm the interpreter and dependencies. Run from the project root and set `$env:PYTHONPATH = 'src'` when using an uninstalled checkout. |
| A report shows **Offline demo — fixture data** or all items are **Not checked** | For a test run, this is expected. For live evidence, check the **Live run** setting and inspect any visible error and the report label. |
| The model ID is rejected | Use `ORG/MODEL` for Hugging Face. For a hosted model, enter both the provider and exact model ID. |
| A required key or hosted domain list is missing | Verify the variable name and value locally. Check that `.env` is in the project root and that the terminal does not have a conflicting process environment setting. |
| Pinecone context is empty or indexing reports a blocker | Check the key, index name, vector compatibility, readiness, and namespace. Confirm the corpus was indexed into the same namespace used for retrieval. |
| A service returns a timeout, rate limit, blocked URL, or extraction error | Inspect **Run details and data sources** and the detailed report. Check network access, service credits, host rules, and source accessibility as applicable. If you choose to run again, submit once after addressing the cause. |
| Assessment output is incomplete | Keep the partial report for diagnosis. The live assessment request has a 4,000-output-token cap. Review the reported stage, reason, and token counts before initiating another paid run. |
| A provider document exists but the brief does not cite it | Verify that the version matches and the document was discovered, allowed, and extracted. Record the gap in the human review. |
| A downloaded report is missing from **Saved reports** | The selector lists only the two bundled examples. Open the downloaded Markdown from your browser's downloads folder or from the destination you selected. |

For repository setup and implementation details, see the [README](README.md).
