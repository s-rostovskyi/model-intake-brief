from __future__ import annotations

from typing import Any, Callable, TypedDict
from urllib.parse import urlparse

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .adapters import hugging_face_model_metadata, jina_reader_extract, tavily_search_urls
from .assessment import (
    AssessmentFailure,
    build_offline_assessment,
    build_partial_assessment,
    call_openai_assessment,
    check_public_urls,
    normalize_assessment_output,
    render_markdown as render_assessment_markdown,
)
from .checklist import ChecklistItem, load_checklist
from .fixtures import FIXTURE_RECORD
from .regulatory import default_query_for_item, load_rag_config_from_env, retrieve_regulatory_context_batch
from .settings import load_live_settings


CHECKLIST_ITEM_BY_TARGET_KIND = {'hugging_face': 'DT-03', 'hosted': 'US-03'}
HF_MODEL_CARD_URL = 'https://huggingface.co/Qwen/Qwen3-8B'
GPT52_TRAINING_SUMMARY_URL = 'https://cdn.openai.com/pdf/gpt-5-2-eu-ai-act-public-summary-of-training-content.pdf'


class IntakeState(TypedDict, total=False):
    live_mode: bool
    provider: str | None
    model: str | None
    normalized_provider: str | None
    exact_model_id: str | None
    target_kind: str | None
    checklist_item_id: str | None
    checklist: list[ChecklistItem]
    provider_domains: list[str]
    model_provider: str | None
    hosting_platform: str | None
    draft_label: str | None
    tool_outcomes: list[Any]
    regulatory_contexts: dict[str, Any]
    assessment: dict[str, Any]
    report_markdown: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Live pipeline events
#
# Nodes and the tool calls inside them emit small dict events through the
# LangGraph custom stream. They are only delivered when the graph is run with
# stream_mode that includes "custom" (see intake.runner). With invoke(), the
# CLI, or direct calls the events are silently dropped, so behaviour is unchanged.
# ---------------------------------------------------------------------------

def _emit(event: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except Exception:  # called outside a running graph
        return
    try:
        writer(event)
    except Exception:
        pass


def _describe_outcome(result: Any) -> tuple[str, str]:
    """Return (status, detail) for a ToolOutcome-like object without trusting its shape."""
    http = getattr(result, 'http_status', None)
    usable = getattr(result, 'usable', None)
    error = getattr(result, 'error', None)
    urls = getattr(result, 'urls', None) or []
    parts: list[str] = []
    if http is not None:
        parts.append(f'HTTP {http}')
    if urls:
        parts.append(f'{len(urls)} URLs')
    if usable is False and error:
        parts.append(str(error)[:80])
    elif usable:
        parts.append('usable')
    return ('done' if usable is not False else 'warn'), ' · '.join(parts)


def _step(node: str, label: str, fn: Callable[..., Any], *args: Any,
          describe: Callable[[Any], tuple[str, str]] = _describe_outcome, **kwargs: Any) -> Any:
    _emit({'type': 'step', 'node': node, 'label': label, 'status': 'start'})
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        _emit({'type': 'step', 'node': node, 'label': label, 'status': 'error', 'detail': exc.__class__.__name__})
        raise
    try:
        status, detail = describe(result)
    except Exception:
        status, detail = 'done', ''
    _emit({'type': 'step', 'node': node, 'label': label, 'status': status, 'detail': detail})
    return result


def _note_step(node: str, label: str, detail: str = '', status: str = 'done') -> None:
    _emit({'type': 'step', 'node': node, 'label': label, 'status': status, 'detail': detail})


def _describe_retrieval(result: Any) -> tuple[str, str]:
    hits = sum(len(getattr(item, 'hits', None) or []) for item in result.values())
    blocked = sum(1 for item in result.values() if getattr(item, 'blockers', None))
    detail = f'{hits} passages'
    if blocked:
        detail += f' · {blocked} with blockers'
    return ('warn' if blocked and not hits else 'done'), detail


def _describe_assessment_call(result: Any) -> tuple[str, str]:
    usage = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    if isinstance(usage, dict):
        tokens_in = usage.get('input_tokens') or usage.get('prompt_tokens')
        tokens_out = usage.get('output_tokens') or usage.get('completion_tokens')
        if tokens_in is not None or tokens_out is not None:
            return 'done', f'{tokens_in or 0:,} in / {tokens_out or 0:,} out tokens'
    return 'done', ''


def _describe_verification(result: Any) -> tuple[str, str]:
    findings = (result or {}).get('findings') or []
    found = sum(1 for item in findings if item.get('status') == 'Evidence found')
    return 'done', f'{found} of {len(findings)} items supported'


def _describe_link_checks(result: Any) -> tuple[str, str]:
    checks = list((result or {}).values())
    ok = sum(1 for item in checks if isinstance(item, dict) and item.get('outcome') == 'ok')
    return 'done', f'{ok} of {len(checks)} reachable'


def _url_label(url: str) -> str:
    parsed = urlparse(url)
    tail = parsed.path.rsplit('/', 1)[-1] or parsed.path
    return f'{parsed.hostname or url}/{tail}'[:44]


def _checklist_item_id(target_kind: str | None) -> str | None:
    return CHECKLIST_ITEM_BY_TARGET_KIND.get(target_kind or '')


def validate_and_normalize_target(state: IntakeState) -> IntakeState:
    provider = (state.get('provider') or '').strip()
    model = (state.get('model') or '').strip()
    if not model:
        return {'error': 'Missing --model.'}
    if provider:
        _note_step('validate_and_normalize_target', 'Resolve input format', 'hosted provider + model ID')
        return {
            'provider': provider,
            'normalized_provider': provider.lower(),
            'exact_model_id': model,
            'target_kind': 'hosted',
            'checklist_item_id': _checklist_item_id('hosted'),
        }
    parts = model.split('/')
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return {'error': 'For Hugging Face targets, use --model ORG/MODEL without --provider.'}
    _note_step('validate_and_normalize_target', 'Resolve input format', 'Hugging Face ORG/MODEL')
    return {
        'provider': 'Hugging Face',
        'normalized_provider': 'huggingface',
        'exact_model_id': model,
        'target_kind': 'hugging_face',
        'checklist_item_id': _checklist_item_id('hugging_face'),
    }


def load_checklist_node(state: IntakeState) -> IntakeState:
    if state.get('error') and not state.get('assessment'):
        return {}
    checklist = load_checklist()
    _note_step('load_checklist', 'Parse model-intake-checklist.md', f'{len(checklist)} items')
    return {'checklist': checklist}


def _offline_fixture_outcome(model_id: str, provider: str | None, target_kind: str | None):
    from .adapters import ToolOutcome

    return ToolOutcome(
        name='Offline fixture',
        requested_url=None,
        http_status=None,
        usable=True,
        urls=[],
        snippet=f'Deterministic fixture for {model_id}.',
        metadata={
            'id': FIXTURE_RECORD['id'],
            'label': FIXTURE_RECORD['label'],
            'status': FIXTURE_RECORD['status'],
            'provider': provider,
            'target_kind': target_kind,
        },
    )


def _model_provider_from_hf_metadata(tool_outcomes: list[Any]) -> str | None:
    for outcome in tool_outcomes:
        if getattr(outcome, 'name', None) == 'Hugging Face Hub':
            metadata = getattr(outcome, 'metadata', {}) or {}
            author = metadata.get('author')
            if isinstance(author, str) and author.strip():
                return author.strip()
    return None


def _collect_hf_sources(exact_model_id: str, live_settings) -> list[Any]:
    outcomes = []
    primary_search = _step(
        'collect_sources', 'Tavily search · documentation', tavily_search_urls,
        query=f'{exact_model_id} official documentation',
        api_key=live_settings.tavily_api_key,
        timeout_seconds=live_settings.tavily_timeout_seconds,
        include_domains=['huggingface.co'],
    )
    outcomes.append(primary_search)
    follow_up_search = _step(
        'collect_sources', 'Tavily search · architecture, data, licence', tavily_search_urls,
        query=f'{exact_model_id} architecture parameter size training data licence',
        api_key=live_settings.tavily_api_key,
        timeout_seconds=live_settings.tavily_timeout_seconds,
        include_domains=['huggingface.co'],
    )
    outcomes.append(follow_up_search)
    outcomes.append(_step('collect_sources', 'Hugging Face Hub · model metadata', hugging_face_model_metadata,
                          exact_model_id, timeout_seconds=live_settings.hf_timeout_seconds))

    candidate_urls: list[str] = [HF_MODEL_CARD_URL]
    for outcome in (primary_search, follow_up_search):
        for url in outcome.urls:
            if url not in candidate_urls:
                candidate_urls.append(url)

    for url in candidate_urls[:3]:
        reader_outcome = _step('collect_sources', f'Jina Reader · {_url_label(url)}', jina_reader_extract,
                               url, timeout_seconds=live_settings.jina_timeout_seconds)
        outcomes.append(reader_outcome)
    return outcomes


def _collect_hosted_sources(provider: str | None, exact_model_id: str, live_settings) -> list[Any]:
    outcomes = []
    primary_search = _step(
        'collect_sources', 'Tavily search · documentation', tavily_search_urls,
        query=f'{provider} {exact_model_id} official documentation',
        api_key=live_settings.tavily_api_key,
        timeout_seconds=live_settings.tavily_timeout_seconds,
        include_domains=live_settings.provider_domains,
    )
    outcomes.append(primary_search)
    follow_up_search = _step(
        'collect_sources', 'Tavily search · limits, access, terms', tavily_search_urls,
        query=f'{provider} {exact_model_id} model documentation API limits access terms',
        api_key=live_settings.tavily_api_key,
        timeout_seconds=live_settings.tavily_timeout_seconds,
        include_domains=live_settings.provider_domains,
    )
    outcomes.append(follow_up_search)

    candidate_urls: list[str] = []
    if (provider or '').strip().lower() == 'openai' and exact_model_id == 'gpt-5.2-2025-12-11':
        candidate_urls.append(GPT52_TRAINING_SUMMARY_URL)
    for outcome in (primary_search, follow_up_search):
        for url in outcome.urls:
            if url not in candidate_urls:
                candidate_urls.append(url)
    for url in candidate_urls[:3]:
        reader_outcome = _step(
            'collect_sources', f'Jina Reader · {_url_label(url)}', jina_reader_extract,
            url,
            timeout_seconds=live_settings.jina_timeout_seconds,
            allowed_domains=live_settings.provider_domains,
        )
        outcomes.append(reader_outcome)
    return outcomes


def collect_sources(state: IntakeState) -> IntakeState:
    if state.get('error'):
        return {}
    exact_model_id = state['exact_model_id']
    provider = state.get('provider')
    target_kind = state.get('target_kind')
    if not state.get('live_mode'):
        _note_step('collect_sources', 'Offline fixture · no network calls', 'deterministic data')
        return {
            'draft_label': 'Offline demo — fixture data',
            'model_provider': 'unknown',
            'hosting_platform': 'offline',
            'tool_outcomes': [_offline_fixture_outcome(exact_model_id, provider, target_kind)],
        }
    try:
        live_settings = load_live_settings()
    except ValueError as exc:
        return {'error': str(exc)}

    if target_kind == 'hugging_face':
        tool_outcomes = _collect_hf_sources(exact_model_id, live_settings)
        provider_domains = live_settings.provider_domains or ['huggingface.co']
        model_provider = _model_provider_from_hf_metadata(tool_outcomes) or 'unknown'
        hosting_platform = 'Hugging Face'
    else:
        if not live_settings.provider_domains:
            return {'error': 'Missing INTAKE_PROVIDER_DOMAINS for live hosted mode.'}
        tool_outcomes = _collect_hosted_sources(provider, exact_model_id, live_settings)
        provider_domains = live_settings.provider_domains
        model_provider = provider or 'unknown'
        hosting_platform = provider or 'unknown'

    return {
        'draft_label': 'Full report',
        'provider_domains': provider_domains,
        'model_provider': model_provider,
        'hosting_platform': hosting_platform,
        'tool_outcomes': tool_outcomes,
    }


def retrieve_regulatory_contexts(state: IntakeState) -> IntakeState:
    if state.get('error'):
        return {}
    if not state.get('live_mode'):
        _note_step('retrieve_regulatory_contexts', 'Pinecone retrieval skipped', 'offline mode', status='skip')
        return {'regulatory_contexts': {}}
    checklist = state.get('checklist') or []
    config = load_rag_config_from_env()
    item_ids = [item.item_id for item in checklist]
    retrievals = _step(
        'retrieve_regulatory_contexts', f'Embeddings + Pinecone · {len(item_ids)} queries',
        retrieve_regulatory_context_batch, item_ids, config, timeout_seconds=15.0, top_k=3,
        describe=_describe_retrieval,
    )
    contexts: dict[str, Any] = {}
    for item_id, result in retrievals.items():
        contexts[str(item_id)] = {
            'checklist_item_id': result.item_id,
            'query': result.query,
            'hits': [
                {
                    'chunk_id': hit.chunk_id,
                    'provision': hit.provision,
                    'official_url': hit.official_url,
                    'score': hit.score,
                    'document_version': hit.document_version,
                    'excerpt': hit.excerpt,
                }
                for hit in result.hits
            ],
            'blockers': list(result.blockers),
        }
    return {'regulatory_contexts': contexts}


def _source_payloads(tool_outcomes: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for outcome in tool_outcomes:
        if hasattr(outcome, 'as_dict'):
            payload = outcome.as_dict()
            payload['full_text'] = getattr(outcome, 'full_text', None)
            payloads.append(payload)
    return payloads


def assess_evidence(state: IntakeState) -> IntakeState:
    if state.get('error'):
        return {}
    checklist = state.get('checklist') or []
    tool_outcomes = state.get('tool_outcomes') or []
    if not state.get('live_mode'):
        _note_step('assess_evidence', 'Fixture assessment · no LLM call', 'status: Not checked')
        return {'assessment': build_offline_assessment(checklist)}

    live_settings = load_live_settings()
    source_payloads = _source_payloads(tool_outcomes)
    if not live_settings.tavily_api_key:
        return {'error': 'Missing TAVILY_API_KEY for live mode.'}

    if not state.get('exact_model_id'):
        return {'error': 'Missing exact model ID.'}

    config = load_rag_config_from_env()
    if not config.get('OPENAI_API_KEY'):
        return {'error': 'Missing OPENAI_API_KEY for live assessment.'}

    try:
        raw_output, usage = _step(
            'assess_evidence', 'OpenAI structured assessment', call_openai_assessment,
            config['OPENAI_API_KEY'],
            checklist,
            {
                'provider': state.get('provider'),
                'normalized_provider': state.get('normalized_provider'),
                'model_provider': state.get('model_provider'),
                'hosting_platform': state.get('hosting_platform'),
                'exact_model_id': state.get('exact_model_id'),
                'target_kind': state.get('target_kind'),
            },
            source_payloads,
            timeout_seconds=live_settings.assessment_timeout_seconds,
            describe=_describe_assessment_call,
        )
    except AssessmentFailure as exc:
        usage_text = None
        if exc.usage:
            prompt_tokens = exc.usage.get('input_tokens') or exc.usage.get('prompt_tokens')
            completion_tokens = exc.usage.get('output_tokens') or exc.usage.get('completion_tokens')
            if prompt_tokens is not None or completion_tokens is not None:
                usage_text = f'input tokens {prompt_tokens if prompt_tokens is not None else "n/a"}, output tokens {completion_tokens if completion_tokens is not None else "n/a"}'
        error = f'Assessment request failed at {exc.stage} stage: {exc.reason}'
        if usage_text:
            error = f'{error} Usage: {usage_text}.'
        elif not error.endswith('.'):
            error = f'{error}.'
        return {
            'assessment': build_partial_assessment(checklist, exc.reason, failure_stage=exc.stage, usage=exc.usage),
            'error': error,
        }
    except Exception as exc:
        return {'error': f'Assessment request failed: {exc.__class__.__name__}.'}

    preliminary = _step(
        'assess_evidence', 'Verify quotes against sources', normalize_assessment_output,
        raw_output,
        checklist,
        source_payloads,
        state['exact_model_id'],
        regulatory_context=state.get('regulatory_contexts') or {},
        describe=_describe_verification,
    )
    candidate_urls: list[str] = []
    for finding in preliminary.get('findings', []):
        for key in ('source_url', 'link_check_url'):
            url = finding.get(key)
            if isinstance(url, str) and url and url not in candidate_urls:
                candidate_urls.append(url)
    link_checks = _step(
        'assess_evidence', f'Link checks · {min(len(candidate_urls), 6)} URLs', check_public_urls,
        candidate_urls,
        timeout_seconds=15.0,
        max_checks=6,
        allowed_domains=state.get('provider_domains') or [],
        describe=_describe_link_checks,
    )
    assessment = normalize_assessment_output(
        raw_output,
        checklist,
        source_payloads,
        state['exact_model_id'],
        regulatory_context=state.get('regulatory_contexts') or {},
        link_checks=link_checks,
    )
    assessment['request_usage'] = usage
    assessment['aggregate_usage'] = usage
    assessment['usage'] = usage
    assessment['request_output_token_cap'] = 4000
    return {'assessment': assessment}


def render_report(state: IntakeState) -> IntakeState:
    if state.get('error') and not state.get('assessment'):
        return {}
    checklist = state.get('checklist') or []
    assessment = state.get('assessment') or {}
    markdown = _step(
        'render_report', 'Assemble Markdown brief', render_assessment_markdown,
        {
            'live_mode': state.get('live_mode'),
            'exact_model_id': state.get('exact_model_id'),
            'provider': state.get('provider'),
            'model_provider': state.get('model_provider'),
            'hosting_platform': state.get('hosting_platform'),
            'target_kind': state.get('target_kind'),
        },
        checklist,
        _source_payloads(state.get('tool_outcomes') or []),
        assessment,
        state.get('regulatory_contexts') or {},
        describe=lambda text: ('done', f'{len(text):,} characters'),
    )
    return {'report_markdown': markdown}


# ---------------------------------------------------------------------------
# Node summaries for the live pipeline view. They describe what a node passed on
# in the shared state (counts and labels only, never raw source text or secrets).
# ---------------------------------------------------------------------------

def _summarize(name: str, before: dict[str, Any], delta: dict[str, Any]) -> tuple[str, list[list[str]]]:
    merged = {**before, **delta}
    if delta.get('error') and name != 'assess_evidence':
        return 'error', [['error', str(delta['error'])[:120]]]
    if name == 'validate_and_normalize_target':
        return 'done', [['model', str(merged.get('exact_model_id'))], ['route', str(merged.get('target_kind'))]]
    if name == 'load_checklist':
        return 'done', [['checklist', f"{len(merged.get('checklist') or [])} items"]]
    if name == 'collect_sources':
        outcomes = merged.get('tool_outcomes') or []
        usable = sum(1 for item in outcomes if getattr(item, 'usable', False))
        return 'done', [['sources', f'{usable} of {len(outcomes)} usable'], ['label', str(merged.get('draft_label'))]]
    if name == 'retrieve_regulatory_contexts':
        contexts = merged.get('regulatory_contexts') or {}
        hits = sum(len(item.get('hits') or []) for item in contexts.values())
        return 'done', [['regulatory', f'{hits} passages' if contexts else 'none (offline)']]
    if name == 'assess_evidence':
        assessment = merged.get('assessment') or {}
        counts = (assessment.get('summary') or {}).get('status_counts') or {}
        data = [[status, str(count)] for status, count in counts.items() if count]
        usage = assessment.get('usage')
        if isinstance(usage, dict):
            data.append(['tokens', f"{usage.get('input_tokens') or usage.get('prompt_tokens') or 0:,} in"
                                   f" / {usage.get('output_tokens') or usage.get('completion_tokens') or 0:,} out"])
        return ('error' if delta.get('error') else 'done'), data or [['findings', 'none']]
    if name == 'render_report':
        report = merged.get('report_markdown')
        return ('done', [['report', f'{len(report):,} chars']]) if report else ('error', [['report', 'not produced']])
    return 'done', []


def _traced(name: str, fn: Callable[[IntakeState], IntakeState]) -> Callable[[IntakeState], IntakeState]:
    def node(state):
        _emit({'type': 'node_started', 'node': name})
        try:
            delta = fn(state) or {}
        except Exception as exc:
            _emit({'type': 'node_finished', 'node': name, 'status': 'error',
                   'data': [['error', exc.__class__.__name__]]})
            raise
        status, data = _summarize(name, state, delta)
        _emit({'type': 'node_finished', 'node': name, 'status': status, 'data': data})
        return delta

    node.__name__ = name
    return node


def _route_on_error(next_node: str) -> Callable[[IntakeState], str]:
    def route(state: IntakeState) -> str:
        return 'fail' if state.get('error') else 'continue'

    route.__name__ = f'route_to_{next_node}'
    return route


NODE_FUNCTIONS = {
    'validate_and_normalize_target': validate_and_normalize_target,
    'load_checklist': load_checklist_node,
    'collect_sources': collect_sources,
    'retrieve_regulatory_contexts': retrieve_regulatory_contexts,
    'assess_evidence': assess_evidence,
    'render_report': render_report,
}


def build_graph():
    graph = StateGraph(IntakeState)
    for name, fn in NODE_FUNCTIONS.items():
        graph.add_node(name, _traced(name, fn))
    graph.add_edge(START, 'validate_and_normalize_target')
    # A failure in the first two stages ends the run early: go straight to the renderer,
    # which returns the error without a report. Previously every later node skipped itself.
    graph.add_conditional_edges('validate_and_normalize_target', _route_on_error('load_checklist'),
                                {'continue': 'load_checklist', 'fail': 'render_report'})
    graph.add_edge('load_checklist', 'collect_sources')
    graph.add_conditional_edges('collect_sources', _route_on_error('retrieve_regulatory_contexts'),
                                {'continue': 'retrieve_regulatory_contexts', 'fail': 'render_report'})
    graph.add_edge('retrieve_regulatory_contexts', 'assess_evidence')
    graph.add_edge('assess_evidence', 'render_report')
    graph.add_edge('render_report', END)
    return graph.compile()


APP = build_graph()
