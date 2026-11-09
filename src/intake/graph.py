from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .adapters import hugging_face_model_metadata, jina_reader_extract, tavily_search_urls
from .fixtures import FIXTURE_RECORD
from .regulatory import default_query_for_item, load_rag_config_from_env, retrieve_regulatory_context
from .settings import load_live_settings


CHECKLIST_ITEM_BY_TARGET_KIND = {'hugging_face': 'DT-03', 'hosted': 'US-03'}


class IntakeState(TypedDict, total=False):
    live_mode: bool
    provider: str | None
    model: str | None
    normalized_provider: str | None
    exact_model_id: str | None
    target_kind: str | None
    checklist_item_id: str | None
    provider_domains: list[str]
    draft_label: str | None
    tool_outcomes: list[dict[str, Any]]
    regulatory_context: dict[str, Any]
    report_markdown: str | None
    error: str | None


def _checklist_item_id(target_kind: str | None) -> str | None:
    return CHECKLIST_ITEM_BY_TARGET_KIND.get(target_kind or '')


def validate_and_normalize_target(state: IntakeState) -> IntakeState:
    provider = (state.get('provider') or '').strip()
    model = (state.get('model') or '').strip()
    if not model:
        return {'error': 'Missing --model.'}
    if provider:
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
    return {
        'provider': 'Hugging Face',
        'normalized_provider': 'huggingface',
        'exact_model_id': model,
        'target_kind': 'hugging_face',
        'checklist_item_id': _checklist_item_id('hugging_face'),
    }


def _offline_fixture_outcome(model_id: str, provider: str | None, target_kind: str | None) -> dict[str, Any]:
    return {
        'name': 'Offline fixture',
        'requested_url': None,
        'http_status': None,
        'usable': True,
        'urls': [],
        'snippet': f'Deterministic fixture for {model_id}.',
        'error': None,
        'metadata': {
            'id': FIXTURE_RECORD['id'],
            'label': FIXTURE_RECORD['label'],
            'status': FIXTURE_RECORD['status'],
            'provider': provider,
            'target_kind': target_kind,
        },
    }


def _build_regulatory_context(checklist_item_id: str | None, regulatory_result) -> dict[str, Any]:
    return {
        'checklist_item_id': checklist_item_id,
        'query': regulatory_result.query,
        'hits': [
            {
                'chunk_id': hit.chunk_id,
                'provision': hit.provision,
                'official_url': hit.official_url,
                'score': hit.score,
                'document_version': hit.document_version,
                'excerpt': hit.excerpt,
            }
            for hit in regulatory_result.hits
        ],
        'blockers': list(regulatory_result.blockers),
    }


def collect_sources(state: IntakeState) -> IntakeState:
    if state.get('error'):
        return {}
    exact_model_id = state['exact_model_id']
    provider = state.get('provider')
    target_kind = state.get('target_kind')
    checklist_item_id = state.get('checklist_item_id')
    if not state.get('live_mode'):
        return {
            'draft_label': 'Offline demo ? fixture data',
            'tool_outcomes': [_offline_fixture_outcome(exact_model_id, provider, target_kind)],
        }
    try:
        live_settings = load_live_settings()
    except ValueError as exc:
        return {'error': str(exc)}
    tool_outcomes: list[dict[str, Any]] = []
    provider_domains: list[str] = []
    if target_kind == 'hugging_face':
        search_outcome = tavily_search_urls(query=f'{exact_model_id} official documentation', api_key=live_settings.tavily_api_key, timeout_seconds=live_settings.tavily_timeout_seconds)
        tool_outcomes.append(search_outcome.as_dict())
        if search_outcome.usable and search_outcome.urls:
            reader_outcome = jina_reader_extract(search_outcome.urls[0], timeout_seconds=live_settings.jina_timeout_seconds)
            tool_outcomes.append(reader_outcome.as_dict())
        tool_outcomes.append(hugging_face_model_metadata(exact_model_id, timeout_seconds=live_settings.hf_timeout_seconds).as_dict())
        provider_domains = live_settings.provider_domains
    else:
        if not live_settings.provider_domains:
            return {'error': 'Missing INTAKE_PROVIDER_DOMAINS for live hosted mode.'}
        search_outcome = tavily_search_urls(query=f'{provider} {exact_model_id} official documentation', api_key=live_settings.tavily_api_key, timeout_seconds=live_settings.tavily_timeout_seconds, include_domains=live_settings.provider_domains)
        tool_outcomes.append(search_outcome.as_dict())
        if search_outcome.usable and search_outcome.urls:
            reader_outcome = jina_reader_extract(search_outcome.urls[0], timeout_seconds=live_settings.jina_timeout_seconds, allowed_domains=live_settings.provider_domains)
            tool_outcomes.append(reader_outcome.as_dict())
        provider_domains = live_settings.provider_domains
    regulatory_result = retrieve_regulatory_context(default_query_for_item(checklist_item_id), load_rag_config_from_env(), item_id=checklist_item_id, timeout_seconds=live_settings.rag_timeout_seconds)
    return {
        'draft_label': 'Integration draft',
        'provider_domains': provider_domains,
        'tool_outcomes': tool_outcomes,
        'regulatory_context': _build_regulatory_context(checklist_item_id, regulatory_result),
    }


def _format_outcome(outcome: dict[str, Any]) -> list[str]:
    lines = []
    name = outcome.get('name', 'Tool')
    http_status = outcome.get('http_status')
    usable = 'yes' if outcome.get('usable') else 'no'
    lines.append(f"- {name}: HTTP {http_status if http_status is not None else 'n/a'}; usable data: {usable}")
    requested_url = outcome.get('requested_url')
    if requested_url and name != 'Tavily Search':
        lines.append(f'  - Source URL: {requested_url}')
    error = outcome.get('error')
    if error:
        lines.append(f'  - Error: {error}')
    urls = outcome.get('urls') or []
    if urls:
        lines.append('  - Source URLs:')
        for url in urls:
            lines.append(f'    - {url}')
    snippet = outcome.get('snippet')
    if snippet:
        if name == 'Tavily Search':
            lines.append(f'  - Search snippet (candidate material only): {snippet}')
        elif name == 'Jina Reader':
            lines.append(f'  - Extracted text: {snippet}')
    metadata = outcome.get('metadata') or {}
    if metadata:
        visible = {key: value for key, value in metadata.items() if value not in (None, '', [], {})}
        if visible:
            lines.append('  - Metadata:')
            for key, value in visible.items():
                lines.append(f'    - {key}: {value}')
    return lines


def _format_regulatory_context(context: dict[str, Any]) -> list[str]:
    lines = [f"- Checklist item: {context.get('checklist_item_id') or 'n/a'}", f"- Retrieval query: {context.get('query') or 'n/a'}"]
    blockers = context.get('blockers') or []
    if blockers:
        lines.append('- Retrieval blockers:')
        for blocker in blockers:
            lines.append(f'  - {blocker}')
    hits = context.get('hits') or []
    if hits:
        lines.append('- Retrieved references:')
        for hit in hits:
            lines.append(f"  - {hit.get('provision', 'unknown')} ({hit.get('chunk_id', 'unknown')})")
            lines.append(f"    - Official URL: {hit.get('official_url') or 'n/a'}")
            lines.append(f"    - Document version: {hit.get('document_version') or 'n/a'}")
            lines.append(f"    - Match score: {hit.get('score')}")
            excerpt = hit.get('excerpt')
            if excerpt:
                lines.append(f'    - Excerpt: {excerpt}')
    else:
        lines.append('- No regulatory references were returned.')
    return lines


def render_markdown(state: IntakeState) -> IntakeState:
    if state.get('error'):
        return {}
    exact_model_id = state['exact_model_id']
    provider = state['provider'] or ''
    target_kind = state['target_kind'] or ''
    checklist_item_id = state.get('checklist_item_id') or 'n/a'
    provider_domains = state.get('provider_domains') or []
    draft_label = state.get('draft_label') or 'Integration draft'
    tool_outcomes = state.get('tool_outcomes') or []
    regulatory_context = state.get('regulatory_context')
    lines = [
        '# Model Intake Brief',
        '',
        f'## {draft_label}',
        '',
        f'- Exact model ID: `{exact_model_id}`',
        f'- Provider: {provider}',
        '- Notice: Information, not legal advice.',
        f'- Target kind: {target_kind}',
        f'- Checklist item: {checklist_item_id}',
    ]
    if provider_domains:
        lines.append(f"- Configured provider domains: {', '.join(provider_domains)}")
    lines.extend(['', '### Per-tool outcomes'])
    for outcome in tool_outcomes:
        lines.extend(_format_outcome(outcome))
    if not tool_outcomes:
        lines.append('- No tools were run.')
    if regulatory_context:
        lines.extend(['', '### Regulatory context'])
        lines.extend(_format_regulatory_context(regulatory_context))
    return {'report_markdown': '\n'.join(lines)}


def build_graph():
    graph = StateGraph(IntakeState)
    graph.add_node('validate_and_normalize_target', validate_and_normalize_target)
    graph.add_node('collect_sources', collect_sources)
    graph.add_node('render_markdown', render_markdown)
    graph.add_edge(START, 'validate_and_normalize_target')
    graph.add_edge('validate_and_normalize_target', 'collect_sources')
    graph.add_edge('collect_sources', 'render_markdown')
    graph.add_edge('render_markdown', END)
    return graph.compile()


APP = build_graph()
