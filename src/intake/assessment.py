from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from typing import Any
from urllib.parse import urlparse

import requests

from .checklist import ChecklistItem
from .fixtures import FIXTURE_RECORD
from .settings import LiveSettings


ASSESSMENT_MODEL = 'gpt-5.4-mini'
ASSESSMENT_TOKENS = 4000
USER_AGENT = 'model-intake-brief/1.0'
ALLOWED_STATUSES = ['Evidence found', 'Partial', 'Not found', 'Not checked']


class AssessmentFailure(RuntimeError):
    def __init__(self, stage: str, reason: str, usage: dict[str, Any] | None = None, status_code: int | None = None):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.usage = usage or {}
        self.status_code = status_code


def _truncate(text: str, limit: int = 1200) -> str:
    collapsed = ' '.join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + '?'


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip().lower()


def _sanitize_failure_reason(reason: Any, limit: int = 240) -> str:
    cleaned = re.sub(r'[\r\n]+', ' ', str(reason or '')).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if not cleaned:
        return 'No additional detail provided.'
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + '?'


def _format_usage(usage: dict[str, Any] | None) -> str | None:
    if not usage:
        return None
    prompt_tokens = usage.get('input_tokens') or usage.get('prompt_tokens')
    completion_tokens = usage.get('output_tokens') or usage.get('completion_tokens')
    if prompt_tokens is None and completion_tokens is None:
        return None
    return (
        f'input tokens {prompt_tokens if prompt_tokens is not None else "n/a"}, '
        f'output tokens {completion_tokens if completion_tokens is not None else "n/a"}'
    )


def _find_text(source: dict[str, Any]) -> str:
    return str(source.get('full_text') or source.get('snippet') or '')


def _family_tokens(model_id: str) -> set[str]:
    tail = model_id.split('/')[-1].lower()
    tokens = {model_id.lower(), tail}
    if tail.startswith('gpt-'):
        match = re.match(r'^(gpt-\d+\.\d+)', tail)
        if match:
            tokens.add(match.group(1))
    elif tail.startswith('qwen'):
        match = re.match(r'^(qwen\d+)', tail)
        if match:
            tokens.add(match.group(1))
    elif '-' in tail:
        tokens.add(tail.rsplit('-', 1)[0])
    return tokens


def _conflicting_variant_tokens(exact_model_id: str) -> list[str]:
    tail = exact_model_id.split('/')[-1].lower()
    if tail.startswith('qwen3'):
        return ['qwen2.5', 'qwen 2.5', 'qwen2', 'qwen 2']
    if tail.startswith('qwen2.5') or tail.startswith('qwen2'):
        return ['qwen3', 'qwen 3']
    if tail.startswith('gpt-5.2'):
        return ['gpt-5.1', 'gpt 5.1', 'gpt-5.0', 'gpt 5.0']
    if tail.startswith('gpt-5.1') or tail.startswith('gpt-5.0'):
        return ['gpt-5.2', 'gpt 5.2']
    return []


def _mentions_conflicting_variant(text: str, exact_model_id: str) -> bool:
    normalized_text = _normalize_text(text)
    return any(token in normalized_text for token in _conflicting_variant_tokens(exact_model_id))


def quote_applies_to_target(quote: str, exact_model_id: str, model_applicability: str | None = None) -> bool:
    if isinstance(model_applicability, str) and model_applicability.strip():
        normalized_applicability = _normalize_text(model_applicability)
        if not _mentions_conflicting_variant(normalized_applicability, exact_model_id):
            return True
    normalized_quote = _normalize_text(quote)
    if _mentions_conflicting_variant(normalized_quote, exact_model_id):
        return False
    tokens = _family_tokens(exact_model_id)
    return any(token in normalized_quote for token in tokens)


def quote_supported(quote: str, source_text: str) -> bool:
    if not quote or not source_text:
        return False
    return _normalize_text(quote) in _normalize_text(source_text)


def check_public_url(url: str, timeout_seconds: float) -> dict[str, Any]:
    outcome: dict[str, Any] = {
        'url': url,
        'status_code': None,
        'final_url': None,
        'outcome': 'not checked',
        'error_category': None,
        'detail': None,
    }
    try:
        response = requests.get(url, timeout=timeout_seconds, allow_redirects=True, headers={'User-Agent': USER_AGENT}, stream=True)
    except requests.exceptions.ConnectTimeout:
        outcome['outcome'] = 'timeout'
        outcome['error_category'] = 'timeout'
        outcome['detail'] = 'Connect timeout.'
        return outcome
    except requests.exceptions.ReadTimeout:
        outcome['outcome'] = 'timeout'
        outcome['error_category'] = 'timeout'
        outcome['detail'] = 'Read timeout.'
        return outcome
    except requests.exceptions.SSLError:
        outcome['outcome'] = 'tls'
        outcome['error_category'] = 'tls'
        outcome['detail'] = 'TLS handshake failed.'
        return outcome
    except requests.exceptions.ProxyError:
        outcome['outcome'] = 'proxy'
        outcome['error_category'] = 'proxy'
        outcome['detail'] = 'Proxy error.'
        return outcome
    except requests.exceptions.ConnectionError as exc:
        message = str(exc).lower()
        if 'name or service not known' in message or 'temporary failure in name resolution' in message or 'gaierror' in message:
            category = 'dns'
        elif 'connection refused' in message:
            category = 'connection refused'
        else:
            category = 'connection'
        outcome['outcome'] = category
        outcome['error_category'] = category
        outcome['detail'] = exc.__class__.__name__
        return outcome
    except requests.RequestException as exc:
        outcome['outcome'] = 'request error'
        outcome['error_category'] = 'request'
        outcome['detail'] = exc.__class__.__name__
        return outcome

    outcome['status_code'] = response.status_code
    outcome['final_url'] = response.url
    if 200 <= response.status_code < 400:
        outcome['outcome'] = 'ok'
    elif response.status_code in {401, 403}:
        outcome['outcome'] = 'blocked'
    else:
        outcome['outcome'] = f'http {response.status_code}'
    return outcome


def check_public_urls(urls: list[str], timeout_seconds: float, max_checks: int = 6) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for url in urls:
        if url in results:
            continue
        if len(results) >= max_checks:
            break
        results[url] = check_public_url(url, timeout_seconds)
    return results


def _extract_response_text(payload: dict[str, Any]) -> str:
    direct_text = payload.get('output_text')
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()
    outputs = payload.get('output')
    pieces: list[str] = []
    if isinstance(outputs, list):
        for item in outputs:
            if not isinstance(item, dict):
                continue
            content = item.get('content')
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                text = chunk.get('text')
                if isinstance(text, str) and text.strip():
                    pieces.append(text)
    text = ''.join(pieces).strip()
    if not text:
        raise ValueError('OpenAI Responses payload did not contain structured text output.')
    return text


def _response_has_refusal(payload: dict[str, Any]) -> bool:
    outputs = payload.get('output')
    if not isinstance(outputs, list):
        return False
    for item in outputs:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'refusal' or item.get('refusal'):
            return True
        content = item.get('content')
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            if chunk.get('type') == 'refusal' or chunk.get('refusal'):
                return True
    return False


def build_assessment_schema(checklist_items: list[ChecklistItem]) -> dict[str, Any]:
    item_ids = [item.item_id for item in checklist_items]
    findings_schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'item_id': {'type': 'string', 'enum': item_ids},
            'status': {'type': 'string', 'enum': ALLOWED_STATUSES},
            'summary': {'type': 'string'},
            'quote': {'type': ['string', 'null']},
            'source_name': {'type': ['string', 'null']},
            'source_url': {'type': ['string', 'null']},
            'source_section': {'type': ['string', 'null']},
            'model_applicability': {'type': ['string', 'null']},
            'link_check_url': {'type': ['string', 'null']},
            'searched_sources': {
                'type': 'array',
                'items': {'type': 'string'},
            },
            'failure_reason': {'type': ['string', 'null']},
            'regulatory_reference': {'type': ['string', 'null']},
            'provider_question': {'type': ['string', 'null']},
        },
        'required': [
            'item_id',
            'status',
            'summary',
            'quote',
            'source_name',
            'source_url',
            'source_section',
            'model_applicability',
            'link_check_url',
            'searched_sources',
            'failure_reason',
            'regulatory_reference',
            'provider_question',
        ],
    }
    return {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'summary': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'headline': {'type': 'string'},
                    'major_gaps': {'type': 'array', 'items': {'type': 'string'}},
                    'status_counts': {
                        'type': 'object',
                        'additionalProperties': False,
                        'properties': {status: {'type': 'integer'} for status in ALLOWED_STATUSES},
                        'required': ALLOWED_STATUSES,
                    },
                },
                'required': ['headline', 'major_gaps', 'status_counts'],
            },
            'findings': {
                'type': 'array',
                'minItems': len(item_ids),
                'maxItems': len(item_ids),
                'items': findings_schema,
            },
            'prioritized_provider_questions': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'priority': {'type': 'integer'},
                        'question': {'type': 'string'},
                        'related_item_ids': {'type': 'array', 'items': {'type': 'string'}},
                    },
                    'required': ['priority', 'question', 'related_item_ids'],
                },
            },
            'methods_and_limitations': {'type': 'array', 'items': {'type': 'string'}},
        },
        'required': ['summary', 'findings', 'prioritized_provider_questions', 'methods_and_limitations'],
    }


def call_openai_assessment(
    api_key: str,
    checklist_items: list[ChecklistItem],
    target: dict[str, Any],
    source_outcomes: list[dict[str, Any]],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = build_assessment_schema(checklist_items)
    prompt = {
        'task': 'Assess public evidence for the fixed checklist.',
        'rules': [
            'Use only the provided provider evidence.',
            'Do not use regulatory context or outside knowledge for provider evidence.',
            'Return one finding per checklist item.',
            'Keep each finding short and direct.',
            'Use at most one brief verbatim quote per finding when supported.',
            'If a quote is unsupported or attributed to a different variant, do not keep it as evidence found.',
            'For unresolved items, include searched sources or a failure reason.',
            'Keep provider questions separate from regulatory context.',
            'Reuse the checklist labels and references already provided.',
        ],
        'target': target,
        'checklist': [
            {
                'item_id': item.item_id,
                'check': item.check,
                'reference': item.reference,
                'evidence_sufficient': item.evidence_sufficient,
                'is_context': item.is_context,
            }
            for item in checklist_items
        ],
        'provider_evidence': [
            {
                'name': outcome.get('name'),
                'requested_url': outcome.get('requested_url'),
                'http_status': outcome.get('http_status'),
                'usable': outcome.get('usable'),
                'urls': outcome.get('urls') or [],
                'snippet': outcome.get('snippet'),
                'full_text': _truncate(str(outcome.get('full_text') or ''), 6000) if outcome.get('full_text') else None,
                'error': outcome.get('error'),
                'metadata': outcome.get('metadata') or {},
            }
            for outcome in source_outcomes
        ],
    }
    try:
        response = requests.post(
            'https://api.openai.com/v1/responses',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': ASSESSMENT_MODEL,
                'input': json.dumps(prompt, ensure_ascii=False),
                'store': False,
                'max_output_tokens': ASSESSMENT_TOKENS,
                'text': {
                    'format': {
                        'type': 'json_schema',
                        'name': 'model_intake_assessment',
                        'strict': True,
                        'schema': schema,
                    }
                },
            },
            timeout=timeout_seconds,
        )
    except requests.exceptions.Timeout as exc:
        raise AssessmentFailure('timeout', f'OpenAI assessment request timed out after {timeout_seconds:g} seconds.') from exc
    except requests.RequestException as exc:
        raise AssessmentFailure('http_error', f'OpenAI request failed before a response was received: {exc.__class__.__name__}.') from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = getattr(response, 'status_code', None)
        raise AssessmentFailure('http_error', f'OpenAI returned HTTP {status_code}.', status_code=status_code) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise AssessmentFailure('non_json_body', 'OpenAI response body was not valid JSON.') from exc

    if not isinstance(payload, dict):
        raise AssessmentFailure('non_json_body', 'OpenAI response body was not a JSON object.')

    usage = payload.get('usage') or {}
    if payload.get('status') == 'incomplete':
        incomplete_details = payload.get('incomplete_details') or {}
        incomplete_reason = 'response status was incomplete'
        if isinstance(incomplete_details, dict):
            incomplete_reason = incomplete_details.get('reason') or incomplete_details.get('message') or incomplete_reason
        raise AssessmentFailure('incomplete', f'OpenAI response was incomplete: {_sanitize_failure_reason(incomplete_reason)}', usage=usage)

    if _response_has_refusal(payload):
        raise AssessmentFailure('refusal', 'OpenAI response contained a refusal and no usable output text.', usage=usage)

    try:
        text = _extract_response_text(payload)
    except ValueError as exc:
        raise AssessmentFailure('missing_output_text', 'OpenAI response did not include usable output text.', usage=usage) from exc

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssessmentFailure('invalid_structured_json', 'OpenAI completed response did not contain valid structured JSON.', usage=usage) from exc

    if not isinstance(parsed, dict):
        raise AssessmentFailure('invalid_structured_json', 'OpenAI completed response JSON was not an object.', usage=usage)

    return parsed, usage


def _source_texts(source_outcomes: list[dict[str, Any]]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for outcome in source_outcomes:
        requested_url = outcome.get('requested_url')
        full_text = outcome.get('full_text') or outcome.get('snippet') or ''
        if isinstance(requested_url, str) and requested_url and isinstance(full_text, str) and full_text:
            texts[requested_url] = full_text
        for url in outcome.get('urls') or []:
            if isinstance(url, str) and url and isinstance(full_text, str) and full_text:
                texts.setdefault(url, full_text)
    return texts


def _normalize_status(status: Any) -> str:
    if isinstance(status, str) and status in ALLOWED_STATUSES:
        return status
    return 'Not checked'


def _default_failure_reason(item: ChecklistItem, source_outcomes: list[dict[str, Any]]) -> str:
    failed_sources = [outcome.get('name') for outcome in source_outcomes if not outcome.get('usable')]
    if failed_sources:
        return f"No supported evidence remained after source issues: {', '.join(str(name) for name in failed_sources if name)}."
    return f'No supported evidence found for {item.item_id}.'


def normalize_assessment_output(
    raw_output: dict[str, Any],
    checklist_items: list[ChecklistItem],
    source_outcomes: list[dict[str, Any]],
    exact_model_id: str,
    regulatory_context: dict[str, Any] | None = None,
    link_checks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    findings_by_id = {finding.get('item_id'): finding for finding in raw_output.get('findings', []) if isinstance(finding, dict)}
    source_texts = _source_texts(source_outcomes)
    normalized_findings: list[dict[str, Any]] = []
    unresolved_questions: list[dict[str, Any]] = []

    for item in checklist_items:
        finding = dict(findings_by_id.get(item.item_id) or {})
        status = _normalize_status(finding.get('status'))
        summary = str(finding.get('summary') or '').strip() or f'No supported evidence found for {item.item_id}.'
        quote = finding.get('quote')
        source_url = finding.get('source_url')
        source_name = finding.get('source_name')
        source_section = finding.get('source_section')
        model_applicability = finding.get('model_applicability')
        link_check_url = finding.get('link_check_url')
        searched_sources = finding.get('searched_sources') if isinstance(finding.get('searched_sources'), list) else []
        failure_reason = finding.get('failure_reason')
        regulatory_reference = finding.get('regulatory_reference')
        provider_question = finding.get('provider_question')

        source_text = ''
        if isinstance(source_url, str) and source_url:
            source_text = source_texts.get(source_url, '')
        if not source_text and isinstance(source_name, str) and source_name:
            for outcome in source_outcomes:
                if outcome.get('name') == source_name:
                    source_text = _find_text(outcome)
                    if not source_url and isinstance(outcome.get('requested_url'), str):
                        source_url = outcome.get('requested_url')
                    break

        if quote and isinstance(quote, str):
            if not quote_applies_to_target(quote, exact_model_id, model_applicability if isinstance(model_applicability, str) else None):
                status = 'Partial' if status == 'Evidence found' else status
                if not failure_reason:
                    failure_reason = 'Quoted text appeared to describe a different model variant.'
                quote = None
                model_applicability = None
            elif not source_text or not quote_supported(quote, source_text):
                if status == 'Evidence found':
                    status = 'Partial'
                if not failure_reason:
                    failure_reason = 'Quoted text was not supported by the extracted source text.'
                quote = None
                model_applicability = None
        elif status in {'Evidence found', 'Partial'} and not source_text:
            status = 'Not found'
            if not failure_reason:
                failure_reason = _default_failure_reason(item, source_outcomes)

        if status in {'Not found', 'Not checked'} and not failure_reason:
            failure_reason = _default_failure_reason(item, source_outcomes)

        if not searched_sources:
            searched_sources = [outcome.get('requested_url') or outcome.get('name') for outcome in source_outcomes if outcome.get('requested_url') or outcome.get('name')][:3]

        link_check = None
        if isinstance(source_url, str) and link_checks:
            link_check = link_checks.get(source_url)
        if not link_check and isinstance(link_check_url, str) and link_checks:
            link_check = link_checks.get(link_check_url)
        link_check_outcome = None
        if isinstance(link_check, dict):
            outcome_value = link_check.get('outcome')
            status_code = link_check.get('status_code')
            if status_code is not None:
                link_check_outcome = f"HTTP {status_code} ({outcome_value})"
            else:
                link_check_outcome = str(outcome_value)

        normalized_findings.append(
            {
                'item_id': item.item_id,
                'check': item.check,
                'reference': item.reference,
                'evidence_sufficient': item.evidence_sufficient,
                'status': status,
                'summary': summary,
                'quote': quote if isinstance(quote, str) and quote else None,
                'source_name': source_name if isinstance(source_name, str) else None,
                'source_url': source_url if isinstance(source_url, str) else None,
                'source_section': source_section if isinstance(source_section, str) else None,
                'model_applicability': model_applicability if isinstance(model_applicability, str) else None,
                'link_check_url': link_check_url if isinstance(link_check_url, str) else None,
                'link_check_outcome': link_check_outcome,
                'searched_sources': [str(value) for value in searched_sources if value],
                'failure_reason': str(failure_reason) if failure_reason else None,
                'regulatory_reference': str(regulatory_reference) if regulatory_reference else None,
                'provider_question': str(provider_question) if provider_question else None,
            }
        )
        if status != 'Evidence found':
            provider_question = provider_question or f'Please provide public documentation for {item.check}.'
            unresolved_questions.append(
                {
                    'priority': len(unresolved_questions) + 1,
                    'question': str(provider_question),
                    'related_item_ids': [item.item_id],
                }
            )

    counts = Counter(finding['status'] for finding in normalized_findings)
    summary = raw_output.get('summary') if isinstance(raw_output.get('summary'), dict) else {}
    summary_counts = {status: counts.get(status, 0) for status in ALLOWED_STATUSES}
    summary_counts.update({status: int(summary.get('status_counts', {}).get(status, summary_counts[status])) for status in ALLOWED_STATUSES})
    summary_counts = {status: counts.get(status, 0) for status in ALLOWED_STATUSES}
    usage = raw_output.get('usage') if isinstance(raw_output.get('usage'), dict) else {}
    return {
        'summary': {
            'headline': str(summary.get('headline') or 'Assessment complete.'),
            'major_gaps': [str(value) for value in summary.get('major_gaps', []) if value],
            'status_counts': summary_counts,
        },
        'findings': normalized_findings,
        'prioritized_provider_questions': raw_output.get('prioritized_provider_questions') or unresolved_questions,
        'methods_and_limitations': [str(value) for value in (raw_output.get('methods_and_limitations') or []) if value],
        'request_usage': usage,
        'aggregate_usage': usage,
        'usage': usage,
        'regulatory_context': regulatory_context or {},
    }


def build_offline_assessment(checklist_items: list[ChecklistItem]) -> dict[str, Any]:
    findings = []
    for item in checklist_items:
        findings.append(
            {
                'item_id': item.item_id,
                'check': item.check,
                'reference': item.reference,
                'evidence_sufficient': item.evidence_sufficient,
                'status': 'Not checked',
                'summary': f'Offline fixture data for {item.item_id}.',
                'quote': None,
                'source_name': FIXTURE_RECORD['label'],
                'source_url': None,
                'source_section': FIXTURE_RECORD['section'],
                'model_applicability': 'Offline demo only.',
                'link_check_url': None,
                'link_check_outcome': None,
                'searched_sources': [FIXTURE_RECORD['source']],
                'failure_reason': 'Offline demo — fixture data only.',
                'regulatory_reference': None,
                'provider_question': None,
            }
        )
    return {
        'summary': {
            'headline': 'Offline demo — fixture data.',
            'major_gaps': ['No live source collection was run in offline mode.'],
            'status_counts': {status: (0 if status != 'Not checked' else len(findings)) for status in ALLOWED_STATUSES},
        },
        'findings': findings,
        'prioritized_provider_questions': [],
        'methods_and_limitations': [
            'Offline mode uses deterministic fixture data only.',
            'No external network requests were made.',
        ],
        'usage': None,
        'regulatory_context': {},
    }


def build_partial_assessment(
    checklist_items: list[ChecklistItem],
    reason: str,
    failure_stage: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    findings = []
    for item in checklist_items:
        findings.append(
            {
                'item_id': item.item_id,
                'check': item.check,
                'reference': item.reference,
                'evidence_sufficient': item.evidence_sufficient,
                'status': 'Not checked',
                'summary': reason,
                'quote': None,
                'source_name': None,
                'source_url': None,
                'source_section': None,
                'model_applicability': None,
                'link_check_url': None,
                'link_check_outcome': None,
                'searched_sources': [],
                'failure_reason': reason,
                'regulatory_reference': None,
                'provider_question': None,
            }
        )
    label = 'Partial integration draft - assessment failure'
    if failure_stage == 'timeout':
        label = 'Partial integration draft - assessment timeout'
    elif failure_stage == 'incomplete':
        label = 'Partial integration draft - assessment incomplete'
    return {
        'draft_label': label,
        'summary': {
            'headline': f'{label}.',
            'major_gaps': [reason],
            'status_counts': {status: (0 if status != 'Not checked' else len(findings)) for status in ALLOWED_STATUSES},
        },
        'findings': findings,
        'prioritized_provider_questions': [],
        'methods_and_limitations': [
            'The source collection and regulatory context were preserved, but the assessment call did not complete.',
            f'Failure stage: {failure_stage or "unknown"}.',
        ],
        'failure_stage': failure_stage,
        'failure_reason': reason,
        'request_usage': usage or None,
        'aggregate_usage': usage or None,
        'usage': usage or None,
        'request_output_token_cap': ASSESSMENT_TOKENS,
        'regulatory_context': {},
    }


def render_markdown(
    target: dict[str, Any],
    checklist_items: list[ChecklistItem],
    source_outcomes: list[dict[str, Any]],
    assessment: dict[str, Any],
    regulatory_contexts: dict[str, Any],
) -> str:
    exact_model_id = target.get('exact_model_id') or 'n/a'
    model_provider = target.get('model_provider') or 'unknown'
    hosting_platform = target.get('hosting_platform') or 'n/a'
    target_kind = target.get('target_kind') or 'n/a'
    checklist_version = 'v0.1'
    draft_label = assessment.get('draft_label') or ('Full report' if target.get('live_mode') else 'Offline demo — fixture data')
    summary = assessment.get('summary') or {}
    findings = assessment.get('findings') or []
    provider_questions = assessment.get('prioritized_provider_questions') or []
    methods_and_limitations = assessment.get('methods_and_limitations') or []
    request_usage = assessment.get('request_usage') or assessment.get('usage') or {}
    aggregate_usage = assessment.get('aggregate_usage') or assessment.get('usage') or request_usage
    request_output_token_cap = assessment.get('request_output_token_cap') or ASSESSMENT_TOKENS
    failure_stage = assessment.get('failure_stage')
    failure_reason = assessment.get('failure_reason')

    lines = [
        '# Model Intake Brief',
        '',
        f'## {draft_label}',
        '',
        f'- Exact model ID: `{exact_model_id}`',
        f'- Model provider: {model_provider}',
        f'- Hosting platform: {hosting_platform}',
        f'- Target kind: {target_kind}',
        f'- Checklist version: {checklist_version}',
        f'- Run date: {date.today().isoformat()}',
        '- Notice: Information, not legal advice.',
    ]
    if request_usage:
        prompt_tokens = request_usage.get('input_tokens') or request_usage.get('prompt_tokens')
        completion_tokens = request_usage.get('output_tokens') or request_usage.get('completion_tokens')
        if prompt_tokens is not None or completion_tokens is not None:
            lines.append(f'- Assessment request output token cap: {request_output_token_cap}')
            lines.append(f"- Assessment request usage: input tokens {prompt_tokens if prompt_tokens is not None else 'n/a'}, output tokens {completion_tokens if completion_tokens is not None else 'n/a'}")
    if aggregate_usage:
        prompt_tokens = aggregate_usage.get('input_tokens') or aggregate_usage.get('prompt_tokens')
        completion_tokens = aggregate_usage.get('output_tokens') or aggregate_usage.get('completion_tokens')
        if prompt_tokens is not None or completion_tokens is not None:
            lines.append(f"- Aggregate assessment usage: input tokens {prompt_tokens if prompt_tokens is not None else 'n/a'}, output tokens {completion_tokens if completion_tokens is not None else 'n/a'}")
    lines.extend(['', '### Summary'])
    lines.append(f"- Headline: {summary.get('headline') or 'Assessment complete.'}")
    counts = summary.get('status_counts') or {}
    for status in ALLOWED_STATUSES:
        lines.append(f"- {status}: {counts.get(status, 0)}")
    major_gaps = summary.get('major_gaps') or []
    if major_gaps:
        lines.append('- Major gaps:')
        for gap in major_gaps:
            lines.append(f'  - {gap}')

    if failure_stage or failure_reason:
        lines.extend(['', '### Assessment failure'])
        lines.append(f'- Stage: {failure_stage or "unknown"}')
        if failure_reason:
            lines.append(f'- Reason: {failure_reason}')
        usage_text = _format_usage(request_usage)
        if usage_text:
            lines.append(f'- Usage: {usage_text}')

    lines.extend(['', '### Source collection'])
    if source_outcomes:
        for outcome in source_outcomes:
            name = outcome.get('name', 'Tool')
            http_status = outcome.get('http_status')
            usable = 'yes' if outcome.get('usable') else 'no'
            lines.append(f"- {name}: HTTP {http_status if http_status is not None else 'n/a'}; usable data: {usable}")
            requested_url = outcome.get('requested_url')
            if requested_url:
                lines.append(f'  - Requested URL: {requested_url}')
            urls = outcome.get('urls') or []
            if urls:
                lines.append('  - Source URLs:')
                for url in urls:
                    lines.append(f'    - {url}')
            snippet = outcome.get('snippet')
            if snippet:
                label = 'Search snippet (candidate material only)' if name == 'Tavily Search' else 'Extracted text'
                lines.append(f'  - {label}: {snippet}')
            error = outcome.get('error')
            if error:
                lines.append(f'  - Error: {error}')
            metadata = outcome.get('metadata') or {}
            if metadata:
                visible = {key: value for key, value in metadata.items() if value not in (None, '', [], {})}
                if visible:
                    lines.append('  - Metadata:')
                    for key, value in visible.items():
                        lines.append(f'    - {key}: {value}')
    else:
        lines.append('- No source collection was run.')

    lines.extend(['', '### Findings'])
    finding_map = {finding.get('item_id'): finding for finding in findings if isinstance(finding, dict)}
    for item in checklist_items:
        finding = finding_map.get(item.item_id, {})
        status = finding.get('status', 'Not checked')
        lines.append(f"- {item.item_id} — {item.check}: {status}")
        lines.append(f"  - Reference: {item.reference}")
        lines.append(f"  - Evidence sufficient: {item.evidence_sufficient}")
        summary_text = finding.get('summary')
        if summary_text:
            lines.append(f'  - Summary: {summary_text}')
        quote = finding.get('quote')
        if quote:
            lines.append(f'  - Quote: {quote}')
        source_name = finding.get('source_name')
        if source_name:
            lines.append(f'  - Source: {source_name}')
        source_url = finding.get('source_url')
        if source_url:
            lines.append(f'  - Source URL: {source_url}')
        source_section = finding.get('source_section')
        if source_section:
            lines.append(f'  - Source section: {source_section}')
        model_applicability = finding.get('model_applicability')
        if model_applicability:
            lines.append(f'  - Model applicability: {model_applicability}')
        link_check_url = finding.get('link_check_url')
        link_check_outcome = finding.get('link_check_outcome')
        if link_check_url or link_check_outcome:
            lines.append(f"  - Link check: {link_check_url or 'n/a'} -> {link_check_outcome or 'not checked'}")
        regulatory_reference = finding.get('regulatory_reference')
        if regulatory_reference:
            lines.append(f'  - Regulatory reference: {regulatory_reference}')
        searched_sources = finding.get('searched_sources') or []
        if searched_sources:
            lines.append('  - Searched sources:')
            for source in searched_sources[:5]:
                lines.append(f'    - {source}')
        failure_reason = finding.get('failure_reason')
        if failure_reason:
            lines.append(f'  - Failure reason: {failure_reason}')

    lines.extend(['', '### Prioritised provider questions'])
    if provider_questions:
        for question in provider_questions:
            if isinstance(question, dict):
                priority = question.get('priority', 'n/a')
                related = ', '.join(question.get('related_item_ids') or [])
                lines.append(f"- {priority}. {question.get('question')}" + (f' [{related}]' if related else ''))
            else:
                lines.append(f'- {question}')
    else:
        lines.append('- No unresolved provider questions were identified.')

    lines.extend(['', '### Methods and limitations'])
    for item in methods_and_limitations or []:
        lines.append(f'- {item}')
    if not methods_and_limitations:
        lines.append('- No limitations recorded.')

    lines.extend(['', '### Regulatory context'])
    if regulatory_contexts:
        for item_id, context in regulatory_contexts.items():
            lines.append(f'- Checklist item: {item_id}')
            lines.append(f"  - Retrieval query: {context.get('query') or 'n/a'}")
            blockers = context.get('blockers') or []
            if blockers:
                lines.append('  - Retrieval blockers:')
                for blocker in blockers:
                    lines.append(f'    - {blocker}')
            hits = context.get('hits') or []
            if hits:
                lines.append('  - Retrieved references:')
                for hit in hits:
                    lines.append(f"    - {hit.get('provision', 'unknown')} ({hit.get('chunk_id', 'unknown')})")
                    lines.append(f"      - Official URL: {hit.get('official_url') or 'n/a'}")
                    lines.append(f"      - Document version: {hit.get('document_version') or 'n/a'}")
                    lines.append(f"      - Match score: {hit.get('score')}")
                    excerpt = hit.get('excerpt')
                    if excerpt:
                        lines.append(f'      - Excerpt: {excerpt}')
    else:
        lines.append('- No regulatory references were returned.')

    return '\n'.join(lines)
