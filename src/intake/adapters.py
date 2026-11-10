from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class ToolOutcome:
    name: str
    requested_url: str | None
    http_status: int | None
    usable: bool
    urls: list[str] = field(default_factory=list)
    snippet: str | None = None
    full_text: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'requested_url': self.requested_url,
            'http_status': self.http_status,
            'usable': self.usable,
            'urls': self.urls,
            'snippet': self.snippet,
            'error': self.error,
            'metadata': self.metadata,
        }


USER_AGENT = 'model-intake-brief/1.0'


def _truncate(text: str, limit: int = 240) -> str:
    cleaned = ' '.join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + '…'


def _validate_https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f'Rejected non-HTTPS or malformed URL: {value!r}')
    if any(ch.isspace() for ch in parsed.netloc):
        raise ValueError(f'Rejected invalid host in URL: {value!r}')
    return value


def _host_matches(url: str, allowed_domains: list[str]) -> bool:
    if not allowed_domains:
        return True
    host = urlparse(url).hostname or ''
    return any(host == domain or host.endswith(f'.{domain}') for domain in allowed_domains)


def tavily_search_urls(
    query: str,
    api_key: str,
    timeout_seconds: float,
    include_domains: list[str] | None = None,
) -> ToolOutcome:
    request_url = 'https://api.tavily.com/search'
    payload: dict[str, Any] = {
        'query': query,
        'max_results': 3,
        'include_answer': False,
        'include_raw_content': False,
        'search_depth': 'basic',
    }
    if include_domains:
        payload['include_domains'] = include_domains

    try:
        response = requests.post(
            request_url,
            json=payload,
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return ToolOutcome(
            name='Tavily Search',
            requested_url=request_url,
            http_status=None,
            usable=False,
            error=f'Tavily Search request failed: {exc.__class__.__name__}.',
        )

    http_status = response.status_code
    if http_status != 200:
        return ToolOutcome(
            name='Tavily Search',
            requested_url=request_url,
            http_status=http_status,
            usable=False,
            error=f'Tavily Search returned HTTP {http_status}.',
        )

    try:
        payload_data = response.json()
    except ValueError:
        return ToolOutcome(
            name='Tavily Search',
            requested_url=request_url,
            http_status=http_status,
            usable=False,
            error='Tavily Search returned invalid JSON.',
        )

    raw_results = payload_data.get('results')
    if not isinstance(raw_results, list):
        return ToolOutcome(
            name='Tavily Search',
            requested_url=request_url,
            http_status=http_status,
            usable=False,
            error='Tavily Search response did not include a results list.',
        )

    urls: list[str] = []
    snippets: list[str] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = item.get('url')
        if isinstance(url, str):
            try:
                safe_url = _validate_https_url(url)
            except ValueError:
                continue
            urls.append(safe_url)
            content = item.get('content') or item.get('title')
            if isinstance(content, str) and content.strip():
                snippets.append(_truncate(content))

    if not urls:
        return ToolOutcome(
            name='Tavily Search',
            requested_url=request_url,
            http_status=http_status,
            usable=False,
            error='Tavily Search returned no usable HTTPS URLs.',
        )

    return ToolOutcome(
        name='Tavily Search',
        requested_url=request_url,
        http_status=http_status,
        usable=True,
        urls=urls,
        snippet=snippets[0] if snippets else None,
        metadata={'result_count': len(raw_results)},
    )


def jina_reader_extract(url: str, timeout_seconds: float, allowed_domains: list[str] | None = None) -> ToolOutcome:
    try:
        safe_url = _validate_https_url(url)
    except ValueError as exc:
        return ToolOutcome(
            name='Jina Reader',
            requested_url=url,
            http_status=None,
            usable=False,
            error=str(exc),
        )

    if allowed_domains is not None and not _host_matches(safe_url, allowed_domains):
        return ToolOutcome(
            name='Jina Reader',
            requested_url=safe_url,
            http_status=None,
            usable=False,
            error='Jina Reader URL host did not match the configured allowlist.',
        )

    request_url = f'https://r.jina.ai/{safe_url}'
    try:
        response = requests.get(request_url, timeout=timeout_seconds, headers={'User-Agent': USER_AGENT})
    except requests.RequestException as exc:
        return ToolOutcome(
            name='Jina Reader',
            requested_url=safe_url,
            http_status=None,
            usable=False,
            error=f'Jina Reader request failed: {exc.__class__.__name__}.',
        )

    http_status = response.status_code
    if http_status != 200:
        return ToolOutcome(
            name='Jina Reader',
            requested_url=safe_url,
            http_status=http_status,
            usable=False,
            error=f'Jina Reader returned HTTP {http_status}.',
        )

    text = response.text.strip()
    if not text:
        return ToolOutcome(
            name='Jina Reader',
            requested_url=safe_url,
            http_status=http_status,
            usable=False,
            error='Jina Reader returned an empty document.',
        )

    return ToolOutcome(
        name='Jina Reader',
        requested_url=safe_url,
        http_status=http_status,
        usable=True,
        snippet=_truncate(text),
        full_text=text,
    )


def hugging_face_model_metadata(model_id: str, timeout_seconds: float) -> ToolOutcome:
    request_url = f'https://huggingface.co/api/models/{model_id}'
    try:
        response = requests.get(request_url, timeout=timeout_seconds, headers={'User-Agent': USER_AGENT})
    except requests.RequestException as exc:
        return ToolOutcome(
            name='Hugging Face Hub',
            requested_url=request_url,
            http_status=None,
            usable=False,
            error=f'Hugging Face Hub request failed: {exc.__class__.__name__}.',
        )

    http_status = response.status_code
    if http_status != 200:
        return ToolOutcome(
            name='Hugging Face Hub',
            requested_url=request_url,
            http_status=http_status,
            usable=False,
            error=f'Hugging Face Hub returned HTTP {http_status}.',
        )

    try:
        payload = response.json()
    except ValueError:
        return ToolOutcome(
            name='Hugging Face Hub',
            requested_url=request_url,
            http_status=http_status,
            usable=False,
            error='Hugging Face Hub returned invalid JSON.',
        )

    model_name = payload.get('modelId') or payload.get('id')
    if not isinstance(model_name, str) or not model_name:
        return ToolOutcome(
            name='Hugging Face Hub',
            requested_url=request_url,
            http_status=http_status,
            usable=False,
            error='Hugging Face Hub response did not include a model identifier.',
        )

    card_data = payload.get('cardData') if isinstance(payload.get('cardData'), dict) else {}
    license_value = payload.get('license')
    if not isinstance(license_value, str) or not license_value:
        card_license = card_data.get('license') if isinstance(card_data, dict) else None
        license_value = card_license if isinstance(card_license, str) else None
    license_url = payload.get('license_url') or payload.get('licenseUrl')
    if not isinstance(license_url, str) or not license_url:
        for candidate in (
            card_data.get('license_url') if isinstance(card_data, dict) else None,
            card_data.get('licenseUrl') if isinstance(card_data, dict) else None,
            card_data.get('license_link') if isinstance(card_data, dict) else None,
            card_data.get('licenseLink') if isinstance(card_data, dict) else None,
        ):
            if isinstance(candidate, str) and candidate:
                license_url = candidate
                break
    author = payload.get('author')
    if not isinstance(author, str) or not author:
        card_author = card_data.get('author') if isinstance(card_data, dict) else None
        author = card_author if isinstance(card_author, str) else None

    metadata = {
        'modelId': model_name,
        'pipeline_tag': payload.get('pipeline_tag'),
        'license': license_value,
        'license_url': license_url,
        'author': author,
        'downloads': payload.get('downloads'),
        'likes': payload.get('likes'),
        'private': payload.get('private'),
    }
    return ToolOutcome(
        name='Hugging Face Hub',
        requested_url=request_url,
        http_status=http_status,
        usable=True,
        metadata=metadata,
    )
