from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiveSettings:
    tavily_api_key: str
    provider_domains: list[str]
    tavily_timeout_seconds: float
    jina_timeout_seconds: float
    hf_timeout_seconds: float
    assessment_timeout_seconds: float
    rag_timeout_seconds: float = 15.0


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _resolve_setting(name: str, env_values: dict[str, str], default: str | None = None) -> str | None:
    value = env_values.get(name)
    if value is None or value == '':
        return default
    return value


def load_live_settings() -> LiveSettings:
    import os

    env_values = load_env_file(Path('.env'))
    merged = {**env_values, **os.environ}

    tavily_api_key = _resolve_setting('TAVILY_API_KEY', merged)
    if not tavily_api_key:
        raise ValueError('Missing TAVILY_API_KEY for live mode.')

    provider_domains_raw = _resolve_setting('INTAKE_PROVIDER_DOMAINS', merged, '') or ''
    provider_domains = [domain.strip() for domain in provider_domains_raw.split(',') if domain.strip()]

    return LiveSettings(
        tavily_api_key=tavily_api_key,
        provider_domains=provider_domains,
        tavily_timeout_seconds=float(_resolve_setting('INTAKE_TAVILY_TIMEOUT_SECONDS', merged, '15') or '15'),
        jina_timeout_seconds=float(_resolve_setting('INTAKE_JINA_TIMEOUT_SECONDS', merged, '15') or '15'),
        hf_timeout_seconds=float(_resolve_setting('INTAKE_HF_TIMEOUT_SECONDS', merged, '15') or '15'),
        assessment_timeout_seconds=float(_resolve_setting('INTAKE_ASSESSMENT_TIMEOUT_SECONDS', merged, '90') or '90'),
        rag_timeout_seconds=float(_resolve_setting('INTAKE_RAG_TIMEOUT_SECONDS', merged, '15') or '15'),
    )
