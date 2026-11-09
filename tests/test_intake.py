from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from intake.adapters import ToolOutcome
from intake.cli import main
from intake.regulatory import RetrievalHit, RetrievalResult
from intake.settings import LiveSettings


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def test_offline_cli_renders_fixture_without_live_config(monkeypatch):
    def should_not_run():
        raise AssertionError('load_live_settings should not be called in offline mode')

    monkeypatch.setattr('intake.graph.load_live_settings', should_not_run)

    code, stdout, stderr = run_cli(['run', '--model', 'Qwen/Qwen3-8B'])

    assert code == 0
    assert 'Offline demo ? fixture data' in stdout
    assert 'Information, not legal advice.' in stdout
    assert 'FIXTURE-01' in stdout
    assert stderr == ''


def test_index_regulatory_cli_reports_needed_config(monkeypatch):
    monkeypatch.setattr('intake.cli.load_rag_config_from_env', lambda: {})

    code, stdout, stderr = run_cli(['index-regulatory'])

    assert code == 2
    assert '# Regulatory index status' in stdout
    assert 'PINECONE_API_KEY' in stdout
    assert 'PINECONE_INDEX_NAME' in stdout
    assert stderr == ''


def test_live_qwen_path_retains_results_and_adds_regulatory_context(monkeypatch):
    settings = LiveSettings(
        tavily_api_key='tvly-test',
        provider_domains=['openai.com', 'platform.openai.com'],
        tavily_timeout_seconds=5,
        jina_timeout_seconds=5,
        hf_timeout_seconds=5,
        rag_timeout_seconds=5,
    )

    def fake_settings():
        return settings

    def fake_rag_config():
        return {
            'OPENAI_API_KEY': 'oa-test',
            'PINECONE_API_KEY': 'pc-test',
            'PINECONE_INDEX_NAME': 'regulatory-index',
            'PINECONE_NAMESPACE': 'model-intake-brief-regulatory',
        }

    def fake_search(*args, **kwargs):
        return ToolOutcome(
            name='Tavily Search',
            requested_url='https://api.tavily.com/search',
            http_status=200,
            usable=True,
            urls=['https://example.com/qwen-docs'],
            snippet='Search result snippet',
            metadata={'result_count': 1},
        )

    def fake_reader(*args, **kwargs):
        return ToolOutcome(
            name='Jina Reader',
            requested_url='https://example.com/qwen-docs',
            http_status=502,
            usable=False,
            error='Jina Reader returned HTTP 502.',
        )

    def fake_hf(*args, **kwargs):
        return ToolOutcome(
            name='Hugging Face Hub',
            requested_url='https://huggingface.co/api/models/Qwen/Qwen3-8B',
            http_status=200,
            usable=True,
            metadata={'modelId': 'Qwen/Qwen3-8B', 'license': 'apache-2.0'},
        )

    def fake_retrieve(query, config, item_id=None, timeout_seconds=15, top_k=3):
        assert item_id == 'DT-03'
        return RetrievalResult(
            query=query,
            item_id=item_id,
            hits=[
                RetrievalHit(
                    chunk_id='ai-act-art53-1d',
                    provision='Article 53(1)(d)',
                    official_url='https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ%3AL_202401689',
                    score=0.93,
                    document_version='Regulation (EU) 2024/1689, OJ L, 12.7.2024',
                    excerpt='Providers must draw up and publicly make available a sufficiently detailed summary of the content used for training.',
                )
            ],
            blockers=[],
        )

    monkeypatch.setattr('intake.graph.load_live_settings', fake_settings)
    monkeypatch.setattr('intake.graph.load_rag_config_from_env', fake_rag_config)
    monkeypatch.setattr('intake.graph.tavily_search_urls', fake_search)
    monkeypatch.setattr('intake.graph.jina_reader_extract', fake_reader)
    monkeypatch.setattr('intake.graph.hugging_face_model_metadata', fake_hf)
    monkeypatch.setattr('intake.graph.retrieve_regulatory_context', fake_retrieve)

    code, stdout, stderr = run_cli(['run', '--live', '--model', 'Qwen/Qwen3-8B'])

    assert code == 0
    assert '## Integration draft' in stdout
    assert 'Exact model ID: `Qwen/Qwen3-8B`' in stdout
    assert 'Tavily Search: HTTP 200; usable data: yes' in stdout
    assert 'Jina Reader: HTTP 502; usable data: no' in stdout
    assert 'Hugging Face Hub: HTTP 200; usable data: yes' in stdout
    assert 'Search snippet (candidate material only): Search result snippet' in stdout
    assert '### Regulatory context' in stdout
    assert 'Checklist item: DT-03' in stdout
    assert 'Article 53(1)(d)' in stdout
    assert stderr == ''


def test_live_hosted_route_uses_provider_domains_and_skips_hf(monkeypatch):
    settings = LiveSettings(
        tavily_api_key='tvly-test',
        provider_domains=['openai.com', 'platform.openai.com'],
        tavily_timeout_seconds=5,
        jina_timeout_seconds=5,
        hf_timeout_seconds=5,
        rag_timeout_seconds=5,
    )

    def fake_settings():
        return settings

    def fake_rag_config():
        return {
            'OPENAI_API_KEY': 'oa-test',
            'PINECONE_API_KEY': 'pc-test',
            'PINECONE_INDEX_NAME': 'regulatory-index',
            'PINECONE_NAMESPACE': 'model-intake-brief-regulatory',
        }

    def fake_search(*args, **kwargs):
        assert kwargs['include_domains'] == ['openai.com', 'platform.openai.com']
        return ToolOutcome(
            name='Tavily Search',
            requested_url='https://api.tavily.com/search',
            http_status=200,
            usable=True,
            urls=['https://openai.com/index/model-release/'],
            snippet='OpenAI docs snippet',
            metadata={'result_count': 1},
        )

    def fake_reader(*args, **kwargs):
        assert kwargs['allowed_domains'] == ['openai.com', 'platform.openai.com']
        return ToolOutcome(
            name='Jina Reader',
            requested_url='https://openai.com/index/model-release/',
            http_status=200,
            usable=True,
            snippet='OpenAI reader text',
        )

    def should_not_run(*args, **kwargs):
        raise AssertionError('Hugging Face Hub should be skipped for hosted runs')

    def fake_retrieve(query, config, item_id=None, timeout_seconds=15, top_k=3):
        assert item_id == 'US-03'
        return RetrievalResult(
            query=query,
            item_id=item_id,
            hits=[
                RetrievalHit(
                    chunk_id='ai-act-art53-1b',
                    provision='Article 53(1)(b)',
                    official_url='https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ%3AL_202401689',
                    score=0.88,
                    document_version='Regulation (EU) 2024/1689, OJ L, 12.7.2024',
                    excerpt='Providers must prepare and keep up to date information and documentation for AI system providers.',
                )
            ],
            blockers=[],
        )

    monkeypatch.setattr('intake.graph.load_live_settings', fake_settings)
    monkeypatch.setattr('intake.graph.load_rag_config_from_env', fake_rag_config)
    monkeypatch.setattr('intake.graph.tavily_search_urls', fake_search)
    monkeypatch.setattr('intake.graph.jina_reader_extract', fake_reader)
    monkeypatch.setattr('intake.graph.hugging_face_model_metadata', should_not_run)
    monkeypatch.setattr('intake.graph.retrieve_regulatory_context', fake_retrieve)

    code, stdout, stderr = run_cli(['run', '--live', '--provider', 'openai', '--model', 'gpt-5.2-2025-12-11'])

    assert code == 0
    assert '## Integration draft' in stdout
    assert 'Exact model ID: `gpt-5.2-2025-12-11`' in stdout
    assert 'Tavily Search: HTTP 200; usable data: yes' in stdout
    assert 'Jina Reader: HTTP 200; usable data: yes' in stdout
    assert 'openai.com' in stdout
    assert 'platform.openai.com' in stdout
    assert 'Hugging Face Hub' not in stdout
    assert 'Checklist item: US-03' in stdout
    assert 'Article 53(1)(b)' in stdout
    assert stderr == ''
