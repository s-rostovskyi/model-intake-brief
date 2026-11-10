from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import requests

from intake.adapters import ToolOutcome
from intake.assessment import ASSESSMENT_TOKENS, AssessmentFailure, call_openai_assessment, normalize_assessment_output
from intake.cli import main
from intake.checklist import load_checklist
from intake.regulatory import RetrievalHit, RetrievalResult
from intake.settings import LiveSettings


CHECKLIST_IDS = [item.item_id for item in load_checklist()]


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _fake_assessment(positive_item: str, source_url: str, source_name: str = 'Jina Reader', expected_timeout_seconds: float | None = None):
    def fake_assessment(api_key, checklist_items, target, source_outcomes, timeout_seconds):
        if expected_timeout_seconds is not None:
            assert float(timeout_seconds) == float(expected_timeout_seconds)
        findings = []
        for item in checklist_items:
            if item.item_id == positive_item:
                findings.append(
                    {
                        'item_id': item.item_id,
                        'status': 'Evidence found',
                        'summary': f'Public evidence supports {item.item_id}.',
                        'quote': f'{item.item_id} excerpt',
                        'source_name': source_name,
                        'source_url': source_url,
                        'source_section': 'Model card',
                        'model_applicability': f'Exact model applies to {target["exact_model_id"]}.',
                        'link_check_url': source_url,
                        'searched_sources': [source_url],
                        'failure_reason': None,
                        'regulatory_reference': item.reference,
                        'provider_question': None,
                    }
                )
            else:
                findings.append(
                    {
                        'item_id': item.item_id,
                        'status': 'Not found',
                        'summary': f'No public evidence located for {item.item_id}.',
                        'quote': None,
                        'source_name': None,
                        'source_url': None,
                        'source_section': None,
                        'model_applicability': None,
                        'link_check_url': None,
                        'searched_sources': [outcome.get('requested_url') for outcome in source_outcomes if outcome.get('requested_url')],
                        'failure_reason': f'No supported evidence found for {item.item_id}.',
                        'regulatory_reference': item.reference,
                        'provider_question': f'Please provide public documentation for {item.check}.',
                    }
                )
        return (
            {
                'summary': {
                    'headline': 'Assessment complete.',
                    'major_gaps': ['Most checklist items need additional public evidence.'],
                    'status_counts': {'Evidence found': 1, 'Partial': 0, 'Not found': len(checklist_items) - 1, 'Not checked': 0},
                },
                'findings': findings,
                'prioritized_provider_questions': [
                    {
                        'priority': 1,
                        'question': f'Please supply public documentation for {positive_item}.',
                        'related_item_ids': [positive_item],
                    }
                ],
                'methods_and_limitations': ['Mocked assessment for verification.'],
            },
            {'input_tokens': 111, 'output_tokens': 222},
        )

    return fake_assessment


def test_offline_cli_renders_fixture_without_live_config(monkeypatch):
    def should_not_run():
        raise AssertionError('load_live_settings should not be called in offline mode')

    monkeypatch.setattr('intake.graph.load_live_settings', should_not_run)

    code, stdout, stderr = run_cli(['run', '--model', 'Qwen/Qwen3-8B'])

    assert code == 0
    assert 'Offline demo' in stdout
    assert 'Information, not legal advice.' in stdout
    assert 'FIXTURE-01' in stdout
    for item_id in CHECKLIST_IDS:
        assert item_id in stdout
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
        assessment_timeout_seconds=42,
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
            urls=['https://huggingface.co/Qwen/Qwen3-8B', 'https://huggingface.co/Qwen/Qwen3-8B/blob/main/README.md'],
            snippet='Search result snippet',
            metadata={'result_count': 2},
        )

    def fake_reader(url, timeout_seconds, allowed_domains=None):
        if 'blob/main/README.md' in url:
            return ToolOutcome(
                name='Jina Reader',
                requested_url=url,
                http_status=502,
                usable=False,
                error='Jina Reader returned HTTP 502.',
            )
        return ToolOutcome(
            name='Jina Reader',
            requested_url=url,
            http_status=200,
            usable=True,
            snippet='Qwen/Qwen3-8B public summary',
            full_text='Qwen/Qwen3-8B public summary and training-content description.',
        )

    def fake_hf(*args, **kwargs):
        return ToolOutcome(
            name='Hugging Face Hub',
            requested_url='https://huggingface.co/api/models/Qwen/Qwen3-8B',
            http_status=200,
            usable=True,
            metadata={'modelId': 'Qwen/Qwen3-8B', 'license': 'apache-2.0', 'license_url': 'https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE', 'author': 'Qwen'},
        )

    def fake_retrieve_batch(item_ids, config, timeout_seconds=15.0, top_k=3):
        result = {}
        for item_id in item_ids:
            result[item_id] = RetrievalResult(
                query='public summary of training content' if item_id == 'DT-03' else 'relevant regulatory context',
                item_id=item_id,
                hits=[
                    RetrievalHit(
                        chunk_id='ai-act-art53-1d',
                        provision='Article 53(1)(d)',
                        official_url='https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ%3AL_202401689',
                        score=0.93,
                        document_version='Regulation (EU) 2024/1689, OJ L, 12.7.2024',
                        excerpt='Providers must draw up and publicly make available a sufficiently detailed summary of the content used for training.',
                    ) if item_id == 'DT-03' else RetrievalHit(
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
        return result

    monkeypatch.setattr('intake.graph.load_live_settings', fake_settings)
    monkeypatch.setattr('intake.graph.load_rag_config_from_env', fake_rag_config)
    monkeypatch.setattr('intake.graph.tavily_search_urls', fake_search)
    monkeypatch.setattr('intake.graph.jina_reader_extract', fake_reader)
    monkeypatch.setattr('intake.graph.hugging_face_model_metadata', fake_hf)
    monkeypatch.setattr('intake.graph.retrieve_regulatory_context_batch', fake_retrieve_batch)
    monkeypatch.setattr('intake.graph.call_openai_assessment', _fake_assessment('DT-03', 'https://huggingface.co/Qwen/Qwen3-8B', expected_timeout_seconds=42))
    monkeypatch.setattr('intake.graph.check_public_urls', lambda urls, timeout_seconds, max_checks=6: {url: {'url': url, 'status_code': 200, 'final_url': url, 'outcome': 'ok'} for url in urls})

    code, stdout, stderr = run_cli(['run', '--live', '--model', 'Qwen/Qwen3-8B'])

    assert code == 0
    assert '## Full report' in stdout
    assert 'Exact model ID: `Qwen/Qwen3-8B`' in stdout
    assert 'Model provider: Qwen' in stdout
    assert 'Hosting platform: Hugging Face' in stdout
    assert 'Tavily Search: HTTP 200; usable data: yes' in stdout
    assert 'Jina Reader: HTTP 502; usable data: no' in stdout
    assert 'Hugging Face Hub: HTTP 200; usable data: yes' in stdout
    assert 'Search snippet (candidate material only): Search result snippet' in stdout
    assert 'license: apache-2.0' in stdout
    assert 'license_url: https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE' in stdout
    assert '### Findings' in stdout
    for item_id in CHECKLIST_IDS:
        assert item_id in stdout
    assert 'Evidence found' in stdout
    assert '### Regulatory context' in stdout
    assert 'Checklist item: DT-03' in stdout
    assert 'Article 53(1)(d)' in stdout
    assert stderr == ''


def test_live_assessment_timeout_preserves_partial_report(monkeypatch):
    settings = LiveSettings(
        tavily_api_key='tvly-test',
        provider_domains=['openai.com', 'platform.openai.com'],
        tavily_timeout_seconds=5,
        jina_timeout_seconds=5,
        hf_timeout_seconds=5,
        assessment_timeout_seconds=33,
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
            urls=['https://openai.com/index/model-release/'],
            snippet='OpenAI docs snippet',
        )

    def fake_reader(*args, **kwargs):
        return ToolOutcome(
            name='Jina Reader',
            requested_url='https://openai.com/index/model-release/',
            http_status=200,
            usable=True,
            snippet='OpenAI reader text',
            full_text='OpenAI reader text with integration instructions.',
        )

    def fake_timeout_assessment(api_key, checklist_items, target, source_outcomes, timeout_seconds):
        assert float(timeout_seconds) == 33.0
        raise AssessmentFailure('timeout', 'OpenAI assessment request timed out after 33 seconds.')

    def fake_retrieve_batch(item_ids, config, timeout_seconds=15.0, top_k=3):
        result = {}
        for item_id in item_ids:
            result[item_id] = RetrievalResult(
                query='official integration instructions and API call' if item_id == 'US-03' else 'relevant regulatory context',
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
        return result

    monkeypatch.setattr('intake.graph.load_live_settings', fake_settings)
    monkeypatch.setattr('intake.graph.load_rag_config_from_env', fake_rag_config)
    monkeypatch.setattr('intake.graph.tavily_search_urls', fake_search)
    monkeypatch.setattr('intake.graph.jina_reader_extract', fake_reader)
    monkeypatch.setattr('intake.graph.retrieve_regulatory_context_batch', fake_retrieve_batch)
    monkeypatch.setattr('intake.graph.call_openai_assessment', fake_timeout_assessment)
    monkeypatch.setattr('intake.graph.check_public_urls', lambda urls, timeout_seconds, max_checks=6: {url: {'url': url, 'status_code': 200, 'final_url': url, 'outcome': 'ok'} for url in urls})

    code, stdout, stderr = run_cli(['run', '--live', '--provider', 'openai', '--model', 'gpt-5.2-2025-12-11'])

    assert code == 2
    assert '## Partial integration draft - assessment timeout' in stdout
    assert '### Assessment failure' in stdout
    assert 'Stage: timeout' in stdout
    assert 'OpenAI assessment request timed out after 33 seconds.' in stdout
    assert '### Source collection' in stdout
    assert 'Tavily Search: HTTP 200; usable data: yes' in stdout
    assert '### Regulatory context' in stdout
    assert 'Checklist item: US-03' in stdout
    assert stdout.count(': Not checked') == len(CHECKLIST_IDS)
    assert 'Assessment request failed at timeout stage: OpenAI assessment request timed out after 33 seconds.' in stderr

def test_live_hosted_route_uses_provider_domains_and_skips_hf(monkeypatch):
    settings = LiveSettings(
        tavily_api_key='tvly-test',
        provider_domains=['openai.com', 'platform.openai.com'],
        tavily_timeout_seconds=5,
        jina_timeout_seconds=5,
        hf_timeout_seconds=5,
        assessment_timeout_seconds=42,
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
            urls=['https://openai.com/index/model-release/', 'https://platform.openai.com/docs/models/gpt-5-2'],
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
            full_text='OpenAI reader text with integration instructions.',
        )

    def should_not_run(*args, **kwargs):
        raise AssertionError('Hugging Face Hub should be skipped for hosted runs')

    def fake_retrieve_batch(item_ids, config, timeout_seconds=15.0, top_k=3):
        result = {}
        for item_id in item_ids:
            result[item_id] = RetrievalResult(
                query='official integration instructions and API call' if item_id == 'US-03' else 'relevant regulatory context',
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
        return result

    monkeypatch.setattr('intake.graph.load_live_settings', fake_settings)
    monkeypatch.setattr('intake.graph.load_rag_config_from_env', fake_rag_config)
    monkeypatch.setattr('intake.graph.tavily_search_urls', fake_search)
    monkeypatch.setattr('intake.graph.jina_reader_extract', fake_reader)
    monkeypatch.setattr('intake.graph.hugging_face_model_metadata', should_not_run)
    monkeypatch.setattr('intake.graph.retrieve_regulatory_context_batch', fake_retrieve_batch)
    monkeypatch.setattr('intake.graph.call_openai_assessment', _fake_assessment('US-03', 'https://openai.com/index/model-release/', expected_timeout_seconds=42))
    monkeypatch.setattr('intake.graph.check_public_urls', lambda urls, timeout_seconds, max_checks=6: {url: {'url': url, 'status_code': 200, 'final_url': url, 'outcome': 'ok'} for url in urls})

    code, stdout, stderr = run_cli(['run', '--live', '--provider', 'openai', '--model', 'gpt-5.2-2025-12-11'])

    assert code == 0
    assert '## Full report' in stdout
    assert 'Exact model ID: `gpt-5.2-2025-12-11`' in stdout
    assert 'Model provider: openai' in stdout
    assert 'Hosting platform: openai' in stdout
    assert 'Tavily Search: HTTP 200; usable data: yes' in stdout
    assert 'Jina Reader: HTTP 200; usable data: yes' in stdout
    assert 'openai.com' in stdout
    assert 'platform.openai.com' in stdout
    assert 'Hugging Face Hub' not in stdout
    assert 'Checklist item: US-03' in stdout
    assert 'Article 53(1)(b)' in stdout
    assert stderr == ''


def test_call_openai_assessment_uses_output_cap(monkeypatch):
    checklist_item = load_checklist()[0]
    captured_request = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'output_text': '{"summary":{"headline":"ok","major_gaps":[],"status_counts":{"Evidence found":0,"Partial":0,"Not found":0,"Not checked":1}},"findings":[{"item_id":"%s","status":"Not checked","summary":"ok","quote":null,"source_name":null,"source_url":null,"source_section":null,"model_applicability":null,"link_check_url":null,"searched_sources":[],"failure_reason":null,"regulatory_reference":null,"provider_question":null}],"prioritized_provider_questions":[],"methods_and_limitations":[]}' % checklist_item.item_id,
                'usage': {'input_tokens': 12, 'output_tokens': 34},
            }

    def fake_post(url, headers, json, timeout):
        captured_request['url'] = url
        captured_request['json'] = json
        captured_request['timeout'] = timeout
        return FakeResponse()

    monkeypatch.setattr('intake.assessment.requests.post', fake_post)

    raw_output, usage = call_openai_assessment(
        'oa-test',
        [checklist_item],
        {'provider': 'openai', 'normalized_provider': 'openai', 'exact_model_id': 'gpt-5.2-2025-12-11', 'target_kind': 'hosted'},
        [],
        timeout_seconds=17,
    )

    assert captured_request['url'] == 'https://api.openai.com/v1/responses'
    assert captured_request['json']['max_output_tokens'] == ASSESSMENT_TOKENS
    assert captured_request['timeout'] == 17
    assert 'Keep each finding short and direct.' in captured_request['json']['input']
    assert 'Use at most one brief verbatim quote per finding when supported.' in captured_request['json']['input']
    assert usage == {'input_tokens': 12, 'output_tokens': 34}
    assert raw_output['findings'][0]['item_id'] == checklist_item.item_id


def test_call_openai_assessment_reports_incomplete_response(monkeypatch):
    checklist_item = load_checklist()[0]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'status': 'incomplete',
                'incomplete_details': {'reason': 'max_output_tokens'},
                'usage': {'input_tokens': 7, 'output_tokens': ASSESSMENT_TOKENS},
            }

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setattr('intake.assessment.requests.post', fake_post)

    try:
        call_openai_assessment(
            'oa-test',
            [checklist_item],
            {'provider': 'openai', 'normalized_provider': 'openai', 'exact_model_id': 'gpt-5.2-2025-12-11', 'target_kind': 'hosted'},
            [],
            timeout_seconds=17,
        )
    except AssessmentFailure as exc:
        assert exc.stage == 'incomplete'
        assert 'max_output_tokens' in exc.reason
        assert exc.usage == {'input_tokens': 7, 'output_tokens': ASSESSMENT_TOKENS}
    else:
        raise AssertionError('AssessmentFailure was not raised for an incomplete response')


def test_call_openai_assessment_reports_non_json_body(monkeypatch):
    checklist_item = load_checklist()[0]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError('not json')

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setattr('intake.assessment.requests.post', fake_post)

    try:
        call_openai_assessment(
            'oa-test',
            [checklist_item],
            {'provider': 'openai', 'normalized_provider': 'openai', 'exact_model_id': 'gpt-5.2-2025-12-11', 'target_kind': 'hosted'},
            [],
            timeout_seconds=17,
        )
    except AssessmentFailure as exc:
        assert exc.stage == 'non_json_body'
        assert 'valid JSON' in exc.reason
        assert exc.usage == {}
    else:
        raise AssertionError('AssessmentFailure was not raised for a non-JSON response body')


def test_call_openai_assessment_reports_invalid_structured_json(monkeypatch):
    checklist_item = load_checklist()[0]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'status': 'completed',
                'output_text': '{"summary": {"headline": "ok"}',
                'usage': {'input_tokens': 9, 'output_tokens': 10},
            }

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setattr('intake.assessment.requests.post', fake_post)

    try:
        call_openai_assessment(
            'oa-test',
            [checklist_item],
            {'provider': 'openai', 'normalized_provider': 'openai', 'exact_model_id': 'gpt-5.2-2025-12-11', 'target_kind': 'hosted'},
            [],
            timeout_seconds=17,
        )
    except AssessmentFailure as exc:
        assert exc.stage == 'invalid_structured_json'
        assert 'structured JSON' in exc.reason
        assert exc.usage == {'input_tokens': 9, 'output_tokens': 10}
    else:
        raise AssertionError('AssessmentFailure was not raised for invalid structured JSON')


def test_assessment_normalizer_keeps_exact_section_quote_with_incidental_variant_reference():
    checklist_items = load_checklist()
    raw_output = {
        'summary': {'headline': 'draft', 'major_gaps': [], 'status_counts': {'Evidence found': 1, 'Partial': 0, 'Not found': 0, 'Not checked': 0}},
        'findings': [
            {
                'item_id': 'DT-03',
                'status': 'Evidence found',
                'summary': 'Exact section quote with incidental related-variant mention.',
                'quote': 'Qwen2.5 appears elsewhere on the page, but this exact section remains about Qwen/Qwen3-8B.',
                'source_name': 'Jina Reader',
                'source_url': 'https://huggingface.co/Qwen/Qwen3-8B',
                'source_section': 'Model card / exact section',
                'model_applicability': 'Exact model applies to Qwen/Qwen3-8B.',
                'link_check_url': 'https://huggingface.co/Qwen/Qwen3-8B',
                'searched_sources': ['https://huggingface.co/Qwen/Qwen3-8B'],
                'failure_reason': None,
                'regulatory_reference': 'Article 53(1)(d)',
                'provider_question': None,
            }
        ],
        'prioritized_provider_questions': [],
        'methods_and_limitations': [],
    }
    normalized = normalize_assessment_output(
        raw_output,
        checklist_items,
        [
            {
                'name': 'Jina Reader',
                'requested_url': 'https://huggingface.co/Qwen/Qwen3-8B',
                'http_status': 200,
                'usable': True,
                'urls': ['https://huggingface.co/Qwen/Qwen3-8B'],
                'snippet': 'Qwen/Qwen3-8B public summary',
                'full_text': 'Qwen2.5 appears elsewhere on the page, but this exact section remains about Qwen/Qwen3-8B.',
                'error': None,
                'metadata': {},
            }
        ],
        'Qwen/Qwen3-8B',
        link_checks={'https://huggingface.co/Qwen/Qwen3-8B': {'status_code': 200, 'final_url': 'https://huggingface.co/Qwen/Qwen3-8B', 'outcome': 'ok'}},
    )

    finding = next(item for item in normalized['findings'] if item['item_id'] == 'DT-03')
    assert finding['status'] == 'Evidence found'
    assert finding['quote'] is not None
    assert finding['failure_reason'] is None


def test_assessment_normalizer_rejects_variant_quote():
    checklist_items = load_checklist()
    raw_output = {
        'summary': {'headline': 'draft', 'major_gaps': [], 'status_counts': {'Evidence found': 1, 'Partial': 0, 'Not found': 0, 'Not checked': 0}},
        'findings': [
            {
                'item_id': 'DT-03',
                'status': 'Evidence found',
                'summary': 'Variant quote should be rejected.',
                'quote': 'Qwen2.5-7B public summary',
                'source_name': 'Jina Reader',
                'source_url': 'https://huggingface.co/Qwen/Qwen3-8B',
                'source_section': 'Model card',
                'model_applicability': 'Qwen2.5',
                'link_check_url': 'https://huggingface.co/Qwen/Qwen3-8B',
                'searched_sources': ['https://huggingface.co/Qwen/Qwen3-8B'],
                'failure_reason': None,
                'regulatory_reference': 'Article 53(1)(d)',
                'provider_question': None,
            }
        ],
        'prioritized_provider_questions': [],
        'methods_and_limitations': [],
    }
    normalized = normalize_assessment_output(
        raw_output,
        checklist_items,
        [
            {
                'name': 'Jina Reader',
                'requested_url': 'https://huggingface.co/Qwen/Qwen3-8B',
                'http_status': 200,
                'usable': True,
                'urls': ['https://huggingface.co/Qwen/Qwen3-8B'],
                'snippet': 'Qwen/Qwen3-8B public summary',
                'full_text': 'Qwen/Qwen3-8B public summary and training-content description.',
                'error': None,
                'metadata': {},
            }
        ],
        'Qwen/Qwen3-8B',
        link_checks={'https://huggingface.co/Qwen/Qwen3-8B': {'status_code': 200, 'final_url': 'https://huggingface.co/Qwen/Qwen3-8B', 'outcome': 'ok'}},
    )

    finding = next(item for item in normalized['findings'] if item['item_id'] == 'DT-03')
    assert finding['status'] == 'Partial'
    assert finding['quote'] is None
    assert 'different model variant' in finding['failure_reason']
