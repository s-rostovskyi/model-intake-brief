from __future__ import annotations

import pytest

from intake import pipeline_view as pv
from intake.adapters import ToolOutcome
from intake.graph import APP, _emit
from intake.regulatory import RetrievalHit, RetrievalResult
from intake.runner import iter_run
from intake.settings import LiveSettings
from test_intake import _fake_assessment

NODES = [
    'validate_and_normalize_target', 'load_checklist', 'collect_sources',
    'retrieve_regulatory_contexts', 'assess_evidence', 'render_report',
]
OFFLINE = {'live_mode': False, 'provider': None, 'model': 'Qwen/Qwen3-8B'}


def _collect(payload):
    events, result = [], None
    for kind, item in iter_run(payload):
        if kind == 'event':
            events.append(item)
        else:
            result = item
    return events, result


def test_topology_is_read_from_the_compiled_graph():
    topo = pv.topology()
    assert topo.order == (pv.START, *NODES, pv.END)
    assert ('validate_and_normalize_target', 'render_report', True) in topo.edges
    assert ('collect_sources', 'render_report', True) in topo.edges
    # widest bypass first, so the arcs nest instead of crossing
    assert topo.bypasses == (('validate_and_normalize_target', 'render_report'), ('collect_sources', 'render_report'))


def test_offline_run_emits_every_node_in_order_and_matches_invoke():
    events, result = _collect(OFFLINE)
    started = [e['node'] for e in events if e['type'] == 'node_started']
    finished = [(e['node'], e['status']) for e in events if e['type'] == 'node_finished']
    assert started == NODES
    assert finished == [(name, 'done') for name in NODES]
    for index, name in enumerate(NODES):  # a node starts only after the previous one finished
        first = next(i for i, e in enumerate(events) if e['node'] == name)
        if index:
            previous_done = next(i for i, e in enumerate(events)
                                 if e['node'] == NODES[index - 1] and e['type'] == 'node_finished')
            assert previous_done < first
    assert result == APP.invoke(OFFLINE)
    assert result['report_markdown']


def test_events_do_not_leak_into_invoke_or_direct_calls():
    _emit({'type': 'step', 'node': 'x'})  # outside a running graph: must not raise
    assert 'report_markdown' in APP.invoke(OFFLINE)


def test_failure_goes_straight_to_the_renderer():
    events, result = _collect({'live_mode': False, 'provider': None, 'model': 'not-a-valid-id'})
    started = [e['node'] for e in events if e['type'] == 'node_started']
    assert started == ['validate_and_normalize_target', 'render_report']
    assert result['error'].startswith('For Hugging Face targets')
    assert not result.get('report_markdown')

    run = pv.PipelineRun(pv.topology())
    for t, event in enumerate(events):
        run.apply(event, t)
    run.finish()
    assert run.status == 'error'
    assert run.nodes['load_checklist'].status == 'skipped'
    assert run.nodes['validate_and_normalize_target'].status == 'error'
    assert ('validate_and_normalize_target', 'render_report') in run.taken
    assert ('collect_sources', 'render_report') not in run.taken
    html = pv.render_html(run)
    assert 'data-state="taken"' in html and 'Stopped with an error' in html


def test_live_run_streams_tool_steps_with_outcomes(monkeypatch):
    settings = LiveSettings('tvly', ['huggingface.co'], 5, 5, 5, 42, 5)
    url = 'https://huggingface.co/Qwen/Qwen3-8B'

    def search(*args, **kwargs):
        return ToolOutcome('Tavily Search', 'https://api.tavily.com/search', 200, True, urls=[url, url + '/blob/main/README.md'])

    def reader(url, timeout_seconds, allowed_domains=None):
        return ToolOutcome('Jina Reader', url, 502, False, error='Jina Reader returned HTTP 502.')

    def hub(*args, **kwargs):
        return ToolOutcome('Hugging Face Hub', 'https://huggingface.co/api/models/Qwen/Qwen3-8B', 200, True,
                           metadata={'author': 'Qwen'})

    def retrieve(item_ids, config, timeout_seconds=15.0, top_k=3):
        hit = RetrievalHit('c', 'Article 53(1)(b)', 'https://eur-lex.europa.eu/x', 0.9, 'v', 'excerpt')
        return {i: RetrievalResult('q', i, [hit], []) for i in item_ids}

    monkeypatch.setattr('intake.graph.load_live_settings', lambda: settings)
    monkeypatch.setattr('intake.graph.load_rag_config_from_env', lambda: {'OPENAI_API_KEY': 'k'})
    monkeypatch.setattr('intake.graph.tavily_search_urls', search)
    monkeypatch.setattr('intake.graph.jina_reader_extract', reader)
    monkeypatch.setattr('intake.graph.hugging_face_model_metadata', hub)
    monkeypatch.setattr('intake.graph.retrieve_regulatory_context_batch', retrieve)
    monkeypatch.setattr('intake.graph.call_openai_assessment', _fake_assessment('DT-03', url))
    monkeypatch.setattr('intake.graph.check_public_urls', lambda urls, timeout_seconds, max_checks=6, allowed_domains=None:
                        {u: {'url': u, 'status_code': 200, 'outcome': 'ok'} for u in urls})

    events, result = _collect({'live_mode': True, 'provider': None, 'model': 'Qwen/Qwen3-8B'})
    assert result['report_markdown']
    steps = [e for e in events if e['type'] == 'step' and e['node'] == 'collect_sources' and e['status'] != 'start']
    assert [e['label'].split(' · ')[0] for e in steps] == [
        'Tavily search', 'Tavily search', 'Hugging Face Hub', 'Jina Reader', 'Jina Reader']
    jina = [e for e in steps if e['label'].startswith('Jina')]
    assert all(e['status'] == 'warn' and 'HTTP 502' in e['detail'] for e in jina)
    labels = {e['label'] for e in events if e['type'] == 'step'}
    assert any(label.startswith('Embeddings + Pinecone') for label in labels)
    assert {'OpenAI structured assessment', 'Verify quotes against sources'} <= labels

    run = pv.replay(pv.topology(), list(enumerate(events)))
    assert run.status == 'done' and all(n.status == 'done' for n in run.nodes.values())
    assert any(s.status == 'warn' for s in run.nodes['collect_sources'].steps)
    assert run.nodes['collect_sources'].data[0][0] == 'sources'


def test_live_run_without_keys_ends_early_with_the_original_error(monkeypatch):
    monkeypatch.setattr('intake.graph.load_live_settings',
                        lambda: (_ for _ in ()).throw(ValueError('Missing TAVILY_API_KEY for live mode.')))
    events, result = _collect({'live_mode': True, 'provider': None, 'model': 'Qwen/Qwen3-8B'})
    assert [e['node'] for e in events if e['type'] == 'node_started'] == [
        'validate_and_normalize_target', 'load_checklist', 'collect_sources', 'render_report']
    assert result['error'] == 'Missing TAVILY_API_KEY for live mode.'
    assert result == APP.invoke({'live_mode': True, 'provider': None, 'model': 'Qwen/Qwen3-8B'})


def test_reducer_handles_idle_running_and_unknown_events():
    topo = pv.topology()
    run = pv.PipelineRun(topo)
    assert run.status == 'idle' and run.element_state(pv.START) == 'pending'
    run.apply({'type': 'node_started', 'node': 'validate_and_normalize_target'}, 0.0)
    run.apply({'type': 'step', 'node': 'unknown_node', 'label': 'x', 'status': 'done'}, 0.1)  # ignored
    assert run.status == 'running' and run.current() == 'validate_and_normalize_target'
    assert run.edge_state(pv.START, 'validate_and_normalize_target') == 'flow'
    run.finish(1.0)  # an interrupted run is closed as failed, never left spinning
    assert run.nodes['validate_and_normalize_target'].status == 'error' and run.status == 'error'


def test_render_escapes_user_supplied_text():
    run = pv.PipelineRun(pv.topology())
    run.apply({'type': 'node_started', 'node': 'collect_sources'}, 0)
    run.apply({'type': 'step', 'node': 'collect_sources', 'label': '<script>alert(1)</script>',
               'status': 'done', 'detail': '<img src=x onerror=1>'}, 0.1)
    run.apply({'type': 'node_finished', 'node': 'collect_sources', 'status': 'done',
               'data': [['model', '<b>x</b>']]}, 0.2)
    html = pv.render_html(run)
    assert '<script>' not in html and '<img' not in html and '<b>x</b>' not in html
    assert '&lt;script&gt;' in html


@pytest.mark.parametrize('seconds, text', [(None, ''), (0.001, ''), (0.25, '250 ms'), (3.14, '3.1 s')])
def test_seconds_format(seconds, text):
    assert pv._seconds(seconds) == text
