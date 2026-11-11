from __future__ import annotations

import time

import pytest

from intake import assessment, pipeline_view as pv
from intake.adapters import ToolOutcome
from intake.graph import APP, _emit, _parallel
from intake.regulatory import RetrievalHit, RetrievalResult
from intake.runner import iter_run
from intake.settings import LiveSettings
from test_intake import _fake_assessment

NODES = [
    'validate_and_normalize_target', 'load_checklist', 'collect_sources',
    'retrieve_regulatory_contexts', 'assess_evidence', 'render_report',
]
OFFLINE = {'live_mode': False, 'provider': None, 'model': 'Qwen/Qwen3-8B'}
HF_URL = 'https://huggingface.co/Qwen/Qwen3-8B'


def _collect(payload):
    events, result = [], None
    for kind, item in iter_run(payload):
        if kind == 'event':
            events.append((time.perf_counter(), item))
        else:
            result = item
    return events, result


def _kinds(events, kind):
    return [e for _, e in events if e['type'] == kind]


def _index(events, node, kind):
    return next(i for i, (_, e) in enumerate(events) if e['node'] == node and e['type'] == kind)


def _live_fakes(monkeypatch, *, search_delay=(0, 0), hub_delay=0, read_delay=0, retrieve_delay=0):
    settings = LiveSettings('tvly', ['huggingface.co'], 5, 5, 5, 42, 5)

    def search(*args, query='', **kwargs):
        time.sleep(search_delay[0] if 'official documentation' in query else search_delay[1])
        return ToolOutcome('Tavily Search', 'https://api.tavily.com/search', 200, True,
                           urls=[HF_URL, HF_URL + '/blob/main/README.md'], metadata={'query': query})

    def reader(url, timeout_seconds, allowed_domains=None):
        time.sleep(read_delay)
        return ToolOutcome('Jina Reader', url, 502 if 'README' in url else 200, 'README' not in url,
                           error='Jina Reader returned HTTP 502.' if 'README' in url else None)

    def hub(*args, **kwargs):
        time.sleep(hub_delay)
        return ToolOutcome('Hugging Face Hub', 'https://huggingface.co/api/models/Qwen/Qwen3-8B', 200, True,
                           metadata={'author': 'Qwen'})

    def retrieve(item_ids, config, timeout_seconds=15.0, top_k=3):
        time.sleep(retrieve_delay)
        hit = RetrievalHit('c', 'Article 53(1)(b)', 'https://eur-lex.europa.eu/x', 0.9, 'v', 'excerpt')
        return {i: RetrievalResult('q', i, [hit], []) for i in item_ids}

    monkeypatch.setattr('intake.graph.load_live_settings', lambda: settings)
    monkeypatch.setattr('intake.graph.load_rag_config_from_env', lambda: {'OPENAI_API_KEY': 'k'})
    monkeypatch.setattr('intake.graph.tavily_search_urls', search)
    monkeypatch.setattr('intake.graph.jina_reader_extract', reader)
    monkeypatch.setattr('intake.graph.hugging_face_model_metadata', hub)
    monkeypatch.setattr('intake.graph.retrieve_regulatory_context_batch', retrieve)
    monkeypatch.setattr('intake.graph.call_openai_assessment', _fake_assessment('DT-03', HF_URL))
    monkeypatch.setattr('intake.graph.check_public_urls', lambda urls, timeout_seconds, max_checks=6, allowed_domains=None:
                        {u: {'url': u, 'status_code': 200, 'outcome': 'ok'} for u in urls})


LIVE = {'live_mode': True, 'provider': None, 'model': 'Qwen/Qwen3-8B'}


# --- topology ---------------------------------------------------------------

def test_topology_is_read_from_the_compiled_graph():
    topo = pv.topology()
    assert topo.order == (pv.START, *NODES, pv.END)
    assert ('validate_and_normalize_target', 'render_report', True) in topo.edges
    assert ('load_checklist', 'collect_sources', False) in topo.edges
    assert ('load_checklist', 'retrieve_regulatory_contexts', False) in topo.edges
    assert ('collect_sources', 'assess_evidence', False) in topo.edges
    assert ('retrieve_regulatory_contexts', 'assess_evidence', False) in topo.edges
    layers = topo.layers
    assert layers['collect_sources'] == layers['retrieve_regulatory_contexts']   # they can run side by side
    assert layers['assess_evidence'] == layers['collect_sources'] + 1
    assert topo.arcs[0] == ('validate_and_normalize_target', 'render_report', True)   # widest first
    assert topo.bypasses == (('validate_and_normalize_target', 'render_report'),)
    assert {(s, t) for s, t, _ in topo.arcs} == {
        ('validate_and_normalize_target', 'render_report'),
        ('load_checklist', 'retrieve_regulatory_contexts'),
        ('collect_sources', 'assess_evidence'),
    }


# --- events and equivalence with invoke -----------------------------------

def test_offline_run_respects_dependencies_and_matches_invoke():
    events, result = _collect(OFFLINE)
    assert sorted(e['node'] for e in _kinds(events, 'node_started')) == sorted(NODES)
    assert all(e['status'] == 'done' for e in _kinds(events, 'node_finished'))
    topo = pv.topology()
    for source, target, _ in topo.edges:
        if source in NODES and target in NODES:   # a node starts only after everything feeding it has finished
            assert _index(events, source, 'node_finished') < _index(events, target, 'node_started')
    assert result == APP.invoke(OFFLINE)
    assert result['report_markdown']


def test_events_do_not_leak_into_invoke_or_direct_calls():
    _emit({'type': 'step', 'node': 'x'})  # outside a running graph: must not raise
    assert 'report_markdown' in APP.invoke(OFFLINE)


def test_failure_goes_straight_to_the_renderer():
    events, result = _collect({'live_mode': False, 'provider': None, 'model': 'not-a-valid-id'})
    assert [e['node'] for e in _kinds(events, 'node_started')] == ['validate_and_normalize_target', 'render_report']
    assert result['error'].startswith('For Hugging Face targets')
    assert not result.get('report_markdown')

    run = pv.PipelineRun(pv.topology())
    for t, (_, event) in enumerate(events):
        run.apply(event, t)
    run.finish()
    assert run.status == 'error'
    assert {n: run.nodes[n].status for n in NODES[1:5]} == dict.fromkeys(NODES[1:5], 'skipped')
    assert run.nodes['validate_and_normalize_target'].status == 'error'
    assert ('validate_and_normalize_target', 'render_report') in run.taken
    html = pv.render_html(run)
    assert 'data-state="taken"' in html and 'Stopped with an error' in html


def test_live_run_without_keys_skips_assessment_and_reports_the_original_error(monkeypatch):
    monkeypatch.setattr('intake.graph.load_live_settings',
                        lambda: (_ for _ in ()).throw(ValueError('Missing TAVILY_API_KEY for live mode.')))
    events, result = _collect(LIVE)
    started = [e['node'] for e in _kinds(events, 'node_started')]
    assert 'assess_evidence' not in started and started[-1] == 'render_report'
    assessment_event = next(e for e in _kinds(events, 'node_finished') if e['node'] == 'assess_evidence')
    assert assessment_event['status'] == 'skipped'
    assert result['error'] == 'Missing TAVILY_API_KEY for live mode.'
    assert result == APP.invoke(LIVE)

    run = pv.replay(pv.topology(), [(i, e) for i, (_, e) in enumerate(events)])
    assert run.nodes['collect_sources'].status == 'error' and run.nodes['assess_evidence'].status == 'skipped'
    assert not run.taken   # nothing was bypassed: collection ran, assessment skipped itself


# --- live run with mocked services -------------------------------------------

def test_live_run_streams_tool_steps_from_worker_threads(monkeypatch):
    _live_fakes(monkeypatch)
    events, result = _collect(LIVE)
    assert result['report_markdown']
    finished = [(i, e) for i, (_, e) in enumerate(events)
                if e['type'] == 'step' and e['node'] == 'collect_sources' and e['status'] != 'start']
    labels = [e['label'] for _, e in finished]
    assert sorted(label.split(' · ')[0] for label in labels) == [
        'Hugging Face Hub', 'Jina Reader', 'Jina Reader', 'Tavily search', 'Tavily search']
    jina = [(i, e) for i, e in finished if e['label'].startswith('Jina')]
    assert all(e['status'] == 'warn' and 'HTTP 502' in e['detail'] for _, e in jina if 'README' in e['label'])
    first_page = min(i for i, _ in jina)   # pages are read only after the searches they depend on
    assert all(i < first_page for i, e in finished if e['label'].startswith(('Tavily', 'Hugging')))
    all_labels = {e['label'] for e in _kinds(events, 'step')}
    assert any(label.startswith('Embeddings + Pinecone') for label in all_labels)
    assert {'OpenAI structured assessment', 'Verify quotes against sources'} <= all_labels

    run = pv.replay(pv.topology(), [(i, e) for i, (_, e) in enumerate(events)])
    assert run.status == 'done' and all(n.status == 'done' for n in run.nodes.values())
    assert any(s.status == 'warn' for s in run.nodes['collect_sources'].steps)
    assert run.nodes['collect_sources'].data[0][0] == 'sources'


def test_branches_overlap_and_source_order_does_not_depend_on_timing(monkeypatch):
    # The primary search is the slowest, so a timing-ordered result would put it last.
    _live_fakes(monkeypatch, search_delay=(0.50, 0.05), hub_delay=0.10, read_delay=0.15, retrieve_delay=0.65)
    started = time.perf_counter()
    events, result = _collect(LIVE)
    wall = time.perf_counter() - started

    outcomes = result['tool_outcomes']
    assert [o.name for o in outcomes] == ['Tavily Search', 'Tavily Search', 'Hugging Face Hub', 'Jina Reader', 'Jina Reader']
    assert 'official documentation' in outcomes[0].metadata['query']
    assert [o.requested_url for o in outcomes[3:]] == [HF_URL, HF_URL + '/blob/main/README.md']

    # collect_sources and retrieve_regulatory_contexts ran at the same time
    assert _index(events, 'retrieve_regulatory_contexts', 'node_started') < _index(events, 'collect_sources', 'node_finished')
    assert _index(events, 'collect_sources', 'node_started') < _index(events, 'retrieve_regulatory_contexts', 'node_finished')
    # and they really overlapped: 0.50 + 0.15 for collection and 0.65 for retrieval would be 1.3 s one after another
    assert wall < 1.0

    run = pv.replay(pv.topology(), [(t - events[0][0], e) for t, e in events])
    assert run.busy_seconds - run.elapsed >= 0.5          # the view reports the time the overlap saved
    assert 'Parallel branches saved' in pv.render_html(run)


def test_parallel_helper_keeps_input_order_and_reraises_the_first_failure():
    assert _parallel([lambda: (time.sleep(0.1), 'slow')[1], lambda: 'fast']) == ['slow', 'fast']

    def boom():
        raise RuntimeError('first')

    def other():
        raise ValueError('second')

    with pytest.raises(RuntimeError, match='first'):
        _parallel([boom, other])


def test_link_checks_run_concurrently_in_order_with_dedupe_and_cap(monkeypatch):
    calls = []

    def fake_check(url, timeout_seconds, allowed_domains=None):
        calls.append(url)
        time.sleep(0.2)
        return {'url': url, 'outcome': 'ok'}

    monkeypatch.setattr(assessment, 'check_public_url', fake_check)
    urls = [f'https://example.com/{n}' for n in range(8)]
    started = time.perf_counter()
    results = assessment.check_public_urls(urls[:3] + urls[:2] + urls[3:], timeout_seconds=1, max_checks=5)
    assert list(results) == urls[:5] and len(calls) == 5
    assert time.perf_counter() - started < 0.6   # five 0.2 s checks, not one second
    assert assessment.check_public_urls([], timeout_seconds=1) == {}


# --- reducer and rendering -------------------------------------------------------

def test_reducer_handles_idle_running_parallel_and_unknown_events():
    topo = pv.topology()
    run = pv.PipelineRun(topo)
    assert run.status == 'idle' and run.element_state(pv.START) == 'pending'
    run.apply({'type': 'node_started', 'node': 'collect_sources'}, 0.0)
    run.apply({'type': 'node_started', 'node': 'retrieve_regulatory_contexts'}, 0.1)   # a sibling, not a successor
    run.apply({'type': 'step', 'node': 'unknown_node', 'label': 'x', 'status': 'done'}, 0.1)  # ignored
    assert run.nodes['retrieve_regulatory_contexts'].status == 'running'
    assert run.nodes['collect_sources'].status == 'running'     # starting the sibling must not skip it
    assert run.nodes['validate_and_normalize_target'].status == 'skipped'   # earlier layers are over
    assert run.current() == ['collect_sources', 'retrieve_regulatory_contexts']
    assert 'Running in parallel: Collect sources, Regulatory context' in pv.render_html(run)
    run.finish(1.0)   # an interrupted run is closed as failed, never left spinning
    assert run.nodes['collect_sources'].status == 'error' and run.status == 'error'


def test_failure_route_is_marked_only_when_the_skipped_stages_are_between():
    topo = pv.topology()
    run = pv.PipelineRun(topo)
    for name in ('validate_and_normalize_target', 'load_checklist', 'collect_sources', 'retrieve_regulatory_contexts'):
        run.apply({'type': 'node_started', 'node': name}, 0)
        run.apply({'type': 'node_finished', 'node': name, 'status': 'done', 'data': []}, 1)
    run.apply({'type': 'node_finished', 'node': 'assess_evidence', 'status': 'skipped', 'data': []}, 2)
    run.apply({'type': 'node_started', 'node': 'render_report'}, 3)
    assert not run.taken   # validation passed and stages ran, so the validation failure route was not used


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


def test_parallel_neighbours_get_a_marker_instead_of_an_arrow():
    html = pv.render_html(pv.PipelineRun(pv.topology()))
    assert html.count('pv-edge pv-par') == 1 and 'data-kind="parallel"' in html and 'data-kind="error"' in html


def test_layer_gate_shows_parallel_siblings_running_together_without_changing_events():
    topo = pv.topology()
    sequential = [   # what an instant offline run produces: one branch is over before the other starts
        {'type': 'node_started', 'node': 'collect_sources'},
        {'type': 'step', 'node': 'collect_sources', 'label': 'a', 'status': 'done'},
        {'type': 'node_finished', 'node': 'collect_sources', 'status': 'done', 'data': []},
        {'type': 'node_started', 'node': 'retrieve_regulatory_contexts'},
        {'type': 'node_finished', 'node': 'retrieve_regulatory_contexts', 'status': 'done', 'data': []},
    ]
    gate, run, shown, both_running = pv.LayerGate(topo), pv.PipelineRun(topo), [], False
    for t, event in enumerate(sequential):
        for ready, ready_t in gate.push(event, t):
            run.apply(ready, ready_t)
            shown.append(ready)
            both_running |= len(run.current()) == 2
    shown.extend(e for e, _ in gate.flush())
    assert both_running
    assert sorted(map(str, shown)) == sorted(map(str, sequential))   # nothing added or dropped
    per_node = lambda events, n: [e['type'] for e in events if e['node'] == n]   # noqa: E731
    assert per_node(shown, 'collect_sources') == per_node(sequential, 'collect_sources')   # order kept per node


def test_layer_gate_flushes_a_node_whose_sibling_never_starts():
    gate = pv.LayerGate(pv.topology())
    assert gate.push({'type': 'node_started', 'node': 'collect_sources'}, 0)
    assert gate.push({'type': 'node_finished', 'node': 'collect_sources', 'status': 'done', 'data': []}, 1) == []
    assert [e['type'] for e, _ in gate.flush()] == ['node_finished']
    assert gate.flush() == []


@pytest.mark.parametrize('seconds, text', [(None, ''), (0.001, ''), (0.25, '250 ms'), (3.14, '3.1 s')])
def test_seconds_format(seconds, text):
    assert pv._seconds(seconds) == text


def test_link_check_step_counts_off_allowlist_pages_as_reachable():
    from intake.graph import _describe_link_checks
    result = {
        'a': {'status_code': 200, 'outcome': 'ok'},
        'b': {'status_code': 200, 'outcome': 'blocked'},
        'c': {'status_code': 401, 'outcome': 'blocked'},
    }
    status, detail = _describe_link_checks(result)
    assert status == 'done'
    assert detail == '2 of 3 reachable · 1 off the approved domains · 1× blocked'


def test_hugging_face_run_ignores_the_hosted_provider_domains_from_the_environment(monkeypatch):
    _live_fakes(monkeypatch)
    monkeypatch.setattr('intake.graph.load_live_settings',
                        lambda: LiveSettings('tvly', ['openai.com', 'cdn.openai.com'], 5, 5, 5, 42, 5))
    seen = {}

    def checks(urls, timeout_seconds, max_checks=6, allowed_domains=None):
        seen['allowed'] = allowed_domains
        return {u: {'url': u, 'status_code': 200, 'outcome': 'ok'} for u in urls}

    monkeypatch.setattr('intake.graph.check_public_urls', checks)
    _, result = _collect(LIVE)
    assert result['provider_domains'] == ['huggingface.co']
    assert seen['allowed'] == ['huggingface.co']
