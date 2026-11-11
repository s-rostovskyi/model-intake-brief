from __future__ import annotations

from pathlib import Path

import pytest

from intake import dashboard as dash
from intake.report_review import read_findings, sections

SAMPLES = Path(__file__).resolve().parent.parent / 'samples'


def overview(name: str, model: str, **kwargs):
    report = (SAMPLES / name).read_text(encoding='utf-8')
    parts = sections(report)
    findings = read_findings(parts['Findings'], model)
    return report, parts, findings, dash.build_overview(report, findings, parts, model, kwargs.get('elapsed', 8.9))


def test_core_checks_exclude_context_and_counts_add_up():
    _, _, findings, o = overview('gpt-5.2.md', 'gpt-5.2-2025-12-11')
    assert len(o.core) + len(o.context) == len(findings)
    assert all(not f.id.startswith('CT-') for f in o.core)
    assert sum(o.counts.values()) == len(o.core)
    assert sum(o.context_counts.values()) == len(o.context) == 3
    assert [label for label, _ in o.groups][:2] == ['Identity', 'Distribution and licence']
    assert sum(len(rows) for _, rows in o.groups) == len(o.core)


def test_questions_put_not_found_before_partial_and_skip_context():
    _, _, _, o = overview('qwen3-8b.md', 'Qwen/Qwen3-8B')
    statuses = [f.status for f, _ in o.questions]
    assert statuses == sorted(statuses, key=lambda s: {'Not found': 0, 'Partial': 1}[s])
    assert all(not f.id.startswith('CT-') for f, _ in o.questions)
    assert o.question_total == len(o.questions) == o.counts['Partial'] + o.counts['Not found']


def test_trust_numbers_come_from_the_report():
    _, _, findings, o = overview('gpt-5.2.md', 'gpt-5.2-2025-12-11')
    assert o.quotes_ok == sum(1 for f in findings if f.fields.get('Quote'))
    assert o.quotes_unsupported >= 1          # this sample has unsupported quotes
    assert o.links_total >= 1 and o.links_ok == o.links_total
    assert 0 <= o.cited_official <= o.cited
    assert o.tokens == (4286, 912) and o.cost == '$0.0073'


def test_failed_link_checks_are_counted_with_their_reason():
    report = ('- Exact model ID: `m`\n- Model provider: acme\n- Target kind: hosted\n\n### Findings\n'
              '- ID-01 - Provider and model identity: Evidence found\n  - Quote: q\n  - Source URL: https://acme.com/a\n'
              '  - Link check: https://acme.com/a -> HTTP 403 (blocked)\n')
    parts = sections(report)
    o = dash.build_overview(report, read_findings(parts['Findings'], 'm'), parts, 'm', None)
    assert (o.links_ok, o.links_total) == (0, 1)
    assert o.link_issues == {'HTTP 403 (blocked)': 1}
    assert 'HTTP 403 (blocked)' in dash.render_html(o)
    assert 'data-tone="bad"' in dash.render_html(o)


@pytest.mark.parametrize('url, provider, model, kind, expected', [
    ('https://cdn.openai.com/x.pdf', 'openai', 'gpt-5.2', 'hosted', True),
    ('https://community.openai.com/t/1', 'openai', 'gpt-5.2', 'hosted', False),
    ('https://example.org/openai-news', 'openai', 'gpt-5.2', 'hosted', False),
    ('https://huggingface.co/Qwen/Qwen3-8B', 'Qwen', 'Qwen/Qwen3-8B', 'hugging_face', True),
    ('https://huggingface.co/other/Qwen3-8B', 'Qwen', 'Qwen/Qwen3-8B', 'hugging_face', False),
])
def test_official_source_heuristic(url, provider, model, kind, expected):
    assert dash._official(url, provider, model, kind) is expected


def test_regulatory_rows_only_list_open_gaps():
    _, _, findings, o = overview('gpt-5.2.md', 'gpt-5.2-2025-12-11')
    gap_ids = {f.id for f in o.core if f.status != 'Evidence found'}
    assert o.provisions and len(o.provisions) <= dash.TOP_PROVISIONS
    assert all(set(ids) <= gap_ids for _, ids in o.provisions)
    assert all('(ai-act-' not in title for title, _ in o.provisions)


def test_render_escapes_report_text_and_offline_has_no_verdict():
    _, _, _, o = overview('qwen3-8b.md', 'Qwen/Qwen3-8B')
    o.questions[0][0].criterion = '<img src=x onerror=alert(1)>'
    o.questions[0] = (o.questions[0][0], '<script>alert(1)</script>')
    html = dash.render_html(o)
    assert '<script>' not in html and '<img' not in html
    off = dash.render_offline_html(0.03, '$0')
    assert 'Not assessed in offline preview' in off and 'core checks' not in off


def test_tone_thresholds():
    assert [dash._tone(p, 10) for p in (10, 9, 6, 5)] == ['good', 'good', 'warn', 'bad']
    assert dash._tone(0, 0) == 'neutral'


def test_page_off_the_allowlist_counts_as_reachable_but_is_called_out():
    report = ('- Exact model ID: `m`\n- Model provider: acme\n- Target kind: hosted\n\n### Findings\n'
              '- ID-01 - Provider and model identity: Evidence found\n  - Quote: q\n  - Source URL: https://acme.com/a\n'
              '  - Link check: https://acme.com/a -> HTTP 200 (blocked)\n'
              '- ID-02 - Release date: Evidence found\n  - Quote: q\n  - Source URL: https://acme.com/b\n'
              '  - Link check: https://acme.com/b -> HTTP 401 (blocked)\n'
              '- ID-03 - Base model dependencies: Evidence found\n  - Quote: q\n  - Source URL: https://acme.com/c\n'
              '  - Link check: https://acme.com/c -> HTTP 200 (ok)\n')
    parts = sections(report)
    o = dash.build_overview(report, read_findings(parts['Findings'], 'm'), parts, 'm', None)
    assert (o.links_ok, o.links_offlist, o.links_total) == (2, 1, 3)
    assert o.link_issues == {'HTTP 401 (blocked)': 1}
    html = dash.render_html(o)
    assert '2 of 3' in html and '1 answered but is off the approved domains' in html and '1 HTTP 401 (blocked)' in html
