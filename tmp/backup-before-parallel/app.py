from __future__ import annotations

import base64
import re
from collections import Counter
from urllib.parse import urlencode
from html import escape
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import streamlit as st

from intake import pipeline_view as pv
from intake.graph import APP
from intake.runner import iter_run
from intake.report_review import (
    QA_GAPS, STATUSES, Finding, public_link, questions_for, read_findings, regulatory_rows, sections, source_rows,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_REPORTS = {
    'Qwen/Qwen3-8B': ROOT / 'samples' / 'qwen3-8b.md',
    'gpt-5.2-2025-12-11': ROOT / 'samples' / 'gpt-5.2.md',
}


def sample_keys() -> list[str]:
    return list(SAMPLE_REPORTS)


def sample_path(sample_key: str) -> Path:
    return SAMPLE_REPORTS[sample_key]


def load_saved_report(sample_key: str) -> str:
    return sample_path(sample_key).read_text(encoding='utf-8')


def build_run_payload(live_mode: bool, provider: str | None, model: str) -> dict[str, Any]:
    return {
        'live_mode': bool(live_mode),
        'provider': provider.strip() if isinstance(provider, str) and provider.strip() else None,
        'model': model.strip(),
    }


def run_graph(live_mode: bool, provider: str | None, model: str) -> dict[str, Any]:
    return APP.invoke(build_run_payload(live_mode, provider, model))


# Seconds to hold each kind of event on screen when "Demo pace" is on (offline runs finish in milliseconds).
PACE = {'node_started': 0.55, 'step': 0.3, 'node_finished': 0.35}
PACE_NOTE = 'Demo pace: steps are slowed on purpose'
REPLAY_NOTE = 'Replay of recorded events, no API calls'


def _animate(view, topo, events, *, note: str, paced: bool):
    """Replay recorded events, redrawing the diagram after each one. Returns the finished PipelineRun."""
    run = pv.PipelineRun(topo, note)
    view.html(pv.render_html(run))
    previous = 0.0
    for t, event in events:
        wait = min(max(t - previous, 0.0), 1.0)   # keep long real waits short, they are only being replayed
        if paced:
            wait = max(wait, PACE.get(event.get('type'), 0.0))
        previous = t
        if wait:
            sleep(wait)
        run.apply(event, t)
        view.html(pv.render_html(run))
    if events:
        run.finish(events[-1][0])
        view.html(pv.render_html(run))
    return run


def _download_name(model: str, live_mode: bool) -> str:
    suffix = 'live' if live_mode else 'offline'
    slug = ''.join(ch.lower() if ch.isalnum() else '-' for ch in model).strip('-') or 'report'
    return f'{slug}-{suffix}.md'



def _report_field(report: str, label: str) -> str:
    prefix = f'- {label}: '
    return next((line[len(prefix):].strip().strip('`') for line in report.splitlines()
                 if line.startswith(prefix)), 'Not recorded')


def _render_dashboard(report: str, mode: str, elapsed: float | None,
                      counts: Counter, sources: list[dict[str, str]]) -> None:
    usage = re.search(r'Aggregate assessment usage: input tokens (\d+), output tokens (\d+)', report)
    cost = 'Not recorded'
    if usage:
        inputs, outputs = map(int, usage.groups())
        cost = f'${(inputs * 0.75 + outputs * 4.50) / 1_000_000:.4f}'
    elif mode == 'Offline preview':
        cost = '$0'
    with st.container(border=True, key='intake-overview'):
        st.header('Report overview')
        st.markdown('**Target model**')
        st.code(_report_field(report, 'Exact model ID'), language=None)
        if mode == 'Offline preview':
            st.caption('Provider and platform are not applicable in offline preview.')
        else:
            st.caption(f"{mode} · Provider: {_report_field(report, 'Model provider')} · "
                   f"Platform: {_report_field(report, 'Hosting platform')}")
        metrics = [
            (status, str(counts[status]) if mode != 'Offline preview' else 'n/a')
            for status in STATUSES
        ]
        metrics.extend([
            ('Run time', f'{elapsed:.1f} s' if elapsed is not None else 'Not recorded'),
            ('Est. assessment cost', cost),
        ])
        status_styles = dict(zip(STATUSES, ('found', 'partial', 'missing', 'unchecked'), strict=True))
        cards = []
        for label, value in metrics:
            title = escape(label)
            if label in STATUSES and mode == 'Saved live report':
                query = urlencode({'report': _report_field(report, 'Exact model ID'), 'status': label})
                title = f'<a href="?{escape(query)}#findings">{title}</a>'
            style = status_styles.get(label, 'metric')
            cards.append(f'<div class="intake-metric" data-status="{style}">'
                         f'<dt>{title}</dt><dd>{escape(value)}</dd></div>')
        cards = ''.join(cards)
        st.html(f'<dl class="intake-metrics" aria-label="Report metrics">{cards}</dl>')
        st.caption('Cost is in USD at standard uncached OpenAI rates. It covers assessment tokens only. '
                   'Embeddings, search, extraction, and storage are excluded.')
        with st.expander('Run details and data sources'):
            st.write('Assessment model:', 'gpt-5.4-mini' if usage else 'Not recorded' if mode != 'Offline preview' else 'Not used')
            st.write('Run date:', _report_field(report, 'Run date'))
            st.write('Checklist version:', _report_field(report, 'Checklist version'))
            if usage:
                st.write(f'Assessment tokens: {inputs:,} input / {outputs:,} output')
            _show_sources(sources)


STATUS_SYMBOLS = {'Evidence found': '✓', 'Partial': '◐', 'Not found': '×', 'Not checked': '–'}


def _show_sources(rows: list[dict[str, str]]) -> None:
    if not rows:
        st.caption('No source requests were recorded.')
    for row in rows:
        st.markdown(f"**{row['tool']}** · {row['outcome']}")
        if public_link(row['url']):
            st.markdown(f"<{row['url']}>")
        st.caption(row['use'])


def _review_export(parts: dict[str, str], findings: list[Finding],
                   model: str, sources: list[dict[str, str]]) -> str:
    counts = Counter(f.status for f in findings)
    summary = re.sub(
        r'^- (Evidence found|Partial|Not found|Not checked): \d+$',
        lambda match: f'- {match[1]}: {counts[match[1]]}', parts.get('Summary', ''), flags=re.M
    )
    summary = '\n'.join(
        line for line in summary.splitlines()
        if not line.removeprefix('  - ').startswith(QA_GAPS)
    )
    if any(f.fields.get('Model applicability', '').startswith('Other model') for f in findings):
        summary = summary.split('- Major gaps:', 1)[0]
        summary = '\n'.join(line for line in summary.splitlines() if not line.startswith('- Headline:'))
    header = parts['Header'].replace('Target kind: hugging_face', 'Model type: Hugging Face repository')
    header = header.replace('Target kind: hosted', 'Model type: Hosted by provider')
    chunks = [header.strip(), '### Summary', summary.strip(), '### Source collection']
    for row in sources:
        link = f" <{row['url']}>" if public_link(row['url']) else ''
        chunks.append(f"- {row['tool']} · {row['outcome']} · {row['use']}{link}")
    chunks.append('\n### Findings')
    for f in findings:
        chunks.append(f'- {f.id} - {f.criterion}: {f.status}')
        for label in ('Evidence sufficient', 'Quote', 'Source ID', 'Source', 'Source URL',
                      'Source section', 'Model applicability', 'Link check', 'Regulatory reference'):
            value = f.fields.get(label)
            if value and (label != 'Source URL' or public_link(value)):
                chunks.append(f'  - {label}: {value}')
        if f.gap and f.status != 'Evidence found':
            chunks.append(f'  - Gap: {_reader_gap(f)}')
    chunks.extend(['\n### Prioritised provider questions',
                   '\n'.join(f'- {q}' for q in questions_for(findings, model)) or '- None generated.',
                   '\n### Context only',
                   '\n'.join(f'- {f.id} · {f.criterion} · {f.status}' for f in findings if f.id.startswith('CT-')),
                   '\n### Methods and limitations', parts.get('Methods and limitations', '').strip()])
    checks = [f'- {f.id}: {f.gap}' for f in findings if f.gap.startswith(QA_GAPS)]
    chunks.extend(checks)
    chunks.append('\n### Regulatory context')
    chunks.append('Match score ranks retrieved text for a query. It is not an evidence confidence or compliance score.')
    provisions, blockers = regulatory_rows(parts.get('Regulatory context', ''))
    for title, record in provisions.items():
        fields = record['fields']
        chunks.append(f"- {title} · {', '.join(record['items'])}")
        for label in ('Official URL', 'Document version', 'Excerpt'):
            if fields.get(label):
                chunks.append(f'  - {label}: {fields[label]}')
        if record['scores']:
            chunks.append(f"  - Best match score: {max(record['scores']):.2f}")
    chunks.extend(f'- {blocker}' for blocker in blockers)
    return '\n'.join(chunks) + '\n'


def _reader_gap(finding: Finding) -> str:
    if finding.gap.startswith(QA_GAPS):
        return 'Public evidence for this item could not be verified. Review the source or ask the provider.'
    return finding.gap or 'No supporting public evidence was established in this run.'


def _render_report(report: str, filename: str, mode: str, elapsed: float | None = None) -> None:
    parts = sections(report)
    model = _report_field(report, 'Exact model ID')
    findings = read_findings(parts.get('Findings', ''), model)
    sources = source_rows(parts.get('Source collection', ''), findings, model)
    counts = Counter(f.status for f in findings)
    if mode == 'Offline preview':
        st.info('This run used fixture data. Turn on Live run to assess public sources.', title='Offline preview')
    _render_dashboard(report, mode, elapsed, counts, sources)
    export = _review_export(parts, findings, model, sources)
    st.download_button('Download Markdown', data=export, file_name=filename,
                       mime='text/markdown', on_click='ignore')
    st.header('Report details')
    st.markdown('[Summary](#summary) · [Findings](#findings) · [Provider questions](#provider-questions) · [Regulatory context](#regulatory-context)')
    st.markdown('### Summary')
    target_kind = _report_field(report, 'Target kind')
    model_type = {'hugging_face': 'Hugging Face repository', 'hosted': 'Hosted by provider'}.get(target_kind)
    if model_type:
        st.caption(f'Model type · {model_type}')
    gaps = re.search(r'^- Major gaps:\s*\n((?:  - .+\n?)+)', parts.get('Summary', ''), re.M)
    if gaps and not any(f.fields.get('Model applicability', '').startswith('Other model') for f in findings):
        with st.expander('Major gaps'):
            st.markdown(gaps[1])
    for status in STATUSES:
        st.markdown(f'- **{status}** · {counts[status] if mode != "Offline preview" else "n/a"}')
    st.markdown('### Findings')
    choices = ['All statuses', *STATUSES]
    requested = st.query_params.get('status')
    selected = st.radio('Filter by status', choices, horizontal=True,
                        index=choices.index(requested) if requested in choices else 0,
                        format_func=lambda status: f'{status} ({counts[status]})' if status in counts else status)
    search = st.text_input('Search findings', placeholder='Search by ID, criterion, or evidence', type='search')
    matches = [f for f in findings if (selected == choices[0] or selected == f.status)
               and search.casefold() in ' '.join([f.id, f.criterion, *f.fields.values()]).casefold()]
    st.caption(f'Showing {len(matches)} of {len(findings)} items')
    for f in matches:
        with st.expander(f'{f.id} · {f.criterion} · {STATUS_SYMBOLS[f.status]} {f.status}'):
            st.markdown(f"**Evidence sufficient**  \n{f.fields.get('Evidence sufficient', f.criterion)}")
            st.markdown('**Evidence**')
            st.write(f.fields.get('Quote') or 'No verified excerpt.')
            if f.fields.get('Source'):
                st.caption(f.fields['Source'])
            if public_link(f.fields.get('Source URL', '')):
                st.markdown(f"<{f.fields['Source URL']}>")
            if f.fields.get('Source section'):
                st.caption(f.fields['Source section'])
            st.write(f"**Model applicability** · {f.fields.get('Model applicability', 'Not established')}")
            if f.fields.get('Link check'):
                st.write(f"**Link check** · {f.fields['Link check']}")
            if f.status != 'Evidence found':
                st.write(f'**Gap** · {_reader_gap(f)}')
            if f.fields.get('Regulatory reference'):
                st.caption(f"Regulatory reference · {f.fields['Regulatory reference']}")

    st.markdown('### Provider questions')
    questions = questions_for(findings, model)
    if questions:
        with st.popover('Copy questions'):
            st.caption('Use the copy icon in the text box to copy all questions.')
            st.code('\n'.join(questions), language=None)
        st.download_button('Download questions', '\n'.join(questions), 'provider-questions.txt', 'text/plain',
                           on_click='ignore')
        for question in questions:
            st.write(f'- {question}')
    else:
        st.caption('No provider questions were generated for this report.')
    with st.expander('Context only · training resources'):
        for f in findings:
            if f.id.startswith('CT-'):
                st.write(f'{f.id} · {f.criterion} · {f.status}')
    with st.expander('Source collection'):
        st.markdown('### Source collection')
        _show_sources(sources)
    with st.expander('Methods and limitations'):
        st.markdown('### Methods and limitations')
        st.markdown(parts.get('Methods and limitations', '').strip() or 'No limitations recorded.')
        for f in findings:
            if f.gap.startswith(QA_GAPS):
                st.write(f'{f.id} · {f.gap}')
    st.markdown('### Regulatory context')
    st.caption('Match score ranks retrieved text for a query. It is not an evidence confidence or compliance score.')
    provisions, blockers = regulatory_rows(parts.get('Regulatory context', ''))
    if not provisions:
        st.caption('No regulatory references were returned.')
    for title, record in provisions.items():
        with st.expander(f"{title} · {', '.join(record['items'])}"):
            fields = record['fields']
            st.write(fields.get('Excerpt', ''))
            st.caption(fields.get('Document version', ''))
            if public_link(fields.get('Official URL', '')):
                st.markdown(f"<{fields['Official URL']}>")
            if record['scores']:
                st.caption(f"Best match score · {max(record['scores']):.2f}")
    if blockers:
        with st.expander('Retrieval limitations'):
            for blocker in blockers:
                st.write(blocker)



def _render_brand() -> None:
    artwork = base64.b64encode((ROOT / 'assets' / 'intake-hero-topography.webp').read_bytes()).decode('ascii')
    assets = ROOT / 'assets'
    base_css = (assets / 'intake-ui.css').read_text(encoding='utf-8').replace('__HERO_ARTWORK__', artwork)
    theme_css = '\n'.join(
        (assets / filename).read_text(encoding='utf-8')
        for filename in ('tokens.css', 'theme-layer.css', 'pipeline.css')
    )
    st.markdown(f'{base_css}<style>{theme_css}</style>', unsafe_allow_html=True)
    st.html('''
        <header class="intake-hero">
            <p class="intake-kicker">AI governance / Evidence review</p>
            <h1>Model Intake Brief</h1>
            <p class="intake-tagline">Review public model documentation before integration.</p>
        </header>
    ''')


def _fill_model(provider: str, model: str, hosted: bool) -> None:
    st.session_state['hosted-model' if hosted else 'hf-model'] = model
    if hosted:
        st.session_state['provider'] = provider


def _render_help() -> None:
    guide = (ROOT / 'USER_GUIDE.md').read_text(encoding='utf-8-sig')
    reviewer = guide.split('## Reviewer guide', 1)[-1].split('## Setup and CLI', 1)[0]
    with st.popover('User guide', icon=':material/help:'):
        st.markdown(reviewer)
        st.download_button('Download full user guide', guide, 'USER_GUIDE.md', 'text/markdown',
                           on_click='ignore')


def main() -> None:
    st.set_page_config(page_title='Model Intake Brief', layout='wide')
    _render_brand()
    st.caption('Information, not legal advice.')
    _render_help()
    mode = st.radio('Workspace', ['Saved reports', 'Run a new analysis'], horizontal=True,
                    key='workspace', bind='query-params')
    if mode == 'Saved reports':
        sample_key = st.selectbox('Report', sample_keys(), key='report', bind='query-params')
        st.set_page_config(page_title=f'{sample_key} · Model Intake Brief')
        st.caption('These reports were generated with live sources. Viewing them uses no API credits.')
        _render_report(load_saved_report(sample_key), sample_path(sample_key).name, 'Saved live report')
        return

    st.set_page_config(page_title='New analysis · Model Intake Brief')
    target_mode = st.radio('Model input format', ['Hugging Face ORG/MODEL', 'Hosted provider + exact model ID'],
                           key='target-format', persist_state='session')
    hosted = target_mode == 'Hosted provider + exact model ID'
    example = 'gpt-5.2-2025-12-11' if hosted else 'Qwen/Qwen3-8B'
    st.button('Use example', on_click=_fill_model, args=('openai', example, hosted))
    recent = [row for row in st.session_state.get('recent_models', []) if row[2] == hosted]
    if recent:
        with st.popover('Recent models'):
            for index, (recent_provider, recent_model, recent_hosted) in enumerate(recent):
                st.button(recent_model, key=f'recent-{index}', on_click=_fill_model,
                          args=(recent_provider, recent_model, recent_hosted))
    live_mode = st.checkbox('Live run', value=False, key='live-mode', persist_state='session')
    if live_mode:
        st.warning('Uses search, extraction and OpenAI calls. May consume credits.')
    else:
        st.caption('Offline preview uses fixture data and makes no external API calls.')
        st.checkbox('Demo pace', value=True, key='demo-pace', persist_state='session',
                    help='Offline runs finish in milliseconds. Demo pace holds each pipeline step on screen '
                         'so you can follow it. The run time shown in the report ignores these pauses.')
    with st.form('run-form'):
        provider = st.text_input('Provider', placeholder='openai', key='provider',
                                 persist_state='session', required=True) if hosted else None
        if hosted:
            st.caption('Enter the provider name used in its documentation.')
        validation = (
            (r'(?!.*://)\S+', 'Enter the exact model ID without a URL or spaces.')
            if hosted else (r'[^/\s]+/[^/\s]+', 'Use ORG/MODEL, for example Qwen/Qwen3-8B.')
        )
        model = st.text_input('Model', placeholder=example, key='hosted-model' if hosted else 'hf-model',
                              persist_state='session', required=True, validate=validation)
        st.caption('Exact model ID from provider documentation. Enter the ID without a URL.' if hosted else
                   'Repository ID in ORG/MODEL format. Enter the ID without a URL.')
        field_error = st.empty()
        submitted = st.form_submit_button('Run live analysis' if live_mode else 'Run offline preview',
                                          type='primary', icon=':material/play_arrow:')
    topo = pv.topology()
    trace = st.session_state.get('last_trace')
    with st.expander('Pipeline view: how the data moves through the graph', expanded=True,
                     icon=':material/account_tree:'):
        view = st.empty()
        replay_clicked = st.button('Replay last run', icon=':material/replay:', key='replay-run',
                                   disabled=trace is None, help='Redraws the recorded events. Makes no API calls.')
    if replay_clicked and trace:
        _animate(view, topo, trace['events'], note=REPLAY_NOTE, paced=trace['paced'])
    else:
        view.html(pv.render_html(pv.replay(topo, trace['events'], trace['note']) if trace else pv.PipelineRun(topo)))

    if submitted:
        if hosted and not (provider or '').strip():
            field_error.error('Enter a provider name, for example openai.')
            return
        if not (model or '').strip():
            field_error.error(f'Enter a model ID, for example {example}.')
            return
        if not hosted and not re.fullmatch(r'[^/\s]+/[^/\s]+', model.strip()):
            field_error.error('Use ORG/MODEL, for example Qwen/Qwen3-8B. For hosted models, switch the input format.')
            return
        if hosted and not re.fullmatch(validation[0], model.strip()):
            field_error.error(validation[1])
            return
        entry = ((provider or '').strip(), model.strip(), hosted)
        st.session_state['recent_models'] = [entry, *[r for r in st.session_state.get('recent_models', [])
                                                   if r != entry]][:5]

        paced = (not live_mode) and bool(st.session_state.get('demo-pace', True))
        note = PACE_NOTE if paced else ''
        payload = build_run_payload(live_mode, provider, model)
        waiting = st.empty()
        if live_mode:
            waiting.html('''
                <div class="intake-wait" role="status">
                    <div class="intake-reels" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
                    <div><strong>Analysis in progress</strong><br>
                    <span>Evidence counts will appear when the report is ready.</span></div>
                </div>
            ''')
        start, paused = perf_counter(), 0.0
        run, events, result = pv.PipelineRun(topo, note), [], None
        view.html(pv.render_html(run))
        try:
            for kind, item in iter_run(payload):
                if kind == 'result':
                    result = item
                    continue
                t = perf_counter() - start - paused
                events.append((t, item))
                run.apply(item, t)
                view.html(pv.render_html(run))
                if paced and PACE.get(item.get('type')):
                    sleep(PACE[item['type']])
                    paused += PACE[item['type']]
        finally:
            run.finish(perf_counter() - start - paused)
            view.html(pv.render_html(run))
            waiting.empty()
        elapsed = perf_counter() - start - paused
        st.session_state['last_trace'] = {'events': events, 'paced': paced, 'note': note}
        st.session_state['last_analysis'] = (result or {}, live_mode, model, elapsed)

    previous = st.session_state.get('last_analysis')
    if previous is None:
        st.info('Your analysis will appear here after you run it.')
        return
    result, was_live, run_model, elapsed = previous
    if result.get('error'):
        st.error(result['error'])
    report = result.get('report_markdown')
    if report:
        _render_report(report, _download_name(run_model, was_live),
                       'Live analysis' if was_live else 'Offline preview', elapsed)
    elif not result.get('error'):
        st.warning('The analysis returned no report.')


if __name__ == '__main__':
    main()
