"""Report overview: what a reviewer needs to decide, computed from the report Markdown.

Everything here is derived from the parsed report (findings, sources, regulatory section), so it works the
same for a fresh run and for a saved report. Rendering is plain HTML and CSS (st.html strips inline SVG).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from urllib.parse import urlencode, urlsplit

from .assessment import provider_question
from .report_review import QA_GAPS, STATUSES, Finding, public_link, regulatory_rows

GROUPS = (
    ('ID', 'Identity'),
    ('DS', 'Distribution and licence'),
    ('TP', 'Technical properties'),
    ('US', 'Intended use'),
    ('DT', 'Training data'),
)
CONTEXT_PREFIX = 'CT-'
STATUS_KEY = {'Evidence found': 'found', 'Partial': 'partial', 'Not found': 'missing', 'Not checked': 'unchecked'}
TOP_QUESTIONS = 3
TOP_PROVISIONS = 4
# community and forum hosts sit on the provider's domain but are not provider documentation
_NOT_OFFICIAL_LABELS = {'community', 'forum', 'forums', 'discuss', 'discourse', 'support'}


@dataclass
class Overview:
    core: list[Finding]
    context: list[Finding]
    counts: Counter
    context_counts: Counter
    groups: list[tuple[str, list[Finding]]]
    questions: list[tuple[Finding, str]]
    question_total: int
    quotes_ok: int
    quotes_unsupported: int
    other_model: int
    links_ok: int          # the page answered (2xx/3xx), including pages that are off the allowlist
    links_offlist: int     # answered, but the final address is not on the approved domains
    links_total: int
    link_issues: Counter
    cited: int
    cited_official: int
    provisions: list[tuple[str, list[str]]]
    elapsed: float | None
    cost: str
    tokens: tuple[int, int] | None
    extra: dict = field(default_factory=dict)


def _is_core(finding: Finding) -> bool:
    return not finding.id.startswith(CONTEXT_PREFIX)


def _field(report: str, label: str) -> str:
    prefix = f'- {label}: '
    return next((line[len(prefix):].strip().strip('`') for line in report.splitlines()
                 if line.startswith(prefix)), '')


def _official(url: str, provider: str, model: str, kind: str) -> bool:
    """Heuristic by host name: provider domains and the model's own Hugging Face organisation."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or '').lower()
    if not host:
        return False
    if host == 'huggingface.co' or host.endswith('.huggingface.co'):
        org = model.split('/', 1)[0].lower() if '/' in model else ''
        segments = [s for s in parts.path.split('/') if s]
        return kind == 'hugging_face' and bool(org) and bool(segments) and segments[0].lower() == org
    labels = host.split('.')
    if labels[0] in _NOT_OFFICIAL_LABELS:
        return False
    token = re.sub(r'[^a-z0-9]', '', provider.lower())
    return len(token) >= 3 and token in re.sub(r'[^a-z0-9.]', '', host).replace('.', '')


def _cost(report: str, offline: bool) -> tuple[str, tuple[int, int] | None]:
    usage = re.search(r'Aggregate assessment usage: input tokens (\d+), output tokens (\d+)', report)
    if usage:
        inputs, outputs = map(int, usage.groups())
        return f'${(inputs * 0.75 + outputs * 4.50) / 1_000_000:.4f}', (inputs, outputs)
    return ('$0' if offline else 'Not recorded'), None


def _answered(outcome: str) -> bool:
    """'HTTP 200 (ok)' and 'HTTP 200 (blocked)' both mean the page answered; the second is off the allowlist."""
    match = re.match(r'HTTP (\d{3}) \(', outcome)
    return bool(match) and 200 <= int(match[1]) < 400


def build_overview(report: str, findings: list[Finding], parts: dict[str, str], model: str,
                   elapsed: float | None, offline: bool = False) -> Overview:
    core = [f for f in findings if _is_core(f)]
    context = [f for f in findings if not _is_core(f)]

    groups = []
    for prefix, label in GROUPS:
        rows = [f for f in core if f.id.startswith(prefix + '-')]
        if rows:
            groups.append((label, rows))
    known = {f.id for _, rows in groups for f in rows}
    rest = [f for f in core if f.id not in known]
    if rest:  # a checklist row from a future group still shows up
        groups.append(('Other checks', rest))

    order = {'Not found': 0, 'Partial': 1}
    gaps = sorted((f for f in core if f.status in order), key=lambda f: order[f.status])
    questions = [(f, provider_question(f.criterion, model)) for f in gaps]

    quoted = [f for f in findings if f.fields.get('Quote')]
    unsupported = [f for f in findings if f.gap.startswith(QA_GAPS)]
    other = [f for f in findings if f.fields.get('Model applicability', '').startswith('Other model')]

    checks: dict[str, str] = {}
    for f in quoted:
        url, _, outcome = f.fields.get('Link check', '').rpartition(' -> ')
        if url and url not in checks:
            checks[url] = outcome.strip()
    answered = {url: _answered(outcome) for url, outcome in checks.items()}
    off_list = sum(1 for url, outcome in checks.items() if answered[url] and not outcome.endswith('(ok)'))
    issues = Counter(outcome for url, outcome in checks.items() if not answered[url])

    provider = _field(report, 'Model provider')
    kind = _field(report, 'Target kind')
    cited_urls = list(dict.fromkeys(f.fields.get('Source URL', '') for f in quoted if public_link(f.fields.get('Source URL', ''))))
    official = [u for u in cited_urls if _official(u, provider, model, kind)]

    gap_ids = {f.id for f in gaps}
    records, _ = regulatory_rows(parts.get('Regulatory context', ''))
    provisions = []
    for title, record in records.items():
        ids = [i for i in record['items'] if i in gap_ids]
        if ids:
            provisions.append((re.sub(r'\s+\([a-z]+(?:-[a-z0-9]+)+\)$', '', title), ids))
    provisions.sort(key=lambda row: -len(row[1]))

    cost, tokens = _cost(report, offline)
    return Overview(
        core=core, context=context,
        counts=Counter(f.status for f in core), context_counts=Counter(f.status for f in context),
        groups=groups, questions=questions, question_total=len(questions),
        quotes_ok=len(quoted), quotes_unsupported=len(unsupported), other_model=len(other),
        links_ok=sum(answered.values()), links_offlist=off_list, links_total=len(checks), link_issues=issues,
        cited=len(cited_urls), cited_official=len(official),
        provisions=provisions[:TOP_PROVISIONS], elapsed=elapsed, cost=cost, tokens=tokens,
    )


# ---------------------------------------------------------------------------------------------- rendering

def _secs(value: float) -> str:
    return f'{value * 1000:.0f} ms' if value < 1 else f'{value:.1f} s'


def _tone(part: int, whole: int) -> str:
    if whole <= 0:
        return 'neutral'
    ratio = part / whole
    return 'good' if ratio >= 0.9 else 'warn' if ratio >= 0.6 else 'bad'


def _verdict(o: Overview, link_model: str | None = None) -> str:
    total = len(o.core)
    found = o.counts['Evidence found']
    segments = ''.join(
        f'<span class="idash-seg" data-s="{STATUS_KEY[s]}" style="flex:{o.counts[s]}" title="{escape(s)}: {o.counts[s]}"></span>'
        for s in STATUSES if o.counts[s])
    def item(status: str) -> str:
        body = f'<b>{o.counts[status]}</b> {escape(status.lower())}'
        if link_model:  # saved reports: jump to the findings list filtered by this status
            query = escape(urlencode({'report': link_model, 'status': status}))
            body = f'<a href="?{query}#findings" title="Show in the findings list (it also lists context checks)">{body}</a>'
        return f'<li data-s="{STATUS_KEY[status]}">{body}</li>'

    legend = ''.join(item(s) for s in STATUSES)
    context = ''
    if o.context:
        bits = ', '.join(f'{o.context_counts[s]} {s.lower()}' for s in STATUSES if o.context_counts[s])
        context = (f'<p class="idash-note">Context only ({len(o.context)} training-resource checks, '
                   f'not part of the count): {escape(bits)}.</p>')
    return (f'<section class="idash-verdict"><h3>Evidence coverage</h3>'
            f'<p class="idash-big"><b>{found}</b> of {total} core checks fully evidenced</p>'
            f'<div class="idash-bar" role="img" aria-label="{found} of {total} core checks fully evidenced">{segments}</div>'
            f'<ul class="idash-legend">{legend}</ul>{context}</section>')


def _groups(o: Overview) -> str:
    rows = []
    for label, items in o.groups:
        found = sum(1 for f in items if f.status == 'Evidence found')
        cells = ''.join(
            f'<i data-s="{STATUS_KEY[f.status]}" title="{escape(f.id)} {escape(f.criterion)}: {escape(f.status)}"></i>'
            for f in items)
        rows.append(f'<li aria-label="{escape(label)}: {found} of {len(items)} fully evidenced">'
                    f'<span class="idash-gname">{escape(label)}</span><span class="idash-cells">{cells}</span>'
                    f'<span class="idash-gcount">{found}/{len(items)}</span></li>')
    return (f'<section><h3>Where the gaps are</h3><ul class="idash-groups">{"".join(rows)}</ul>'
            f'<p class="idash-note">One square per check. Hover a square for its ID and status.</p></section>')


def _questions(o: Overview) -> str:
    if not o.questions:
        body = '<p class="idash-note">No open gaps, so no provider questions were generated.</p>'
    else:
        shown = o.questions[:TOP_QUESTIONS]
        items = ''.join(
            f'<li><span class="idash-id" data-s="{STATUS_KEY[f.status]}">{escape(f.id)}</span>'
            f'<span>{escape(text)}<em>{escape(f.status)}</em></span></li>' for f, text in shown)
        more = o.question_total - len(shown)
        tail = f' {more} more in the full list below.' if more > 0 else ''
        body = (f'<ol class="idash-questions">{items}</ol>'
                f'<p class="idash-note">Most important first: not found before partial.{escape(tail)}</p>')
    return f'<section><h3>Ask the provider</h3>{body}</section>'


def _tile(title: str, value: str, detail: str, tone: str) -> str:
    return (f'<div class="idash-tile" data-tone="{tone}"><dt>{escape(title)}</dt><dd>{escape(value)}</dd>'
            f'<p>{escape(detail)}</p></div>')


def _trust(o: Overview) -> str:
    quote_total = o.quotes_ok + o.quotes_unsupported
    quote_detail = (f'{o.quotes_unsupported} quote(s) not found in the source text, dropped'
                    if o.quotes_unsupported else 'every proposed quote matched its source')
    if o.links_total:
        parts = []
        if o.links_offlist:
            parts.append(f'{o.links_offlist} answered but is off the approved domains')
        parts.extend(f'{count} {reason}' for reason, count in o.link_issues.items())
        link_detail = ', '.join(parts) or 'every cited page answered'
    else:
        link_detail = 'no cited links to check'
    other = ('no quotes about another model' if not o.other_model
             else f'{o.other_model} finding(s) cited another model and were downgraded')
    tiles = ''.join((
        _tile('Quotes verified in source text', f'{o.quotes_ok} of {quote_total}' if quote_total else 'n/a',
              quote_detail, _tone(o.quotes_ok, quote_total)),
        _tile('Cited links reachable', f'{o.links_ok} of {o.links_total}' if o.links_total else 'n/a',
              link_detail, _tone(o.links_ok, o.links_total)),
        _tile('Cited from provider domains', f'{o.cited_official} of {o.cited}' if o.cited else 'n/a',
              'matched by host name, a heuristic', _tone(o.cited_official, o.cited) if o.cited else 'neutral'),
        _tile('Evidence about the exact model', 'yes' if not o.other_model else f'{o.other_model} flagged',
              other, 'good' if not o.other_model else 'warn'),
    ))
    return (f'<section class="idash-trust"><h3>How far to trust this report</h3><dl class="idash-tiles">{tiles}</dl></section>')


def _regulatory(o: Overview) -> str:
    if not o.provisions:
        return ''
    rows = ''.join(f'<li><b>{escape(title)}</b><span>{escape(", ".join(ids))}</span></li>'
                   for title, ids in o.provisions)
    return (f'<section><h3>Open gaps and AI Act provisions</h3><ul class="idash-reg">{rows}</ul>'
            f'<p class="idash-note">Provisions retrieved for the open gaps by similarity search. '
            f'This is a pointer for the reviewer, not a compliance finding.</p></section>')


def _footer(o: Overview, model_note: str) -> str:
    bits = [f'Run time <b>{_secs(o.elapsed)}</b>' if o.elapsed is not None else 'Run time <b>not recorded</b>',
            f'Est. assessment cost <b>{escape(o.cost)}</b>']
    if o.tokens:
        bits.append(f'{o.tokens[0]:,} in / {o.tokens[1]:,} out tokens')
    if model_note:
        bits.append(escape(model_note))
    return f'<p class="idash-foot">{" · ".join(bits)}</p>'


def render_html(o: Overview, model_note: str = '', link_model: str | None = None) -> str:
    return (f'<div class="idash">{_verdict(o, link_model)}<div class="idash-pair">{_groups(o)}{_questions(o)}</div>'
            f'{_trust(o)}{_regulatory(o)}{_footer(o, model_note)}</div>')


def render_offline_html(elapsed: float | None, cost: str) -> str:
    run = f'Run time <b>{_secs(elapsed)}</b>' if elapsed is not None else 'Run time <b>not recorded</b>'
    return ('<div class="idash"><section class="idash-verdict"><h3>Evidence coverage</h3>'
            '<p class="idash-big">Not assessed in offline preview</p>'
            '<p class="idash-note">The findings below are fixture data that shows the report layout. '
            'Turn on Live run to see evidence coverage, gaps and source checks for a real model.</p></section>'
            f'<p class="idash-foot">{run} · Est. assessment cost <b>{escape(cost)}</b></p></div>')
