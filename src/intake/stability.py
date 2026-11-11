"""Run the same analysis several times and show which checks keep their status.

The assessment is an LLM call over whatever the search returned, so two runs of the same model can differ.
This module compares finished reports; it does not call any service itself.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .report_review import STATUSES, read_findings, sections, source_rows

SYMBOL = {'Evidence found': '✓', 'Partial': '◐', 'Not found': '×', 'Not checked': '–'}


@dataclass
class Row:
    id: str
    criterion: str
    statuses: list[str | None]      # one per run, None when the run produced no report

    @property
    def seen(self) -> list[str]:
        return [s for s in self.statuses if s is not None]

    @property
    def stable(self) -> bool:
        return len(set(self.seen)) <= 1


@dataclass
class Stability:
    model: str
    runs: int
    rows: list[Row]
    pages: list[set[str] | None]     # pages read by Jina, per run
    seconds: list[float | None]
    failures: list[str | None]


def _pages(report: str, findings: list, model: str) -> set[str]:
    parts = sections(report)
    rows = source_rows(parts.get('Source collection', ''), findings, model)
    return {row['url'] for row in rows if row['tool'].startswith('Jina Reader') and row['url']}


def compare(reports: list[str | None], model: str, seconds: list[float | None] | None = None,
            failures: list[str | None] | None = None) -> Stability:
    """`reports` holds one Markdown report per run, or None for a run that failed."""
    by_run: list[dict[str, tuple[str, str]] | None] = []
    pages: list[set[str] | None] = []
    for report in reports:
        if not report:
            by_run.append(None)
            pages.append(None)
            continue
        findings = read_findings(sections(report).get('Findings', ''), model)
        by_run.append({f.id: (f.criterion, f.status) for f in findings})
        pages.append(_pages(report, findings, model))
    ids: list[str] = []
    criteria: dict[str, str] = {}
    for run in by_run:
        for item_id, (criterion, _) in (run or {}).items():
            if item_id not in criteria:
                ids.append(item_id)
                criteria[item_id] = criterion
    rows = [Row(i, criteria[i], [(run or {}).get(i, (None, None))[1] if run else None for run in by_run]) for i in ids]
    return Stability(model, len(reports), rows, pages, seconds or [None] * len(reports),
                     failures or [None] * len(reports))


def _is_core(row: Row) -> bool:
    return not row.id.startswith('CT-')


def render(result: Stability) -> str:
    done = [i for i, p in enumerate(result.pages) if p is not None]
    lines = [f'# Stability check: {result.model}', '',
             f'- Runs: {len(done)} completed of {result.runs}']
    for index, reason in enumerate(result.failures, start=1):
        if reason:
            lines.append(f'  - Run {index} failed: {reason}')
    if len(done) < 2:
        lines.append('- Not enough completed runs to compare.')
        return '\n'.join(lines) + '\n'

    core = [r for r in result.rows if _is_core(r)]
    stable = [r for r in core if r.stable]
    lines.append(f'- Core checks with the same status in every run: {len(stable)} of {len(core)}')
    lines.append('')
    lines.append('## Per run (core checks)')
    lines.append('')
    lines.append('| Run | ' + ' | '.join(STATUSES) + ' | Time |')
    lines.append('|---|' + '---|' * (len(STATUSES) + 1))
    for index in range(result.runs):
        if result.pages[index] is None:
            lines.append(f'| {index + 1} | failed | | | | |')
            continue
        counts = Counter(r.statuses[index] for r in core)
        seconds = result.seconds[index]
        lines.append(f'| {index + 1} | ' + ' | '.join(str(counts[s]) for s in STATUSES)
                     + f" | {f'{seconds:.1f} s' if seconds is not None else 'n/a'} |")

    lines += ['', '## Per check', '', '| ID | Check | ' + ' | '.join(f'Run {i + 1}' for i in range(result.runs)) + ' | Stable |',
              '|---|---|' + '---|' * (result.runs + 1)]
    for row in result.rows:
        cells = ' | '.join(SYMBOL.get(s, '·') if s else 'n/a' for s in row.statuses)
        if row.stable:
            verdict = 'yes'
        else:
            spread = Counter(row.seen)
            verdict = 'no: ' + ', '.join(f'{SYMBOL[s]} {n}' for s, n in spread.most_common())
        context = ' (context)' if not _is_core(row) else ''
        lines.append(f'| {row.id} | {row.criterion}{context} | {cells} | {verdict} |')
    lines.append('')
    lines.append(f'Legend: {", ".join(f"{SYMBOL[s]} {s.lower()}" for s in STATUSES)}.')

    page_sets = [p for p in result.pages if p is not None]
    union = set().union(*page_sets)
    common = set.intersection(*page_sets)
    lines += ['', '## Where the difference comes from', '',
              f'- Pages read by Jina: {len(common)} in every run, {len(union)} in at least one run.']
    if union == common:
        lines.append('- The same pages were read every time, so any difference in status comes from the assessment step.')
    else:
        lines.append('- The set of pages changed between runs, so the search results differ. '
                     'Statuses can move even if the assessment were deterministic.')
        for url in sorted(union - common):
            seen_in = ', '.join(str(i + 1) for i, p in enumerate(result.pages) if p is not None and url in p)
            lines.append(f'  - {url} (runs {seen_in})')
    unstable = [r for r in core if not r.stable]
    if unstable:
        lines += ['', '## Unstable core checks', '']
        lines += [f'- {r.id} {r.criterion}: ' + ' / '.join(SYMBOL.get(s, 'n/a') if s else 'n/a' for s in r.statuses)
                  for r in unstable]
    return '\n'.join(lines) + '\n'


def report_model_id(report: str, fallback: str) -> str:
    match = re.search(r'^- Exact model ID: `(.+)`$', report, re.M)
    return match[1] if match else fallback
