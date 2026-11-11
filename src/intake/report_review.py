"""Read both historic and current intake Markdown for the reviewer interface."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlsplit

from .assessment import other_model, provider_question

STATUSES = ('Evidence found', 'Partial', 'Not found', 'Not checked')
FINDING = re.compile(r'^- ([A-Z]{2}-\d{2}) (?:-|—) (.+): (Evidence found|Partial|Not found|Not checked)$', re.M)
QA_GAPS = ('Quoted text was not supported', 'Quoted text appeared to describe')


@dataclass
class Finding:
    id: str
    criterion: str
    status: str
    fields: dict[str, str]

    @property
    def gap(self) -> str:
        return self.fields.get('Gap') or self.fields.get('Gap explanation') or self.fields.get('Failure reason', '')


def read_findings(section: str, model: str) -> list[Finding]:
    matches = list(FINDING.finditer(section))
    findings = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        fields = dict(re.findall(r'^  - ([^:\n]+): (.+)$', section[match.end():end], re.M))
        item_id, criterion, status = match.groups()
        variant = other_model(fields.get('Quote', ''), fields.get('Source URL', ''), model)
        if variant:
            fields['Model applicability'] = f'Other model ({variant})'
            fields['Gap'] = 'Find public evidence that applies to the exact target.'
            fields.pop('Quote', None)
            fields.pop('Link check', None)
            if status == 'Evidence found':
                status = 'Partial'
        elif not fields.get('Model applicability'):
            fields['Model applicability'] = 'Review the source for exact-model or family coverage.' if fields.get('Quote') else 'Not established'
        if not fields.get('Quote'):
            fields.pop('Link check', None)
        findings.append(Finding(item_id, criterion, status, fields))
    return findings


def sections(report: str) -> dict[str, str]:
    parts = re.split(r'^### (.+)\n', report, flags=re.M)
    return {'Header': parts[0], **dict(zip(parts[1::2], parts[2::2]))}


def public_link(value: str) -> bool:
    try:
        url = urlsplit(value)
        return url.scheme == 'https' and bool(url.hostname) and not url.username and not any(
            char.isspace() or char in '<>…' for char in value
        ) and '%e2%80%a6' not in value.lower()
    except ValueError:
        return False


def source_rows(section: str, findings: list[Finding], model: str) -> list[dict[str, str]]:
    citations = Counter(f.fields.get('Source URL') for f in findings if f.fields.get('Quote'))
    rows = []
    for index, block in enumerate(re.split(r'\n(?=- )', section.strip()), start=1):
        if not block.startswith('- '):
            continue
        first, *details = block.splitlines()
        tool, _, outcome = first[2:].partition(': ')
        url = next((s.split(': ', 1)[1] for s in details if s.startswith('  - Requested URL: ')), '')
        variant = other_model('', url, model)
        if tool == 'Tavily Search':
            candidates = [s.strip()[2:] for s in details if s.startswith('    - https://')]
            hits = [urlsplit(u).hostname for u in candidates if public_link(u)]
            use = 'Discovery only'
            if hits:
                use += ' · ' + ', '.join(dict.fromkeys(hits))
        elif variant:
            use = f'Other model ({variant}) · not used as evidence'
        else:
            use = f'Cited in {citations[url]} findings' if citations[url] else 'No verified quote cited'
        rows.append({'tool': f'{tool} · {index}', 'outcome': outcome, 'url': url, 'use': use})
    return rows


def regulatory_rows(section: str) -> tuple[dict[str, dict], list[str]]:
    records: dict[str, dict] = {}
    blockers = []
    for block in re.split(r'^- Checklist item: ', section, flags=re.M)[1:]:
        item, _, content = block.partition('\n')
        blocked = re.search(r'^  - Retrieval blockers:\n((?:    - .*\n?)*)', content, re.M)
        if blocked:
            blockers.extend(f'{item} · {line.strip()[2:]}' for line in blocked[1].splitlines())
        references = content.partition('  - Retrieved references:\n')[2]
        for entry in re.split(r'^    - ', references, flags=re.M)[1:]:
            title, _, body = entry.partition('\n')
            fields = dict(re.findall(r'^      - ([^:\n]+): (.+)$', body, re.M))
            record = records.setdefault(title, {'fields': fields, 'items': [], 'scores': []})
            if item not in record['items']:
                record['items'].append(item)
            try:
                record['scores'].append(float(fields.get('Match score', '')))
            except ValueError:
                pass
    return records, blockers


def questions_for(findings: list[Finding], model: str) -> list[str]:
    return [f'{provider_question(f.criterion, model)} [{f.id}]' for f in findings
            if f.status in {'Partial', 'Not found'} and not f.id.startswith('CT-')]

