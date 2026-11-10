from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChecklistItem:
    item_id: str
    check: str
    evidence_sufficient: str
    reference: str
    is_context: bool = False


def checklist_path() -> Path:
    return Path(__file__).resolve().parents[2] / 'model-intake-checklist.md'


def load_checklist() -> list[ChecklistItem]:
    rows: list[ChecklistItem] = []
    in_table = False
    for raw_line in checklist_path().read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if line.startswith('| ID |'):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith('|'):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) != 4 or cells[0] in {'ID', '---'}:
            continue
        item_id, check, evidence_sufficient, reference = cells
        rows.append(
            ChecklistItem(
                item_id=item_id,
                check=check,
                evidence_sufficient=evidence_sufficient,
                reference=reference,
                is_context=item_id.startswith('CT-'),
            )
        )
    return rows
