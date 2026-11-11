from __future__ import annotations

import app as ui


def test_saved_reports_load_without_graph_calls(monkeypatch):
    invoked = []

    def fake_invoke(payload):
        invoked.append(payload)
        raise AssertionError('graph should not run while previewing saved reports')

    monkeypatch.setattr(ui.APP, 'invoke', fake_invoke, raising=False)

    for sample_key in ui.sample_keys():
        report = ui.load_saved_report(sample_key)
        assert '# Model Intake Brief' in report
        assert 'Information, not legal advice.' in report

    assert invoked == []


def test_build_run_payload_supports_both_model_formats():
    hugging_face = ui.build_run_payload(False, None, 'Qwen/Qwen3-8B')
    hosted = ui.build_run_payload(True, 'openai', 'gpt-5.2-2025-12-11')

    assert hugging_face == {'live_mode': False, 'provider': None, 'model': 'Qwen/Qwen3-8B'}
    assert hosted == {'live_mode': True, 'provider': 'openai', 'model': 'gpt-5.2-2025-12-11'}


def test_review_preserves_training_summary_and_flags_codex_variant():
    from intake.report_review import questions_for, read_findings, sections

    model = 'gpt-5.2-2025-12-11'
    parts = sections(ui.load_saved_report(model))
    findings = {item.id: item for item in read_findings(parts['Findings'], model)}
    assert findings['DS-01'].status == 'Partial'
    assert 'codex' in findings['DS-01'].fields['Model applicability']
    assert 'Quote' not in findings['DS-01'].fields
    assert 'Link check' not in findings['DS-01'].fields
    assert findings['DT-03'].status == 'Evidence found'
    assert 'public-summary-of-training-content' in findings['DT-03'].fields['Source URL']
    questions = questions_for(list(findings.values()), model)
    assert any('? [DS-01]' in question for question in questions)
    assert all('? [' in question and '[CT-' not in question for question in questions)


def test_review_omits_truncated_links_and_groups_regulatory_passages():
    from intake.report_review import public_link, regulatory_rows, sections

    assert not public_link('https://example.com/preview%E2%80%A6')
    assert not public_link('https://example.com/preview…')
    assert public_link('https://example.com/source')
    report = sections(ui.load_saved_report('gpt-5.2-2025-12-11'))
    provisions, _ = regulatory_rows(report['Regulatory context'])
    assert len(provisions) == 13
    assert any(len(record['items']) > 1 for record in provisions.values())


def test_export_keeps_verification_notes_in_methods():
    from intake.report_review import read_findings, sections, source_rows

    model = 'gpt-5.2-2025-12-11'
    parts = sections(ui.load_saved_report(model))
    findings = read_findings(parts['Findings'], model)
    sources = source_rows(parts['Source collection'], findings, model)
    exported = sections(ui._review_export(parts, findings, model, sources))
    verification_note = 'Quoted text was not supported by the extracted source text.'
    assert verification_note not in exported['Summary']
    assert verification_note not in exported['Prioritised provider questions']
    assert verification_note in exported['Methods and limitations']
    assert 'Other model' in exported['Findings']
