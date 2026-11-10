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
