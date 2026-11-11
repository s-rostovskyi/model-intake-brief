from __future__ import annotations

from pathlib import Path

from intake import cli, stability

SAMPLE = (Path(__file__).resolve().parent.parent / 'samples' / 'qwen3-8b.md').read_text(encoding='utf-8')
MODEL = 'Qwen/Qwen3-8B'


def changed(report: str, item_id: str, old: str, new: str) -> str:
    start = report.index(f'- {item_id} ') + 2   # historic samples use an em dash, new reports a hyphen
    end = report.index('\n', start)
    line = report[start:end]
    assert line.endswith(old)
    return report[:start] + line[:-len(old)] + new + report[end:]


def test_identical_reports_are_stable():
    result = stability.compare([SAMPLE, SAMPLE, SAMPLE], MODEL, [1.0, 2.0, 3.0])
    assert all(row.stable for row in result.rows)
    text = stability.render(result)
    assert 'Core checks with the same status in every run: 15 of 15' in text
    assert 'The same pages were read every time' in text
    assert 'Unstable core checks' not in text


def test_a_changed_status_is_reported_per_check_and_per_run():
    other = changed(SAMPLE, 'ID-02', 'Not found', 'Partial')
    result = stability.compare([SAMPLE, other, SAMPLE], MODEL)
    unstable = [r.id for r in result.rows if not r.stable]
    assert unstable == ['ID-02']
    text = stability.render(result)
    assert 'Core checks with the same status in every run: 14 of 15' in text
    assert '| ID-02 |' in text and 'no: × 2, ◐ 1' in text
    assert '- ID-02 Release date: × / ◐ / ×' in text


def test_context_checks_do_not_count_toward_stability():
    other = changed(SAMPLE, 'CT-01', 'Not found', 'Partial')
    text = stability.render(stability.compare([SAMPLE, other], MODEL))
    assert 'same status in every run: 15 of 15' in text
    assert '(context)' in text


def test_failed_runs_are_listed_and_skipped():
    result = stability.compare([SAMPLE, None, SAMPLE], MODEL, [1, 2, 3], [None, 'Missing key', None])
    text = stability.render(result)
    assert 'Runs: 2 completed of 3' in text and 'Run 2 failed: Missing key' in text
    only_one = stability.render(stability.compare([SAMPLE, None], MODEL, failures=[None, 'x']))
    assert 'Not enough completed runs to compare.' in only_one


def test_cli_runs_n_times_and_writes_the_file(monkeypatch, tmp_path, capsys):
    reports = iter([SAMPLE, changed(SAMPLE, 'ID-03', 'Not found', 'Partial'), SAMPLE])
    monkeypatch.setattr(cli.APP, 'invoke', lambda payload: {'report_markdown': next(reports)})
    out = tmp_path / 'stability.md'
    code = cli.main(['stability', '--model', MODEL, '-n', '3', '--pause', '0', '--out', str(out)])
    assert code == 0
    assert '14 of 15' in out.read_text(encoding='utf-8')
    assert '# Stability check' in capsys.readouterr().out


def test_cli_rejects_a_silly_number_of_runs(capsys):
    assert cli.main(['stability', '--model', MODEL, '-n', '1']) == 2
    assert cli.main(['stability', '--model', MODEL, '-n', '99']) == 2
