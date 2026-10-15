"""Campaign reuse must be explicit and bound to the tested inputs."""

import csv
from pathlib import Path

import pytest

from bbo_via_fmqa import fmqa_qci as runner, fmqa_simulated


def configure_campaign(tmp_path, monkeypatch):
    dataset_dir = tmp_path / 'grids'
    dataset_dir.mkdir()
    dataset = dataset_dir / 'tiny.csv'
    dataset.write_text('x,y,Objective\n0,0,0\n0,1,1\n1,0,2\n1,1,3\n')
    monkeypatch.setattr(runner, 'DATASET_DIR', str(dataset_dir))
    monkeypatch.setattr(runner, 'OUTPUT_DIR', str(tmp_path / 'results'))
    monkeypatch.delenv('FMQA_RESUME_SUMMARY', raising=False)
    monkeypatch.setenv('FMQA_RUNS_PER_GRAPH', '1')
    monkeypatch.setenv('FMQA_MAX_CYCLES', '2')
    monkeypatch.setenv('FMQA_CONVERGENCE_FRACTION', '0.1')
    return dataset


def fake_result():
    return {'summary_row': dict(zip(runner.SUMMARY_FIELDNAMES,
            ['', '', 0, 0, 0.1, 0.0, 0.1, 0.0, 0.0, 1, 'T'])), 'improvement_rows': []}


def test_default_campaign_starts_new_work(tmp_path, monkeypatch):
    configure_campaign(tmp_path, monkeypatch)
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        return fake_result()
    monkeypatch.setattr(runner, 'run_single_fmqa', run)
    runner.main()
    runner.main()
    assert len(calls) == 2
    assert len(list(Path(runner.OUTPUT_DIR).glob('fmqa_qci_all_runs_*.csv'))) == 2


@pytest.mark.parametrize('changed', ['dataset', 'budget', 'fraction'])
def test_explicit_resume_rejects_changed_inputs(tmp_path, monkeypatch, changed):
    dataset = configure_campaign(tmp_path, monkeypatch)
    summary, _, _ = runner.open_campaign([str(dataset)], 2, 0.1)
    monkeypatch.setenv('FMQA_RESUME_SUMMARY', summary)
    if changed == 'dataset':
        dataset.write_text(dataset.read_text().replace('1,1,3', '1,1,8'))
    with pytest.raises(ValueError, match='does not match'):
        runner.open_campaign([str(dataset)], 3 if changed == 'budget' else 2,
                             0.5 if changed == 'fraction' else 0.1)


def test_identical_resume_skips_completed_work(tmp_path, monkeypatch):
    dataset = configure_campaign(tmp_path, monkeypatch)
    summary, timestamp, _ = runner.open_campaign([str(dataset)], 2, 0.1)
    row = fake_result()['summary_row']
    row.update(graph_type='tiny', run_index=1)
    runner.append_summary_row(summary, row)
    monkeypatch.setenv('FMQA_RESUME_SUMMARY', summary)
    restored, restored_timestamp, keys = runner.open_campaign([str(dataset)], 2, 0.1)
    assert (restored, restored_timestamp, keys) == (summary, timestamp, {('tiny', 1)})


def test_simulated_fraction_controls_patience(tmp_path, monkeypatch):
    configure_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(fmqa_simulated, 'DATASET_DIR', runner.DATASET_DIR)
    monkeypatch.setattr(fmqa_simulated, 'OUTPUT_DIR', runner.OUTPUT_DIR)
    monkeypatch.setenv('FMQA_MAX_CYCLES', '150')
    monkeypatch.setenv('FMQA_CONVERGENCE_FRACTION', '0.5')
    # 1000 cells => 100 proposal cycles; requested patience is bounded by that budget.
    import read_grid
    grid = {(x, 0): float(x) for x in range(1000)}
    monkeypatch.setattr(fmqa_simulated.read_grid, 'load_grid', lambda **kwargs: (grid, 0.0, 999.0, 999, 0))
    captured = []
    def run(*args):
        captured.append(args[-1])
        return dict(zip(fmqa_simulated.SUMMARY_FIELDNAMES, ['tiny', 1, 0, 0, 0.0, 0.0, 0.0, 1, 'T']))
    monkeypatch.setattr(fmqa_simulated, 'run_single_fmqa', run)
    fmqa_simulated.main()
    assert captured == [100]
