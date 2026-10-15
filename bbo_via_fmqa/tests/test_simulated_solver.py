"""Regression checks for candidate filtering and runner convergence logs."""

import csv
from pathlib import Path
import runpy

import dimod
import numpy as np
import pytest

import bbo_via_fmqa
import ising_machine
import ml_surrogate


class FixedSampler:
    def __init__(self, samples):
        self.samples = samples

    def sample(self, model, **kwargs):
        return self.samples


def test_sa_decodes_rectangular_grid():
    samples = dimod.SampleSet.from_samples([[1, 1, 0, 1]], 'BINARY', energy=[0.0])
    assert ising_machine.solve_surrogate_SA(
        None, 1, 7, set(), FixedSampler(samples), {(1, 5): 2.0}
    ) == (1, 5)


def test_sa_filters_invalid_duplicate_and_nonfinite_points():
    samples = dimod.SampleSet.from_samples(
        [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0]],
        'BINARY', energy=[-5.0, -4.0, -3.0, 2.0, 1.0],
    )
    grid = {(0, 0): 0.0, (1, 0): np.nan, (2, 1): 8.0, (3, 0): 9.0}
    assert ising_machine.solve_surrogate_SA(
        None, 3, 3, {(0, 0)}, FixedSampler(samples), grid
    ) == (3, 0)


def test_sa_handles_sampler_failure():
    class BrokenSampler:
        def sample(self, model, **kwargs):
            raise RuntimeError('sampler unavailable')

    assert ising_machine.solve_surrogate_SA(None, 1, 1, set(), BrokenSampler(), {}) == (None, None)


def test_runner_stops_and_logs_updated_incumbent(tmp_path, monkeypatch):
    """Two valid but worse proposals must consume patience and retain the initial best."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    monkeypatch.setenv('FMQA_MAX_CYCLES', '10')
    monkeypatch.setenv('FMQA_CONVERGENCE_PATIENCE', '2')
    dataset = tmp_path / 'tiny.csv'
    dataset.write_text('x,y,Objective\n0,0,0\n0,1,1\n1,0,2\n1,1,3\n')
    monkeypatch.setenv('FMQA_DATASET', str(dataset))
    import random
    monkeypatch.setattr(random, 'shuffle', lambda points: None)
    monkeypatch.setattr(ml_surrogate, 'train_surrogate_model', lambda *args, **kwargs: object())
    monkeypatch.setattr(ml_surrogate, 'print_final_equation', lambda model: None)
    proposals = iter([(1, 0), (1, 1)])
    monkeypatch.setattr(ising_machine, 'solve_surrogate_SA', lambda *args: next(proposals))
    runner = Path(bbo_via_fmqa.__file__).with_name('fmqa_simulated.py')
    runpy.run_path(str(runner), run_name='__main__')
    logs = list((tmp_path / 'output').glob('*SA_log*.csv'))
    assert len(logs) == 1
    with logs[0].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [float(row['best_val']) for row in rows] == [0.0, 0.0]
    assert [int(row['no_improvement_count']) for row in rows] == [1, 2]
    assert [row['status'] for row in rows] == ['evaluated', 'evaluated']
