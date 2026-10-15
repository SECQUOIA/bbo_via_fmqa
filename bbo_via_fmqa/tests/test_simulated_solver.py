"""Regression checks for candidate filtering and runner convergence logs."""

import csv

import dimod
import numpy as np
import pytest

import bbo_via_fmqa
import ising_machine


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
    """Two valid worse proposals consume patience and preserve the single-start best."""
    from bbo_via_fmqa import fmqa_simulated as runner
    monkeypatch.setattr(runner, 'train_surrogate_model', lambda *args, **kwargs: object())
    proposals = iter([(1, 0), (1, 1)])
    monkeypatch.setattr(runner, 'solve_surrogate_SA', lambda *args: next(proposals))
    monkeypatch.setattr(runner, 'OUTPUT_DIR', str(tmp_path))
    grid = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 2.0, (1, 1): 3.0}
    row = runner.run_single_fmqa(grid, 0.0, 3.0, 1, 1, (0, 0), 10, 2)
    assert row['iterations'] == 2
    assert row['final_value'] == 0.0
    assert row['solution_found'] == 'T'
    row.update(graph_type='tiny', run_index=1)
    path = runner.create_summary_csv()
    runner.append_summary_row(path, row)
    with open(path) as f:
        records = list(csv.DictReader(f))
    assert len(records) == 1
    assert float(records[0]['final_value']) == 0.0
    assert int(records[0]['iterations']) == 2
