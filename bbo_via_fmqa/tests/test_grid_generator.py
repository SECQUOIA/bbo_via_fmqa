"""Analytic reference values and round trips for generated benchmark grids."""

import numpy as np
import pytest

from bbo_via_fmqa import qhd_grid_generator as generator, read_grid


def test_shubert_reference_and_symmetry():
    # Yang (2010), equation 19: both factors sum i*cos((i+1)*x+i), i=1,...,5.
    assert generator.shubert(0.0, 0.0) == pytest.approx(19.875836249802127)
    assert generator.shubert(0.4, -1.3) == pytest.approx(generator.shubert(-1.3, 0.4))


def test_xin_she_yang_reference_and_symmetry():
    # Yang (2010), equation 22, beta=15, m=5.
    assert generator.xin_she_yang3(0.0, 0.0) == pytest.approx(-1.0)
    assert generator.xin_she_yang3(0.4, 1.3) == pytest.approx(generator.xin_she_yang3(1.3, 0.4))


def test_csendes_zero_extension_is_finite():
    with np.errstate(divide='raise', invalid='raise'):
        values = generator.csendes(np.array([0.0, 0.5]), np.array([0.0, 0.0]))
    assert values[0] == 0.0
    assert np.isfinite(values).all()


@pytest.mark.parametrize('graph_type', sorted(generator.GRAPH_SPECS))
def test_generated_grids_load_as_normalized_finite_values(tmp_path, graph_type):
    path = generator.make_csv(2, 3, graph_type, str(tmp_path))
    grid, low, high, x_bound, y_bound = read_grid.load_grid(path)
    assert len(grid) == 12
    assert (x_bound, y_bound) == (2, 3)
    assert np.isfinite(list(grid.values())).all()
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)


def test_tiny_nonconstant_objective_retains_variation(tmp_path, monkeypatch):
    monkeypatch.setitem(generator.GRAPH_SPECS, 'tiny', (lambda x, y: x * 1e-12, (0.0, 1.0), (0.0, 1.0)))
    path = generator.make_csv(1, 1, 'tiny', str(tmp_path))
    grid, low, high, *_ = read_grid.load_grid(path)
    assert grid[(0, 0)] == 0.0
    assert grid[(1, 1)] == 1.0
    assert (low, high) == (0.0, 1.0)
