"""Offline regression checks for codecs, sampler labels, and optional QCI use."""

import sys

import dimod
import pytest

import ising_machine
from bbo_via_fmqa import read_grid


@pytest.mark.parametrize('bounds,point', [((3, 3, 3), (1, 2, 3)), ((1, 7, 0), (1, 5, 0))])
@pytest.mark.parametrize('lsb_first', [False, True])
def test_three_dimensional_codec_roundtrip(bounds, point, lsb_first):
    bits = read_grid.coord_bits_3d(*point, *bounds, lsb_first=lsb_first)
    assert read_grid.bits_to_int_3d(bits, *bounds, lsb_first=lsb_first) == point


def test_bounded_two_dimensional_lsb_roundtrip():
    bits = read_grid.coord_bits(1, 5, 1, 7, lsb_first=True)
    assert read_grid.bits_to_int(bits, 1, 7, lsb_first=True) == (1, 5)


def test_sa_decoding_uses_variable_labels():
    samples = dimod.SampleSet.from_samples(([[0, 1]], [1, 0]), 'BINARY', energy=[-2.0], sort_labels=False)
    class Sampler:
        def sample(self, model, **kwargs):
            return samples
    model = dimod.BinaryQuadraticModel({1: 1.0, 0: -2.0}, {}, 0.0, 'BINARY')
    assert ising_machine.solve_surrogate_SA(model, 1, 1, set(), Sampler(), {(1, 0): 0.0, (0, 1): 1.0}) == (1, 0)


def test_energy_fallback_uses_variable_labels():
    model = dimod.BinaryQuadraticModel({1: 1.0, 0: -2.0}, {}, 0.0, 'BINARY')
    assert ising_machine._select_lowest_energy_candidate(model, [((1, 0), '10'), ((0, 1), '01')], 'test') == (1, 0)


@pytest.mark.parametrize('has_credentials', [False, True])
def test_qci_falls_back_without_optional_sdk(monkeypatch, has_credentials):
    monkeypatch.setitem(sys.modules, 'qci_client', None)
    monkeypatch.delenv('QCI_API_TOKEN', raising=False)
    monkeypatch.setenv('QCI_TOKEN', 'offline-test-placeholder' if has_credentials else '')
    model = dimod.BinaryQuadraticModel({0: -2.0, 1: 1.0}, {}, 0.0, 'BINARY')
    x, y, runtime = ising_machine.solve_surrogate_qci(model, 1, 1, set(), {(1, 0): 2.0, (0, 1): 3.0})
    assert (x, y) == (1, 0)
    assert runtime == 0.0


@pytest.mark.parametrize('copy_alias', [False, True])
def test_qci_device_runtime_counts_alias_once(copy_alias):
    # qci-client 5.0.3 adds a dirac-3 alias for its concrete device metric.
    device = {'samples': {'runtime': [1_000_000_000]}}
    alias = {'samples': {'runtime': [1_000_000_000]}} if copy_alias else device
    response = {'job_metrics': {'time_ns': {'device': {'dirac-3_qudit': device, 'dirac-3': alias}}}}
    assert ising_machine._extract_qci_metrics_device_runtime_seconds(response) == 1.0


def test_qci_sdk_shaped_response_and_runtime(monkeypatch):
    """Exercise process_job's documented results envelope with an offline client."""
    from types import SimpleNamespace
    class Client:
        def __init__(self, **kwargs):
            pass
        def upload_file(self, **kwargs):
            return {'file_id': 'offline-file'}
        def build_job_body(self, **kwargs):
            return kwargs
        def process_job(self, **kwargs):
            return {'job_info': {'job_id': 'offline-job'}, 'status': 'COMPLETED',
                    'results': {'solutions': [[1, 0, 1]]}}
        def get_job_metrics(self, **kwargs):
            node = {'samples': {'runtime': [1_000_000_000]}}
            return {'job_metrics': {'time_ns': {'device': {'dirac-3_qudit': node, 'dirac-3': node}}}}
    monkeypatch.setitem(sys.modules, 'qci_client', SimpleNamespace(QciClient=Client))
    monkeypatch.setenv('QCI_TOKEN', 'offline-test-placeholder')
    model = dimod.BinaryQuadraticModel({0: -2.0, 1: 1.0}, {}, 0.0, 'BINARY')
    assert ising_machine.solve_surrogate_qci(model, 1, 1, set(), {(1, 0): 2.0, (0, 1): 3.0}) == (1, 0, 1.0)
