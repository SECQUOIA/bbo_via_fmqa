import os

import read_grid
import numpy as np
from math import isfinite

DEFAULT_QCI_API_URL = "https://api.qci-prod.com"
QCI_TOKEN_ENV_VARS = ("QCI_TOKEN", "QCI_API_TOKEN")
QCI_API_URL_ENV_VAR = "QCI_API_URL"


def _get_qci_connection_settings():
    """Load QCI credentials from the environment."""
    qci_token = next(
        (os.environ.get(name, "").strip() for name in QCI_TOKEN_ENV_VARS if os.environ.get(name, "").strip()),
        "",
    )
    qci_api_url = os.environ.get(QCI_API_URL_ENV_VAR, DEFAULT_QCI_API_URL).strip()
    if not qci_api_url:
        qci_api_url = DEFAULT_QCI_API_URL
    return qci_token, qci_api_url

def solve_surrogate_dwave(
    fm_model,
    x_bound,
    y_bound,
    evaluated_points,
    grid,
):
    """
    Propose the next 2D candidate point from the surrogate model with D-Wave.

    The function converts the factorization-machine surrogate into a binary
    quadratic model, samples it with `LeapHybridSampler`, decodes sampled bit
    strings into `(x, y)` coordinates, and returns the lowest-energy candidate
    that is still inside the valid grid and has not been evaluated yet. If the
    D-Wave samples do not contain a usable point, it falls back to scoring all
    remaining grid points directly with the BQM energy and returns the best one.

    Args:
        fm_model: Trained surrogate model represented as a
            `dimod.BinaryQuadraticModel`, or an FM-like object that exposes
            `.linear`, `.quadratic`, `.offset`, and `.energy()`.
        x_bound (int): Maximum valid x index in the search grid.
        y_bound (int): Maximum valid y index in the search grid.
        evaluated_points (set[tuple[int, int]]): Coordinates that have already
            been evaluated and therefore should not be proposed again.
        grid (dict[tuple[int, int], float]): Mapping from grid coordinates to
            objective values. Entries with non-finite values are treated as
            invalid candidates.

    Returns:
        tuple[int | None, int | None]: The selected `(x, y)` coordinate, or
        `(None, None)` if no valid unevaluated point can be produced.
    """
    import dimod
    from dwave.system import LeapHybridSampler

    # ---------- 0. Build a BQM from fm_model (if needed) ----------
    try:
        if isinstance(fm_model, dimod.BinaryQuadraticModel):
            bqm = fm_model
        else:
            # Assume fm_model has .linear, .quadratic, .offset like FMBQM
            linear_raw = getattr(fm_model, "linear", {})
            quad_raw = getattr(fm_model, "quadratic", {})
            offset = float(getattr(fm_model, "offset", 0.0))

            linear = {int(i): float(h) for i, h in linear_raw.items()}
            quadratic = {(int(i), int(j)): float(J)
                         for (i, j), J in quad_raw.items()}

            # binary variables (0/1) because coord_bits uses 0/1 bits
            bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)
    except Exception as e:
        print(f"[solve_surrogate_dwave] Error building BQM from fm_model: {e}")
        return None, None

    # ---------- 1. Try D-Wave hybrid sampler ----------
    candidates = []

    try:
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm)
    except Exception as e:
        print(f"[solve_surrogate_dwave] Error calling LeapHybridSampler: {e}")
        sampleset = None

    if sampleset is not None and len(sampleset):
        variables = sorted(sampleset.variables)
        for sample, energy in sampleset.data(['sample', 'energy']):
            # sample: {var_label: value, ...}
            bitlist = []
            for variable in variables:
                v = sample[variable]
                # handle SPIN or BINARY just in case
                if v in (-1, 1):
                    bit = 0 if v == -1 else 1
                else:
                    bit = int(v)
                bitlist.append(bit)

            bitstring = "".join(str(b) for b in bitlist)

            # Decode bits -> (x, y) using your existing convention
            try:
                cand_x, cand_y = read_grid.bits_to_int(
                    bitstring, x_max=x_bound, y_max=y_bound, lsb_first=False
                )
            except Exception:
                continue

            # Bounds
            if not (0 <= cand_x <= x_bound and 0 <= cand_y <= y_bound):
                continue
            # Must exist in grid
            if (cand_x, cand_y) not in grid:
                continue
            # Must not have NaN objective
            val = grid[(cand_x, cand_y)]
            if val is None or not np.isfinite(val):
                continue
            # Must be a new point
            if (cand_x, cand_y) in evaluated_points:
                continue

            candidates.append((cand_x, cand_y, energy))

    if candidates:
        # choose the candidate with the lowest energy
        cand_x, cand_y, _ = min(candidates, key=lambda t: t[2])
        print(f"[solve_surrogate_dwave] New candidate from D-Wave: ({cand_x}, {cand_y})")
        return cand_x, cand_y

    print("[solve_surrogate_dwave] No valid *new* samples from D-Wave. "
          "Falling back to BQM-energy-based search.")

    # ---------- 2. Fallback: BQM-energy-based search over remaining grid ----------
    remaining_candidates = [
        ((x, y), read_grid.coord_bits(x, y, x_bound, y_bound))
        for (x, y), val in grid.items()
        if (x, y) not in evaluated_points and np.isfinite(val)
    ]
    candidate = _select_lowest_energy_candidate(
        bqm, remaining_candidates, log_prefix="solve_surrogate_dwave"
    )
    if candidate is None:
        return None, None
    return candidate

def _coerce_nonnegative_float(value):
    """Return a finite non-negative float, or None if coercion fails."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def _extract_duration_seconds_from_runtime_node(runtime_node):
    """Convert a runtime payload expressed in nanoseconds into seconds."""
    if runtime_node is None:
        return None

    if isinstance(runtime_node, dict):
        child_durations = [
            _extract_duration_seconds_from_runtime_node(child)
            for child in runtime_node.values()
        ]
        child_durations = [duration for duration in child_durations if duration is not None]
        if child_durations:
            return sum(child_durations)
        return None

    if isinstance(runtime_node, (list, tuple)):
        child_durations = [
            _extract_duration_seconds_from_runtime_node(child)
            for child in runtime_node
        ]
        child_durations = [duration for duration in child_durations if duration is not None]
        if child_durations:
            return sum(child_durations)
        return None

    numeric = _coerce_nonnegative_float(runtime_node)
    if numeric is None:
        return None
    return numeric / 1e9


def _extract_qci_metrics_device_runtime_seconds(metrics_response):
    """Return device-only runtime from QCI metrics, if present."""
    if not isinstance(metrics_response, dict):
        return None

    job_metrics = metrics_response.get("job_metrics", {})
    if not isinstance(job_metrics, dict):
        return None

    time_ns = job_metrics.get("time_ns", {})
    if not isinstance(time_ns, dict):
        return None

    device_metrics = time_ns.get("device", {})
    if not isinstance(device_metrics, dict):
        return None

    durations = []
    seen_nodes = set()
    concrete_dirac = any(name.startswith("dirac-3_") for name in device_metrics)

    for device_name, device_node in device_metrics.items():
        # The SDK exposes dirac-3 as an alias of the concrete device metric.
        if device_name == "dirac-3" and concrete_dirac:
            continue
        if id(device_node) in seen_nodes:
            continue
        seen_nodes.add(id(device_node))
        if not isinstance(device_node, dict):
            continue

        samples_node = device_node.get("samples", {})
        if isinstance(samples_node, dict):
            start_job_ts = _coerce_nonnegative_float(samples_node.get("start_job_ts"))
            end_job_ts = _coerce_nonnegative_float(samples_node.get("end_job_ts"))
            if (
                start_job_ts is not None
                and end_job_ts is not None
                and end_job_ts >= start_job_ts
            ):
                durations.append((end_job_ts - start_job_ts) / 1e9)
                continue

            runtime_seconds = _extract_duration_seconds_from_runtime_node(
                samples_node.get("runtime")
            )
            if runtime_seconds is not None:
                durations.append(runtime_seconds)
                continue

        runtime_seconds = _extract_duration_seconds_from_runtime_node(
            device_node.get("runtime")
        )
        if runtime_seconds is not None:
            durations.append(runtime_seconds)
            continue

        print(
            f"[solve_surrogate_qci] No device-runtime fields found in metrics for {device_name}."
        )

    if durations:
        return sum(durations)
    return None


def _extract_qci_device_runtime_seconds(job_response, client):
    """
    Return QCI device-only runtime in seconds.

    Preference order:
    1. Detailed `get_job_metrics(...)` device timestamps/runtime.
    2. Rounded `job_info -> job_result -> device_usage_s`.
    3. 0.0 if neither source is available.
    """
    job_id = None
    if isinstance(job_response, dict):
        job_id = job_response.get("job_id")
        job_info = job_response.get("job_info", {})
        if isinstance(job_info, dict):
            job_id = job_info.get("job_id", job_id)

    if job_id is not None and hasattr(client, "get_job_metrics"):
        try:
            metrics_response = client.get_job_metrics(job_id=job_id)
            metrics_runtime_seconds = _extract_qci_metrics_device_runtime_seconds(
                metrics_response
            )
            if metrics_runtime_seconds is not None:
                return metrics_runtime_seconds
        except Exception as e:
            print(f"[solve_surrogate_qci] Unable to fetch QCI job metrics: {e}")

    if isinstance(job_response, dict):
        job_info = job_response.get("job_info", {})
        if isinstance(job_info, dict):
            job_result = job_info.get("job_result", {})
            if isinstance(job_result, dict):
                device_usage_seconds = _coerce_nonnegative_float(
                    job_result.get("device_usage_s")
                )
                if device_usage_seconds is not None:
                    return device_usage_seconds

    return 0.0


def solve_surrogate_qci(fm_model, x_bound, y_bound, evaluated_points, grid):
    """
    Propose the next 2D candidate point by submitting the surrogate to QCI.

    The function rewrites the surrogate model as a quadratic polynomial with an
    ancillary variable, sends that polynomial to the QCI Dirac-3 sampler,
    parses the returned bit strings, decodes them into `(x, y)` coordinates, and
    returns the lowest-surrogate-energy valid point that is inside the grid and
    has not already been evaluated. If QCI returns no usable point or the API
    call fails, it falls back to the surrogate's best remaining grid point.

    Args:
        fm_model: Trained surrogate model with `.linear` and `.quadratic`
            coefficient dictionaries describing the binary quadratic objective.
        x_bound (int): Maximum valid x index in the search grid.
        y_bound (int): Maximum valid y index in the search grid.
        evaluated_points (set[tuple[int, int]]): Coordinates that have already
            been sampled and should be skipped.
        grid (dict[tuple[int, int], float]): Mapping from valid grid
            coordinates to objective values. Non-finite entries are rejected.

    Returns:
        tuple[int | None, int | None, float]: A valid unevaluated `(x, y)`
        candidate plus the QCI device-only runtime in seconds. If no valid
        candidate can be produced, the coordinate entries are `None` while the
        runtime still reflects any device time spent by the attempted job.
    """
    from math import isfinite

    def fallback_candidate(qci_runtime_sec=0.0):
        candidate = _select_lowest_energy_candidate(
            fm_model,
            [
                ((x, y), read_grid.coord_bits(x, y, x_bound, y_bound))
                for (x, y), val in grid.items()
                if (x, y) not in evaluated_points and np.isfinite(val)
            ],
            log_prefix="solve_surrogate_qci",
        )
        if candidate is None:
            return None, None, qci_runtime_sec
        return candidate[0], candidate[1], qci_runtime_sec

    # ---------- 1. Convert FM model to homogeneous quadratic polynomial ----------
    num_original_vars = len(read_grid.coord_bits(0, 0, x_bound, y_bound))
    ancillary_idx = num_original_vars
    total_vars = num_original_vars + 1
    poly_data = []

    linear_coeffs = list(fm_model.linear.values())
    quadratic_coeffs = list(fm_model.quadratic.values())
    all_coeffs = np.abs(np.array(linear_coeffs + quadratic_coeffs)) if (linear_coeffs or quadratic_coeffs) else np.array([1.0])
    penalty = 10 * np.max(all_coeffs)

    # strong ancilla bias → encourage a = 1
    poly_data.append({"idx": [ancillary_idx, ancillary_idx], "val": -penalty})

    # linear terms  (x_i * a)
    for i, h in fm_model.linear.items():
        if h != 0:
            poly_data.append({"idx": sorted([int(i), ancillary_idx]), "val": float(h)})

    # quadratic terms (x_i * x_j)
    for (i, j), J_val in fm_model.quadratic.items():
        if J_val != 0:
            poly_data.append({"idx": sorted([int(i), int(j)]), "val": float(J_val)})

    # ---------- 2. Submit polynomial to Dirac-3 ----------
    print("Submitting Hamiltonian job to QCI (for Dirac-3)...")

    qci_token, qci_api_url = _get_qci_connection_settings()
    if not qci_token:
        print(
            "[solve_surrogate_qci] Missing QCI credentials. "
            "Set QCI_TOKEN or QCI_API_TOKEN to enable QCI sampling. "
            "Falling back to surrogate energy ranking."
        )
        return fallback_candidate()
    try:
        from qci_client import QciClient

        client = QciClient(api_token=qci_token, url=qci_api_url)

        polynomial_file = {
            "file_name": "fmqa-poly-file-dirac3",
            "file_config": {
                "polynomial": {
                    "num_variables": total_vars,
                    "max_degree": 2,
                    "min_degree": 1,
                    "data": poly_data,
                }
            },
        }

        poly_file_id = client.upload_file(file=polynomial_file)["file_id"]

        job_body = client.build_job_body(
            job_type="sample-hamiltonian-integer",
            job_params={
                "num_samples": 100,
                "device_type": "dirac-3",
                "num_levels": [2] * total_vars,
                "return_samples": True,
            },
            polynomial_file_id=poly_file_id,
        )

        print("Job submitted. Waiting for completion...")
        response = client.process_job(job_body=job_body, wait=True)
        print("Job completed successfully.")
    except Exception as e:
        print(f"An error occurred while communicating with QCI: {e}")
        return fallback_candidate()

    qci_runtime_sec = _extract_qci_device_runtime_seconds(response, client)
    print(f"[solve_surrogate_qci] QCI device runtime = {qci_runtime_sec:.6f} s")

    # ---------- 3. Extract and normalize solutions ----------
    # Accept multiple possible keys / formats
    candidates = []
    def coerce_bitlist(item):
        if item is None:
            return None
        if isinstance(item, dict):
            sol = item.get("solution") or item.get("values") or item
            if isinstance(sol, dict):
                out = [0] * total_vars
                for k, v in sol.items():
                    try:
                        k2 = int(k)
                        if 0 <= k2 < total_vars:
                            out[k2] = int(v)
                    except Exception:
                        continue
                return out
            if isinstance(sol, (list, tuple)):
                arr = [int(x) for x in sol]
                if len(arr) < total_vars:
                    arr += [0] * (total_vars - len(arr))
                return arr[:total_vars]
        if isinstance(item, (list, tuple)):
            arr = [int(x) for x in item]
            if len(arr) < total_vars:
                arr += [0] * (total_vars - len(arr))
            return arr[:total_vars]
        return None

    for key in ("results", "result", "data", ""):
        node = response.get(key, response) if isinstance(response, dict) else None
        if not isinstance(node, dict):
            continue
        for arr_key in ("solutions", "samples", "states"):
            arr = node.get(arr_key, [])
            if isinstance(arr, dict):
                arr = arr.get("data", []) or arr.get("list", [])
            if isinstance(arr, list):
                for it in arr:
                    bits = coerce_bitlist(it)
                    if bits is not None:
                        candidates.append(bits)

    if not candidates:
        print("QCI job returned no solutions (after parsing).")
        return fallback_candidate(qci_runtime_sec=qci_runtime_sec)

    # ---------- 4. Post-process and decode to (x, y) ----------
    valid_candidates = []
    for bits in candidates:
        # drop ancilla for decoding
        fm_bits = bits[:num_original_vars]
        bitstring = "".join(map(str, fm_bits))

        try:
            cand_x, cand_y = read_grid.bits_to_int(
                bitstring,
                x_max=x_bound,
                y_max=y_bound,
                lsb_first=False,
            )
        except Exception:
            continue

        # Bounds and grid membership check
        if not (0 <= cand_x <= x_bound and 0 <= cand_y <= y_bound):
            continue
        if (cand_x, cand_y) not in grid:
            continue
        val = grid[(cand_x, cand_y)]
        if val is None or not isfinite(val):
            continue
        if (cand_x, cand_y) in evaluated_points:
            continue

        valid_candidates.append(((cand_x, cand_y), bitstring))

    if valid_candidates:
        candidate = _select_lowest_energy_candidate(
            fm_model,
            valid_candidates,
            log_prefix="solve_surrogate_qci",
        )
        if candidate is None:
            return None, None, qci_runtime_sec
        print(f"Found valid new candidate from QCI: {candidate}")
        return candidate[0], candidate[1], qci_runtime_sec

    print("QCI returned solutions, but all were previously evaluated or invalid.")
    return fallback_candidate(qci_runtime_sec=qci_runtime_sec)


def _sample_bits_and_energy(sampleset):
    """Yield decoded bit strings and energies from sampler results."""
    variables = sorted(sampleset.variables)
    for sample, energy in sampleset.data(['sample', 'energy']):
        bitlist = []
        for variable in variables:
            v = sample[variable]
            if v in (-1, 1):
                bit = 0 if v == -1 else 1
            else:
                bit = int(v)
            bitlist.append(bit)
        yield "".join(map(str, bitlist)), energy


def _select_lowest_energy_candidate(fm_model, encoded_candidates, log_prefix):
    """Score remaining encoded candidates directly with the surrogate energy."""
    encoded_candidates = list(encoded_candidates)
    if not encoded_candidates:
        print(f"[{log_prefix}] No remaining unevaluated grid points.")
        return None

    try:
        variables = sorted(fm_model.variables)
        best_candidate = None
        best_energy = None

        for candidate, bitstring in encoded_candidates:
            bits = [int(c) for c in bitstring]
            if len(bits) < len(variables):
                bits.extend([0] * (len(variables) - len(bits)))
            elif len(bits) > len(variables):
                bits = bits[: len(variables)]

            sample = {
                variable: bits[index]
                for index, variable in enumerate(variables)
            }
            energy = fm_model.energy(sample)

            if best_energy is None or energy < best_energy:
                best_candidate = candidate
                best_energy = energy

        print(f"[{log_prefix}] Fallback energy-ranked candidate: {best_candidate}")
        return best_candidate
    except Exception as e:
        print(f"[{log_prefix}] Energy fallback failed: {e}")
        return None


def _solve_surrogate_sa_common(
    fm_model,
    sampler,
    num_reads,
    decode_fn,
    is_valid_candidate,
    log_prefix,
):
    """Shared SA candidate selection for both 2D and 3D decoders."""
    try:
        sampleset = sampler.sample(fm_model, num_reads=num_reads)
    except Exception as e:
        print(f"[{log_prefix}] Sampler error: {e}")
        return None

    candidates = []
    for bitstring, energy in _sample_bits_and_energy(sampleset):
        try:
            candidate = decode_fn(bitstring)
        except Exception:
            continue

        if not is_valid_candidate(candidate):
            continue

        candidates.append((candidate, energy))

    if candidates:
        candidate, _ = min(candidates, key=lambda item: item[1])
        print(f"[{log_prefix}] New candidate from SA: {candidate}")
        return candidate

    print(f"[{log_prefix}] No valid new candidate found.")
    return None


def solve_surrogate_SA(fm_model, x_bound, y_bound, evaluated_points, sampler, grid):
    """
    Propose the next 2D candidate point with a simulated annealing sampler.

    The function samples the surrogate BQM, decodes each sampled bit string into
    an `(x, y)` coordinate, filters out out-of-bounds, invalid, and previously
    evaluated points, and returns the valid candidate with the lowest sampled
    energy.

    Args:
        fm_model: Trained surrogate binary quadratic model to be sampled.
        x_bound (int): Maximum valid x index in the search grid.
        y_bound (int): Maximum valid y index in the search grid.
        evaluated_points (set[tuple[int, int]]): Coordinates that have already
            been evaluated and should not be returned again.
        sampler: Sampler object with a `.sample(...)` method compatible with the
            supplied BQM, such as a dimod simulated annealing sampler.
        grid (dict[tuple[int, int], float]): Mapping from grid coordinates to
            objective values, used to reject invalid or missing points.

    Returns:
        tuple[int | None, int | None]: The best valid `(x, y)` candidate found
        in the sampled states, or `(None, None)` if none are usable.
    """
    def decode_candidate(bitstring):
        return read_grid.bits_to_int(
            bitstring, x_max=x_bound, y_max=y_bound, lsb_first=False
        )

    def is_valid_candidate(candidate):
        cand_x, cand_y = candidate
        if not (0 <= cand_x <= x_bound and 0 <= cand_y <= y_bound):
            return False
        if (cand_x, cand_y) not in grid:
            return False
        val = grid[(cand_x, cand_y)]
        if val is None or not np.isfinite(val):
            return False
        if (cand_x, cand_y) in evaluated_points:
            return False
        return True

    candidate = _solve_surrogate_sa_common(
        fm_model=fm_model,
        sampler=sampler,
        num_reads=50,
        decode_fn=decode_candidate,
        is_valid_candidate=is_valid_candidate,
        log_prefix="solve_surrogate_SA",
    )
    if candidate is None:
        candidate = _select_lowest_energy_candidate(
            fm_model,
            [
                ((x, y), read_grid.coord_bits(x, y, x_bound, y_bound))
                for (x, y), val in grid.items()
                if (x, y) not in evaluated_points and np.isfinite(val)
            ],
            log_prefix="solve_surrogate_SA",
        )
        if candidate is None:
            return None, None
    return candidate


def solve_surrogate_SA_3d(fm_model, x_bound, y_bound, z_bound, evaluated_points, sampler, grid):
    """
    Propose the next 3D candidate point with a simulated annealing sampler.

    This wrapper reuses the same simulated annealing selection flow as the 2D
    solver, but keeps a separate public function because 3D decoding requires
    different bounds, a different bit-string decoder, and a different return
    shape.

    Args:
        fm_model: Trained surrogate binary quadratic model to be sampled.
        x_bound (int): Maximum valid x index in the search grid.
        y_bound (int): Maximum valid y index in the search grid.
        z_bound (int): Maximum valid z index in the search grid.
        evaluated_points (set[tuple[int, int, int]]): 3D coordinates that have
            already been evaluated and should be skipped.
        sampler: Sampler object with a `.sample(...)` method compatible with the
            supplied BQM.
        grid (dict[tuple[int, int, int], float]): Mapping from 3D coordinates to
            objective values, used to validate decoded candidates.

    Returns:
        tuple[int | None, int | None, int | None]: The best valid
        `(x, y, z)` candidate found, or `(None, None, None)` if no usable point
        is produced.
    """
    def decode_candidate(bitstring):
        return read_grid.bits_to_int_3d(
            bitstring, x_bound, y_bound, z_bound, lsb_first=False
        )

    def is_valid_candidate(candidate):
        cand_x, cand_y, cand_z = candidate
        if not (0 <= cand_x <= x_bound and 0 <= cand_y <= y_bound and 0 <= cand_z <= z_bound):
            return False
        if (cand_x, cand_y, cand_z) not in grid:
            return False
        val = grid[(cand_x, cand_y, cand_z)]
        if val is None or not np.isfinite(val):
            return False
        if (cand_x, cand_y, cand_z) in evaluated_points:
            return False
        return True

    candidate = _solve_surrogate_sa_common(
        fm_model=fm_model,
        sampler=sampler,
        num_reads=50,
        decode_fn=decode_candidate,
        is_valid_candidate=is_valid_candidate,
        log_prefix="solve_surrogate_SA_3d",
    )
    if candidate is None:
        candidate = _select_lowest_energy_candidate(
            fm_model,
            [
                (
                    (x, y, z),
                    read_grid.coord_bits_3d(x, y, z, x_bound, y_bound, z_bound),
                )
                for (x, y, z), val in grid.items()
                if (x, y, z) not in evaluated_points and np.isfinite(val)
            ],
            log_prefix="solve_surrogate_SA_3d",
        )
        if candidate is None:
            return None, None, None
    return candidate
