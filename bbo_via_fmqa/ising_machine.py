
import read_grid
import numpy as np
from qci_client import QciClient
from math import isfinite

# QCI_TOKEN = 'your_api_token'
# QCI_API_URL = 'your_qci_api_url'


def solve_surrogate_qci(fm_model, x_bound, y_bound, evaluated_points, grid):
    """
    Solve the surrogate model via QCI Dirac-3 to propose the next (x, y) point.

    Fixes:
        • Removes accidental (x,y) swap
        • Enforces dataset constraint y ≤ x
        • Filters out invalid/out-of-grid points
        • Robustly parses QCI job results
    """
    import numpy as np
    from math import isfinite
    import read_grid

    # ---------- 1. Convert FM model to homogeneous quadratic polynomial ----------
    num_original_vars = len(fm_model.linear)
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
    try:
        client = QciClient(api_token=QCI_TOKEN, url=QCI_API_URL)

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
        return None, None

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
        return None, None

    # ---------- 4. Post-process and decode to (x, y) ----------
    for bits in candidates:
        # drop ancilla for decoding
        fm_bits = bits[:num_original_vars]
        bitstring = "".join(map(str, fm_bits))

        try:
            cand_x, cand_y = read_grid.bits_to_int(bitstring, lsb_first=False)
        except Exception:
            continue

        # Enforce dataset symmetry y ≤ x
        if cand_x < cand_y:
            cand_x, cand_y = cand_y, cand_x

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

        print(f"Found valid new candidate from QCI: ({cand_x}, {cand_y})")
        return cand_x, cand_y

    print("QCI returned solutions, but all were previously evaluated or invalid.")
    return None, None

def solve_surrogate(fm_model, x_bound, y_bound, evaluated_points, sampler):
    """
    Solves the surrogate model using an simulated quantum annealer (Ising machine sampler) 
    to find the best new candidate point.

    Args:
        fm_model (FMBQM): The trained surrogate model.
        x_bound (int): The maximum value for the x-coordinate.
        y_bound (int): The maximum value for the y-coordinate.
        evaluated_points (set): A set of (x, y) tuples that have already been evaluated.
        sampler: The dimod sampler to use for solving.

    Returns:
        tuple or None: An (x, y) tuple for the new candidate, or None if no valid new
                       candidate was found.
    """
    sampleset = sampler.sample(fm_model, num_reads=100)

    for sample, energy in sampleset.data(['sample', 'energy']):
        bitlist = [sample[i] for i in sorted(sample.keys())]
        bitstring = "".join(map(str, bitlist))

        # Decode normally first
        cand_x, cand_y = read_grid.bits_to_int(bitstring, lsb_first=False)
        # Swap x and y because coord_bits encoded them in the opposite order
        cand_x, cand_y = cand_y, cand_x

        # Validate candidate
        if (0 <= cand_x <= x_bound) and (0 <= cand_y <= y_bound):
            if (cand_x, cand_y) not in evaluated_points:
                return cand_x, cand_y

    return None, None


