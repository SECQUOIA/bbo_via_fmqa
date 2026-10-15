from datetime import datetime
import sys, os
import time
import csv
import random
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "fmqa")))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import numpy as np

import read_grid
from FM_surrogate import train_surrogate_model
from ising_machine import solve_surrogate_qci


DATASET_DIR = os.path.abspath(
    os.environ.get("FMQA_DATASET_DIR", os.path.join(os.getcwd(), "qhd_2D_graphs"))
)
OUTPUT_DIR = os.path.abspath(
    os.environ.get("FMQA_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
)
DEFAULT_RUNS_PER_GRAPH = 10
DEFAULT_MAX_CYCLES = 150
DEFAULT_CONVERGENCE_FRACTION = 0.10
SUMMARY_FIELDNAMES = [
    "graph_type",
    "run_index",
    "startx",
    "starty",
    "fm_runtime_sec",
    "qci_runtime_sec",
    "total_runtime_sec",
    "final_value",
    "optimal_value",
    "iterations",
    "solution_found",
]
RESULT_FIELDNAMES = [
    "graph_type",
    "x",
    "y",
    "percent_away_from_optimal",
    "fm_runtime_sec",
    "qci_runtime_sec",
    "total_runtime_sec",
    "iteration",
    "run_index",
]


def _display_path(path):
    """Return a repo-relative path for logging when possible."""
    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, start=PROJECT_ROOT)
    except ValueError:
        return os.path.basename(abs_path)
    return rel_path if not rel_path.startswith("..") else os.path.basename(abs_path)


def evaluate(x, y, grid):
    """Evaluate objective from grid."""
    return read_grid.obj_funct([x, y], grid)


def scale_value(y_val, obj_min, obj_max):
    """Normalize objective values to [0,1] range."""
    if y_val == np.inf:
        return np.inf
    if obj_max == obj_min:
        return 0.0
    return (y_val - obj_min) / (obj_max - obj_min)


def compute_gap_metrics(y_val, obj_min, obj_max):
    """Return the percent gap from the optimum on the normalized 0-1 scale."""
    final_value_scaled = scale_value(y_val, obj_min, obj_max)
    optimal_value_scaled = scale_value(obj_min, obj_min, obj_max)
    if final_value_scaled == np.inf:
        percent_away_from_optimal = np.inf
    else:
        percent_away_from_optimal = abs(final_value_scaled - optimal_value_scaled) * 100.0
    return percent_away_from_optimal


def compute_fmqa_budget(x_bound, y_bound, requested_max_cycles, patience_fraction):
    """
    Derive a simple FMQA budget from the binary search dimension.

    The encoded problem size grows with the number of bits, not with the raw
    number of grid cells, so budgeting from bit-count scales more naturally
    from 30x30 to 1000x1000.
    """
    num_binary_vars = len(read_grid.coord_bits(0, 0, x_bound, y_bound))
    budget_from_dimension = max(40, 6 * num_binary_vars)
    dataset_max_cycles = min(requested_max_cycles, budget_from_dimension)
    convergence_patience = min(
        dataset_max_cycles,
        max(1, int(np.ceil(dataset_max_cycles * patience_fraction))),
    )
    return dataset_max_cycles, convergence_patience, num_binary_vars


def select_start_points(all_feasible_points, runs_per_graph):
    """Pick the first N points from a random shuffle."""
    shuffled_points = list(all_feasible_points)
    random.seed(0)
    random.shuffle(shuffled_points)
    return shuffled_points[: min(runs_per_graph, len(shuffled_points))]


def run_single_fmqa(
    grid,
    obj_min,
    obj_max,
    x_bound,
    y_bound,
    start_point,
    max_cycles,
    convergence_patience,
):
    start_val_raw = evaluate(*start_point, grid)
    start_bits = read_grid.coord_bits(start_point[0], start_point[1], x_bound, y_bound)
    start_gap_percent = compute_gap_metrics(start_val_raw, obj_min, obj_max)

    evaluated_points = {start_point}
    xs = [start_bits]
    x_vectors = [[int(bit) for bit in start_bits]]
    ys = [scale_value(start_val_raw, obj_min, obj_max)]

    best_val_raw = start_val_raw
    best_coord = start_point
    best_iteration = 0
    no_improvement_count = 0
    iterations_completed = 0
    converged = False
    fm_runtime_sec = 0.0
    qci_runtime_sec = 0.0
    improvement_rows = [
        {
            "graph_type": "",
            "x": start_point[0],
            "y": start_point[1],
            "percent_away_from_optimal": start_gap_percent,
            "fm_runtime_sec": 0.0,
            "qci_runtime_sec": 0.0,
            "total_runtime_sec": 0.0,
            "iteration": 0,
            "run_index": "",
        }
    ]

    for t in range(max_cycles):
        fmqa_cycle = t + 1
        iterations_completed = fmqa_cycle

        print(f"\n=== FMQA Cycle {fmqa_cycle}/{max_cycles} ===")
        print(f"FMQA cycle {fmqa_cycle}: best so far = {best_val_raw} at {best_coord}")

        fm_start_time = time.perf_counter()
        fm = train_surrogate_model(xs, ys, x_vectors=x_vectors)
        fm_runtime_sec += time.perf_counter() - fm_start_time

        px, py, qci_cycle_runtime_sec = solve_surrogate_qci(
            fm,
            x_bound,
            y_bound,
            evaluated_points,
            grid,
        )
        qci_runtime_sec += qci_cycle_runtime_sec
        total_runtime_sec = fm_runtime_sec + qci_runtime_sec
        print(
            f"FM runtime so far = {fm_runtime_sec:.6f} s, "
            f"QCI device runtime so far = {qci_runtime_sec:.6f} s, "
            f"total tracked runtime = {total_runtime_sec:.6f} s"
        )

        print((px, py), "in grid?", (px, py) in grid if px is not None and py is not None else False)
        if px is not None and py is not None and (px, py) in grid:
            print("grid value:", grid[(px, py)])

        if px is None or py is None:
            print("All proposed samples were already evaluated or invalid.")
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} consecutive cycles.")

            if no_improvement_count >= convergence_patience:
                converged = True
                print(f"\nConvergence reached after {fmqa_cycle} FMQA cycles. Stopping.")
                break

            continue

        obj_val_raw = evaluate(px, py, grid)
        evaluated_points.add((px, py))

        if np.isfinite(obj_val_raw):
            print(
                f"FMQA cycle {fmqa_cycle}: objective = {obj_val_raw:.6f}, "
                f"current best = {best_val_raw:.6f}"
            )
        else:
            print(f"FMQA cycle {fmqa_cycle}: objective = inf (invalid or out of bounds)")

        if obj_val_raw < best_val_raw:
            print(f"New best found: {obj_val_raw} at ({px}, {py})")
            best_val_raw = obj_val_raw
            best_coord = (px, py)
            best_iteration = fmqa_cycle
            no_improvement_count = 0
            improvement_rows.append(
                {
                    "graph_type": "",
                    "x": px,
                    "y": py,
                    "percent_away_from_optimal": compute_gap_metrics(
                        best_val_raw, obj_min, obj_max
                    ),
                    "fm_runtime_sec": fm_runtime_sec,
                    "qci_runtime_sec": qci_runtime_sec,
                    "total_runtime_sec": total_runtime_sec,
                    "iteration": fmqa_cycle,
                    "run_index": "",
                }
            )
        else:
            print("No improvement this iteration.")
            no_improvement_count += 1
            if no_improvement_count >= convergence_patience:
                converged = True
                print(f"\nConvergence reached after {fmqa_cycle} FMQA cycles. Stopping.")
                break

        obj_val_scaled = scale_value(obj_val_raw, obj_min, obj_max)
        if obj_val_scaled == np.inf:
            obj_val_scaled = 1.1

        new_bits = read_grid.coord_bits(px, py, x_bound, y_bound)
        xs.append(new_bits)
        x_vectors.append([int(bit) for bit in new_bits])
        ys.append(obj_val_scaled)

        print(f"No improvement for {no_improvement_count} consecutive cycles.")
        print(f"Evaluated so far: {len(evaluated_points)} / total {len(grid)}")

    if max_cycles > 0 and iterations_completed == max_cycles and not converged:
        print(f"\nMax FMQA cycles reached without convergence.")

    total_runtime_sec = fm_runtime_sec + qci_runtime_sec
    found_optimal = best_val_raw <= obj_min
    percent_away_from_optimal = compute_gap_metrics(best_val_raw, obj_min, obj_max)

    print(f"\n--- Final Results ---")
    print(f"Best objective found by FMQA = {best_val_raw} at {best_coord}")
    print(f"The known global minimum for the entire dataset is = {obj_min}")
    print(f"FM runtime = {fm_runtime_sec:.6f} s")
    print(f"QCI device runtime = {qci_runtime_sec:.6f} s")
    print(f"Total tracked runtime (FM + QCI) = {total_runtime_sec:.6f} s")
    print(
        f"Percent away from optimal on normalized scale = "
        f"{percent_away_from_optimal:.2f}%"
    )

    return {
        "summary_row": {
            "graph_type": "",
            "run_index": "",
            "startx": start_point[0],
            "starty": start_point[1],
            "fm_runtime_sec": fm_runtime_sec,
            "qci_runtime_sec": qci_runtime_sec,
            "total_runtime_sec": total_runtime_sec,
            "final_value": best_val_raw,
            "optimal_value": obj_min,
            "iterations": iterations_completed,
            "solution_found": "T" if found_optimal else "F",
        },
        "improvement_rows": improvement_rows,
    }


def create_graph_results_csv(graphtype, timestamp):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(
        OUTPUT_DIR, f"{graphtype}_fmqa_qci_results_{timestamp}.csv"
    )

    if os.path.exists(results_path):
        if not csv_has_expected_header(results_path, RESULT_FIELDNAMES):
            raise ValueError(
                f"Existing result CSV has incompatible columns: {_display_path(results_path)}"
            )
    else:
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
            writer.writeheader()

    return results_path


def append_result_row(results_path, row):
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writerow(row)


def create_summary_csv(timestamp):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(
        OUTPUT_DIR, f"fmqa_qci_all_runs_{timestamp}.csv"
    )

    if os.path.exists(summary_path):
        if not csv_has_expected_header(summary_path, SUMMARY_FIELDNAMES):
            raise ValueError(
                f"Existing summary CSV has incompatible columns: {_display_path(summary_path)}"
            )
    else:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            writer.writeheader()

    return summary_path


def append_summary_row(summary_path, row):
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writerow(row)


def csv_has_expected_header(csv_path, expected_fieldnames):
    """Return True if a CSV exists and its header matches the expected schema."""
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    return header == list(expected_fieldnames)


def find_latest_summary_csv(expected_fieldnames=None):
    """Return the most recent all-runs summary CSV, optionally matching a schema."""
    pattern = os.path.join(OUTPUT_DIR, "fmqa_qci_all_runs_*.csv")
    summary_paths = sorted(glob.glob(pattern))
    if expected_fieldnames is None:
        return summary_paths[-1] if summary_paths else None

    for summary_path in reversed(summary_paths):
        if csv_has_expected_header(summary_path, expected_fieldnames):
            return summary_path
    return None


def extract_timestamp_from_summary_path(summary_path):
    """Recover the run timestamp from a summary CSV filename."""
    basename = os.path.basename(summary_path)
    prefix = "fmqa_qci_all_runs_"
    suffix = ".csv"
    if basename.startswith(prefix) and basename.endswith(suffix):
        return basename[len(prefix):-len(suffix)]
    raise ValueError(f"Unexpected summary CSV name: {basename}")


def load_completed_runs(summary_path):
    """Load completed (graph_type, run_index) pairs from an existing summary CSV."""
    completed_runs = set()
    if not os.path.exists(summary_path):
        return completed_runs

    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph_type = row.get("graph_type", "").strip()
            run_index = row.get("run_index", "").strip()
            if not graph_type or not run_index:
                continue
            try:
                completed_runs.add((graph_type, int(run_index)))
            except ValueError:
                continue

    return completed_runs


def prune_result_rows_for_run(results_path, graphtype, run_index):
    """Remove any partially written result rows for a run before re-appending."""
    if not os.path.exists(results_path):
        return

    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))

    filtered_rows = [
        row
        for row in rows
        if not (
            row.get("graph_type") == graphtype
            and row.get("run_index") == str(run_index)
        )
    ]
    if len(filtered_rows) == len(rows):
        return

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(filtered_rows)


def main():
    runs_per_graph = int(os.environ.get("FMQA_RUNS_PER_GRAPH", str(DEFAULT_RUNS_PER_GRAPH)))
    if runs_per_graph <= 0:
        raise ValueError("FMQA_RUNS_PER_GRAPH must be a positive integer.")

    requested_max_cycles = int(os.environ.get("FMQA_MAX_CYCLES", str(DEFAULT_MAX_CYCLES)))
    if requested_max_cycles <= 0:
        raise ValueError("FMQA_MAX_CYCLES must be a positive integer.")

    convergence_patience_fraction = float(
        os.environ.get("FMQA_CONVERGENCE_FRACTION", str(DEFAULT_CONVERGENCE_FRACTION))
    )
    if not (0 < convergence_patience_fraction <= 1):
        raise ValueError("FMQA_CONVERGENCE_FRACTION must be in the interval (0, 1].")

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {_display_path(DATASET_DIR)}"
        )

    dataset_paths = sorted(
        os.path.join(DATASET_DIR, name)
        for name in os.listdir(DATASET_DIR)
        if name.endswith(".csv")
    )
    if not dataset_paths:
        raise ValueError(
            f"No CSV files found in dataset directory: {_display_path(DATASET_DIR)}"
        )

    print(f"Resolved {len(dataset_paths)} dataset(s) from {_display_path(DATASET_DIR)}")

    latest_summary_path = find_latest_summary_csv()
    existing_summary_path = find_latest_summary_csv(
        expected_fieldnames=SUMMARY_FIELDNAMES
    )
    if existing_summary_path is not None:
        summary_path = existing_summary_path
        run_timestamp = extract_timestamp_from_summary_path(summary_path)
        completed_run_keys = load_completed_runs(summary_path)
        print(f"Resuming from existing summary CSV: {_display_path(summary_path)}")
        print(f"Found {len(completed_run_keys)} completed run(s) to skip.")
    else:
        if latest_summary_path is not None:
            print(
                "Latest summary CSV uses an older runtime schema and will not be "
                f"resumed: {_display_path(latest_summary_path)}"
            )
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        summary_path = create_summary_csv(run_timestamp)
        completed_run_keys = set()
        print(f"Original summary CSV will be updated at: {_display_path(summary_path)}")

    completed_runs = len(completed_run_keys)

    for dataset_index, dataset_path in enumerate(dataset_paths, start=1):
        graphtype = os.path.splitext(os.path.basename(dataset_path))[0]
        grid, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(filename=dataset_path)
        all_feasible_points = [p for p, v in grid.items() if np.isfinite(v)]

        if not all_feasible_points:
            raise ValueError(f"No feasible points are available for dataset: {graphtype}")

        dataset_max_cycles, convergence_patience, num_binary_vars = compute_fmqa_budget(
            x_bound=x_bound,
            y_bound=y_bound,
            requested_max_cycles=requested_max_cycles,
            patience_fraction=convergence_patience_fraction,
        )

        start_points = select_start_points(all_feasible_points, runs_per_graph)

        print(f"\n=== Dataset {dataset_index}/{len(dataset_paths)}: {graphtype} ===")
        print(f"Grid loaded: {len(grid)} points, x in [0,{x_bound}], y in [0,{y_bound}]")
        print(f"Objective range: [{obj_min}, {obj_max}]")
        print(f"Total feasible points: {len(all_feasible_points)}")
        print(f"Binary decision variables: {num_binary_vars}")
        print(f"Max FMQA cycles set to {dataset_max_cycles}")
        print(
            f"Convergence patience set to {convergence_patience} "
            f"(fraction={convergence_patience_fraction})"
        )
        print(f"Using {len(start_points)} seeded single-start runs.")

        results_path = create_graph_results_csv(graphtype, run_timestamp)
        print(
            f"Results CSV for {graphtype} will be written to: {_display_path(results_path)}"
        )

        for run_index, start_point in enumerate(start_points, start=1):
            run_key = (graphtype, run_index)
            if run_key in completed_run_keys:
                print(
                    f"Skipping completed run {run_index}/{len(start_points)} for {graphtype}."
                )
                continue

            print(
                f"\n--- Run {run_index}/{len(start_points)} for {graphtype} "
                f"starting at {start_point} ---"
            )
            prune_result_rows_for_run(results_path, graphtype, run_index)
            run_result = run_single_fmqa(
                grid,
                obj_min,
                obj_max,
                x_bound,
                y_bound,
                start_point,
                dataset_max_cycles,
                convergence_patience,
            )
            for improvement_row in run_result["improvement_rows"]:
                improvement_row["graph_type"] = graphtype
                improvement_row["run_index"] = run_index
                append_result_row(results_path, improvement_row)

            summary_row = run_result["summary_row"]
            summary_row["graph_type"] = graphtype
            summary_row["run_index"] = run_index
            append_summary_row(summary_path, summary_row)
            completed_run_keys.add(run_key)
            completed_runs += 1
            print(
                f"Updated {results_path} with "
                f"{len(run_result['improvement_rows'])} improvement row(s) for run {run_index}."
            )

    print(f"\nCompleted {completed_runs} run(s) across {len(dataset_paths)} dataset(s).")
    print(f"Original summary CSV written to: {_display_path(summary_path)}")
    print(f"Per-graph result CSVs written to: {_display_path(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
