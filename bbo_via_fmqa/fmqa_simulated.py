from datetime import datetime
import sys, os
import time
import csv
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "fmqa")))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import numpy as np

import read_grid
from FM_surrogate import train_surrogate_model
from ising_machine import solve_surrogate_SA
from dimod import SimulatedAnnealingSampler


DATASET_DIR = os.path.abspath(
    os.environ.get("FMQA_DATASET_DIR", os.path.join(os.getcwd(), "qhd_2D_graphs"))
)
OUTPUT_DIR = os.path.abspath(
    os.environ.get("FMQA_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
)
DEFAULT_RUNS_PER_GRAPH = 10
DEFAULT_MAX_CYCLES = 150
DEFAULT_CONVERGENCE_FRACTION = 0.01
SUMMARY_FIELDNAMES = [
    "graph_type",
    "run_index",
    "startx",
    "starty",
    "runtime_sec",
    "final_value",
    "optimal_value",
    "iterations",
    "solution_found",
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


def select_start_points(all_feasible_points, runs_per_graph):
    """Pick the first N points from a random shuffle."""
    shuffled_points = list(all_feasible_points)
    random.seed(0)      # Use a fixed seed for reproducibility of start point selection
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
    sampler = SimulatedAnnealingSampler()
    start_time = time.perf_counter()

    start_val_raw = evaluate(*start_point, grid)
    start_bits = read_grid.coord_bits(start_point[0], start_point[1], x_bound, y_bound)

    evaluated_points = {start_point}
    xs = [start_bits]
    x_vectors = [[int(bit) for bit in start_bits]]
    ys = [scale_value(start_val_raw, obj_min, obj_max)]

    best_val_raw = start_val_raw
    best_coord = start_point
    no_improvement_count = 0
    iterations_completed = 0
    converged = False

    for t in range(max_cycles):
        fmqa_cycle = t + 1
        iterations_completed = fmqa_cycle

        print(f"\n=== FMQA Cycle {fmqa_cycle}/{max_cycles} ===")
        print(f"FMQA cycle {fmqa_cycle}: best so far = {best_val_raw} at {best_coord}")

        fm = train_surrogate_model(xs, ys, x_vectors=x_vectors)

        px, py = solve_surrogate_SA(
            fm,
            x_bound,
            y_bound,
            evaluated_points,
            sampler,
            grid,
        )

        print((px, py), "in grid?", (px, py) in grid)
        if (px, py) in grid:
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
            no_improvement_count = 0
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

    runtime_seconds = time.perf_counter() - start_time
    found_optimal = best_val_raw <= obj_min

    print(f"\n--- Final Results ---")
    print(f"Best objective found by FMQA = {best_val_raw} at {best_coord}")
    print(f"The known global minimum for the entire dataset is = {obj_min}")

    return {
        "startx": start_point[0],
        "starty": start_point[1],
        "runtime_sec": runtime_seconds,
        "final_value": best_val_raw,
        "optimal_value": obj_min,
        "iterations": iterations_completed,
        "solution_found": "T" if found_optimal else "F",
    }


def create_summary_csv():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = os.path.join(
        OUTPUT_DIR, f"fmqa_simulated_all_runs_{timestamp}.csv"
    )

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()

    return summary_path


def append_summary_row(summary_path, row):
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writerow(row)


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

    all_rows = []
    summary_path = create_summary_csv()
    print(f"Summary CSV will be updated at: {_display_path(summary_path)}")

    for dataset_index, dataset_path in enumerate(dataset_paths, start=1):
        graphtype = os.path.splitext(os.path.basename(dataset_path))[0]
        grid, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(filename=dataset_path)
        all_feasible_points = [p for p, v in grid.items() if np.isfinite(v)]

        if not all_feasible_points:
            raise ValueError(f"No feasible points are available for dataset: {graphtype}")

        dataset_max_cycles = min(
            requested_max_cycles,
            1000,
            max(1, int(len(all_feasible_points) * 0.1)),
        )

        convergence_patience = min(
            dataset_max_cycles,
            max(1, int(len(all_feasible_points) * convergence_patience_fraction)),
        )

        start_points = select_start_points(all_feasible_points, runs_per_graph)

        print(f"\n=== Dataset {dataset_index}/{len(dataset_paths)}: {graphtype} ===")
        print(f"Grid loaded: {len(grid)} points, x in [0,{x_bound}], y in [0,{y_bound}]")
        print(f"Objective range: [{obj_min}, {obj_max}]")
        print(f"Total feasible points: {len(all_feasible_points)}")
        print(f"Max FMQA cycles set to {dataset_max_cycles}")
        print(
            f"Convergence patience set to {convergence_patience} "
            f"(fraction={convergence_patience_fraction})"
        )
        print(f"Using {len(start_points)} seeded single-start runs.")

        for run_index, start_point in enumerate(start_points, start=1):
            print(
                f"\n--- Run {run_index}/{len(start_points)} for {graphtype} "
                f"starting at {start_point} ---"
            )
            row = run_single_fmqa(
                grid,
                obj_min,
                obj_max,
                x_bound,
                y_bound,
                start_point,
                dataset_max_cycles,
                convergence_patience,
            )
            row["graph_type"] = graphtype
            row["run_index"] = run_index
            all_rows.append(row)
            append_summary_row(summary_path, row)
            print(f"Updated summary CSV with {len(all_rows)} completed run(s).")

    print(f"\nCompleted {len(all_rows)} run(s) across {len(dataset_paths)} dataset(s).")
    print(f"Summary CSV written to: {_display_path(summary_path)}")


if __name__ == "__main__":
    main()
