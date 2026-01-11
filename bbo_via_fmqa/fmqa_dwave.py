import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fmqa")))

import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import datetime

# Helper and module imports
import read_grid
from ml_surrogate import train_surrogate_model, print_final_equation
from ising_machine import solve_surrogate_dwave

# ============================================================
#                 LOAD DATASET
# ============================================================
# path = "path/to/your/dataset.csv"
grid, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(filename=path)
print(f"Grid loaded: {len(grid)} points, x in [0,{x_bound}], y in [0,{y_bound}]")
print(f"Objective range: [{obj_min}, {obj_max}]")


# ============================================================
#                 HELPER FUNCTIONS
# ============================================================

def evaluate(x, y):
    """Evaluate objective from grid (with built-in penalty handling)."""
    return read_grid.obj_funct([x, y], grid)

def scale_value(val):
    """Normalize objective values to [0,1] range."""
    if val == np.inf:
        return np.inf
    return (val - obj_min) / (obj_max - obj_min)

def bitstring_to_array(bitstring: str) -> np.ndarray:
    """Convert a bitstring like '01011' to a numpy int array [0,1,0,1,1]."""
    return np.fromiter((int(c) for c in bitstring), dtype=np.int8)


# ============================================================
#                 TRAIN / TEST SPLIT
# ============================================================

all_feasible_points = [p for p, v in grid.items() if not np.isnan(v)]
random.shuffle(all_feasible_points)

split_index = int(0.7 * len(all_feasible_points))
train_points = all_feasible_points[:split_index]
test_points  = all_feasible_points[split_index:]

print(f"\nDataset split: {len(train_points)} training candidate points, {len(test_points)} testing points.")


# ============================================================
#                 INITIAL FMQA DATASET (POINT-BY-POINT)
# ============================================================

# Instead of evaluating ALL train_points at once, start with only a few.
# You can change initial_k if you want more initial data.
initial_k = 1
initial_points = train_points[:initial_k]

evaluated_points = set(initial_points)  # track visited points
xs = []
ys = []

for (x0, y0) in initial_points:
    val = evaluate(x0, y0)
    xs.append(read_grid.coord_bits(x0, y0, x_bound, y_bound))
    ys.append(scale_value(val))

print(f"\nInitial evaluated points: {len(evaluated_points)} (out of {len(grid)} total grid points)")
print(f"First initial point: {initial_points[0]} -> objective {evaluate(*initial_points[0])}")


# ============================================================
#                 FMQA LOOP
# ============================================================

max_cycles = 100
convergence_patience = 10

best_val_raw = np.inf
best_coord   = None
history      = []
no_improvement_count = 0

final_model = None

# For CSV logging
loop_records = []


for t in range(max_cycles):
    print("\n====================================================")
    print(f"                 Cycle {t+1}/{max_cycles}")
    print("====================================================")
    print(f"Best so far = {best_val_raw} at {best_coord}")

    # Train surrogate on CURRENT evaluated set
    fm = train_surrogate_model(xs, ys)
    final_model = fm

    # Use D-Wave to propose new candidate
    px, py = solve_surrogate_dwave(
        fm,
        x_bound,
        y_bound,
        evaluated_points,
        grid
    )

    print((px, py), "in grid?", (px, py) in grid)

    # Default logging values for this cycle
    status = "ok"
    obj_val_raw = np.nan

    # Handle invalid cases
    if px is None or py is None:
        print("All proposed samples were already evaluated or invalid.")
        no_improvement_count += 1
        history.append(best_val_raw)
        print(f"No improvement for {no_improvement_count} consecutive cycles.")

        status = "invalid_or_evaluated"

        # Log this cycle
        loop_records.append({
            "cycle": t + 1,
            "px": "",
            "py": "",
            "objective": "",
            "best_val": best_val_raw,
            "evaluated_count": len(evaluated_points),
            "no_improvement_count": no_improvement_count,
            "status": status,
        })

        if no_improvement_count >= convergence_patience:
            print("\nConvergence reached. Stopping.")
            break
        continue

    # TEMPORARY RULE: triangular symmetry
    if px < py:
        print(f"Skipping ({px},{py}) due to triangular symmetry (x < y).")
        # Mark as evaluated so it won't be proposed repeatedly
        evaluated_points.add((px, py))
        history.append(best_val_raw)
        no_improvement_count += 1

        status = "skipped_triangular"

        loop_records.append({
            "cycle": t + 1,
            "px": px,
            "py": py,
            "objective": "",
            "best_val": best_val_raw,
            "evaluated_count": len(evaluated_points),
            "no_improvement_count": no_improvement_count,
            "status": status,
        })

        if no_improvement_count >= convergence_patience:
            print("\nConvergence reached. Stopping.")
            break
        continue

    # Evaluate chosen point (one new point per cycle)
    obj_val_raw = evaluate(px, py)
    evaluated_points.add((px, py))

    print(f"Proposed ({px},{py}) -> objective {obj_val_raw}")

    if np.isfinite(obj_val_raw):
        print(f"Objective = {obj_val_raw:.6f}, current best = {best_val_raw:.6f}")
    else:
        print(f"Objective = inf (invalid or outside grid)")

    # Update best
    improved = False
    if obj_val_raw < best_val_raw:
        print(f"New BEST FOUND: {obj_val_raw} at ({px},{py})")
        best_val_raw = obj_val_raw
        best_coord = (px, py)
        no_improvement_count = 0
        improved = True
    else:
        print("No improvement.")
        no_improvement_count += 1

    # Add new data to training set (one point)
    scaled = scale_value(obj_val_raw)
    if scaled == np.inf:
        scaled = 1.1
    xs.append(read_grid.coord_bits(px, py, x_bound, y_bound))
    ys.append(scaled)

    history.append(best_val_raw)

    print(f"Evaluated: {len(evaluated_points)} / {len(grid)} points")
    print(f"No improvement count = {no_improvement_count}")

    # Log this cycle
    loop_records.append({
        "cycle": t + 1,
        "px": px,
        "py": py,
        "objective": obj_val_raw,
        "best_val": best_val_raw,
        "evaluated_count": len(evaluated_points),
        "no_improvement_count": no_improvement_count,
        "status": "improved" if improved else "no_improvement",
    })

    if no_improvement_count >= convergence_patience:
        print("\nConvergence reached. Stopping.")
        break


# ============================================================
#                 FINAL REPORT
# ============================================================

print("\n================ FINAL RESULTS ================")
print(f"FMQA best = {best_val_raw} at {best_coord}")
print(f"Global minimum in dataset = {obj_min}")
min_test = min(grid[p] for p in test_points)
print(f"Best in (predefined) test set = {min_test}")

if best_val_raw <= min_test:
    print("SUCCESS: FMQA found a test-set-competitive value.")
else:
    print("INFO: FMQA did not outperform the test set minimum.")

if final_model:
    print_final_equation(final_model)


# ============================================================
#                 WRITE CSV LOG (UNIQUE FILE)
# ============================================================

os.makedirs("output", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_path = os.path.join("output", f"fmqa_dwave_log_{timestamp}.csv")

if loop_records:
    fieldnames = [
        "cycle",
        "px",
        "py",
        "objective",
        "best_val",
        "evaluated_count",
        "no_improvement_count",
        "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in loop_records:
            writer.writerow(rec)

    print(f"\nFMQA loop log written to: {csv_path}")
else:
    print("\nNo loop records to write (loop did not run?).")


# ============================================================
#                 TEST SURROGATE ON TEST SET
# ============================================================

# if final_model:
#     # Build feature matrix for test set: one row per test point
#     test_bitstrings = [
#         read_grid.coord_bits(x, y, x_bound, y_bound) for (x, y) in test_points
#     ]
#     X_test = np.stack([bitstring_to_array(bs) for bs in test_bitstrings], axis=0)

#     # True objective values (raw and scaled)
#     y_test_raw = np.array([evaluate(x, y) for (x, y) in test_points], dtype=float)
#     y_test_scaled = np.array([scale_value(v) for v in y_test_raw], dtype=float)

#     # Surrogate predictions (scaled)
#     y_pred_scaled = final_model.predict(X_test)
#     y_pred_scaled = np.squeeze(y_pred_scaled)

#     # Basic metrics on scaled values
#     mse = np.mean((y_pred_scaled - y_test_scaled) ** 2)
#     mae = np.mean(np.abs(y_pred_scaled - y_test_scaled))

#     print("\n================ SURROGATE TEST RESULTS ================")
#     print(f"Test set size: {len(test_points)}")
#     print(f"MSE (scaled objective): {mse:.6e}")
#     print(f"MAE (scaled objective): {mae:.6e}")
#     # Optionally convert predictions back to raw objective:
#     # y_pred_raw = y_pred_scaled * (obj_max - obj_min) + obj_min

# ============================================================
#                 VISUALIZATION
# ============================================================
x_vals = np.arange(x_bound + 1)
y_vals = np.arange(y_bound + 1)
Z = np.full((y_bound + 1, x_bound + 1), np.nan)

for (x, y), val in grid.items():
    Z[y, x] = val

plt.figure(figsize=(8, 6))
plt.title("FMQA Optimization (D-Wave)")

im = plt.imshow(Z, origin="lower", cmap="cividis", aspect="auto")
plt.colorbar(im, label="Objective Value")

train_x, train_y = zip(*train_points)
test_x, test_y   = zip(*test_points)

plt.scatter(train_x, train_y, c="white", edgecolors="black", s=40, label="Training Candidates")
plt.scatter(test_x,  test_y,  c="gray",  edgecolors="black", s=40, label="Testing Points")

if best_coord is not None:
    plt.scatter(*best_coord, color="red", s=120, marker="*", label="FMQA Best")

true_best = min(grid, key=grid.get)
plt.scatter(*true_best, color="cyan", s=120, marker="*", label="Global Opt")

plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
