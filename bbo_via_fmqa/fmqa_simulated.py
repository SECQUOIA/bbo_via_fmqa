import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fmqa")))

import numpy as np
import matplotlib.pyplot as plt
import random

# Helper and module imports
import read_grid
from ml_surrogate import train_surrogate_model, print_final_equation
from ising_machine import solve_surrogate
from dimod import SimulatedAnnealingSampler

# --- Load Full Dataset ---
# path = path/to/your/dataset.csv
grid, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(filename=path)
print(f"Grid loaded: {len(grid)} points, x in [0,{x_bound}], y in [0,{y_bound}]")
print(f"Objective range: [{obj_min}, {obj_max}]")

# --- Parameters ---
max_cycles = 100
convergence_patience = 5
sampler = SimulatedAnnealingSampler()

# --- Helper Functions ---
def evaluate(x, y):
    """Evaluate objective from grid."""
    return read_grid.obj_funct([x, y], grid)

def scale_value(y_val):
    """Normalize objective values to [0,1] range."""
    if y_val == np.inf:
        return np.inf
    return (y_val - obj_min) / (obj_max - obj_min)

# --- Create Training and Testing Split (70/30) ---
all_feasible_points = [p for p, v in grid.items() if not np.isnan(v)]
random.shuffle(all_feasible_points)

split_index = int(0.7 * len(all_feasible_points))
train_points = all_feasible_points[:split_index]
test_points = all_feasible_points[split_index:]

print(f"\nDataset split: {len(train_points)} training points, {len(test_points)} testing points.")

# --- Initial dataset (from training split) ---
evaluated_points = set(train_points)
xs = [read_grid.coord_bits(dx, dy, x_bound, y_bound) for dx, dy in train_points]
ys = [scale_value(evaluate(dx, dy)) for dx, dy in train_points]

# --- Tracking and Convergence ---
best_val_raw = np.inf
best_coord = None
history = []
no_improvement_count = 0

# --- FMQA Loop ---
final_model = None
for t in range(max_cycles):
    print(f"\n=== Cycle {t+1}/{max_cycles} ===")
    print(f"Cycle {t+1}: best so far = {best_val_raw} at {best_coord}")

    # Train surrogate model
    fm = train_surrogate_model(xs, ys)
    final_model = fm

    # Solve surrogate to get candidate
    px, py = solve_surrogate(fm, x_bound, y_bound, evaluated_points, sampler)

    # Debug info
    print((px, py), "in grid?", (px, py) in grid)
    if (px, py) in grid:
        print("grid value:", grid[(px, py)])

    # --- Skip invalid or duplicate proposals ---
    if px is None or py is None:
        print("All proposed samples were already evaluated or invalid.")
        no_improvement_count += 1
        history.append(best_val_raw)
        print(f"No improvement for {no_improvement_count} consecutive cycles.")
        continue

    # --- TEMPORARY RULE: skip x < y points ---
    if px < py:
        print(f"Skipping point ({px}, {py}) because x < y.")
        evaluated_points.add((px, py))  # mark as evaluated so not proposed again
        history.append(best_val_raw)
        print(f"No improvement for {no_improvement_count} consecutive cycles.")
        continue

    # --- Evaluate candidate ---
    obj_val_raw = evaluate(px, py)
    evaluated_points.add((px, py))
    print(f"Proposed ({px},{py}) -> objective {obj_val_raw}")

    # --- Print objective every iteration (even if not best) ---
    if np.isfinite(obj_val_raw):
        print(f"Cycle {t+1}: objective = {obj_val_raw:.6f}, current best = {best_val_raw:.6f}")
    else:
        print(f"Cycle {t+1}: objective = inf (invalid or out of bounds)")

    # --- Update best found if improved ---
    if obj_val_raw < best_val_raw:
        print(f"New best found: {obj_val_raw} at ({px}, {py})")
        best_val_raw = obj_val_raw
        best_coord = (px, py)
        no_improvement_count = 0
    else:
        print("No improvement this iteration.")
        no_improvement_count += 1

    # --- Scale and append data for next iteration ---
    obj_val_scaled = scale_value(obj_val_raw)
    if obj_val_scaled == np.inf:
        obj_val_scaled = 1.1

    xs.append(read_grid.coord_bits(px, py, x_bound, y_bound))
    ys.append(obj_val_scaled)

    history.append(best_val_raw)
    print(f"No improvement for {no_improvement_count} consecutive cycles.")
    print(f"Evaluated so far: {len(evaluated_points)} / total {len(grid)}")

    if no_improvement_count >= convergence_patience:
        print(f"\nConvergence reached after {t+1} cycles. Stopping.")
        break

if t == max_cycles - 1:
    print(f"\nMax cycles reached without convergence.")

# --- Final Report ---
print(f"\n--- Final Results ---")
print(f"Best objective found by FMQA = {best_val_raw} at {best_coord}")
print(f"The known global minimum for the entire dataset is = {obj_min}")

min_in_test = min(grid[p] for p in test_points)
print(f"The best value in the unseen 30% test set was = {min_in_test}")

if best_val_raw <= min_in_test:
    print("SUCCESS: The algorithm found a value as good or better than the best in the test set.")
else:
    print("INFO: The algorithm did not find a value better than the best in the test set.")

if final_model:
    print_final_equation(final_model)

# --- Visualization ---
x_vals = np.arange(x_bound + 1)
y_vals = np.arange(y_bound + 1)
Z = np.full((y_bound + 1, x_bound + 1), np.nan)

for (x, y), v in grid.items():
    Z[y, x] = v

plt.figure(figsize=(8, 6))
plt.title("FMQA Optimization")

# Heatmap of the objective values
im = plt.imshow(Z, origin="lower", cmap="cividis", aspect="auto")
plt.colorbar(im, label="Objective Value")

# Plot training/testing points
train_x, train_y = zip(*train_points)
test_x, test_y = zip(*test_points)
plt.scatter(train_x, train_y, c="white", edgecolors="black", s=40, label="Training Points")
plt.scatter(test_x, test_y, c="gray", edgecolors="black", s=40, label="Testing Points")

# Plot best found and true global minimum
if best_coord is not None:
    plt.scatter(*best_coord, color="red", s=120, marker="*", label=f"FMQA Best")
true_best_coord = min(grid, key=grid.get)
plt.scatter(*true_best_coord, color="cyan", s=120, marker="*", label=f"Global Optimal")

plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
