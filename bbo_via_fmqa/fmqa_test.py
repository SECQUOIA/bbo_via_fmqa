import numpy as np
import dimod
import matplotlib.pyplot as plt
import random

# FMQA repo
import fmqa
from fmqa import FMBQM

# Helper file
import read_grid

path = '/Users/woosik/Documents/Purdue/Research/bbo_via_fmqa/dataset/compl_enum_cstr_17_baron_hull.csv' # Adjust the path as needed
grid, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(filename=path)
print(f"Grid loaded: {len(grid)} points, x in [0,{x_bound}], y in [0,{y_bound}]")
print(f"Objective range: [{obj_min}, {obj_max}]")


# --- Parameters ---
# random.seed(0)
# np.random.seed(0)
num_init = 20
num_cycles = 40
sampler = dimod.SimulatedAnnealingSampler()

# --- Helper Functions ---
def evaluate(x, y):
    return read_grid.obj_funct([x, y], grid)

def scale_value(y_val):
    if y_val == np.inf:
        return np.inf
    # Scales value to be in [0, 1]
    return (y_val - obj_min) / (obj_max - obj_min)

# --- Initial dataset ---
all_points = list(grid.keys())
init_points = []
evaluated_points = set()
while len(init_points) < num_init:
    p = random.choice(all_points)
    # FIX: Added check to prevent duplicate initial points
    if p not in evaluated_points and not np.isnan(grid[p]):
        init_points.append(p)
        evaluated_points.add(p)

xs = [] # List of bit-strings
ys = [] # List of SCALED objective values
for dx, dy in init_points:
    xs.append(read_grid.coord_bits(dx, dy, x_bound, y_bound))
    # FIX: Consistently use the [0, 1] scaler
    ys.append(scale_value(evaluate(dx, dy)))

# Track best UN-SCALED value for reporting
# FIX: Correctly find the initial best value and coordinate
best_val_raw = min(grid[p] for p in init_points)
best_coord = [p for p in init_points if grid[p] == best_val_raw][0]

print(f"Initial best = {best_val_raw} at {best_coord}")

history = [best_val_raw]

# --- FMQA loop ---
for t in range(num_cycles):
    print(f"\n=== Cycle {t+1}/{num_cycles} ===")

    # Train surrogate
    x_vectors = [[int(bit) for bit in b] for b in xs]
    x_vectors_np = np.array(x_vectors, dtype=int)
    fm = FMBQM.from_data(x=x_vectors_np, y=np.array(ys), num_epoch=1000, learning_rate=1.0e-2)

    # Solve surrogate with sampler
    sampleset = sampler.sample(fm, num_reads=100)

    # Find the best new, in-bounds sample to evaluate
    px, py = None, None
    for sample, energy in sampleset.data(['sample', 'energy']):
        bitlist = [sample[i] for i in sorted(sample.keys())]
        bitstring = "".join(map(str, bitlist))
        cand_x, cand_y = read_grid.bits_to_int(bitstring, lsb_first=False)

        if (0 < cand_x <= x_bound) and (0 < cand_y <= y_bound):
            if (cand_x, cand_y) not in evaluated_points:
                px, py = cand_x, cand_y
                break

    if px is None:
        print("All proposed samples were invalid. Skipping cycle.")
        history.append(best_val_raw)
        continue

    obj_val_raw = evaluate(px, py)
    evaluated_points.add((px, py))
    print(f"Proposed ({px},{py}) -> objective {obj_val_raw}")

    # Scale the new value and use a scaled penalty
    obj_val_scaled = scale_value(obj_val_raw)
    if obj_val_scaled == np.inf:
        penalty_scaled = 1.1 # Penalty is slightly worse than the worst possible scaled value (1.0)
        print(f"Infeasible point found. Assigning scaled penalty: {penalty_scaled}")
        obj_val_scaled = penalty_scaled

    # Update dataset with the new point
    xs.append(read_grid.coord_bits(px, py, x_bound, y_bound))
    ys.append(obj_val_scaled)

    # Track best raw (un-scaled) value
    if obj_val_raw < best_val_raw:
        best_val_raw = obj_val_raw
        best_coord = (px, py)
        print(f"New best: {best_val_raw} at {best_coord}")

    history.append(best_val_raw)


print(f"\nFinal best objective = {best_val_raw} at {best_coord}")
print(f"Global minimum = {obj_min}")

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(history, marker='o', linestyle='-')
plt.axhline(y=obj_min, color='r', linestyle='--', label=f'Global Minimum ({obj_min:.4f})')
plt.xlabel("Iteration")
plt.ylabel("Best Objective So Far")
plt.title("FMQA Optimization Progress")
plt.legend()
plt.grid(True)
plt.show()

