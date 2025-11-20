import numpy as np
import csv

def levy13(x, y):
    term1 = np.sin(3 * np.pi * x)**2
    term2 = (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2)
    term3 = (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)
    return term1 + term2 + term3

def alpine2(x, y):
    term1 = - np.sqrt(x*y)
    term2 = np.sin(x) * np.sin(y)
    return term1 * term2

def deflected_corrugated_spring(x, y):
    term1 = 0.1 *((x-5)**2 + (y-5)**2)
    term2 = np.cos(5 * np.sqrt((x-5)**2 + (y-5)**2))
    return term1 - term2

# csv_filename = "levy13_grid.csv"
# csv_filename = "alpine2_grid.csv"
csv_filename = "deflected_corrugated_spring_grid.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Point", "x", "y", "objective"])  # header

    for x in range(1, 33):      # 1 to 32
        for y in range(1, 33):  # 1 to 32
            point_str = f"[{x},{y}]"
            fval = levy13(x, y)
            writer.writerow([point_str, x, y, fval])

print(f"CSV file '{csv_filename}' successfully created.")