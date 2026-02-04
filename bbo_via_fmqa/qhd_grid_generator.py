import numpy as np
import os
from typing import Callable, Dict, Tuple


# Graphs
# Ridges or Valleys
def holder_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x**2 + y**2) / np.pi))))

def deflected_corrugated_spring(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    term1 = 0.1 * ((x - 5)**2 + (y - 5)**2)
    term2 = -np.cos(5 * np.sqrt((x - 5)**2 + (y - 5)**2))
    return term1 + term2

def dropwave(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
    denominator = 0.5 * (x**2 + y**2) + 2
    return -numerator / denominator

def levy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    term1 = np.sin(np.pi * (1 + (x - 1) / 4))**2
    term2 = ((1 + (x - 1) / 4) - 1)**2 * (1 + 10 * np.sin(np.pi * (1 + (x - 1) / 4) + 1)**2)
    term3 = ((1 + (y - 1) / 4) - 1)**2 * (1 + np.sin(2 * np.pi * (1 + (y - 1) / 4))**2)
    return term1 + term2 + term3

def levy13(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    term1 = np.sin(3 * np.pi * x)**2
    term2 = (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2)
    term3 = (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)
    return term1 + term2 + term3

# Basin
def bohachevsky2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 + 2 * y**2 - 0.3 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y) + 0.3

def camel3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2

def csendes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # NOTE: This function has 1/x and 1/y; if x or y is 0 you get inf/nan.
    # We'll handle safety in evaluation by replacing zeros with eps.
    term1 = x**6 * (2 + np.sin(1 / x))**2
    term2 = y**6 * (2 + np.sin(1 / y))**2
    return term1 + term2

def rosenbrock(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (1 - x)**2 + 100 * (y - x**2)**2

# Flat
def michalewicz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.sin(x) * (np.sin(x**2 / np.pi))**(20) - np.sin(y) * (np.sin(2*y**2 / np.pi))**(20)

def easom(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

def xin_she_yang3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    term1 = np.exp(-((x / 15)**6 + (y / 1)**6))
    term2 = -2 * np.exp(-x**2 - y**2) * np.cos(x)**2 * np.cos(y) ** 2
    return term1 - term2

# Studded
def alpine1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(x * np.sin(x) + 0.1 * x) + np.abs(y * np.sin(y) + 0.1 * y)

def griewank(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    term1 = (x**2) / 4000 + (y**2) / 4000
    term2 = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return term1 - term2 + 1

def rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)) + 20

def ackley(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e

# Simple
def alpine2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.sqrt(x * y) * np.sin(x) * np.sin(y)

def hosaki(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (1 - 8*x + 7*x**2 - (7/3)*x**3 + (1/4)*x**4) * y**2 * np.exp(-y)

def shubert(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.cos(2*x + 1) + 2 * np.cos(3*x + 2) + 3 * np.cos(4*x + 3)) * (np.cos(2*y + 1) + np.cos(y+2))

def styblinski_tang(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (1/78) * 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)

def sum_of_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 + 2 * y**2


# ----------------- Domain registry -----------------
# graph_type -> (function, (xmin,xmax), (ymin,ymax))
GRAPH_SPECS: Dict[str, Tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Tuple[float, float], Tuple[float, float]]] = {
    "holder_table": (holder_table, (-10.0, 10.0), (-10.0, 10.0)),
    "deflected_corrugated_spring": (deflected_corrugated_spring, (0.0, 10.0), (0.0, 10.0)),
    "dropwave": (dropwave, (-5.12, 5.12), (-5.12, 5.12)),
    "levy": (levy, (-10.0, 10.0), (-10.0, 10.0)),
    "levy13": (levy13, (-10.0, 10.0), (-10.0, 10.0)),
    "bohachevsky2": (bohachevsky2, (-100.0, 100.0), (-100.0, 100.0)),
    "camel3": (camel3, (-5.0, 5.0), (-5.0, 5.0)),
    "csendes": (csendes, (-1.0, 1.0), (-1.0, 1.0)),
    "rosenbrock": (rosenbrock, (-2.0, 2.0), (-1.0, 3.0)),
    "michalewicz": (michalewicz, (0.0, float(np.pi)), (0.0, float(np.pi))),
    "easom": (easom, (-10.0, 10.0), (-10.0, 10.0)),
    "xin_she_yang3": (xin_she_yang3, (-20.0, 20.0), (-20.0, 20.0)),
    "alpine1": (alpine1, (0.0, 10.0), (0.0, 10.0)),
    "griewank": (griewank, (-600.0, 600.0), (-600.0, 600.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12), (-5.12, 5.12)),
    "ackley": (ackley, (-5.0, 5.0), (-5.0, 5.0)),
    "alpine2": (alpine2, (0.0, 1000.0), (0.0, 1000.0)),
    "hosaki": (hosaki, (0.0, 5.0), (0.0, 6.0)),
    "shubert": (shubert, (0.0, 10.0), (0.0, 10.0)),
    "styblinski_tang": (styblinski_tang, (-5.0, 5.0), (-5.0, 5.0)),
    "sum_of_squares": (sum_of_squares, (-10.0, 10.0), (-10.0, 10.0)),
}


def make_csv(x_bound: int, y_bound: int, graph_type: str, out_dir: str = "./dataset") -> str:
    if graph_type not in GRAPH_SPECS:
        raise ValueError(f"Unknown graph_type='{graph_type}'. Options: {sorted(GRAPH_SPECS)}")

    func, (xmin, xmax), (ymin, ymax) = GRAPH_SPECS[graph_type]

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{graph_type}_{x_bound}x{y_bound}.csv")

    xb = max(x_bound, 1)
    yb = max(y_bound, 1)

    # integer grid
    xi = np.arange(x_bound + 1)
    yi = np.arange(y_bound + 1)
    X_i, Y_i = np.meshgrid(xi, yi, indexing="ij")

    # map to continuous domain
    X = xmin + (X_i / xb) * (xmax - xmin)
    Y = ymin + (Y_i / yb) * (ymax - ymin)

    # avoid csendes singularity at 0
    if graph_type == "csendes":
        eps = 1e-12
        X = np.where(np.isclose(X, 0.0), eps, X)
        Y = np.where(np.isclose(Y, 0.0), eps, Y)

    Z = func(X, Y).astype(float)

    # normalize to [0,1]
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    if np.isclose(z_max - z_min, 0.0):
        Z_norm = np.zeros_like(Z)
    else:
        Z_norm = (Z - z_min) / (z_max - z_min)

    # write CSV (normalized z only)
    with open(filepath, "w") as f:
        f.write("Point,x,y,Objective\n")
        for x in range(x_bound + 1):
            for y in range(y_bound + 1):
                f.write(f"\"[{x},{y}]\",{x},{y},{Z_norm[x, y]}\n")

    return filepath


def main():
    x_bound, y_bound = 30, 30

    # # Generate ONE dataset:
    # path = make_csv(x_bound, y_bound, graph_type="shubert")
    # print(f"Saved: {path}")

    # Or generate ALL datasets:
    for gt in GRAPH_SPECS.keys():
        path = make_csv(x_bound, y_bound, graph_type=gt)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()