import numpy as np
import os

def shubert(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Shubert function."""
    return (np.cos(2*x + 1) + 2 * np.cos(3*x + 2) + 3 * np.cos(4*x + 3)) * (np.cos(2*y + 1) + np.cos(y+2))

def rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Rastrigin function."""
    return (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)) + 20

def michalewicz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Michalewicz function."""
    return -np.sin(x) * (np.sin(x**2 / np.pi))**(20) - np.sin(y) * (np.sin(2*y**2 / np.pi))**(20)

def alpine2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Alpine 2 function."""
    return -np.sqrt(x * y) * np.sin(x) * np.sin(y)

def make_csv(x_bound: int, y_bound: int, filename: str, graph_type: str):
    """Generate a CSV file for the specified function over a grid.
    x_bound: upper limit for x (inclusive)
    y_bound: upper limit for y (inclusive)
    graph_type: type of graph to generate (e.g., 'alpine2', 'michalewicz')
    """
    
    dataset_dir = "./dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    filepath = os.path.join(dataset_dir, filename)

    with open(filepath, "w") as f:
        f.write("Point,x,y,Objective\n")
        if graph_type == "michalewicz":
            for x in range(x_bound + 1):
                for y in range(y_bound + 1):
                    val = michalewicz(np.array(np.pi * x/x_bound), np.array(np.pi * y/y_bound))
                    f.write(f"\"[{x}, {y}]\", {x}, {y}, {val}\n")
        elif graph_type == "alpine2":
            for x in range(x_bound + 1):
                for y in range(y_bound + 1):
                    val = alpine2(np.array(1000 * x/x_bound), np.array(1000 * y/y_bound))
                    f.write(f"\"[{x}, {y}]\", {x}, {y}, {val}\n")
        elif graph_type == "shubert":
            for x in range(x_bound + 1):
                for y in range(y_bound + 1):
                    val = shubert(np.array(10 * x/x_bound), np.array(10 * y/y_bound))
                    f.write(f"\"[{x}, {y}]\", {x}, {y}, {val}\n")
        
    print(
        f"CSV file '{filepath}' generated for QHD type {graph_type} "
        f"with bounds x:[0,{x_bound}], y:[0,{y_bound}]."
    )

x_bound = 30
y_bound = 30

make_csv(
    x_bound=x_bound,
    y_bound=y_bound,
    filename=f"shubert_{x_bound}x{y_bound}.csv",
    graph_type="shubert",
)
