import numpy as np
import os


def alpine2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Alpine 2 function."""
    return -np.sqrt(x * y) * np.sin(x) * np.sin(y)

def alpine2_make_csv(x_bound: int, y_bound: int, filename: str):
    """Generate a CSV file for the Alpine 2 function over a grid.
    x_bound: upper limit for x (inclusive)
    y_bound: upper limit for y (inclusive)
    """
    
    dataset_dir = "./dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    filepath = os.path.join(dataset_dir, filename)

    with open(filepath, "w") as f:
        f.write("Point,x,y,Objective\n")
        for x in range(x_bound + 1):
            for y in range(y_bound + 1):
                val = alpine2(np.array(1000 * x/x_bound), np.array(1000 * y/y_bound))
                f.write(f"\"[{x}, {y}]\", {x}, {y}, {val}\n")

    print(
        f"CSV file '{filepath}' generated for QHD type alpine2 "
        f"with bounds x:[0,{x_bound}], y:[0,{y_bound}]."
    )


x_bound = 30
y_bound = 30

alpine2_make_csv(
    x_bound=x_bound,
    y_bound=y_bound,
    filename=f"alpine2_{x_bound}x{y_bound}.csv",
)
