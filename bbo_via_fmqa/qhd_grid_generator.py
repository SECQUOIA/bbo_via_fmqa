import numpy as np
import os

# Ridges or Valleys
def holder_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Holder Table function."""
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x**2 + y**2) / np.pi))))

def deflected_corrugated_spring(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Deflected Corrugated Spring function."""
    term1 = 0.1 * ((x - 5)**2 + (y - 5)**2)
    term2 = -np.cos(5 * np.sqrt((x - 5)**2 + (y - 5)**2))
    return term1 + term2

def dropwave(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Drop-Wave function."""
    numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
    denominator = 0.5 * (x**2 + y**2) + 2
    return -numerator / denominator

def levy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Levy function."""
    term1 = np.sin(np.pi * (1 + (x - 1) / 4))**2
    term2 = ((1 + (x - 1) / 4) - 1)**2 * (1 + 10 * np.sin(np.pi * (1 + (x - 1) / 4) + 1)**2)
    term3 = ((1 + (y - 1) / 4) - 1)**2 * (1 + np.sin(2 * np.pi * (1 + (y - 1) / 4))**2)
    return term1 + term2 + term3

def levy13(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Levy 13 function."""
    term1 = np.sin(3 * np.pi * x)**2
    term2 = (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2)
    term3 = (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)
    return term1 + term2 + term3

# Basin
def bohachevsky2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Bohachevsky 2 function."""
    return x**2 + 2 * y**2 - 0.3 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y) + 0.3

def camel3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized 3-Hump Camel function."""
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2

def csendes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Csendes function."""
    term1 = x**6 * (2 + np.sin(1 / x))**2
    term2 = y**6 * (2 + np.sin(1 / y))**2
    return term1 + term2

def rosenbrock(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Rosenbrock function."""
    return (1 - x)**2 + 100 * (y - x**2)**2

# Flat
def michalewicz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Michalewicz function."""
    return -np.sin(x) * (np.sin(x**2 / np.pi))**(20) - np.sin(y) * (np.sin(2*y**2 / np.pi))**(20)

def easom(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Easom function."""
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

def xin_she_yang3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Xin-She Yang 3 function."""
    term1 = np.exp(-((x / 15)**6 + (y / 1)**6))
    term2 = -2 * np.exp(-x**2 - y**2) * np.cos(x)**2 * np.cos(y) ** 2
    return term1 - term2

# Studded
def alpine1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Alpine 1 function."""
    return np.abs(x * np.sin(x) + 0.1 * x) + np.abs(y * np.sin(y) + 0.1 * y)

def griewank(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Griewank function."""
    term1 = (x**2) / 4000 + (y**2) / 4000
    term2 = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return term1 - term2 + 1

def rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Rastrigin function."""
    return (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)) + 20

def ackley(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Ackley function."""
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e
# Simple
def alpine2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Alpine 2 function."""
    return -np.sqrt(x * y) * np.sin(x) * np.sin(y)

def hosaki(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Hosaki function."""
    return (1 - 8*x + 7*x**2 - (7/3)*x**3 + (1/4)*x**4) * y**2 * np.exp(-y)

def shubert(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Shubert function."""
    return (np.cos(2*x + 1) + 2 * np.cos(3*x + 2) + 3 * np.cos(4*x + 3)) * (np.cos(2*y + 1) + np.cos(y+2))

def styblinski_tang(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Styblinski-Tang function."""
    return 1/78 * 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)

def sum_of_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Sum of Squares function."""
    return x**2 + 2* y**2

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



def main():
    x_bound = 30    # default x bound
    y_bound = 30    # default y bound
    
    make_csv(
        x_bound=x_bound,
        y_bound=y_bound,
        filename=f"shubert_{x_bound}x{y_bound}.csv",
        graph_type="shubert",   # Change to desired function type
    )

if __name__ == "__main__":
    main()  

