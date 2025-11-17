import numpy as np
import csv

def load_grid(filename):
    """
    Load grid data from CSV.
    Expected CSV columns: "x", "y", "Objective", among others.
    Returns a dictionary mapping (x, y) tuples to the objective value.
    """
    grid_data = {}
    x_bound = 0
    y_bound = 0
    
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Convert x and y values to integers
                x_val = int(row["x"])
                y_val = int(row["y"])
                # Convert objective value to float
                obj_val = float(row["Objective"])
                grid_data[(x_val, y_val)] = obj_val
                if x_val > x_bound:
                    x_bound = x_val
                if y_val > y_bound:
                    y_bound = y_val
                    
            except Exception as e:
                # Skip rows with conversion issues
                print("Skipping row due to error:", e)
                continue

    # Compute the minimum among finite objective values
    finite_values = [v for v in grid_data.values() if not np.isnan(v)]
    x_min = min(finite_values) if finite_values else None
    x_max = max(finite_values) if finite_values else None
    return grid_data, x_min, x_max, x_bound, y_bound


# Integer to binary weighted bit encoding
def int_to_bits(n: int, upper: int, lsb_first: bool = False):
    """
    Encode n in binary using the minimum number of bits for [0, upper].
    Returns a list of 0/1 (LSB first by default).
    """
    if not (0 <= n <= upper):
        raise ValueError(f"value {n} outside [0, {upper}]")

    m = int(np.ceil(np.log2(upper + 1)))
    bits = [(n >> k) & 1 for k in range(m)]
    if not lsb_first:
        bits.reverse()
    return bits

def bits_to_int(bits: str, lsb_first: bool = False):
    """
    Split the binary string to two integers (x,y).

    Parameters:
    - bits: str, binary string (e.g., '101001')
    - lsb_first: bool, if True, the least significant bit is first

    Returns:
    - two integers decoded from the bitstring
    
    Example:
    - bits_to_int('101001', lsb_first=False) -> (5, 1)
    """
    
    if lsb_first:
        bits = bits[::-1]  # Reverse the bits if LSB first

    mid = len(bits) // 2
    x_bits = bits[:mid]
    y_bits = bits[mid:]

    return int(x_bits, 2), int(y_bits, 2)


def coord_bits(x: int, y: int,
               x_max: int, y_max: int,
               lsb_first: bool = False):
    """
    Given (x, y) and their maxima, return a string of concatenated bits:
    (bits for x + bits for y)
    """
    
    x_bits = int_to_bits(x, x_max, lsb_first)
    y_bits = int_to_bits(y, y_max, lsb_first)
    all_bits = x_bits + y_bits
    return ''.join(str(bit) for bit in all_bits)

# PENALTY = 1e6  # adjust this penalty as appropriate
PENALTY = np.inf # use infinity as penalty for infeasible points
# PENALTY = np.nan  # use NaN as penalty for infeasible points
# PENALTY = None  # use None as penalty for infeasible points

def obj_funct(x, grid_data):
    """
    Given a candidate point x (a list or 1-D numpy array with 2 elements),
    this function rounds x to the nearest integers and returns the objective value
    if the point exists in grid_data and is feasible.
    Otherwise, returns a penalty value.
    """
    
    # Round the candidate coordinates to nearest integers
    x_int = int(round(x[0]))
    y_int = int(round(x[1]))
    
    # Check that x_int and y_int are numbers (optional check)
    if not isinstance(x_int, (int, float)) or not isinstance(y_int, (int, float)):
        return PENALTY
    
    # If the point exists in the grid, check its objective value.
    if (x_int, y_int) in grid_data:
        obj_val = grid_data[(x_int, y_int)]
        # Return penalty if the objective is NaN.
        if np.isnan(obj_val):
            return PENALTY
        else:
            return obj_val
    else:
        # If the point is not in the grid, return the penalty.
        return PENALTY
    
    
def scale_point(x, y, grid_data, obj_min, obj_max):
    """
    Compute the arctan-based scaled transformation for a point (x, y).
    Maps x_min -> -1 and infinity -> 1.
    
    Parameters:
      - x, y: coordinates (floats)
      - grid_data: dict from load_grid
      - x_min: minimum objective value from load_grid
      - alpha: scale parameter (default=1)
    
    Returns:
      - scaled value in [-1, 1]
    """
    
    raw_val = obj_funct([x, y], grid_data)

    if raw_val == np.inf:
        return 1.0
    else:
      delta = raw_val - obj_min
      diff = obj_max - obj_min
      scaled = -1 + delta / diff
      return float(scaled)


