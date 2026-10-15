import numpy as np
import csv


def _bit_length_for_upper(upper: int) -> int:
    """Return the minimum number of bits needed to encode [0, upper]."""
    if upper < 0:
        raise ValueError(f"upper bound must be non-negative, got {upper}")
    if upper == 0:
        return 0
    return int(np.ceil(np.log2(upper + 1)))


def _bits_fragment_to_int(bits: str) -> int:
    """Decode a binary fragment, treating the empty string as zero."""
    return int(bits, 2) if bits else 0

def load_grid(filename):
    """
    Load grid data from CSV.
    Expected CSV columns: "x", "y", "Objective", among others.
    Returns a dictionary mapping (x, y) tuples to the objective value.
    """
    grid_data = {}
    x_bound = 0
    y_bound = 0
    
    with open(filename, "r", newline="", encoding="utf-8") as f:
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
    finite_values = [v for v in grid_data.values() if np.isfinite(v)]
    obj_min = min(finite_values) if finite_values else None
    obj_max = max(finite_values) if finite_values else None
    return grid_data, obj_min, obj_max, x_bound, y_bound


def load_grid_3d(filename):
    """
    Load 3D grid data from CSV.
    Expected CSV columns: "x", "y", "z", "Objective".
    Returns a dictionary mapping (x, y, z) tuples to the objective value.
    """
    grid_data = {}
    x_bound = 0
    y_bound = 0
    z_bound = 0

    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x_val = int(row["x"])
                y_val = int(row["y"])
                z_val = int(row["z"])
                obj_val = float(row["Objective"])
                grid_data[(x_val, y_val, z_val)] = obj_val
                if x_val > x_bound:
                    x_bound = x_val
                if y_val > y_bound:
                    y_bound = y_val
                if z_val > z_bound:
                    z_bound = z_val
            except Exception as e:
                print("Skipping row due to error:", e)
                continue

    finite_values = [v for v in grid_data.values() if np.isfinite(v)]
    obj_min = min(finite_values) if finite_values else None
    obj_max = max(finite_values) if finite_values else None
    return grid_data, obj_min, obj_max, x_bound, y_bound, z_bound


# Integer to binary weighted bit encoding
def int_to_bits(n: int, upper: int, lsb_first: bool = False):
    """
    Encode n in binary using the minimum number of bits for [0, upper].
    Returns a list of 0/1 (LSB first by default).
    """
    if not (0 <= n <= upper):
        raise ValueError(f"value {n} outside [0, {upper}]")

    m = _bit_length_for_upper(upper)
    bits = [(n >> k) & 1 for k in range(m)]
    if not lsb_first:
        bits.reverse()
    return bits

def bits_to_int(bits: str, x_max: int = None, y_max: int = None, lsb_first: bool = False):
    """
    Decode a binary string into two integers (x, y).

    Parameters:
    - bits: str, binary string (e.g., '101001')
    - x_max: int | None, optional maximum x value used to determine x bit width
    - y_max: int | None, optional maximum y value used to determine y bit width
    - lsb_first: bool, if True, the least significant bit is first

    Returns:
    - two integers decoded from the bitstring
    
    Example:
    - bits_to_int('101001', lsb_first=False) -> (5, 1)
    """
    
    # Keep the historical whole-string convention only for unbounded decoding.
    if lsb_first and x_max is None and y_max is None:
        bits = bits[::-1]

    if (x_max is None) != (y_max is None):
        raise ValueError("x_max and y_max must both be provided or both be omitted")

    if x_max is None and y_max is None:
        mid = len(bits) // 2
        x_bits = bits[:mid]
        y_bits = bits[mid:]
    else:
        x_len = _bit_length_for_upper(x_max)
        y_len = _bit_length_for_upper(y_max)
        total_len = x_len + y_len
        if len(bits) < total_len:
            bits = bits.rjust(total_len, "0")
        x_bits = bits[:x_len]
        y_bits = bits[x_len:x_len + y_len]
        if lsb_first:
            x_bits, y_bits = x_bits[::-1], y_bits[::-1]

    return _bits_fragment_to_int(x_bits), _bits_fragment_to_int(y_bits)


def bits_to_int_3d(bits: str, x_max: int, y_max: int, z_max: int, lsb_first: bool = False):
    """
    Decode a binary string into (x, y, z) using bounds to determine bit lengths.
    """
    x_len = _bit_length_for_upper(x_max)
    y_len = _bit_length_for_upper(y_max)
    z_len = _bit_length_for_upper(z_max)

    if len(bits) < x_len + y_len + z_len:
        bits = bits.rjust(x_len + y_len + z_len, "0")

    x_bits = bits[:x_len]
    y_bits = bits[x_len:x_len + y_len]
    z_bits = bits[x_len + y_len:x_len + y_len + z_len]

    if lsb_first:
        x_bits, y_bits, z_bits = x_bits[::-1], y_bits[::-1], z_bits[::-1]

    return (
        _bits_fragment_to_int(x_bits),
        _bits_fragment_to_int(y_bits),
        _bits_fragment_to_int(z_bits),
    )


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


def coord_bits_3d(x: int, y: int, z: int,
                  x_max: int, y_max: int, z_max: int,
                  lsb_first: bool = False):
    """
    Given (x, y, z) and their maxima, return a string of concatenated bits:
    (bits for x + bits for y + bits for z)
    """
    x_bits = int_to_bits(x, x_max, lsb_first)
    y_bits = int_to_bits(y, y_max, lsb_first)
    z_bits = int_to_bits(z, z_max, lsb_first)
    all_bits = x_bits + y_bits + z_bits
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
        # Return penalty if the objective is not finite.
        if not np.isfinite(obj_val):
            return PENALTY
        else:
            return obj_val
    else:
        # If the point is not in the grid, return the penalty.
        return PENALTY


def obj_funct_3d(x, grid_data):
    """
    Objective lookup for 3D grid data.
    """
    x_int = int(round(x[0]))
    y_int = int(round(x[1]))
    z_int = int(round(x[2]))

    if not all(isinstance(v, (int, float)) for v in (x_int, y_int, z_int)):
        return PENALTY

    if (x_int, y_int, z_int) in grid_data:
        obj_val = grid_data[(x_int, y_int, z_int)]
        if not np.isfinite(obj_val):
            return PENALTY
        return obj_val
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
        if diff == 0:
            return -1.0
        scaled = -1 + delta / diff
        return float(scaled)
    
