import numpy as np
from fmqa import FMBQM

def train_surrogate_model(xs, ys, num_epoch=1000, learning_rate=1.0e-2):
    """
    Trains the Factorization Machine surrogate model on the provided data.

    Args:
        xs (list of str): A list of binary strings representing the input points.
        ys (list of float): A list of scaled objective values.
        num_epoch (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        FMBQM: A trained Factorization Machine Binary Quadratic Model.
    """
    # Convert data to the format required by the FMBQM class
    x_vectors = [[int(bit) for bit in b] for b in xs]
    x_vectors_np = np.array(x_vectors, dtype=int)
    y_values_np = np.array(ys)

    # Train and return the model
    fm = FMBQM.from_data(x=x_vectors_np, y=y_values_np, num_epoch=num_epoch, learning_rate=learning_rate)
    
    return fm

def print_final_equation(fm_model):
    """
    Prints the final learned QUBO equation from the trained model.

    Args:
        fm_model (FMBQM): The final trained surrogate model.
    """
    print("\n--- Final Learned Quadratic Equation (QUBO) ---")
    equation_parts = []

    # Add linear terms (h_i * x_i)
    for i, h in fm_model.linear.items():
        if abs(h) > 1e-4:  # Only include non-trivial terms
            equation_parts.append(f"{h:+.4f}*x{i}")

    # Add quadratic terms (J_ij * x_i * x_j)
    for (i, j), J in fm_model.quadratic.items():
        if abs(J) > 1e-4:  # Only include non-trivial terms
            equation_parts.append(f"{J:+.4f}*x{i}*x{j}")

    # Add the constant offset
    equation_parts.append(f"{fm_model.offset:+.4f}")

    # Join all parts for the final equation string
    final_equation = "E = " + " ".join(equation_parts)
    print(final_equation)

