"""Single source of truth for synthetic response functions.

Generator and O0 reconstruction MUST call the same function implementation.
"""
import numpy as np

TRUTH_REGISTRY_VERSION = "stage1_truth_v1"

# Active variable 0: f0(x) = tanh(x)
def f0_tanh(x):
    return np.tanh(x)

# Active variable 1: f1(x) = 0.5*x^2 - 0.5  (centered quadratic)
def f1_quadratic(x):
    return 0.5 * x * x - 0.5

# Active variable 2: f2(x) = sin(1.5*x)
def f2_sine(x):
    return np.sin(x * 1.5)

# Registry mapping active variable index -> function
TRUE_FUNCTIONS = {
    0: f0_tanh,
    1: f1_quadratic,
    2: f2_sine,
}

TRUE_FUNCTION_NAMES = {
    0: "tanh(x)",
    1: "0.5*x^2 - 0.5",
    2: "sin(1.5*x)",
}

TRUE_FUNCTION_CENTERING = {
    0: 0.0,   # tanh is odd, E[tanh(x)]≈0 for symmetric dist
    1: 0.5,   # E[x^2]≈1 for std normal, so 0.5*1-0.5=0
    2: 0.0,   # sin is odd
}

def get_true_function(j: int):
    """Return the true response function for variable j."""
    return TRUE_FUNCTIONS[j]

def apply_true_response(x: np.ndarray, j: int) -> np.ndarray:
    """Apply true response function j to input x."""
    return TRUE_FUNCTIONS[j](x)
