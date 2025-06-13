"""
Free Fermion Utilities Module

Common utility functions for the free fermion codebase.

Copyright 2025 James.D.Whitfield@dartmouth.edu
Licensed under MIT License.
"""

import numpy as np

def print_custom(obj, k=9):
    """Custom print function with small number suppression

    Args:
        obj: Any object to be printed
        k: The number of decimal places to print

    Returns:
        None
    """
    if isinstance(obj, (int, float, complex,list, np.ndarray, np.matrix)):
        _print(obj, k)
    else:
        print(obj)

def _print(obj, k=9):
    """Printing with small number suppression (using numpy printoptions)

    Args:
        obj: Any object to be printed
        k: The number of decimal places to print

    Returns:
        None
    """
    try:
        val = np.array(obj)

        # get current precision
        p = np.get_printoptions()["precision"]

        # change to request precision
        np.set_printoptions(precision=k)

        # check if input is completely real
        # If it is don't print complex part
        if np.allclose(val.imag, np.zeros_like(val)):
            val = val.real

        # do the printing
        print(val.round(k))

        # reset precision
        np.set_printoptions(precision=p)
        return
    finally:
        return print(obj)


def clean(obj, threshold=1e-6):
    """
    Clean small numerical values from arrays or matrices.

    Args:
        obj: array, scalar, NumPy array or matrix
        threshold: Values below this threshold are set to zero

    Note: if threshold is an integer, it will be converted to 10^-threshold

    Returns:
        Cleaned obj with small values set to zero and rounded according to the threshold.
    """

    if isinstance(threshold, int):
        # If threshold is an integer, convert to 10^-threshold
        ndigits = threshold
        threshold = 10**(-threshold)
    else:
        ndigits = -round(np.log10(threshold))

    if isinstance(obj,list):
        # If it's a list, convert to numpy array
        obj_array = np.array(obj)
        obj_array = np.round(obj_array, ndigits)
        # Set small values to zero
        obj_array[np.abs(obj_array) < threshold] = 0
        return obj_array.tolist()

    elif isinstance(obj, (np.matrix, np.ndarray)):
        # If it's a numpy matrix or array, ensure it's a numpy array
        if hasattr(obj, 'imag'):  # if complex, check for small imaginary part
            # If it's complex, check the imaginary part
            if np.all(np.abs(obj.imag) < threshold):
                # If the imaginary part is small, return only the real part
                return np.round(obj.real, ndigits)
        # Round the array and set small values to zero
        obj = np.round(obj, ndigits)
        obj[np.abs(obj) < threshold] = 0
        return obj

    if isinstance(obj,str):
        if obj.replace('.', '', 1).isnumeric():
            # If it's a numeric string, convert to float
            obj = float(obj)
            obj = np.round(obj, ndigits)
            return str(obj)
        else:
            # If it's a non-numeric string, return as is
            return obj

    elif isinstance(obj, (int, float)):
        return np.round(obj, ndigits)

    elif isinstance(obj, complex):
        if abs(obj.imag) < threshold:
            # If the imaginary part is small, return only the real part
            return np.round(obj.real, ndigits)
        else:
            # If the imaginary part is significant, return the complex number rounded
            return np.round(obj, ndigits)        
    else:
        raise TypeError("Unsupported type for cleaning: {}".format(type(obj)))

def formatted_output(obj, precision=6):
    """
    Format numerical output with specified precision.

    Args:
        obj: Object to format
        precision: Number of decimal places

    Returns:
        Formatted string representation
    """
    if isinstance(obj, (int, float, complex)):
        if isinstance(obj, complex):
            if abs(obj.imag) < 1e-10:
                return f"{obj.real:.{precision}f}"
            else:
                return f"{obj.real:.{precision}f} + {obj.imag:.{precision}f}j"
        else:
            return f"{obj:.{precision}f}"
    else:
        return str(obj)


def generate_random_bitstring(n, k):
    """Generates a random bit string of length n with Hamming weight k.

    Based on `np.random.choice`

    Args:
        n: The length of the bit string.
        k: The Hamming weight (number of 1s).

    Returns:
        A NumPy array representing the bit string, or None if k is invalid.
    """
    if k < 0 or k > n:
        return None  # Invalid Hamming weight

    bitstring = np.zeros(n, dtype=int)

    indices = np.random.choice(n, size=k, replace=False)
    bitstring[indices] = 1
    return bitstring


def kron_plus(a, b):
    """Computes the direct sum of two matrices

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Direct sum matrix [[a, 0], [0, b]]
    """
    Z01 = np.zeros((a.shape[0], b.shape[1]))
    return np.block([[a, Z01], [Z01.T, b]])
