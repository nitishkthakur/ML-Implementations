import pandas as pd
import numpy as np
from typing import Union


def polynomial_features(X):
    """Takes featurea and polynomial features such that any given pair of features never appears twice as a product"""
    X_poly = np.zeros((X.shape[0], X.shape[1] * (X.shape[1] - 1) // 2)) * np.nan

    # Iterate and store polynomial features
    outer_index = 0
    for index in range(X.shape[1]):
        X_poly[:, outer_index : outer_index + X.shape[1] - index] = (
            X[:, index].reshape(-1, 1) * X[:, index:]
        )
        outer_index += X.shape[1] - index
    return X_poly


def flatten_lower_triangular(matrix):
    """
    Extracts the lower triangular part of the matrix (including the diagonal)
    and returns it as a 1D flattened array.

    Parameters:
        matrix (np.ndarray): A 2D numpy array.

    Returns:
        np.ndarray: A 1D array containing the lower triangular elements of the matrix.
    """
    lower_tri_indices = np.tril_indices_from(matrix)
    return matrix[lower_tri_indices]


def product_of_features_and_coeffs_expansion_included(
    X: Union[pd.DataFrame, np.array], beta: np.array
) -> np.array:
    """1. Assumes beta is a matrix - of coeffs for 2d interactions
    2. Functionextracts lower triangular part and flattens it
    3. returns dot product of that and the polynomial features of X (also represented in that form)"""

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Restructure beta
    beta_reshaped = flatten_lower_triangular(beta).reshape(-1, 1)

    # compute predictions
    predictions = X @ beta_reshaped

    return predictions
