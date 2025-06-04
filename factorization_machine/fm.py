import numpy as np
from sklearn import preprocessing


class FactorizationMachine:
    """A simple Factorization Machine model for regression tasks."""

    def __init__(self, n_factors=10):
        self.n_factors = n_factors
        self.weights = None
        self.factors = None

    def calculate_second_order_terms(self, X):
        """Calculate second order terms for the input features."""
        poly_features = [
            X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])
        ]
        return np.array(poly_features)

    def initialize_weights(self):
        """Initialize weights and factors for the model. Also define the indices for the different kinds of weights"""
        # The number of weights = 1 + number_of_features + n_factors * number_of_features
        self.n_features = n_features = self.X.shape[1]
        n_weights = n_features + 1 + self.n_factors * n_features

        # Need to initialize the weights to a flat array - because that is the required format for the optimizer
        self.weights = np.random.normal(0, 0.1, n_weights)

        # Define the indices for the weights - will be useful in the predict method
        self.intercept_index = 0
        self.first_order_feature_indices = np.arange(1, 1 + n_features)
        self.second_order_feature_indices = np.arange(
            1 + n_features, 1 + n_features + self.n_factors * n_features
        )

    def predict(self, X):
        """predict using stored coeffs in self.weights"""
        if self.weights is None:
            raise ValueError("Model weights are not initialized. Call fit() first.")

        # Initialize predictions
        predictions = 0

        # Add the intercept term
        predictions += self.weights[self.intercept_index]

        # Add the first order terms
        predictions += np.dot(
            X, self.weights[self.first_order_feature_indices].reshape(-1, 1)
        )

        # Add the second order terms
        # We want to write a for loop - and in each loop, evaluate the correct dot product

        X_poly = self.poly_features.transform(X)[:, 1 + self.n_features :]
        matrix_factors = self.weights[self.second_order_feature_indices]
        matrix_factors = matrix_factors.reshape(self.n_features, self.n_factors)
        coeffs = matrix_factors @ matrix_factors.T
        coeffs = coeffs.reshape()
        second_order_low_rank = np.dot(X_poly, matrix_factors.T)  # noqa: F841

        second_order_terms = self.poly_features.transform(X)[:, 1:]  # noqa: F841

    def fit(self, X, y):
        """
        Fit the Factorization Machine model to the data.
        """
        # Create Polynomial Features for second order terms
        self.poly_features = preprocessing.PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=False
        ).fit(X)
