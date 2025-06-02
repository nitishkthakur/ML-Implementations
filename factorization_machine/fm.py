import numpy as np


class FactorizationMachine:
    """A simple Factorization Machine model for regression tasks."""

    def __init__(self, n_factors=10):
        self.n_factors = n_factors
        self.weights = None
        self.factors = None

    def initialize_weights(self):
        """Initialize weights and factors for the model."""
        # The number of weights = 1 + number_of_features + n_factors * number_of_features
        self.n_features = n_features = self.X.shape[1]
        n_weights = n_features + 1 + self.n_factors * n_features
        self.weights = np.random.normal(0, 0.1, n_weights)

        # Define the indices for the weights - will be useful in the predict method
        self.intercept_index = 0
        self.first_order_feature_indices = np.arange(1, 1 + n_features)
        self.second_order_feature_indices = np.arange(
            1 + n_features, 1 + n_features + self.n_factors * n_features
        )

    def fit(self, X, y):
        """
        Fit the Factorization Machine model to the data.
        """
