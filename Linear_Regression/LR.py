# Building Linear Regression Scratch
import numpy as np


class linear_regression:

    def __init__(self, learning_rate, no_of_interations):
        self.learning_rate = learning_rate
        self.no_of_interations = no_of_interations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.no_of_interations):
            # Hypothesis function in Linear Regression
            y_pred = np.dot(X, self.weights) + self.bias

            # Update parameters
            dw = (1 / n_sample) * np.dot(X.T, (y_pred - y))
            db = (1 / n_sample) * np.sum(y_pred - y)

            # Gradient Descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
