import numpy as np


class KNN:
    """
    K-Nearest Neighbors (KNN) classification algorithm.

    Parameters:
        n_neighbors: int, optional (default=5)
        Number of neighbors to use.

    Methods:
        fit(X_train, y_train)
            Stores the values of X_train and y_train.

        predict(X_test)
            Predicts the class labels for each example in X.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def euclidean_distance(self, X1, X2):
        """
        Calculates the Euclidean distance between two data points.

        Parameters:
            X1: numpy.ndarray, shape (n_features,)
                A data point in the dataset.

            X2: numpy.ndarray, shape (n_features,)
                A data point in the dataset.

        Returns:
            distance: float
                The Euclidean distance between X1 and X2.
        """
        return np.linalg.norm(X1 - X2)

    def fit(self, X_train, y_train):
        """
        Stores the values of X_train and y_train.

        Parameters:
            X_train: numpy.ndarray, shape (n_samples, n_features)
                The training data.

            y_train: numpy.ndarray, shape (n_samples, )
                The target labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        """
        Predicts the class labels for each example in X.

        Parameters:
            X: numpy.ndarray, shape (n_samples, n_features)
                The training data.

        Returns:
            Predictions: numpy.ndarray, shape (n_samples,)
                The predicted class labels for each example in X.
        """
        # Create empty array to store the predictions
        predictions = []
        # Loop over X examples
        for x in X:
            # Get prediction using the prediction helper function
            prediction = self._predict(x)
            # Append the prediction to the predictions list
            predictions.append(prediction)
        return np.array(predictions)

    def _predict(self, X):
        """
        Predicts the class labels for single example.

        Parameters:
            X: numpy.ndarray, shape (n_features, )
                A data point in the dataset.

        Returns:
            most_occurring_value: int
                The prediction class label for x.
        """
        # Create empty array to store distances
        distances = []
        # Loop over all training examples and distance between x and all the training examples.
        for x_train in self.X_train:
            distance = self.euclidean_distance(X, x_train)
            distances.append(distance)
        distances = np.array(distances)

        # Sort by ascendingly distance and return
