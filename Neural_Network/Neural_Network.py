import numpy as np
import plotly.express as px


def relu(x):
    """
    Implementation of the ReLU activation function.

    Arguments:
        Z -- Output of the linear layer

    Returns:
        A -- Post-activation parameter
        cache -- used for backpropagation
    """
    A = np.maximum(0, x)
    cache = x
    return A, cache


def relu_backward(dA, cache):
    """
    Implementation of the backward propagation for a single ReLU unit.

    Argments:
        dA -- post-activation gradient
        cache -- 'Z' stored for backpropagation

    Returns:
        dZ -- Gradient of the cost with respect to Z

    """
    Z = cache
    dZ = np.array(dA, copy=True)
    # When Z <= 0, set dZ to 0 as well.
    dZ[Z <= 0] = 0

    return dZ


def sigmoid(Z):
    """
    Implementation the sigmoid function.

    Arguments:
        Z -- Output of the linear layer

    Returns:
        A -- Post-activation parameter
        cache -- a python dictionary containing 'A' for backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implementation of the backward propagation for a single sigmoid unit

    Arguments:
     dA -- post-activation gradient
     cache -- 'Z' stored for backpropagation

    Returns:
        dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


class Neural_Network:
    def __init__(self, layer_dimensions=[25, 16, 16, 1], learning_rate=0.00001):
        """
        Parameters
        ----------

        layer_dimensions : list
            python array (list) containing the dimensions of each layer in our network

        learning_rate :  float
            learning rate of the network.

        """
        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate

    def initialize_parameters(self):
        """initializes the parameters"""
        np.random.seed(3)
        self.n_layers = len(self.layer_dimensions)
        for l in range(1, self.n_layers):
            vars(self)[f"W{l}"] = (
                np.random.randn(self.layer_dimensions[l], self.layer_dimensions[l - 1])
                * 0.01
            )
            vars(self)[f"b{l}"] = np.zeros((self.layer_dimensions[l], 1))

    def _linear_forward(self, A, W, b):
        """
        Implements the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- pre-activation parameter
        cache -- a python tuple containing "A", "W" and "b"  for backpropagation
        """
        # Compute Z
        Z = np.dot(W, A) + b
        # Cache  A, W , b for backpropagation
        cache = (A, W, b)
        return Z, cache

    def _forward_propagation(self, A_prev, W, b, activation):
        """
        Implements the forward propagation for a network layer

        Arguments:
        A_prev -- activations from previous layer, shape : (size of previous layer, number of examples)
        W -- shape : (size of current layer, size of previous layer)
        b -- shape : (size of the current layer, 1)
        activation -- the activation to be used in this layer

        Returns:
        A -- the output of the activation function
        cache -- a python tuple containing "linear_cache" and "activation_cache" for backpropagation
        """

        # Compute Z using the function defined above, compute A using the activaiton function
        if activation == "sigmoid":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            # Store the cache for backpropagation
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X):
        """
        Implements forward propagation for the whole network

        Arguments:
        X --  shape : (input size, number of examples)

        Returns:
        AL -- last post-activation value
        caches -- list of cache returned by _forward_propagation helper function
        """
        # Initialize empty list to store caches
        caches = []
        # Set initial A to X
        A = X
        L = self.n_layers - 1
        for l in range(1, L):
            A_prev = A
            # Forward propagate through the network except the last layer
            A, cache = self._forward_propagation(
                A_prev, vars(self)["W" + str(l)], vars(self)["b" + str(l)], "relu"
            )
            caches.append(cache)
        # Forward propagate through the output layer and get the predictions
        predictions, cache = self._forward_propagation(
            A, vars(self)["W" + str(L)], vars(self)["b" + str(L)], "sigmoid"
        )
        # Append the cache to caches list recall that cache will be (linear_cache, activation_cache)
        caches.append(cache)

        return predictions, caches

    def compute_cost(self, predictions, y):
        """
        Implements the cost function

        Arguments:
        predictions -- The model predictions, shape : (1, number of examples)
        y -- The true values, shape : (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        # Get number of training examples
        m = y.shape[0]
        # Compute cost we're adding small epsilon for numeric stability
        cost = (-1 / m) * (
            np.dot(y, np.log(predictions + 1e-9).T)
            + np.dot((1 - y), np.log(1 - predictions + 1e-9).T)
        )
        # squeeze the cost to set it into the correct shape
        cost = np.squeeze(cost)
        return cost

    def _linear_backward(self, dZ, cache):
        """
        Implements the linear portion of backward propagation

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output of the current layer
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        # Get the cache from forward propagation
        A_prev, W, b = cache
        # Get number of training examples
        m = A_prev.shape[1]
        # Compute gradients for W, b and A
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def _back_propagation(self, dA, cache, activation):
        """
        Implements the backward propagation for a single layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache)
        activation -- the activation to be used in this layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        # get the cache from forward propagation and activation derivates function
        linear_cache, activation_cache = cache
        # compute gradients for Z depending on the activation function
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        # Compute gradients for W, b and A
        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def back_propagation(self, predictions, Y, caches):
        """
        Implements the backward propagation for the NeuralNetwork

        Arguments:
        Prediction --  output of the forward propagation
        Y -- true label
        caches -- list of caches
        """
        L = self.n_layers - 1
        # Get number of examples
        m = predictions.shape[1]
        Y = Y.reshape(predictions.shape)
        # Initializing the backpropagation we're adding a small epsilon for numeric stability
        dAL = -(
            np.divide(Y, predictions + 1e-9) - np.divide(1 - Y, 1 - predictions + 1e-9)
        )
        current_cache = caches[L - 1]  # Last Layer
        # Compute gradients of the predictions
        vars(self)[f"dA{L-1}"], vars(self)[f"dW{L}"], vars(self)[f"db{L}"] = (
            self._back_propagation(dAL, current_cache, "sigmoid")
        )
        for l in reversed(range(L - 1)):
            # update the cache
            current_cache = caches[l]
            # compute gradients of the network layers
            vars(self)[f"dA{l}"], vars(self)[f"dW{l+1}"], vars(self)[f"db{l+1}"] = (
                self._back_propagation(
                    vars(self)[f"dA{l + 1}"], current_cache, activation="relu"
                )
            )

    def update_parameters(self):
        """
        Updates parameters using gradient descent
        """
        L = self.n_layers - 1
        # Loop over parameters and update them using computed gradients
        for l in range(L):
            vars(self)[f"W{l+1}"] = (
                vars(self)[f"W{l+1}"] - self.learning_rate * vars(self)[f"dW{l+1}"]
            )
            vars(self)[f"b{l+1}"] = (
                vars(self)[f"b{l+1}"] - self.learning_rate * vars(self)[f"db{l+1}"]
            )

    def fit(self, X, Y, epochs=2000, print_cost=True):
        """
        Trains the Neural Network using input data

        Arguments:
        X -- input data
        Y -- true "label"
        Epochs -- number of iterations of the optimization loop
        print_cost -- If set to True, this will print the cost every 100 iterations
        """
        # Transpose X to get the correct shape
        X = X.T
        np.random.seed(1)
        # create empty array to store the costs
        costs = []
        # Get number of training examples
        m = X.shape[1]
        # Initialize parameters
        self.initialize_parameters()
        # loop for stated number of epochs
        for i in range(0, epochs):
            # Forward propagate and get the predictions and caches
            predictions, caches = self.forward_propagation(X)
            # compute the cost function
            cost = self.compute_cost(predictions, Y)
            # Calculate the gradient and update the parameters
            self.back_propagation(predictions, Y, caches)

            self.update_parameters()

            # Print the cost every 10000 training example
            if print_cost and i % 5000 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 5000 == 0:
                costs.append(cost)
        if print_cost:
            # Plot the cost over training
            fig = px.line(y=np.squeeze(costs), title="Cost", template="plotly_dark")
            fig.update_layout(
                title_font_color="#00F1FF",
                xaxis=dict(color="#00F1FF"),
                yaxis=dict(color="#00F1FF"),
            )
            fig.show()

    def predict(self, X, y):
        """
        uses the trained model to predict given X value

        Arguments:
        X -- data set of examples you would like to label
        y -- True values of examples; used for measuring the model's accuracy
        Returns:
        predictions -- predictions for the given dataset X
        """
        X = X.T
        # Get predictions from forward propagation
        predictions, _ = self.forward_propagation(X)
        # Predictions Above 0.5 are True otherwise they are False
        predictions = predictions > 0.5
        # Squeeze the predictions into the correct shape and cast true/false values to 1/0
        predictions = np.squeeze(predictions.astype(int))
        # Print the accuracy
        return np.sum((predictions == y) / X.shape[1]), predictions.T
