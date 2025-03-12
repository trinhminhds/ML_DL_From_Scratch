import numpy as np
import plotly.express as px


class NeuralNetwork:
    def __init__(self, layer_dimension=[25, 16, 16, 1], learning_rate=0.00001):
        """
        Parameters
        ----------
        layer_dimension: list
            python array (list) containing the dimensions of each layer in our network

        learning_rate: float
            learning rate of the network

        """
        self.layer_dimension = layer_dimension
        self.learning_rate = learning_rate

    def initialize_parameters(self):
        """initialize the parameters"""
        np.random.seed(3)
        self.n_layers = len(self.layer_dimension)
        for l in range(1, self.n_layers):
            vars(self)[f"W{l}"] = (
                np.random.randn(self.layer_dimension[l], self.layer_dimension[l - 1])
                * 0.01
            )
            vars(self)[f"b{l}"] = np.zeros((self.layer_dimension[l], 1))

    def _linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
             A -- activations from previous layer (size of previous layer, number of examples)
             W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
             b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
            Z -- pre-activation parameter
            cache -- a python tuple containing "A", "W" and "b" for backpropagation
        """
        # compute Z
        Z = np.dot(W, A) + b
        # cache A, W, b for backpropagation
        cache = (A, W, b)
        return Z, cache

    def _forward_propagation(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for a network layer

        Arguments:
            A_prev -- activations from previous layer, shape: (size of previous layer, number of examples)
            W -- shape: (size of current layer, size of previous layer)
            b -- shape: (size of the current layer, 1)
            activation -- the activation to be used in this layer

        Returns:
            A -- the output of the activation function
            cache -- a python tuple containing "linear_cache" and "activation_cache" for backpropagation
        """
        # Compute Z using the function implemented above, compute activation function
        if activation == "self.sigmoid":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "self.relu":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            # Store the cache for backpropagation
        cache = (linear_cache, activation_cache)
        return A, cache

    def relu(self, Z):
        """
        Implement the RELU function.

        Arguments:
            Z -- Output of the linear layer

        Returns:
            A -- Post-activation parameter
            cache -- used for backpropagation
        """
        A = np.maximum(0, Z)
        cache = Z
        return A, cache

    def relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
             dA -- post-activation gradient
             cache -- 'Z' stored for backpropagation

        Returns:
            dZ -- gradient of the cost with respect to Z
        """
        Z = cache
        dZ = np.array(dA, copy=True)
        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

        return dZ

    def sigmoid(self, Z):
        """
        Implement the sigmoid function

        Arguments:
            Z -- output of the linear layer

        Returns:
            A -- post-activation parameter
            cache -- a python dictionary containing "Z" for backpropagation
        """

        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single sigmoid unit.

        Arguments:
            dA -- post-activation gradient
            cache -- 'Z' stored during forward propagation

        Returns:
            dZ -- gradient of the cost with respect to Z
        """
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

    def forward_propagation(self, X):
        """
        Implements forward propagation for the whole network

        Arguments:
            X -- shape: (input size, number of examples)

        Returns:
            AL -- last post-activation value
            caches -- list of cache returned by the _forward_propagation function
        """
        # Initialize empty list to store caches
        caches = []
        #  Set initial A to X
        A = X
        L = self.n_layers - 1
        for l in range(1, L):
            A_prev = A
            # Forward propagate through the network except the last layer
            A, cache = self._forward_propagation(
                A_prev, vars(self)["W" + str(l)], vars(self)["b" + str(l)], "self.relu"
            )
            caches.append(cache)
        # Forward propagate through the output layer and get the prediction
        predictions, cache = self._forward_propagation(
            A, vars(self)["W" + str(L)], vars(self)["b" + str(L)], "self.sigmoid"
        )
        # Append the cache to caches list recall that cache will be (linear_cache, activation_cache)
        caches.append(cache)

        return predictions, caches

    def compute_cost(self, predictions, y):
        """
        Implement the cost function

        Arguments:
            predictions -- The model predictions, shape: (1, number of examples)
            y -- The true values, shape: (1, number of examples)

        Returns:
            cost -- cross-entropy cost
        """

        # Get number of training examples
        m = y.shape[0]
        # Compute the cost we're adding small epsilon for numerical stability
        cost = (-1 / m) * (
            np.dot(y, np.log(predictions + 1e-9).T)
            + np.dot((1 - y), np.log(1 - predictions + 1e-9).T)
        )
        # Squeeze to cost to set it into the correct shape
        cost = np.squeeze(cost)
        return cost

    def _linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation

        Arguments:
            dZ -- Gradient of the cost with respect to the linear output of the current layer
            cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
            dA_prev -- Gradient of the cost with respect to the activation of the previous layer
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        # Get the cache from forward propagation
        A_prev, W, b = cache
        # Get number of training examples
        m = A_prev.shape[1]
        # Compute gradients for W, b, and A
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def _back_propagation(self, dA, cache, activation):
        """
        Implement the backward propagation for a single layer

        Arguments:
            dA -- post-activation gradient for current layer l
            cache -- tuple of values (linear_cache, activation_cache)
            activation -- the activation to be used in this layer

        Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l - 1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        # Get the cache from forward propagation and activation derivative function
        linear_cache, activation_cache = cache
        # compute gradients for Z depending on the activation function
        if activation == "self.relu":
            dZ = self.relu_backward(dA, activation_cache)

        elif activation == "self.sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        # Compute gradients for W, b and A
        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def back_propagation(self, predictions, Y, caches):
        """
        Implements the backward propagation for the Neural Network

        Arguments:
            Predictions -- the output of the forward propagation
            Y -- the true values
            caches -- list of caches

        """
        L = self.n_layers - 1
        # Get number of examples
        m = predictions.shape[0]
        Y = Y.reshape(predictions.shape)
        # Initialize the backpropagation we're adding a small epsilon for numerical stability
        dAL = -(
            np.divide(Y, predictions + 1e-9) - np.divide(1 - Y, 1 - predictions + 1e-9)
        )
        current_cache = caches[L - 1]  # Last Layer
        # Compute gradients of the prediction
        vars(self)[f"dA{L-1}"], vars(self)[f"dW{L}"], vars(self)[f"db{L}"] = (
            self._back_propagation(dAL, current_cache, "self.sigmoid")
        )
        for l in reversed(range(L - 1)):
            # update the cache
            current_cache = caches[l]
            # compute gradients of the neural network
            vars(self)[f"dA{l}"], vars(self)[f"dW{l+1}"], vars(self)[f"db{l+1}"] = (
                self._back_propagation(
                    vars(self)[f"dA{l+1}"], current_cache, "self.relu"
                )
            )

    def momentum(self, beta=0.9):
        """
        Update parameters using Momentum

        Arguments:
             beta -- the momentum hyperparameter, scalar
        """

        L = self.n_layers - 1
        for l in range(L):
            vars(self)[f"vdW{l + 1}"] = np.zeros(
                vars(self)[f"W{l + 1}"].shape[0], vars(self)[f"W{l + 1}"].shape[1]
            )
            vars(self)[f"vdb{l + 1}"] = np.zeros(
                vars(self)[f"b{l + 1}"].shape[0], vars(self)[f"b{l + 1}"].shape[1]
            )

        for l in range(L):
            vars(self)[f"vdW{l + 1}"] = (
                beta * vars(self)[f"vdW{l + 1}"] + (1 - beta) * vars(self)[f"dW{l + 1}"]
            )
            vars(self)[f"vdb{l + 1}"] = (
                beta * vars(self)[f"vdb{l + 1}"] + (1 - beta) * vars(self)[f"db{l + 1}"]
            )

            vars(self)[f"W{l + 1}"] = (
                vars(self)[f"W{l + 1}"] - self.learning_rate * vars(self)[f"vdW{l + 1}"]
            )
            vars(self)[f"b{l + 1}"] = (
                vars(self)[f"b{l + 1}"] - self.learning_rate * vars(self)[f"vdb{l + 1}"]
            )

    def RMSProp(self, beta=0.9):
        """
        Update parameters using RMSProp

        Arguments:
            beat -- the momentum hyperparameter scalar

        """
        L = self.n_layers - 1
        for l in range(L):
            vars(self)[f"sdW{l + 1}"] = np.zeros(
                (vars(self)[f"W{l + 1}"].shape[0], vars(self)[f"W{l + 1}"].shape[1])
            )
            vars(self)[f"sdb{l + 1}"] = np.zeros(
                (vars(self)[f"b{l + 1}"].shape[0], vars(self)[f"b{l + 1}"].shape[1])
            )

        for l in range(L):
            vars(self)[f"sdW{l + 1}"] = beta * vars(self)[f"sdW{l + 1}"] + (
                1 - beta
            ) * np.square(vars(self)[f"dW{l + 1}"])
            vars(self)[f"sdb{l + 1}"] = beta * vars(self)[f"sdb{l + 1}"] + (
                1 - beta
            ) * np.square(vars(self)[f"db{l + 1}"])

            vars(self)[f"sdW{l + 1}"] = vars(self)[f"sdW{l + 1}"] / (1 - beta**2)
            vars(self)[f"sdb{l + 1}"] = vars(self)[f"sdb{l + 1}"] / (1 - beta**2)

            vars(self)[f"W{l + 1}"] = vars(self)[
                f"W{l + 1}"
            ] - self.learning_rate * vars(self)[f"dW{l + 1}"] / np.sqrt(
                vars(self)[f"sdW{l + 1}"] + 1e-9
            )
            vars(self)[f"b{l + 1}"] = vars(self)[
                f"b{l + 1}"
            ] - self.learning_rate * vars(self)[f"db{l + 1}"] / np.sqrt(
                vars(self)[f"sdb{l + 1}"] + 1e-9
            )

    def Adam(self, beta1=0.9, beta2=0.999):
        """
        Update prameters using Adam


        Arguments:
            beta1 -- Exponential decay hyperparameter for the first moment estimates
            beta2 -- Exponential decay hyperparameter for the second moment estimates
        """

        L = self.n_layers - 1
        for l in range(L):
            vars(self)[f"vdW{l + 1}"] = np.zeros(
                (vars(self)[f"W{l + 1}"].shape[0], vars(self)[f"W{l + 1}"].shape[1])
            )
            vars(self)[f"vdb{l + 1}"] = np.zeros(
                (vars(self)[f"b{l + 1}"].shape[0], vars(self)[f"b{l + 1}"].shape[1])
            )
            vars(self)[f"sdW{l + 1}"] = np.zeros(
                (vars(self)[f"W{l + 1}"].shape[0], vars(self)[f"W{l + 1}"].shape[1])
            )
            vars(self)[f"sdb{l + 1}"] = np.zeros(
                (vars(self)[f"b{l + 1}"].shape[0], vars(self)[f"b{l + 1}"].shape[1])
            )

        for l in range(L):
            vars(self)[f"vdW{l + 1}"] = (
                beta1 * vars(self)[f"vdW{l + 1}"]
                + (1 - beta1) * vars(self)[f"dW{l + 1}"]
            )
            vars(self)[f"vdb{l + 1}"] = (
                beta1 * vars(self)[f"vdb{l + 1}"]
                + (1 - beta1) * vars(self)[f"db{l + 1}"]
            )

            vars(self)[f"vdW{l + 1}"] = vars(self)[f"vdW{l + 1}"] / (1 - beta1**2)
            vars(self)[f"vdb{l + 1}"] = vars(self)[f"vdb{l + 1}"] / (1 - beta1**2)

            vars(self)[f"sdW{l + 1}"] = beta2 * vars(self)[f"sdW{l + 1}"] + (
                1 - beta2
            ) * np.square(vars(self)[f"dW{l + 1}"])
            vars(self)[f"sdb{l + 1}"] = beta2 * vars(self)[f"sdb{l + 1}"] + (
                1 - beta2
            ) * np.square(vars(self)[f"db{l + 1}"])

            vars(self)[f"sdW{l + 1}"] = vars(self)[f"sdW{l + 1}"] / (1 - beta2**2)
            vars(self)[f"sdb{l + 1}"] = vars(self)[f"sdb{l + 1}"] / (1 - beta2**2)

            vars(self)[f"W{l + 1}"] = vars(self)[
                f"W{l + 1}"
            ] - self.learning_rate * vars(self)[f"vdW{l + 1}"] / np.sqrt(
                vars(self)[f"sdW{l + 1}"] + 1e-9
            )
            vars(self)[f"b{l + 1}"] = vars(self)[
                f"b{l + 1}"
            ] - self.learning_rate * vars(self)[f"vdb{l + 1}"] / np.sqrt(
                vars(self)[f"sdb{l + 1}"] + 1e-9
            )

    def update_parameters(self, optimizer=None):
        """
        Update parameters

        Arguments:
            optimizer -- the optimizer to be used (default): None
        """
        L = self.n_layers - 1
        if optimizer == "momentum":
            self.momentum(beta=0.9)
        elif optimizer == "rmsprop":
            np.seterr(divide="ignore", invalid="ignore")
            self.RMSProp(beta=0.999)
        elif optimizer == "adam":
            self.Adam()
        else:
            for l in range(L):
                vars(self)[f"W{l + 1}"] = (
                    vars(self)[f"W{l + 1}"]
                    - self.learning_rate * vars(self)[f"dW{l + 1}"]
                )
                vars(self)[f"b{l + 1}"] = (
                    vars(self)[f"b{l + 1}"]
                    - self.learning_rate * vars(self)[f"db{l + 1}"]
                )

    def step_decay(self):
        pass

    def fit(self, X, Y, epochs=2000, optimizer=None, print_cost=True):
        """
        Trains the Neural Network using input data

        Arguments:
            X -- input data
            Y -- true label
            epochs -- number of iterations of the optimization loop
            optimizer -- the optimizer to be used (default): None
            print_cost -- If set to True, prints the cost every 100 iterations
        """
        # Transpose X to get the correct shape
        X = X.T
        np.random.seed(1)
        # Create empty array to store the cost
        costs = []
        # Get number of training examples
        m = X.shape[1]
        # Initialize parameters
        self.initialize_parameters()
        # loop for stated number of epochs
        for i in range(0, epochs):
            # Forward propagation and get the predictions and caches
            predictions, caches = self.forward_propagation(X)
            # compute the cost function
            cost = self.compute_cost(predictions, Y)
            # Calculate the gradient adn update the parameters
            self.back_propagation(predictions, Y, caches)
            self.update_parameters(optimizer=optimizer)

            # Print the cos every 10000 training example
            if print_cost and i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 5000 == 0:
                costs.append(cost)
        if print_cost:
            # Plot the cost over training
            fig = px.line(
                y=np.squeeze(costs), title="Cost Function", template="plotly_dark"
            )
            fig.update_layout(
                title_font_color="#f6abb6",
                xaxis=dict(color="#f6abb6"),
                yaxis=dict(color="#f6abb6"),
            )
            fig.show()

    def predict(self, X, y):
        """
        Uses the trained model to predict given X value

        Arguments:
            X -- input data set of examples you would like to label
            y -- True values of examples; used to calculate accuracy

        Returns:
            predictions -- predictions of the given dataset X
        """
        X = X.T
        # Get predictions from forward propagation
        predictions, _ = self.forward_propagation(X)
        # Predictions Above 0.5 are True otherwise they are False
        predictions = predictions > 0.5
        # Squeeze the predictions into correct shape and cats true/false values to 1/0
        predictions = np.squeeze(predictions.astype(int))
        # Print the accuracy
        return np.sum((predictions == y) / X.shape[1]), predictions.T
