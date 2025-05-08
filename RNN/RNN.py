import os
import numpy as np
import scipy as sp


class RNN:
    """
    A class used to represent a Recurrent Neural Network (RNN)

    Attributes:
        hidden_size: int
            The number of neurons in the hidden layer
        vocab_size: int
            The size of the vocabulary used by the RNN
        sequence_length: int
            The length of the input sequences fed to the RNN
        learning_rate: float
            The learning rate used during training
        is_initialized: bool
            Indicates whether the AdamW parameters has been initialized

    Methods:
        __init__(hidden_size, vocab_size, sequence_length, learning_rate)
            Initializes an instance of the RNN class

        forward(self, X, a_prev)
            Computes the forward pass of the RNN

        softmax(self, X)
            Computes the softmax activation function for a given input array

        backward(self, X, y_preds, targets)
            Implements the backward pass of the RNN

        loss(self, y_preds, targets)
            Computes the cross-entropy loss for a given sequence of predicted probabilities and true targets

        adamw(self, beta1=0.9, beta2=0.999, epsilon=1e-8, L2_reg=1e-4)
            Updates the RNN parameters using the Adamw optimization algorithm.

        train(self, generated_name = 5)
            Train the RNN's on a dataset using backpropagation through time (BPTT)

        predict(self, start)
            Generates a sequence of characters using the trained self, starting from the given sequence.
            The generated sequence may contain a maximum of 50 characters or a newline character.
    """

    def __init__(self, hidden_size, data_generator, sequence_length, learning_rate):
        """
        Initializes an instance of the RNN class.

        Parameters:
            hidden_size: int
                The number of hidden units in the RNN
            vocab_size: int
                The size of the vocabulary used by the RNN
            sequence_length: int
                The length of the input sequences fed to the RNN
            learning_rate: float
                The learning rate used during training.
        """

        # Hyper parameters
        self.hidden_size = hidden_size
        self.data_generator = data_generator
        self.vocab_size = self.data_generator.vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.X = None

        # model parameters
        self.Wax = np.random.uniform(
            -np.sqrt(
                1.0 / (self.vocab_size),
                np.sqrt(1.0 / self.vocab_size),
                (hidden_size, self.vocab_size),
            ),
        )
        self.Waa = np.random.uniform(
            -np.sqrt(1.0 / hidden_size),
            np.sqrt(1.0 / hidden_size),
            (hidden_size, hidden_size),
        )
        self.Wya = np.random.uniform(
            -np.sqrt(1.0 / hidden_size),
            np.sqrt(1.0 / hidden_size),
            (self.vocab_size, hidden_size),
        )
        self.ba = np.zeros((hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

        # Initialize gradients
        self.dWax, self.dWaa, self.dWya = (
            np.zeros_like(self.Wax),
            np.zeros_like(self.Waa),
            np.zeros_like(self.Wya),
        )
        self.dba, self.dby = np.zeros_like(self.ba), np.zeros_like(self.by)

        # Parameters update with AdamW
        self.mWax = np.zeros_like(self.Wax)
        self.vWax = np.zeros_like(self.Wax)
        self.mWaa = np.zeros_like(self.Waa)
        self.vWaa = np.zeros_like(self.Waa)
        self.mWya = np.zeros_like(self.Wya)
        self.vWya = np.zeros_like(self.Wya)
        self.mba = np.zeros_like(self.ba)
        self.vba = np.zeros_like(self.ba)
        self.mby = np.zeros_like(self.by)
        self.vby = np.zeros_like(self.by)

    def softmax(self, X):
        """
        Computes the softmax activation function for a given input array.

        Parameters:
            X (ndarray): Input array

        Returns:
            Array of the same shape as X, containing the softmax activation values.
        """
        # Shift the input to prevent overflow when computing the exponentials
        X = X - np.max(X)
        # Compute the exponentials of the shifted input
        p = np.exp(X)
        # Normaline the exponentials by dividing by their sum
        return p / np.sum(p)

    def forward(self, X, a_prev):
        """
        Computes the forward pass of the RNN.

        Parameters:
            X (ndarray): Input array of shape (seq_length, vocab_size)
            a_prev (ndarray): Activation of the previous time step of shape (vocab_size, 1)

        Returns:
            X (dict): Dictionary of input data of shape (seq_length, vocab_size, 1), with keys from 0 to seq_length - 1
            a (dict): Dictionary of hidden activations for each time step of shape, with keys from 0 to seq_length - 1
            y_pred (dict): Dictionary of output probabilities for each time step, with keys from 0 to seq_length - 1
        """
        # Initialize dictionaries to store activations and output probabilities
        x, a, y_pred = {}, {}, {}

        # Store the input data in class variable for later use in backward pass.
        self.X = X

        # Set the initial activation to the previous activation.
        a[-1] = np.copy(a_prev)
        # iterate over each time step in the previous activation
        for t in range(len(self.X)):
            # Get the input at the current time step
            x[t] = np.zeros((self.vocab_size, 1))
            if self.X[t] != None:
                x[t][self.X[t]] = 1
            # Compute the hidden activation at the current time step
            a[t] = np.tanh(
                np.dot(self.Wax, x[t]) + np.dot(self.Waa, a[t - 1]) + self.ba
            )
            # Compute the output probabilities at the current time step
            y_pred[t] = self.softmax(np.dot(self.Wya, a[t]) + self.by)
            # add an extra dimension to X to make it compatible with the shape of the input to the backward pass
        # Return the input, hidden activations, and output probabilities at each time step
        return x, a, y_pred

    def backward(self, x, a, y_preds, targets):
        """
        Computes the backward pass of the RNN.

        """
