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
        Implement the backward pass of the RNN.

        Arguments:
            x --(dict) of input characters (as one-hot encoding vectors) for each time step, shape of (vocab_size, sequence_length)
            a --(dict) of hidden state vectors for each time step, shape (hidden_size, sequence_length)
            y_pred --(dict) of output probabilities for vectors (after softmax) for each time step, shape of (vocab_size, sequence_length)
            targets --(list) of integer targets characters (indices of characters in the vocabulary) for each time step, shape (1, sequence_length)

        Returns:
            None
        """
        # Initialize derivative of hidden state for the last time-step
        da_next = np.zeros_like(a[0])

        # Loop through the input sequence backwards
        for t in reversed(range(len(self.X))):
            # Calculate derivative of output probability vector
            dy_preds = np.copy(y_preds[t])
            dy_preds[targets[t]] -= 1

            # Calculate derivative of hidden state
            da = np.dot(self.Waa.T, da_next) + np.dot(self.Wya.T, dy_preds)
            dtanh = 1 - np.power(a[t], 2)
            da_unactivated = dtanh * da

            # Calculate gradients
            self.dba += da_unactivated
            self.dWax += np.dot(da_unactivated, x[t].T)
            self.dWaa += np.dot(da_unactivated, a[t - 1].T)

            # Update derivative of hidden state for the next iteration
            da_next = da_unactivated

            # Calculate gradients for output weight matrix
            self.dWya += np.dot(dy_preds, a[t].T)

            # Clip gradients to avoid exploding gradients
            for grad in [self.dWax, self.dWaa, self.dWya, self.dba, self.dby]:
                np.clip(grad, -1, 1, out=grad)

    def loss(self, y_preds, targets):
        """
        Computes the cross entropy loss for a given sequence of predicted probabilities and true targets.

        Parameters:
            y_preds (ndarray): Array of shape (seq_length, vocab_size) containing predicted probabilities for each time step.
            targets (ndarray): Array of shape (seq_length, 1) containing the true targets for each time step.

        Returns:
            float: Cross-entropy loss.
        """
        # Calculate cross-entropy loss.
        return sum(-np.log(y_preds[t][targets[t], 0]) for t in range(len(self.X)))

    def adamw(self, beta1=0.9, beta2=0.999, epsilon=1e-8, L2_reg=1e-4):
        """
        Updates the RNN's parameters using the AdamW optimizer algorithm.
        """
        # AdamW update for Wax
        self.mWax = beta1 * self.mWax + (1 - beta1) * self.dWax
        self.vWax = beta1 * self.vWax + (1 - beta2) * np.square(self.dWax)
        m_hat = self.mWax / (1 - beta1)
        v_hat = self.vWax / (1 - beta2)
        self.Wax -= self.learning_rate * (
            m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wax
        )

        # AdamW update for Waa
        self.mWaa = beta1 * self.mWaa + (1 - beta1) * self.dWaa
        self.vWaa = beta2 * self.vWaa + (1 - beta2) * np.square(self.dWaa)
        m_hat = self.mWaa / (1 - beta1)
        v_hat = self.vWaa / (1 - beta2)
        self.Waa -= self.learning_rate * (
            m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Waa
        )

        # AdamW update to Wya
        self.mWya = beta1 * self.mWya + (1 - beta1) * self.dWya
        self.vWya = beta2 * self.vWya + (1 - beta2) * np.square(self.dWya)
        m_hat = self.mWya / (1 - beta1)
        v_hat = self.vWya / (1 - beta2)
        self.Wya -= self.learning_rate * (
            m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wya
        )

        # AdamW update for ba
        self.mba = beta1 * self.mba + (1 - beta1) * self.dba
        self.vba = beta2 * self.vba + (1 - beta2) * np.square(self.dba)
        m_hat = self.mba / (1 - beta1)
        v_hat = self.vba / (1 - beta2)
        self.ba -= self.learning_rate * (
            m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.ba
        )

        # AdamW update for by
        self.mby = beta1 * self.mby + (1 - beta1) * self.dby
        self.vby = beta2 * self.vby + (1 - beta2) * np.square(self.dby)

        def sample(self):
            """
            Sample a sequence of characters from the RNN

            Arguments:
                None

            Returns:
                list: A list of integers representing the generated sequence.
            """
