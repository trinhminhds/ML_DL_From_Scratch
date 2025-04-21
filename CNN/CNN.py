import numpy as np


def zero_pad(X, padding):
    """
    Pad with zeros all the images of the dataset X.

    Arguments:
        X -- numpy array of shape (m, n_H, n_W, n_C) representing m images
        pad -- integer, amount of padding around each image vertical and horizontal dimensions

    Returns:
        X_pad -- numpy array of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    X_pad = np.pad(
        X,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="constant",
        constant_values=(0, 0),
    )
    return X_pad


class Conv2D:
    """
    A 2D Convolution layer in a Neural Network.

    Attributes:
        filters (int): The number of filters in the convolution layer.
        filter_size (int): The size of the filters.
        input_channels (int, optional): The number of input channels. Defaults to 3.
        padding (int, optional): The number of zeros padding to be added to the input image. Defaults to 0.
        stride (int, optional): The stride length. Defaults to 1.
        learning_rate (float, optional): The learning rate to be used during training. Defaults to 0.001.
        optimizer (object, optional): The optimization method to be used during training. Defaults to None.
        cache (dict, optional): A dictionary to store intermediate values during forward and backward pass. Defaults to None.
        initializer (bool, optional): A flag to keep track of whether the layer has been initialized. Defaults to False.

    """

    def __init__(
        self,
        filters,
        filter_size,
        input_channels=3,
        padding=0,
        stride=1,
        learning_rate=0.001,
        optimizer=None,
    ):
        """
        Initialize the Conv2D layer with the given parameters.

        Args:
            filters (int): The number of filters in the convolution layer.
            filter_size (int): The size of the filters.
            input_channels (int, optional): The number of input channels. Defaults to 3.
            padding (int, optional): The number of zeros padding to be added to the input image. Defaults to 0.
            stride (int, optional): The stride length. Defaults to 1.
            learning_rate (float, optional): The learning rate to be used during training. Defaults to 0.001.
            optimizer (object, optional): The optimization method to be used during training. Defaults to None.
        """
        self.filters = filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.padding = padding
        self.stride = stride
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.cache = None
        self.initializer = False

    def relu(self, Z):
        """
        Implement the ReLU activation function.

        Arguments:
            Z -- numpy array of any shape

        Returns:
            A -- Post-activation parameter
            cache -- used for backpropagation
        """
        A = np.maximum(Z, 0)
        cache = Z
        return A, cache

    def relu_backward(self, dA, activation_cache):
        """
        Implement the backward propagation for a single ReLU unit.

        Arguments:
            dA -- post-activation gradient, of any shape
            activation_cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
            dZ -- Gradient of the cost with respect to Z
        """
        Z = activation_cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well,
        dZ[Z <= 0] = 0
        return dZ

    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation of the previous layers.

        Arguments:
            a_slice_prev -- slice of the output activation of shape (f, f, n_C_prev)
            W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
            b -- Bias parameter contained in a window - matrix of shape (1, 1, 1)

        Returns:
            A -- result of applying the activation function to Z
            cache -- used for backpropagation
        """
        s = np.multiply(a_slice_prev, W)
        Z = np.sum(s)
        Z = Z + float(b)
        return Z

    def forward(self, A_prev):
        """
        Implement the forward propagation for a convolution function.

        Parameter:
            A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev).

        Returns:
            Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
            cache -- cache of values needed for the conv_backward() function
        """
        # Create list to store activation cache for backprop
        activation_cache = []
        # Initialize neural network
        if self.initializer == False:
            np.random.seed(0)
            self.W = np.random.randn(
                self.filter_size, self.filters, A_prev.shape[-1], self.filters
            )
            self.b = np.random.randn(
                1,
                1,
                1,
                self.filters,
            )
            self.initializer = True
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = self.W.shape

        # Compute the dimensions of the output volume
        n_H = int((n_H_prev - f + (2 * self.padding)) / self.stride) + 1
        n_W = int((n_W_prev - f + (2 * self.padding)) / self.stride) + 1

        # Initialize the output volume Z with zeros
        Z = np.zeros((m, n_H, n_W, n_C))

        # add padding to A_prev
        A_prev_pad = zero_pad(A_prev, padding=self.padding)

        # Loop over the batch of training examples
        for i in range(m):
            # Select ith training example's padded activation
            a_prev_pad = A_prev_pad[i]
            for h in range(n_H):
                # Find the vertical start and end
                vert_start = h * self.stride
                vert_end = (h + 1) * self.stride
                for w in range(n_W):
                    # Find the horizontal start
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + f
                    # Loop over the channels
                    for c in range(n_C):
                        # Use the correct to define the slice from a_prev_pad
                        a_slice_prev = a_prev_pad[
                            vert_start:vert_end, horiz_start:horiz_end, :
                        ]
                        # Convolve the slice with the filter W and bias b
