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
                        weight = self.W[:, :, :, c]
                        bias = self.b[:, :, :, c]
                        Z[i, h, w, c] = self.conv_single_step(
                            a_slice_prev, weight, bias
                        )
                # Apply ReLU activation and store the cache for backpropagation
                Z[i], activation_cache = self.relu(Z[i])
                # Append the cache to the list
                activation_cache.append(activation_cache)

        self.cache = (A_prev, np.array(activation_cache))

        return Z

    def backward(self, dZ):
        """
        Implement the backward propagation for a convolution function.

        Parameter:
            dz -- gradient of the cost with respect to the output of the conv layer(Z), numypy array of shape (m, n_H, n_W, n_C)
            cache -- cache of values needed for the conv_backward() function, output of the conv_forward() function

        Returns:
            dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
            dW -- gradient of the cost with respect to the weight of the conv layer (W),
                    numpy array of shape (f, f, n_C_prev, n_C)
            db -- gradient of the cost with respect to the bias of the conv layer (b),
                    numpy array of shape (1, 1, 1, n_C)
        """
        # Retrieve information from cache
        A_prev, activation_cache = self.cache
        W, b = self.W, self.b

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve stride and padding information
        stride = self.stride
        padding = self.padding
        # Activation Gradient

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape

        # Initialize dA_prev, dW, db
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        self.dW = np.zeros((f, f, n_C_prev, n_C))
        self.db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = zero_pad(A_prev, padding)
        dA_prev_pad = zero_pad(dA_prev, padding)

        # Loop over the training examples
        for i in range(m):
            # Compute gradients of the activation function
            dZ[i] = self.relu_backward(dA_prev[i], activation_cache[i])
            # Select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            # Loop over vertical axis of the output volume
            for h in range(n_H):
                vert_start = h * stride
                vert_end = vert_start + f
                # Loop over horizontal axis of the output volume
                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Loop over the channels of the output volume
                    for c in range(n_C):
                        # Find a slice using the dimensions of the filter W
                        a_slice = a_prev_pad[
                            vert_start:vert_end, horiz_start:horiz_end, :
                        ]
                        # Update gradients for the window and the filter's parameters
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += (
                            W[:, :, :, c] * dZ[i, h, w, c]
                        )
                        self.dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        self.db[:, :, :, c] += dZ[i, h, w, c]

            # Set the ith training example's dA_prev to the updated dA_prev_pad
            if padding:
                dA_prev[i, :, :, :] = da_prev_pad[padding:-padding, padding:-padding, :]
            else:
                dA_prev[i, :, :, :] = da_prev_pad[i, :, :, :]

        self.update_parameters()
        return dA_prev

    def Adam(self, beta1=0.9, beta2=0.999):
        """ "
        Update parameters using Adam

        Parameters:
            beta1 -- Exponential decay hyperparameter for the first moment estimates
            beta2 -- Exponential decay hyperparameter for the second moment estimates
        """
        self.epsilon = 1e-8
        self.v_dW = np.zeros(self.W.shape)
        self.v_db = np.zeros(self.b.shape)
        self.s_dW = np.zeros(self.W.shape)
        self.s_db = np.zeros(self.b.shape)
        self.t = 1

        self.v_dW = beta1 * self.v_dW + (1 - beta1) * self.dW
        self.v_db = beta1 * self.v_db + (1 - beta1) * self.db
        self.v_dW_corrected = self.v_dW / (1 - beta1**self.t)
        self.v_db_corrected = self.v_db / (1 - beta1**self.t)

        self.s_dW = beta2 * self.s_dW + (1 - beta2) * np.square(self.dW)
        self.s_db = beta2 * self.s_db + (1 - beta2) * np.square(self.db)
        self.s_dW_corrected = self.s_dW / (1 - beta2**self.t)
        self.s_db_corrected = self.s_db / (1 - beta2**self.t)

        self.t += 1

        self.W = self.W - self.learning_rate * (
            self.v_dW_corrected / (np.sqrt(self.s_dW_corrected) + self.epsilon)
        )
        self.b = self.b - self.learning_rate * (
            self.v_db_corrected / (np.sqrt(self.s_db_corrected) + self.epsilon)
        )

    def update_parameters(self, optimizer=None):
        """
        Updates parameters

        Parameters:
            Optimizer -- the optimizer to be used (default): None
        """
        if optimizer == "adam":
            self.Adam()
        else:
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db


class Pooling2D:
    """
    2 Pooling layer for down-sampling image data.

    Parameters:
        filter_size (int) -- size of the pooling window
        stride (int) -- stride of the pooling window
        mode (str, optional) -- pooling mode, either 'max' or 'average'. Defaults to 'max'.
    """

    def __init__(self, filter_size, stride, mode="max"):
        """
        Initialize the Pooling2D layer with the given parameters.

        Parameters:
            filter_size (int): The size of the pooling window.
            stride (int): The stride length.
            mode (str, optional): Pooling mode, either 'max' or 'average'. Defaults to 'max'.
        """
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

    def forward(self, A_prev):
        """
        Implement the forward pass of the pooling layer.

        Parameters:
            A_prev (numpy.ndarray): The input data of shape (m, n_H_prev, n_W_prev, n_C_prev).
            mode -- the pooling mode you would like to use, defined as a string ('max' or 'average')

        Returns:
            A -- output of the pooling layer, a numpy array of shape (m, n_H, n_W, n_C)
            cache -- cache used in the backward pass of the pooling layer, contains the input and parameters
        """

        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve hyperparameters from "hyperparameters"
        f = self.filter_size
        stride = self.stride

        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        # Initialize the output matrix A
        A = np.zeros((m, n_H, n_W, n_C))
        # loop over the training examples
        for i in range(m):
            # loop on the vertical axis of the output volume
            for h in range(n_H):
                # Find the vertical start and end
                vert_start = h * stride
                vert_end = vert_start + f
                # loop on the horizontal axis of the output volume
                for w in range(n_W):
                    # Find the vertical start and end
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # loop over the channels of the output volume
                    for c in range(n_C):
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c
                        a_prev_slice = A_prev[
                            i, vert_start:vert_end, horiz_start:horiz_end, c
                        ]
                        # Compute the pooling operation on the slice
                        # Determine the pooling mode
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)

        # Store the input for backpropagation
        self.cache = (A_prev, A)

        return A

    def create_mask_from_window(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.

        Parameters:
            x -- Array of shape (f, f)

        Returns:
            mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        mask = x == x.max()
        return mask

    def distribute_value(self, dz, shape):
        """
        Distributes the input value in the matrix of dimension shape.

        Parameters:
            dz -- input scalar
            shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute dz value of dz

        Returns:
            a -- Array of shape (n_H, n_W) for which we distribute dz value of dz
        """
        # Retrieve dimensions from shape
        (n_H, n_W) = shape

        # Compute the value to distribute on the matrix
        average = dz / (n_H * n_W)

        # Create a matrix where every entry is the "average" value
        a = np.ones((n_H, n_W)) * average
        return a

    def backward(self, dA):
        """
        Implement the backward pass of the pooling layer

        Parameters:
            dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
            cache -- cache output from the forward pass of the pooling layer, contains the layer's input

        Returns:
            dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        # Retrieve information from cache
        A_prev = self.cache

        stride = self.stride
        f = self.filter_size

        # Retrieve dimensions from A_prev's shape and dA's shape
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Initialize dA_prev with zeros
        dA_prev = np.zeros((A_prev.shape))

        # Loop over the training examples
        for i in range(m):
            # Select training example from A_prev
            a_prev = A_prev[i, :, :, :]
            # loop on the vertical axis
            for h in range(n_H):
                # loop on the horizontal axis
                for w in range(n_W):
                    # loop over the channels
                    for c in range(n_C):
                        # Find the corners
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Compute the backward propagation in the case of max pooling
                        if self.mode == "max":

                            # Use the corners 'c' to define the current slice from a_prev
                            a_prev_slice = A_prev[
                                vert_start:vert_end, horiz_start:horiz_end, c
                            ]
                            # Create the mask from a_prev_slice
                            mask = self.create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiply by the corrent entry of dA)
                            dA_prev[
                                i, vert_start:vert_end, horiz_start:horiz_end, c
                            ] += (mask * dA[i, h, w, c])
                        elif self.mode == "average":
                            # Get the value from dA
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf
                            shape = (f, f)
                            # Distribute it to get the slice of dA_prev
                            dA_prev[
                                i, vert_start:vert_end, horiz_start:horiz_end, c
                            ] += self.distribute_value(da, shape)
        return dA_prev


class Flatten:
    """
    A class for flattening the input tensor in a neural network.
    """

    def __init__(self):
        """
        Initialize the input shape to None
        """

        self.input_shape = None

    def forward(self, X):
        """
        Implement the forward pass.

        Parameters:
            X (numpy.ndarray): The input tensor

        Returns:
            numpy.ndarray: The flattened input tensor
        """
        # Store the input shape
        self.input_shape = X.shape
        # Flatten the input tensor
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        """
        Implement the backward pass.

        Parameters:
            dout (numpy.ndarray): The gradient of the loss with respect to the output of the layer.

        Returns:
            numpy.ndarray: The reshaped gradiend tensor.
        """
        # Reshape the gradient to the original input shape
        return dout.reshape(self.input_shape)


class Dense:
    """
    A class representing a dense layer in a neural network.
    """

    def __init__(self, units, activation="relu", optimizer=None, learning_rate=0.001):
        """
        Implement the dense layer with the given number of units and activation function.

        Parameters:
            units (int): The number of units in the dense layer.
            activation (str): The activation function to be use, either 'relu' or 'sortmax'
            optimizer (str): The optimization to use for updating the weights.
            learning_rate (float): The learning rate to use during training.
        """
        self.units = units
        self.W = None
        self.b = None
        self.activation = activation
        self.input_shape = None
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, A):
        """
        Perform the forward pass of the dense layer.

        Parameters:
            A (numpy.ndarray): The input data of shape (batch_size, input_shape).
        Returns:
            numpy.ndarray: The output of the dense layer.
        """
        # if the weights are not initialized, initialize weight
        if self.W is None:
            self.initialize_weights(A.shape[1])
        # Store the input for use in backward pass
        self.A = A
        # Compute the linear transformation
        out = np.dot(A, self.W) + self.b
        # Apply the activation function
        if self.activation == "relu":
            out = np.maximum(out, 0)
        elif self.activation == "sortmax":
            out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)

        return out

    def initialize_weights(self, input_shape):
        """
        Initialize the weights of the dense layer.

        Parameters:
            input_shape (tuple): The shape of the input data.
        """
        # Initialize the weights and biases
        np.random.seed(0)
        # store the input shape
        self.input_shape = input_shape
        # initialize weights with small random values
        self.W = np.random.randn(input_shape, self.units) * 0.01
        # initialize biases with zeros
        self.b = np.zeros((1, self.units))
