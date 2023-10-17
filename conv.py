import torch
import torch.nn.functional as F


def zero_pad(X: torch.tensor, pad: int) -> torch.tensor:
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.

    Argument:
    X -- torch tensor of shape (m, n_C, n_H, n_W) representing a batch of m images.
        n_C, n_H, n_W denote respectively the number of channels, height and width
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_C, n_H + 2 * pad, n_W + 2 * pad)
    """
    pad_per_dim = (pad, pad, pad, pad)
    X_pad = F.pad(X, pad_per_dim, "constant", 0)

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev).

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameter contained in a window - scalar value

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    s = torch.mul(a_slice_prev, W)
    Z = torch.sum(s) + b

    return Z


class Conv2D:
    """
        Implements the class with forward propagation for a convolution function

        Arguments:
        A_prev -- input for current conv layer, torch tensor of shape (m, n_C, n_H, n_W)
        weight -- Weights, torch tensor of shape (n_C, n_C_prev, f, f)
        bias -- Biases, torch tensor of shape (n_C)
        stride -- stride hparameter, int
        pad -- pad hparameter, int

    """

    def __init__(self, weight, bias, stride, pad):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.pad = pad

    def __call__(self, *args, **kwargs):
        self._conv_forward(*args, **kwargs)

    def _conv_forward(self, input_tensor: torch.tensor):

        """
            Implements the forward propagation function

            Arguments:
            input_tensor -- input for current conv layer, torch tensor of shape (m, n_C, n_H, n_W)

            Returns:
            Z -- conv output, torch tensor of shape (m, n_C, n_H, n_W)
        """

        (m, n_C_prev, n_H_prev, n_W_prev) = input_tensor.shape
        (n_C, n_C_prev, f, f) = self.weight.shape

        # Compute the dimensions of the CONV output volume
        n_H = int((n_H_prev + (2 * self.pad) - f) / self.stride) + 1
        n_W = int((n_W_prev + (2 * self.pad) - f) / self.stride) + 1
        # Initialize the output volume Z with zeros.
        Z = torch.zeros(m, n_C, n_H, n_W)

        input_tensor_pad = zero_pad(input_tensor, self.pad)

        for i in range(m):
            input_tensor_pad = input_tensor_pad[i]
            for h in range(n_H):
                vert_start = self.stride * h
                vert_end = vert_start + f

                for w in range(n_W):
                    horiz_start = self.stride * w
                    horiz_end = horiz_start + f

                    for c in range(n_C):
                        a_slice_prev = input_tensor_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                        weights = self.weight[c, :, :, :]
                        biases = self.bias[c]
                        Z[i, c, h, w] = conv_single_step(a_slice_prev, weights, biases)

        return Z


class Pool2d:
    """
        Implements the forward pass of the pooling layer

        Arguments:
        stride -- stride hparameter, int
        filter -- filter size hparameter, int
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        A -- output of the pool layer, a torch tensor of shape (m, n_C, n_H, n_W)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    def __init__(self, filter, stride, mode = "max"):
        self.filter = filter
        self.stride = stride
        self.mode = mode

    def __call__(self, *args, **kwargs):
        self._pool_forward(*args, **kwargs)

    def _pool_forward(self, input_tensor: torch.tensor):
        """
            Implements the forward pass of the pooling layer

            Arguments:
            input_tensor -- input data, torch tensor of shape (m, n_C_prev, n_H_prev, n_W_prev)

            Returns:
            A -- output of the pool layer, a torch tensor of shape (m, n_C, n_H, n_W)
        """

        (m, n_C_prev, n_H_prev, n_W_prev) = input_tensor.shape

        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - self.filter) / self.stride)
        n_W = int(1 + (n_W_prev - self.filter) / self.stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = torch.zeros(m, n_C, n_H, n_W)

        for i in range(m):
            a_prev_slice = input_tensor[i]
            for h in range(n_H):
                vert_start = self.stride * h
                vert_end = vert_start + self.filter

                for w in range(n_W):
                    horiz_start = self.stride * w
                    horiz_end = horiz_start + self.filter

                    for c in range(n_C):
                        # Use the corners to define the (3D) slice of a_prev_pad
                        a_slice_prev = a_prev_slice[c, vert_start:vert_end, horiz_start:horiz_end]

                        if self.mode == "max":
                            A[i, c, h, w] = torch.max(a_slice_prev)
                        elif self.mode == "average":
                            A[i, c, h, w] = torch.mean(a_slice_prev)
                        else:
                            print(self.mode + "-type pooling layer NOT Defined")

        return A