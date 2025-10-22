"""
Docstring to be added later.
"""

import torch
import numpy as np

def reparameterization_trick(mu, logvar):
    """
    Reparameterization trick to sample from a Gaussian distribution
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def KL_div_modified(var1, var2):
    """
    Computes KL divergence between two normal dists. with identical means based only on
    the variances.
    """
    return torch.log(torch.sqrt(var2) / torch.sqrt(var1)) + var1 / (2 * var2) - 0.5

def to_categorical(x, num_classes=None):
    """
    The "to_categorical" function copied directly from the TensorFlow documentation.
    NOT my own code, but included here to avoid TensorFlow dependency.

    Converts a class vector (integers) to binary class matrix (i.e., one-hot encoding).

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical