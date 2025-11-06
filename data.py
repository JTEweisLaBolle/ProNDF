"""
data.py
This module contains classes and functions for generating and handling datasets to be 
used with ProNDF. User may add additional or custom datasets via files, which can then 
be loaded.
"""

# To include:
# General analytic dataset generation function (NOTE - ensure we can pass a random generator object)
# File to generate the dataset itself will be in the appropriate folder
# Dataset classes to:
#    - Batch data
#    - Load from file
#    - Save to file
# Utility functions such as data normalization, etc. (should these go in eiher utils.py or in the dataset class instead?)
# Start with the dataset class

import torch
import numpy as np
from scipy.stats import qmc
from torch.utils.data import Dataset, DataLoader
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple


# Main dataset class
class MultiSourceDataset(Dataset):
    """
    # NOTE: Add docstring later. For now, let's list intended behaviors:
    # NOTE: Write plotting code to take dataset objects as inputs
    - Init should take in each piece of data (num, cat, source, out), and metadata dict (too much abstraction?)
    - Needs to have __len__ manually defined, as wel as __getitem__
    - Assumes any data processing (e.g., scaling, normalization, etc.) has already been done prior to init
    Potential additional methods:
    - save and load
    """
    def __init__(
            self,
            source: np.ndarray = None,
            cat: np.ndarray = None,
            num: np.ndarray = None,
            targets: np.ndarray = None,
            meta: dict = {},
    ):
        super(MultiSourceDataset, self).__init__()
        # Store parameters
        self.source = source
        self.cat = cat
        self.num = num
        self.targets = targets
        self.meta = meta
        # Unpack meta (NOTE: Update at a later date if needed - may need more / fewer params)
        self.quant_in = meta.get('quant_in', None)
        self.qual_in = meta.get('qual_in', None)

    def __len__(self):
        """Returns number of samples in the dataset."""
        return self.targets.shape[0]  # Targets are always present in the dataset
        
    def __getitem__(self, idx):
        """
        Returns a dict containing a single sample from the dataset.
        Args:
            idx (int): index of the sample to retrieve.
        Returns:
            sample: a dict containing the sample data.
            Keys: source, cat, num, targets
        """
        source_sample = self.source[idx, :]
        targets_sample = self.targets[idx, :]
        # Retrieve samples from categorical and numerical inputs if they exist
        cat_sample = self.cat[idx, :] if self.qual_in else torch.empty_like(targets_sample)
        num_sample = self.num[idx, :] if self.quant_in else torch.empty_like(targets_sample)
        out = {
            'source': source_sample,
            'cat': cat_sample,
            'num': num_sample,
            'targets': targets_sample
        }
        return out
    
    def load(self, path):
        """
        Loads dataset from a folder defined by path.
        NOTE: Complete after we decide what should go in the metadata.
        NOTE: Should we be using a folder path or a file path? Ideally I'd like to pass just a folder, but that might be too much abstraction.
        """
        pass

    def save(self, path):
        """
        Saves dataset files to a folder defined by path.
        # NOTE: When saving, we'll also be savin the metadata. This means the dataset needs to be able to fill in the metadata from existing inputs (source, cat, num, targets).
        NOTE: We should probably have a method called "build_metadata" or something like that - maybe call it in the init?
        """
        pass


# Functions and classes dealing with normalizing/scaling or otherwise preprocessing data

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

ArrayLike = np.ndarray  # For type hinting

def _ensure_2d(array: ArrayLike) -> Tuple[ArrayLike, bool]:
    """
    Ensures that input array is 2D. If input is 1D, reshapes to (N, 1).
    """
    x = np.asarray(array)
    was_1d = (x.ndim == 1)
    if was_1d:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Input array must be 1D or 2D array-like.")
    return x, was_1d

@dataclass
class StandardNormalizer:
    """
    Standardizes features (column-wise) to zero mean and unit variance:
    z = (x - mean) / std
    Stores mean_ and scale_ for inverse transform.
    Zero-variance columns have their scale_ set to 1 to avoid division by zero.
    """
    with_mean: bool = True
    with_std: bool = True
    mean_: Optional[np.ndarray] = field(init=False, default=None)
    scale_: Optional[np.ndarray] = field(init=False, default=None)
    n_features_in_: Optional[int] = field(init=False, default=None)

    def fit(self, x: ArrayLike) -> "StandardNormalizer":
        """
        Stores parameters for normalizing / unnormalizing data.
        """
        x2d, _ = _ensure_2d(x)
        self.n_features_in_ = x2d.shape[1]
        if self.with_mean:
            self.mean_ = np.mean(x2d, axis=0)
        else:
            self.mean_ = np.zeros(x2d.shape[1], dtype=x2d.dtype)
        if self.with_std:
            std = np.nanstd(x2d, axis=0, ddof=0)  # ddof=0 to match conventions
            # Handle zero-variance edge cases
            std_safe = std.copy()
            std_safe[std_safe == 0] = 1.0
            self.scale_ = std_safe
        else:
            self.scale_ = np.ones(x2d.shape[1], dtype=x2d.dtype)
        return self
    
    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Normalizes input using the stored parameters.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform().")
        x2d, was_1d = _ensure_2d(x)
        x2d = x2d - self.mean_ if self.with_mean else x2d
        x2d = x2d / self.scale_ if self.with_std else x2d
        return x2d.ravel() if was_1d else x2d  # Return to 1D if input was 1D
    
    def inverse_transform(self, x: ArrayLike) -> ArrayLike:
        """
        Un-normalizes input using the stored parameters.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        x2d, was_1d = _ensure_2d(x)
        x2d = x2d * self.scale_ if self.with_std else x2d
        x2d = x2d + self.mean_ if self.with_mean else x2d
        return x2d.ravel() if was_1d else x2d  # Return to 1D if input was 1D
    
    def to_dict(self) -> dict:
        """
        Stores parameters in dictionary for serialization.
        """
        return {
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "mean_": None if self.mean_ is None else self.mean_.tolist(),
            "scale_": None if self.scale_ is None else self.scale_.tolist(),
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardNormalizer":
        """
        Loads parameters from dictionary for deserialization.
        """
        obj = cls(with_mean=d["with_mean"], with_std=d["with_std"])
        obj.mean_ = None if d["mean_"] is None else np.array(d["mean_"])
        obj.scale_ = None if d["scale_"] is None else np.array(d["scale_"])
        obj.n_features_in_ = d.get("n_features_in_", None)
        return obj
    

@dataclass
class MinMaxNormalizer:
    """
    Scales features (column-wise) to provided range (defaults to (0, 1)):
    z = (x - min) / (max - min)
    Zero-variance columns are mapped to the lower bound of the range.
    """
    feature_range: Tuple[float, float] = (0.0, 1.0)
    data_min_: Optional[np.ndarray] = field(init=False, default=None)
    data_max_: Optional[np.ndarray] = field(init=False, default=None)
    scale_: Optional[np.ndarray] = field(init=False, default=None)  # for convenience for inverse transform
    min_shift_: Optional[np.ndarray] = field(init=False, default=None) # for convenience for inverse transform
    denom_: Optional[np.ndarray] = field(init=False, default=None)  # For handling zero-variance columns
    n_features_in_: Optional[int] = field(init=False, default=None)

    def fit(self, x: ArrayLike) -> "MinMaxNormalizer":
        """
        Stores parameters for normalizing / unnormalizing data.
        """
        x2d, _ = _ensure_2d(x)
        self.n_features_in_ = x2d.shape[1]
        self.data_min_ = np.nanmin(x2d, axis=0)
        self.data_max_ = np.nanmax(x2d, axis=0)
        denom = (self.data_max_ - self.data_min_).astype(float)
        denom[denom == 0] = 1.0  # avoid div-by-zero; constant columns handled below
        self.denom_ = denom

        fr_min, fr_max = self.feature_range
        fr_scale = (fr_max - fr_min)
        self.scale_ = np.full(x2d.shape[1], fr_scale, dtype=float)
        self.min_shift_ = np.full(x2d.shape[1], fr_min, dtype=float)
        return self
    
    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Normalizes input using the stored parameters.
        """
        if any(v is None for v in (self.data_min_, self.denom_, self.scale_, self.min_shift_)):
            raise RuntimeError("Call fit() before transform().")
        x2d, was_1d = _ensure_2d(x)
        z = (x2d - self.data_min_) / self.denom_
        out = z * self.scale_ + self.min_shift_
        # For truly constant columns, map everything to lower bound:
        const_mask = (self.data_max_ == self.data_min_)
        if np.any(const_mask):
            out[:, const_mask] = self.feature_range[0]
        return out.ravel() if was_1d else out  # Return to 1D if input was 1D
    
    def inverse_transform(self, x: ArrayLike) -> ArrayLike:
        """
        Un-normalizes input using the stored parameters.
        """
        if any(v is None for v in (self.data_min_, self.denom_, self.scale_, self.min_shift_)):
            raise RuntimeError("Call fit() before inverse_transform().")
        x2d, was_1d = _ensure_2d(x)
        # Undo feature_range mapping:
        z = (x2d - self.min_shift_) / self.scale_
        out = z * self.denom_ + self.data_min_
        # For constant columns, everything maps back to the constant value:
        const_mask = (self.data_max_ == self.data_min_)
        if np.any(const_mask):
            out[:, const_mask] = self.data_min_[const_mask]
        return out.ravel() if was_1d else out  # Return to 1D if input was 1D
    
    def to_dict(self) -> dict:
        """
        Stores parameters in dictionary for serialization.
        """
        return {
            "feature_range": self.feature_range,
            "data_min_": None if self.data_min_ is None else self.data_min_.tolist(),
            "data_max_": None if self.data_max_ is None else self.data_max_.tolist(),
            "denom_": None if self.denom_ is None else self.denom_.tolist(),
            "scale_": None if self.scale_ is None else self.scale_.tolist(),
            "min_shift_": None if self.min_shift_ is None else self.min_shift_.tolist(),
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxNormalizer":
        """
        Loads parameters from dictionary for deserialization.
        """
        obj = cls(feature_range=tuple(d["feature_range"]))
        obj.data_min_ = None if d["data_min_"] is None else np.array(d["data_min_"])
        obj.data_max_ = None if d["data_max_"] is None else np.array(d["data_max_"])
        obj.denom_ = None if d["denom_"] is None else np.array(d["denom_"])
        obj.scale_ = None if d["scale_"] is None else np.array(d["scale_"])
        obj.min_shift_ = None if d["min_shift_"] is None else np.array(d["min_shift_"])
        obj.n_features_in_ = d.get("n_features_in_", None)
        return obj
    
