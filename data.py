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
    Potential additional methods:
    - save and load
    - scale and unscale (normalization) - maybe we store the parameters used to scale/unscale the data as well as set a state variable that indicates whether the data is currently scaled or unscaled?
    """
    def __init__(
            self,
            source: np.ndarray = None,
            cat: np.ndarray = None,
            num: np.ndarray = None,
            targets: np.ndarray = None,
            meta: dict = {},
            # scaled: bool = False,  # Decide whether we want to do this
            # scale_dict = None,
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


# Functions dealing with normalizing/scaling data

ArrayLike = np.ndarray
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
    
    