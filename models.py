"""
Docstring to be added later


"""

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import warnings


class ProNDF(pl.LightningModule):
    """
    Probabilistic NN for data fusion.
    TODO: UPDATE DOCSTRING AT LATER DATE
    PSEUDOCODE:
    Take in block 1, block 2, and block 3, trainer, laerning algorithm, etc.
    Construct the architecture based on the blocks and the parameters passed in.
    """
    def __init__(
        self,
        dsource: list[int],
        dcat: list[int],
        dnum: int,
        dy: list[int],
        num_outputs: int,
        num_sources: int,
        # qual_in: bool,  # TODO: Delete these and instead check whether dims are 0 or None
        # quant_in: bool,
        # dz_B0: int = 2,
        # dz_B1: int = 2,
        # architecture: dict[str, list[int]] = {  # TODO: Decide on whether we should include bias
        #     "Bias": [8, 4],  # TODO: Decide on whether we should allow the user to construct the architecture differently... Currently we're trying to make this as plug-and-play as possible, but that may not be the best approach
        #     "B0": [8, 4],
        #     "B1": [8, 4],
        #     "B3": [16, 32, 16, 8],
        # },
        var_init_bounds: list[float] = [-5.0, -3.0],  # TODO: Use an initializer object in the calibration version?
        act_fn: nn.functional = nn.Tanh(),  # TODO: Should this be included in the architecture dict instead?
        lr: float = 0.001,  # TODO: Should bundle all of the parameters associated with training into a trainer class instead of passing them here. Current setup constrains the model to be used only with regression. We want it to be usable on classification, etc. as well.
        k_L2: float = 0.01,
        k_KL: float = 0.01,
        k_IS: float = 1.0,
        KL_prior=0.1,
        # alpha1: float = 0.9,  # TODO: Pass a loss-weighting class object (or None) rather than these parameters. Allows for more modularity.
        # alpha2: float = 0.999,
        add_bias: bool = False,  # TODO: Make it more clear that this addresses the data fusion scheme (bias vs no bias) rather than bias in NN blocks(i.e., weights and biases)
        prob_bias: bool = True,  # TODO: Perhaps the user should create each block separately and then pass them to the model individually rather than having the model construct them. This would reduce the number of parameters passed to the model and allow for more flexibility in the architecture.
        prob_B0: bool = False,
        prob_B1: bool = False,
        prob_B2: bool = False,
        prob_out: bool = True,
        num_realizations: int = 1,  # TODO: Again, this should be addressed in each block individually. Do not want to have this hardcoded as is currently is. (Actually just remove it)
        loss_fn: str = "NLL",  # TODO: Should probably pass a loss function class object rather than a string. This would allow for more flexibility in the loss function used.
        loss_weighting: bool = True,  # TODO: Pass a loss weighting class rather than a string.
        loss_weight_ref_idx: int = 0,  # TODO: See above
        log_plots: dict[str, bool | int] = {  # TODO: Should this even be included? Definitely should have this done in a non-hardcoded way.
            "true_pred": True,
            "source_LS": True,
            "freq": 1000,
        },
    ):
        """
        TODO: Add docstring for __init__
        """
        super(ProNDF, self).__init__()
        # TODO: Add sanity checks for inputs, warnings, etc.
        # Save parameters
        self.save_hyperparameters()
        # Build blocks
