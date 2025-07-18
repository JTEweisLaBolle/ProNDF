"""
Docstring to be added later


"""

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import warnings
from .blocks import BLOCK_REGISTRY
from .losses import LOSS_HANDLER_REGISTRY
from .optimizers import OPTIMIZER_REGISTRY


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
        # Data parameters
        dsource: int,  # the number of data sources
        dcat: list[int],  # the number of categories for each categorical input
        dnum: int,  # the dimension of the numerical input
        dout: int,  # the dimenson of the output
        qual_in: bool,  # whether qualitative (categorical) inputs are present
        quant_in: bool,  # whether quantitative (numerical) inputs are present
        # Block / architecture parameters
        B1_type: str,  # Source block
        B1_config: dict[str, any],  # Block 1 parameters
        B2_type: str,  # Categorical block
        B2_config: dict[str, any],  # Block 2 parameters
        B3_type: str,  # Input-output relationship
        B3_config: dict[str, any],  # Block 3 parameters
        # Training parameters such as loss function, optimizer, and regularizers
        loss_handler_type: str,  # Loss handler to use
        loss_handler_config: dict[str, any],  # Loss handler config including loss functions and regularizers
        optimizer_type: str,  # Optimizer choice
        optimizer_config: dict[str, any],  # Optimizer configuration
        # Logging and other misc
        # log_plots: dict[str, bool | int] = {  # TODO: Figure out how to do logging. Should this be done via a "logger" class? Most likely yes.
        #     "true_pred": True,
        #     "source_LS": True,
        #     "freq": 1000,
        # },
    ):
        """
        TODO: Add docstring for __init__
        """
        super(ProNDF, self).__init__()
        # TODO: Add sanity checks for inputs, warnings, etc.
        # Save parameters
        self.save_hyperparameters()
        # Build blocks and loss handler
        # Build source block
        self.B1 = BLOCK_REGISTRY[B1_type(**B1_config)]
        # Build categorical block if necessary
        if qual_in:
            self.B2 = BLOCK_REGISTRY[B2_type(**B2_config)]
        # Build input-output block
        self.B3 = BLOCK_REGISTRY[B3_type(**B3_config)]
        # Build loss handler
        self.loss_handler = LOSS_HANDLER_REGISTRY[loss_handler_type(**loss_handler_config)]

    def forward(self, batch):
        """
        Performs a basic forward pass through the model, returning a tensor (sampled 
        from the output distribution if block 3 is probabilistic).
        TODO: Finish docstring
        """
        source, cat, num, out = batch
        # Get source manifold
        z_B1 = self.B1(source)
        # Get categorical manifold if necessary
        if self.hparams.qual_in:
            z_B2 = self.B2(cat)
        # Concatenate as necessary and pass through block 3
        # Checks to avoid iterative torch.cat operations
        if self.hparams.qual_in and self.hparams.quant_in:  # Both qual and quant inputs
            u = torch.cat((z_B1, z_B2, num), dim = -1)  # u is combined input
        elif self.hparams.qual_in:  # Only qual inputs
            u = torch.cat((z_B1, z_B2), dim = -1)  # u is combined input
        else:  # Only quant inputs
            u = torch.cat((z_B1, num), dim = -1)  # u is combined input
        out = self.B3(u)
        return out
        
    def get_model_outputs(self, batch):
        """
        Builds output dictionary for use in loss handling / loss context object
        TODO: Finish docstring
        """
        source, cat, num, out = batch
        outputs = {}
        # Get source manifold
        B1_outputs = {}
        z_B1 = self.B1(source)  # u is combined input to be concatenated
        B1_outputs["out"] = z_B1
        if self.B1.probabilistic_output:
            z_B1_dist = self.B1.predict_distribution(source)
            B1_outputs["out_dist"] = z_B1_dist
        outputs["B1"] = B1_outputs
        # Get categorical manifold if necessary
        if self.hparams.qual_in:
            B2_outputs = {}
            z_B2 = self.B2(cat)
            B2_outputs["out"] = z_B2
            if self.B2.probabilistic_output:
                z_B2_dist = self.B2.predict_distribution(cat)
                B2_outputs["out_dist"] = z_B2_dist
            outputs["B2"] = B2_outputs
        # Concatenate as necessary and pass through block 3
        # Checks to avoid iterative torch.cat operations
        if self.hparams.qual_in and self.hparams.quant_in:  # Both qual and quant inputs
            u = torch.cat((z_B1, z_B2, num), dim = -1)
        elif self.hparams.qual_in:  # Only qual inputs
            u = torch.cat((z_B1, z_B2), dim = -1)
        else:  # Only quant inputs
            u = torch.cat((z_B1, num), dim = -1)
        B3_outputs = {}
        out = self.B3(u)
        B3_outputs["out"] = out
        if self.B3.probabilistic_output:
            z_B3_dist = self.B3.predict_distribution(u)
            B3_outputs["out_dist"] = z_B3_dist
        outputs["B3"] = B3_outputs
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Model forward pass, loss term calculations, and loss weight updates
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch)
        # Build loss context
        self.loss_handler.build_loss_context(self, batch, outputs)
        # Compute loss terms for weighting
        self.loss_handler.compute_loss_terms()
        # Update loss weights
        self.loss_handler.update_loss_weights()
        # Compute final loss
        loss = self.loss_handler.compute_loss()
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Model forward pass and loss term calculations. No loss weight updates
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch)
        # Build loss context
        self.loss_handler.build_loss_context(self, batch, outputs)
        # Compute loss terms for weighting
        self.loss_handler.compute_loss_terms()
        # Compute final loss
        loss = self.loss_handler.compute_loss()
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Model forward pass and loss term calculations. No loss weight updates
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch)
        # Build loss context
        self.loss_handler.build_loss_context(self, batch, outputs)
        # Compute loss terms for weighting
        self.loss_handler.compute_loss_terms()
        # Compute final loss including regularization
        loss = self.loss_handler.compute_loss()
        return loss
    
    def configure_optimizers(self):
        """
        Initializes and configures optimizers using provided parameters
        """
        optimizer = OPTIMIZER_REGISTRY[self.hparams.optimizer_type](**self.hparams.optimizer_config)
        return optimizer