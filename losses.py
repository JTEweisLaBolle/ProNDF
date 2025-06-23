"""
losses.py
This module contains classes for losses, loss-weighting algorithms, and loss 
computations, as well as registries to store them.
Additional or custom losses and algorithms can be registered by the user if desired, 
either in this file or inline.
Registries are used to enable serialization by PyTorch Lightning's checkpointing 
system and automatic hyperparameter saving, as class objects can not be serialized.
Example usage:
    # Importing and using the registry and adding a custom loss
    from losses import LOSS_REGISTRY, register_loss

    # Registering a custom loss (inline or in this file)
    @register_loss("CustomLoss")
    class CustomLoss(nn.Module):
        def __init__(self):
            super(CustomLoss, self).__init__()
            # Register any buffers or parameters if needed
        def forward(self, mu, y, var, sigma):
            return torch.mean((mu - y) ** 2)  # Example custom loss function
"""

import torch
from torch import nn
from torch.nn import functional as F

# Loss registry for storing different types of losses
LOSS_REGISTRY = {}

def register_loss(name):
    """
    Decorator to register a loss type with the given name.
    
    Args:
        name (str): The name of the loss type to register.
    
    Returns:
        function: The decorator function that registers the loss.
    """
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


# loss functions
@register_loss("IS_loss")
class IS_loss(nn.Module):
    """
    Interval score loss. Assumes 95% CI. Requires network output to be dist. object.
    """

    def __init__(self, alpha=0.05):
        """
        Initializes the IS_loss with a significance level for the interval score.
        Args:
            alpha (float): Significance level for the interval score, default is 0.05.
        """
        super(IS_loss, self).__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, mu, y, var, sigma):
        """
        Assumes predictions is a distribution object
        """
        mu_lb = mu - 1.96 * sigma
        mu_ub = mu + 1.96 * sigma
        loss = mu_ub - mu_lb
        loss += (y > mu_ub).float() * 2 / self.alpha * (y - mu_ub)
        loss += (y < mu_lb).float() * 2 / self.alpha * (mu_lb - y)
        loss = loss.mean()
        return loss


@register_loss("NLL_loss")
class NLL_loss(nn.Module):
    """
    Negative log likelihood loss. Requires network output to be dist. object.
    """

    def __init__(self):
        """
        Initializes the NLL_loss.
        Args:
            None
        """
        super(NLL_loss, self).__init__()

    def forward(self, mu, y, var, sigma):
        loss = F.gaussian_nll_loss(mu, y, var, full=True, eps=1e-6) + 6
        return loss
    
@register_loss("NLL_IS_loss")
class NLL_IS_loss(nn.Module):
    """
    Negative log-likelihood loss with interval score. Assumes 95% CI.
    Requires network output to be dist. object.
    """

    def __init__(self, NLL_weight = 0.5, IS_weight = 0.5, alpha=0.05):
        """
        Initializes the NLL_IS_loss with weights for NLL and IS loss.
        Args:
            weights (list): Weights for NLL and IS loss, respectively.
            alpha (float): Significance level for the interval score.
        """
        super(NLL_IS_loss, self).__init__()
        self.register_buffer("NLL_weight", torch.tensor(NLL_weight))
        self.register_buffer("IS_weight", torch.tensor(IS_weight))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.NLL_loss = NLL_loss()
        self.IS_loss = IS_loss(alpha=alpha)

    def forward(self, mu, y, var, sigma):
        nll_loss = self.NLL_weight * self.NLL_loss(mu, y, var, sigma)
        is_loss = self.IS_weight * self.IS_loss(mu, y, var, sigma)
        return nll_loss + is_loss

@register_loss("MSE_loss")
class MSE_loss(nn.Module):
    """
    MSE loss. Works with either dist. object or raw tensor output.
    """

    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, mu, y, var, sigma):
        loss = F.mse_loss(mu, y)
        return loss

# Loss splitting registry
LOSS_SPLIT_REGISTRY = {}
def register_loss_split(name):
    """
    Decorator to register a loss splitting function with the given name.
    
    Args:
        name (str): The name of the loss splitting function to register.
    
    Returns:
        function: The decorator function that registers the loss splitting function.
    """
    def decorator(cls):
        LOSS_SPLIT_REGISTRY[name] = cls
        return cls
    return decorator

# loss splitting classes
class Base_Loss_split(nn.Module):
    """
    Base class for loss splitting. Should not be instantiated directly.
    """
    def __init__(self):
        super(Base_Loss_split, self).__init__()
        if type(self) is Base_Loss_split:
            raise NotImplementedError("Base_Loss_split should not be instantiated directly.")

    def forward(self, ):
        """
        Splits losses into individual components. Define in subclasses.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")
    

@register_loss_split("Split_by_source")
class Split_by_source(nn.Module):
    """
    Splits losses by source. Assumes losses are in the form of a list of tensors.
    """

    def __init__(self):
        super(Split_by_source, self).__init__()

    def forward(self, losses):
        """
        Splits losses by source.
        Args:
            losses (list): List of loss tensors to be split.
        Returns:
            list: List of loss tensors split by source.
        """
        return [losses[i] for i in range(len(losses))]

# Loss weighting algorithm registry
LW_ALG_REGISTRY = {}

def register_lw_alg(name):
    """
    Decorator to register a loss-weighting algorithm with the given name.
    
    Args:
        name (str): The name of the loss-weighting algorithm type to register.
    
    Returns:
        function: The decorator function that registers the loss-weighting algorithm.
    """
    def decorator(cls):
        LW_ALG_REGISTRY[name] = cls
        return cls
    return decorator


# Loss weighting algoithms
class Base_LW_alg(nn.Module):
    """
    Base class for loss weighting algorithms. Should not be instantiated directly.
    """
    def __init__(self):
        super(Base_LW_alg, self).__init__()
        if type(self) is Base_LW_alg:
            raise NotImplementedError("Base_LW_alg should not be instantiated directly.")

    def forward(self, losses):
        """
        Linear combination of loss terms. Define in subclasses.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")

    def update(self, losses, parameters, step):
        """
        Updates the loss weights.
        Inputs:
            losses: Flattened tensor of losses to be weighted
            parameters: Network params (e.g., model.parameters)
            step: Global step for bias correction (e.g., model.global_step)
        """
        raise NotImplementedError("Update method should be implemented in subclasses.")


@register_lw_alg("Dynamic_Weights")
class Dymanic_Weights(Base_LW_alg):
    """
    TODO: UPDATE DOCSTRING AT LATER DATE
    """

    def __init__(self, num_loss_terms, ref_idx=0, alpha1=0.9, alpha2=0.999, eps=1e-8):
        """TODO: Add docstring for __init__"""
        super(Dymanic_Weights, self).__init__()
        shape = (num_loss_terms,)
        self.register_buffer("lambdas", torch.zeros(shape))
        self.register_buffer("gammas", torch.zeros(shape))
        self.register_buffer("weights", torch.zeros(shape))
        self.register_buffer("alpha1", torch.tensor(alpha1))
        self.register_buffer("alpha2", torch.tensor(alpha2))
        self.register_buffer("eps", torch.tensor(eps))
        self.ref_idx = ref_idx

    def forward(self, losses):
        """
        Applies loss weights to loss terms and sums them.
        """
        return torch.sum(losses * self.weights)

    def update(self, losses, parameters, step):
        """
        Updates the loss weights.
        Inputs:
            losses: Flattened tensor of losses to be weighted
            parameters: Network params (e.g., model.parameters)
            step: Global step for bias correction (e.g., model.global_step)
        """
        # Calculate reference loss gradients, etc.
        ref_loss = losses[self.ref_idx]
        ref_grads = torch.autograd.grad(
            ref_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        ref_grads_flat = torch.cat([g.view(-1) for g in ref_grads if g is not None])
        ref_grads_max = torch.max(torch.abs(ref_grads_flat))
        ref_grads_max_sq = torch.max(torch.abs(ref_grads_flat) ** 2)
        # Update each loss weight
        for idx, loss in enumerate(losses):
            if idx == self.ref_idx:
                self.weights[idx] = torch.tensor(1.0, device=self.weights.device)
            else:
                # Calculate gradients
                grads = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                grads_flat = torch.cat([g.view(-1) for g in grads if g is not None])
                grads_mean = torch.mean(torch.abs(grads_flat))
                grads_mean_sq = torch.mean(torch.abs(grads_flat) ** 2)
                # Calculate moment estimates w/ nugget to avoid instability
                lambda_hat = ref_grads_max / (grads_mean + 1e-6)
                gamma_hat = ref_grads_max_sq / (grads_mean_sq + 1e-6)
                # Calculate moving averages
                lambda_mavg = (1 - self.alpha1) * self.lambdas[
                    self.ref_idx
                ] + self.alpha1 * lambda_hat
                gamma_mavg = (1 - self.alpha2) * self.gammas[
                    self.ref_idx
                ] + self.alpha2 * gamma_hat
                # Bias correction
                m = lambda_mavg / (1 - torch.pow(1 - self.alpha1, step))
                v = gamma_mavg / (1 - torch.pow(1 - self.alpha2, step))
                # Calculate weight and update
                self.weights[idx] = m / (torch.sqrt(v) + self.eps)
                self.lambdas[idx] = lambda_mavg
                self.gammas[idx] = gamma_mavg


@register_lw_alg("Linear_Sum")
class Linear_Sum(Base_LW_alg):
    """
    TODO: UPDATE DOCSTRING AT LATER DATE
    """

    def __init__(self, num_loss_terms, ref_idx=0, alpha1=0.9, alpha2=0.999, eps=1e-8):
        """
        TODO: Add docstring for __init__, remove extraneous inputs
        """
        super(Linear_Sum, self).__init__()
        shape = (num_loss_terms,)
        self.register_buffer("lambdas", torch.zeros(shape))
        self.register_buffer("gammas", torch.zeros(shape))
        self.register_buffer("weights", torch.ones(shape))
        self.register_buffer("alpha1", torch.tensor(alpha1))
        self.register_buffer("alpha2", torch.tensor(alpha2))
        self.register_buffer("eps", torch.tensor(eps))
        self.ref_idx = ref_idx

    def forward(self, losses):
        """
        Applies loss weights to loss terms and sums them.
        """
        return torch.sum(losses * self.weights)

    def update(self, losses, parameters, step):
        """
        No updates are required for multi-task
        """
        pass


@register_lw_alg("No_Weights")
class No_Weights(Base_LW_alg):
    """
    TODO: UPDATE DOCSTRING AT LATER DATE
    This class is used when no loss weighting is desired.
    """
    # TODO: Implement
    pass


# Loss computation registry
LOSS_COMP_REGISTRY = {}

def register_loss_comp(name):
    """
    Decorator to register a loss computation with the given name.
    
    Args:
        name (str): The name of the loss computation type to register.
    
    Returns:
        function: The decorator function that registers the loss computation.
    """
    def decorator(cls):
        LOSS_COMP_REGISTRY[name] = cls
        return cls
    return decorator