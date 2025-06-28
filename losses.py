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
    This is a weighted sum of NLL_loss and IS_loss.
    Requires network output to be dist. object.
    Typicaly not used - instead, IS is used as a regularizer with its own weight.
    
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

# Data splitting registry
DATA_SPLIT_REGISTRY = {}
def register_data_split(name):
    """
    Decorator to register a data splitting function with the given name.
    
    Args:
        name (str): The name of the data splitting function to register.
    
    Returns:
        function: The decorator function that registers the data splitting function.
    """
    def decorator(cls):
        DATA_SPLIT_REGISTRY[name] = cls
        return cls
    return decorator

# data splitting classes
class Base_Data_Split(nn.Module):
    """
    Base class for data splitting. Should not be instantiated directly.
    """
    def __init__(self):
        super(Base_Data_Split, self).__init__()
        if type(self) is Base_Data_Split:
            raise NotImplementedError(
                "Base_Data_Split should not be instantiated directly."
                )

    def forward(self, source, cat, num, y):
        """
        Splits data into individual components. Define in subclasses.
        Args:
            source (torch.Tensor): Source data tensor.
            cat (torch.Tensor): Categorical data tensor.
            num (torch.Tensor): Numerical data tensor.
            y (torch.Tensor): Target data tensor.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")
    

@register_data_split("No_Split")
class No_Split(Base_Data_Split):
    """
    Performs no data splitting. Returns the input data as is.
    """
    def __init__(self):
        super(No_Split, self).__init__()

    def forward(self, source, cat, num, y):
        """
        Returns the input data as is without any splitting.
        Args:
            source (torch.Tensor): Source data tensor.
            cat (torch.Tensor): Categorical data tensor.
            num (torch.Tensor): Numerical data tensor.
            y (torch.Tensor): Target data tensor.
        Returns:
            out: List containing tuple of the input tensors unmodified.
        """
        out = [(source, cat, num, y)]
        return out


@register_data_split("Split_by_Source")
class Split_by_Source(Base_Data_Split):
    """
    Splits data by source. Assumes source is one-hot encoded.
    """
    def __init__(self, config: dict[any, any] = None):
        """
        Initializes the Split_by_Source with the number of sources.
        Args:
            config (dict): Config object. Should contain 'num_sources' key.
        """
        super(Split_by_Source, self).__init__()
        if config is not None and "num_sources" in config:
            self.register_buffer("num_sources", torch.tensor(config["num_sources"]))
        else:
            pass  # num_sources will be set in forward method
        
    def forward(self, source, cat, num, y):
        """
        Splits data by source. Assumes source is one-hot encoded.
        Args:
            source (torch.Tensor): Source data tensor, one-hot encoded.
            cat (torch.Tensor): Categorical data tensor.
            num (torch.Tensor): Numerical data tensor.
            y (torch.Tensor): Target data tensor.
        Returns:
            Out: List of loss tensors split by source.
        """
        if not hasattr(self, "num_sources"):
            # If num_sources is not set, determine it from the source tensor
            self.register_buffer(
                "num_sources", torch.tensor(source.shape[-1]), device=source.device
                )
        out = []
        for ds in range(self.num_sources.item()):
            # Get indices for the current source
            source_mask = source[:, ds] == 1
            # Split data by source
            source_split = source[source_mask, :]
            cat_split = cat[source_mask, :]
            num_split = num[source_mask, :]
            y_split = y[source_mask, :]
            out.append((source_split, cat_split, num_split, y_split))
        return out


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
    Base class for loss weighting algorithms.
    Do not instantiate directly. Subclasses should implement the `forward()` method 
    to compute a weighted sum of loss terms, and optionally `update()` for dynamic 
    schemes.
    """
    def __init__(self):
        super(Base_LW_alg, self).__init__()
        if type(self) is Base_LW_alg:
            raise NotImplementedError(
                "Base_LW_alg should not be instantiated directly."
                )
    
    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Linear combination of loss terms. Define in subclasses.
        Args:
            losses (list[torch.Tensor]): List of loss tensors to be weighted.
        Returns:
            torch.Tensor: Scalar weighted sum of losses.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")
    
    class Context:
        """
        Context object for loss weighting algorithms. Extracts necessary information 
        for updating weights from the losses, model, and optimizer.
        Define methods as necessary.

        """
        def __init__(self, losses, model, optimizer):
            """
            Initializes the context with losses, parameters, and step.
            Args:
                losses (list[torch.Tensor]): List of loss tensors to be weighted.
                model: Model from which to extract information (e.g., model.parameters).
                optimizer: Model optimizer from which to extract information.
            """
            raise NotImplementedError(
                "Context should be defined in a subclass of Base_LW_alg"
            )
    
    def build_context(self, losses, model, optimizer):
        """
        Builds a context object for the loss weighting algorithm.
        Args:
            losses (list[torch.Tensor]): List of loss tensors to be weighted.
            model: Model from which to extract information (e.g., model.parameters).
            optimizer: Model optimizer from which to extract information.
        Returns:
            Context: Context object containing necessary information for updating weights.
        """
        return self.Context(losses, model, optimizer)

    def update(self, context):
        """
        Optionally update the loss weights.
        Override in subclasses when loss weights need to be dynamically updated.
        Args:
            context: Context object containing necessary information for updating 
            weights.
        """
        pass


@register_lw_alg("No_Weighting")
class No_Weighting(Base_LW_alg):
    """
    This class is used when no loss weighting is desired.
    """
    def __init__(self):
        """
        Initializes the No_Weighting loss weighting algorithm which does not apply any 
        weighting to the loss and simply returns it as is.
        """
        super(No_Weighting, self).__init__()
        pass  # No parameters to register

    def forward(self, losses):
        """
        Applies loss weights to loss terms and sums them.
        Args:
            losses (torch.Tensor): Flattened tensor of losses. Should be scalar or 1D 
            if using no weighting.
        Returns:
            torch.Tensor: Sum of losses.
        """
        return torch.sum(losses)


@register_lw_alg("Two_Moment_Weighting")
class Two_Moment_Weighting(Base_LW_alg):
    """
    Loss weighting algorithm that uses two moment estimates to weight losses.
    This algorithm computes the first and second moments of the gradients of each loss
    term with respect to the model parameters, and uses these moments to compute
    adaptive weights for each loss term. The reference loss term is used to normalize
    the weights of the other loss terms.
    The algorithm is inspired by the papers "Multi-Objective Loss Balancing for 
    Physics-Informed Deep Learning" by Rafael Bischof and Michael Kraus (2021) and 
    "Adam: A method for stochastic optimization" by Diederik P Kingm and Jimmy Ba 
    (2014).
    """

    def __init__(self, num_loss_terms, ref_idx=0, alpha1=0.9, alpha2=0.999, eps=1e-8):
        """
        Initializes the Two_Moment_Weighting loss weighting algorithm.
        Args:
            num_loss_terms (int): Number of loss terms to weight.
            ref_idx (int): Index of the reference loss term.
            alpha1 (float): Exponential decay rate for first moment.
            alpha2 (float): Exponential decay rate for second moment.
            eps (float): Small value to avoid division by zero.
        """
        super(Two_Moment_Weighting, self).__init__()
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
        Args:
            losses (torch.Tensor): Flattened tensor of losses. Should be scalar or 1D
        """
        return torch.sum(losses * self.weights)

    def update(self, losses, parameters, step):
        """
        Updates the loss weights.
        Args:
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


# TODO: Add gradnorm as a class here
@register_lw_alg("GradNorm")
class GradNorm(Base_LW_alg):
    """TODO: IMPLEMENT. UPDATE DOCSTRING AT LATER DATE."""
    def __init__(self, num_loss_terms, ref_idx=0, alpha1=0.9, alpha2=0.999, eps=1e-8):
        """
        Initializes the GradNorm loss weighting algorithm.
        Args:
            num_loss_terms (int): Number of loss terms to weight.
            ref_idx (int): Index of the reference loss term.
            alpha1 (float): Exponential decay rate for first moment.
            alpha2 (float): Exponential decay rate for second moment.
            eps (float): Small value to avoid division by zero.
        """
        super(GradNorm, self).__init__()
        shape = (num_loss_terms,)
        self.register_buffer("lambdas", torch.zeros(shape))
        self.register_buffer("gammas", torch.zeros(shape))
        self.register_buffer("weights", torch.zeros(shape))
        self.register_buffer("alpha1", torch.tensor(alpha1))
        self.register_buffer("alpha2", torch.tensor(alpha2))
        self.register_buffer("eps", torch.tensor(eps))
        self.ref_idx = ref_idx


@register_lw_alg("Fixed_Weights")
class Fixed_Weights(Base_LW_alg):
    """
    Fixed weights loss weighting algorithm. Applies fixed weights to each loss term.
    """
    def __init__(self, num_loss_terms: int, weights: list = None):
        """
        Initializes the Fixed_Weights loss weighting algorithm.
        Args:
            num_loss_terms (int): Number of loss terms to weight.
            weights (list, optional): List of fixed weights for each loss term. If None,
                defaults to equal weights for all terms.
        """
        super(Fixed_Weights, self).__init__()
        shape = (num_loss_terms,)
        if weights is not None:
            if len(weights) != num_loss_terms:
                raise ValueError(
                    f"Length of weights ({len(weights)}) does not match number of loss terms ({num_loss_terms})."
                )
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("weights", torch.ones(shape))

    def forward(self, losses):
        """
        Applies loss weights to loss terms and sums them.
        """
        return torch.sum(losses * self.weights)


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