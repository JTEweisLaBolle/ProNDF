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

# Basic loss functions
def NLL_loss(mu, var, targets):
    """
    Computes the negative log likelihood loss.
    Args:
        mu (torch.Tensor): Mean of the predicted distribution.
        var (torch.Tensor): Variance of the predicted distribution.
        targets (torch.Tensor): Target values.
    """
    loss = F.gaussian_nll_loss(mu, targets, var, full=True, eps=1e-6) + 6
    return loss


def IS_loss(mu, sigma, targets, alpha=0.05):
    """
    Computes the interval score loss.
    Args:
        preds_dist (torch.distributions.Distribution): Predicted dist. object.
        y (torch.Tensor): Target values.
        alpha (float): Significance level for the interval score, default is 0.05.
        strength (float): Scaling factor for the loss, default is 1.0.
    Returns:
        torch.Tensor: The computed interval score loss.
    """
    mu_lb = mu - 1.96 * sigma
    mu_ub = mu + 1.96 * sigma
    loss = mu_ub - mu_lb
    loss += (targets > mu_ub).float() * 2 / alpha * (targets - mu_ub)
    loss += (targets < mu_lb).float() * 2 / alpha * (mu_lb - targets)
    loss = torch.mean(loss)
    return loss


def KL_div_var_only_loss(var, targets, prior_var=0.01, eps=1e-8):
    """
    Computes the KL divergence loss focusing on variance.
    Args:
        preds_dist (torch.distributions.Distribution): Predicted dist. object.
        y (torch.Tensor): Target values.
        prior_var (float): Prior variance for KL divergence.
        eps (float): Small value to avoid division by zero.
    Returns:
        torch.Tensor: The computed KL divergence loss.
    """
    prior_vars = prior_var * torch.ones_like(targets)
    KL_divs = torch.log(torch.sqrt(var) / torch.sqrt(prior_vars) + eps) + prior_vars / (2 * var) - 0.5
    return torch.mean(KL_divs)


# Loss registry for storing different types of loss classes
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


# Loss context object to store model parameters/outputs to be accessed by loss classes.
@register_loss("Loss_Context")
class Loss_Context(nn.Module):
    """
    Context object for losses. Stores model parameters and outputs to be accessed by 
    loss classes.
    User can add more objects to the context if future loss classes require them.
    """
    def __init__(self, model, batch, outputs):
        """
        Initializes the Loss_Context with model parameters and outputs.
        Args:
            model (pl.LightningModule): The model from which to extract parameters or 
                other information if needed.
            batch (tuple): The batch of data to be used in loss computation. For typical 
                usage of ProNDF, the batch will have the form 
                (source, cat, num, targets), where source and cat are one-hot encoded.
            outputs (dict[str, str]): The model outputs to be used in loss 
                computation. Should be containing a dictionary for each block with that 
                block's outputs. For example, if B1 and B3 are probabilistic while B2 is 
                deterministic, outputs would take the following form:
                outputs = {
                    "B1": {
                        "out": torch.Tensor, 
                        "out_dist": torch.distributions.Distribution
                        },
                    "B2": {"out": torch.Tensor},
                    "B3": {
                        "out": torch.Tensor, 
                        "out_dist": torch.distributions.Distribution
                        }
                }
        """
        super(Loss_Context, self).__init__()
        self.model = model
        self.batch = batch
        self.outputs = outputs


# loss function classes
@register_loss("Base_Loss")
class Base_Loss(nn.Module):
    """
    Base class for all losses. Should not be instantiated directly.
    """
    def __init__(self):
        """
        Initializes the Base_Loss.
        requires_probabilistic_output (bool): If True, assumes the model outputs a 
            distribution object. If False, assumes the model outputs raw tensors.
        """
        super(Base_Loss, self).__init__()
        self.requires_probabilistic_output = False

    def forward(self, context) -> torch.Tensor:
        """
        Computes the loss. Define in subclasses.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")


@register_loss("Output_MSE_loss")
class Output_MSE_loss(Base_Loss):
    """
    MSE loss on model outputs vs targets. Works with both probabilistic and 
    deterministic outputs.
    """
    def __init__(self):
        """
        Initializes the MSE_loss.
        Args:
            None
        """
        super(Output_MSE_loss, self).__init__()

    def forward(self, context) -> torch.Tensor:
        """
        Computes the mean squared error loss.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        if not self.hasattr("probabilistic_output"):
            self.probabilistic_output = context.model.B3.probabilistic_output
        targets = context.batch[-1]  # Targets should be the last element in the batch
        if self.probabilistic_output:
            # If the model outputs a distribution object, extract the mean
            preds = context.outputs["B3"]["out_dist"]
            preds = preds.mean
        else:
            # If the model outputs raw tensors, use them directly
            preds = context.outputs["B3"]["out"]
        loss = F.mse_loss(preds, targets, reduction="mean")
        return loss


@register_loss("Output_NLL_loss")
class Output_NLL_loss(Base_Loss):
    """
    Negative log likelihood loss on the network output. Network output must be able to 
    output a distribution.
    We add a constant of 6 to the loss to ensure it is positive, as the negative log 
    likelihood can be negative for some distributions. Since a nugget of eps = 1e-6
    is added, this ensures that the loss is always positive and avoids numerical
    instability. See documentation of torch.nn.functional.gaussian_nll_loss:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gaussian_nll_loss.html
    """

    def __init__(self):
        """
        Initializes the NLL_loss.
        Args:
            None
        """
        super(NLL_loss, self).__init__()
        self.requires_probabilistic_output = True

    def forward(self, context) -> torch.Tensor:
        """
        Computes the negative log likelihood loss.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        targets = context.batch[-1]  # Targets should be the last element in the batch
        preds_dist = context.outputs["B3"]["out_dist"]
        mu = preds_dist.mean
        var = preds_dist.variance
        loss = NLL_loss(mu, var, targets)
        return loss


@register_loss("Output_IS_loss")
class Output_IS_loss(Base_Loss):
    """
    Interval score loss. Assumes 95% CI. Requires network output to be dist. object.
    Typically used as a regularizer with a tunable strength parameter.
    For a definition and in-depth discussion of the interval score, see:
    Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of 
    Multi-fidelity Data Sets by Mora and Eweis-LaBolle et. al. (2023).
    https://arxiv.org/abs/2301.13271
    """

    def __init__(self, alpha=0.05, strength=1.0):
        """
        Initializes the IS_loss with a significance level for the interval score.
        Args:
            alpha (float): Significance level for the interval score, default is 0.05.
            strength (float): Scaling factor for the loss, default is 1.0.
        """
        super(Output_IS_loss, self).__init__()
        self.requires_probabilistic_output = True
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("strength", torch.tensor(strength))

    def forward(self, context) -> torch.Tensor:
        """
        Computes the interval score loss.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        targets = context.batch[-1]  # Targets should be the last element in the batch
        preds_dist = context.outputs["B3"]["out_dist"]
        mu = preds_dist.mean
        sigma = preds_dist.stddev
        loss = IS_loss(mu, sigma, targets, alpha = self.alpha.item())
        return loss * self.strength  # Scale by strength factor


@register_loss("Intermediate_KL_Div_Loss")
class Intermediate_KL_Div_Loss(Base_Loss):
    """
    KL divergence loss for variational inference. Requires output of intermediate block 
    to be a distribution object.
    This loss computes the KL divergence between the predicted distribution and a 
    standard normal distribution, focusing only on the variance.
    Should be used to regularize the variance of the outputs of probabillstic 
    intermediate blocks and tuned via strength parameter.
    """

    def __init__(self, block_label = "B1", prior_var = 0.01, eps = 1e-8, strength=1.0):
        """
        Initializes the Intermediate_KL_Div_Loss with a block label, prior variance,
        small epsilon value to avoid division by zero, and a strength factor.
        Args:
            block_label (str): The label of the block whose output to regularize.
            prior_var (float): Prior variance for KL divergence, default is 0.01.
            eps (float): Small value to avoid division by zero, default is 1e-8.
            strength (float): Scaling factor for the loss, default is 1.0.
        """
        super(Intermediate_KL_Div_Loss, self).__init__()
        self.requires_probabilistic_output = True
        self.block_label = block_label
        self.register_buffer("prior_var", torch.tensor(prior_var))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("strength", torch.tensor(strength))

    def forward(self, context) -> torch.Tensor:
        """
        Computes the KL divergence loss focusing on variance.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        targets = context.batch[-1]  # Targets should be the last element in the batch
        preds_dist = context.outputs[self.block_label]["out_dist"]
        var = preds_dist.variance
        loss = KL_div_var_only_loss(var, targets, prior_var=self.prior_var.item(), eps=self.eps.item())
        return loss * self.strength  # Scale by strength factor

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
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")
    

@register_data_split("No_Split")
class No_Split(Base_Data_Split):
    """
    Performs no data splitting. Returns the input data as is.
    """
    def __init__(self):
        super(No_Split, self).__init__()

    def forward(self, context):
        """
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            out: List containing tuple of the input tensors unmodified.
        """
        source, cat, num, y = context.batch
        outputs = context.outputs["B3"]["out"]
        out = [(source, cat, num, y, outputs)]
        return out


@register_data_split("Split_by_Source")
class Split_by_Source(Base_Data_Split):
    """
    Splits data by source. Assumes source is one-hot encoded.
    """
    def __init__(self, config: dict[any, any]):
        """
        Initializes the Split_by_Source with the number of sources.
        Args:
            config (dict): Config object. Should contain 'num_sources' key.
        """
        super(Split_by_Source, self).__init__()
        if "num_sources" not in config:
            raise ValueError(
                "Split_by_Source requires 'num_sources' key in config."
                )
        else:
            self.register_buffer("num_sources", torch.tensor(config["num_sources"]))
            self.register_buffer("num_splits", torch.tensor(config["num_sources"]))
        
    def forward(self, context):
        """
        Splits data by source. Assumes source is one-hot encoded.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            Out: List of loss tensors split by source.
        """
        out = []
        source, cat, num, y = context.batch
        outputs = context.outputs["B3"]["out"]
        for ds in range(self.num_sources.item()):
            # Get indices for the current source
            source_mask = source[:, ds] == 1
            # Split data by source
            source_split = source[source_mask, :]
            cat_split = cat[source_mask, :]
            num_split = num[source_mask, :]
            y_split = y[source_mask, :]
            preds_split = preds[source_mask, :]
            out.append((source_split, cat_split, num_split, y_split, preds_split))
        return out


@register_data_split("Split_by_Output_Dim")
class Split_by_Output_Dim(Base_Data_Split):
    """
    Splits data by output dimension. Used in multi-output regression tasks where each
    output dimension is treated as a separate task
    """
    def __init__(self, config: dict[any, any]):
        """
        Initializes the Split_by_Output_Dim with the number of output dimensions.
        Args:
            config (dict): Config object. Should contain 'num_outputs' key.
        """
        super(Split_by_Output_Dim, self).__init__()
        if "num_outputs" not in config:
            raise ValueError(
                "Split_by_Output_Dim requires 'num_outputs' key in config."
                )
        else:
            self.register_buffer("num_outputs", torch.tensor(config["num_outputs"]))
            self.register_buffer("num_splits", torch.tensor(config["num_outputs"]))
        
    def forward(self, source, cat, num, y, preds):
        """
        Splits data by output dimension.
        Args:
            source (torch.Tensor): Source data tensor.
            cat (torch.Tensor): Categorical data tensor.
            num (torch.Tensor): Numerical data tensor.
            y (torch.Tensor): Target data tensor.
            preds (torch.Tensor): Model predictions tensor.
        Returns:
            out: List of loss tensors split by output dimension.
        """
        out = []
        for ds in range(self.num_outputs.item()):
            # Get indices for the current output dimension
            y_split = y[:, ds].unsqueeze(1)
            preds_split = preds[:, ds].unsqueeze(1)
            out.append((source, cat, num, y_split, preds_split))
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

    def update(self, losses, model, optimizer):
        """
        Optionally update the loss weights.
        Override in subclasses when loss weights need to be dynamically updated.
        Args:
            losses (list[torch.Tensor]): List of loss tensors to be weighted.
            model: Model from which to extract information (e.g., model.parameters).
            optimizer: Model optimizer from which to extract information.
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

    def update(self, losses: list[torch.Tensor], model, optimizer):
        """
        Updates the loss weights.
        Args:
            losses (list[torch.Tensor]): List of loss tensors to be weighted.
            model: Model from which to extract information (e.g., model.parameters).
            optimizer: Model optimizer from which to extract information.
        """
        parameters = list(model.parameters())
        step = model.step if hasattr(model.global_step, 'step') else 0
        # Obtain the reference loss gradients, etc.
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
                lambda_hat = ref_grads_max / (grads_mean + self.eps)
                gamma_hat = ref_grads_max_sq / (grads_mean_sq + self.eps)
                # Calculate moving averages
                lambda_mavg = (1 - self.alpha1) * self.lambdas[
                    self.ref_idx
                ] + self.alpha1 * lambda_hat
                gamma_mavg = (1 - self.alpha2) * self.gammas[
                    self.ref_idx
                ] + self.alpha2 * gamma_hat
                # Bias correction
                m = lambda_mavg / (1 - torch.pow(1 - self.alpha1, step + 1))
                v = gamma_mavg / (1 - torch.pow(1 - self.alpha2, step + 1))
                # Calculate weight and update
                self.weights[idx] = m / (torch.sqrt(v) + self.eps)
                self.lambdas[idx] = lambda_mavg
                self.gammas[idx] = gamma_mavg


# TODO: Add gradnorm as a class here
@register_lw_alg("GradNorm")
class GradNorm(Base_LW_alg):
    """TODO: IMPLEMENT. UPDATE DOCSTRING AT LATER DATE."""
    def __init__(self, num_loss_terms: int, ref_idx=0, alpha1=0.9, alpha2=0.999, eps=1e-8):
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
        raise NotImplementedError(
            "GradNorm is not yet implemented. Please implement the forward and update methods."
        )
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
LOSS_HANDLER_REGISTRY = {}

def register_loss_handler(name):
    """
    Decorator to register a loss computation with the given name.
    
    Args:
        name (str): The name of the loss computation type to register.
    
    Returns:
        function: The decorator function that registers the loss computation.
    """
    def decorator(cls):
        LOSS_HANDLER_REGISTRY[name] = cls
        return cls
    return decorator

@register_loss_handler("Base_Loss_Handler")
class Base_Loss_Handler(nn.Module):
    """
    Base class for loss computation handlers. Should not be instantiated directly.
    Subclasses should implement the `forward()` method to compute the loss.
    """
    def __init__(self):
        super(Base_Loss_Handler, self).__init__()
        if type(self) is Base_Loss_Handler:
            raise NotImplementedError(
                "Base_Loss_Handler should not be instantiated directly."
                )



@register_loss_handler("One_Stage_Loss_Handler")
class One_Stage_Loss_Handler(Base_Loss_Handler):
    """
    Loss handler that computes loss in a single stage, splitting data only once.
    """
    def __init__(
            self,
            config: dict[any, any],
            ):
        """
        Initializes the One_Stage_Loss_Handler with a configuration object.
        Config object should contain the following entries:
            - loss_function_classes (list): List of loss function class names to use.
            - loss_function_configs (list): List of configuration dictionaries for each 
                loss function.
            - data_split_classes (list): Name of the data splitting class to use. For 
                one-stage loss handler, this should be a list of length 1 with only one 
                split.
            - data_split_configs (list): List of configuration dictionaries for each 
                data splitting class.
            - LW_alg_classes (list): List of loss weighting algorithm class names to 
                use. For one-stage loss handler, this should be a list of length 1 with 
                only one algorithm.
            - LW_alg_configs (list): List of configuration dictionaries for each loss 
                weighting algorithm.
            - regularizer_classes (list, optional): List of regularizer class names to 
                use.
            - regularizer_configs (list, optional): List of configuration dictionaries 
                for each regularizer class. 
        """
        super(One_Stage_Loss_Handler, self).__init__()
        # Check that there is only one data split and one loss weighting algorithm
        if len(config["data_split_classes"]) != 1:
            raise ValueError(
                "One_Stage_Loss_Handler requires exactly one data split class."
                )
        if len(config["LW_alg_classes"]) != 1:
            raise ValueError(
                "One_Stage_Loss_Handler requires exactly one loss weighting algorithm class."
                )
        # Instantiate loss functions, data splits, and loss weighting algorithms
        self.loss_functions = nn.ModuleList(
            [LOSS_REGISTRY[loss_fn_class](**config) for loss_fn_class, config in zip(
                config["loss_function_classes"], config["loss_function_configs"]
            )]
        )
        self.data_split = DATA_SPLIT_REGISTRY[config["data_split_classes"][0]](**config["data_split_configs"][0])
        # Extract number of splits from the data split class
        if not hasattr(self.data_split, "num_splits"):
            raise ValueError(
                "Data split class must have 'num_splits' attribute to determine number of splits."
                )
        self.num_splits = self.data_split.num_splits.item()
        # Add number of loss terms to the config for the loss weighting algorithm
        config["LW_alg_configs"][0]["num_loss_terms"] = len(self.loss_functions) * self.num_splits
        # Instantiate the loss weighting algorithm
        self.loss_weighting_algorithm = LW_ALG_REGISTRY[config["LW_alg_classes"][0]](**config["LW_alg_configs"][0])
        # Instantiate regularizers if provided
        if "regularizer_classes" in config and "regularizer_configs" in config:
            self.regularizers = nn.ModuleList(
                [LOSS_REGISTRY[reg_class](**reg_config) for reg_class, reg_config in zip(
                    config["regularizer_classes"], config["regularizer_configs"]
                )]
            )
        
        def compute_loss_terms(self, source, cat, num, y, preds):
            """
            Computes the loss terms for the given data.
            Args:
                source (torch.Tensor): Source data tensor.
                cat (torch.Tensor): Categorical data tensor.
                num (torch.Tensor): Numerical data tensor.
                y (torch.Tensor): Target data tensor.
                preds (torch.Tensor): Model predictions tensor.
            Returns:
                list: List of loss tensors for each loss function and data split.
            """
            # Split data into individual components
            data_splits = self.data_split(source, cat, num, y, preds)
            # Initialize list to store losses
            losses = []
            # Compute loss for each data split and each loss function
            for data_split in data_splits:
                source_split, cat_split, num_split, y_split, preds_split = data_split
                for loss_fn in self.loss_functions:
                    loss = loss_fn(preds_split, y_split)
                    losses.append(loss)
            self.register_buffer("loss_terms", torch.stack(losses))  # Store loss terms
            return losses  # TODO: Decide whether to return losses or not
        
        def update_loss_weights(self, model, optimizer):
            """
            Updates the loss weights using the loss weighting algorithm.
            Args:
                model: Model from which to extract information (e.g., model.parameters).
                optimizer: Model optimizer from which to extract information.
            """
            if not hasattr(self, "loss_terms"):
                raise ValueError(
                    "Loss terms have not been computed. Call compute_loss_terms() first."
                )
            self.loss_weighting_algorithm.update(self.loss_terms, model, optimizer)

        def compute_loss(self, source, cat, num, y, preds):
            """
            Computes the final loss by applying the loss weighting algorithm to the 
            computed loss terms.
            Returns:
                torch.Tensor: The final weighted loss.
            """
            if not hasattr(self, "loss_terms"):
                raise ValueError(
                    "Loss terms have not been computed. Call compute_loss_terms() first."
                )
            weighted_loss = self.loss_weighting_algorithm(self.loss_terms)
            # Apply regularizers if provided
            if hasattr(self, "regularizers"):
                for reg in self.regularizers:
                    weighted_loss += reg(preds, y)
            return weighted_loss


@register_loss_handler("Heirarchical_Loss_Handler")
class Heirarchical_Loss_Handler(Base_Loss_Handler):
    """
    Loss handler that computes loss heirarchically via two data splits, e.g., by both 
    source and output. 
    """
    def __init__(self, config: dict[any, any]):
        """
        Initializes the Heirarchical_Loss_Handler with a configuration object.
        Config object should contain the following entries:
            - loss_function_classes (list): List of loss function class names to use.
            - loss_function_configs (list): List of configuration dictionaries for each 
                loss function.
            - data_split_classes (list): List of data splitting class names to use. 
                Should contain two classes for heirarchical splitting.
            - data_split_configs (list): List of configuration dictionaries for each 
                data splitting class.
            - LW_alg_classes (list): List of loss weighting algorithm class names to 
                use. Should contain two classes for heirarchical splitting, 
                corresponding to the two splits.
            - LW_alg_configs (list): List of configuration dictionaries for each loss 
                weighting algorithm.
            - regularizer_classes (list, optional): List of regularizer class names to 
                use.
            - regularizer_configs (list, optional): List of configuration dictionaries 
                for each regularizer class.
        """
        super(Heirarchical_Loss_Handler, self).__init__()
        # Check that there are two data splits and one loss weighting algorithm
        if len(config["data_split_classes"]) != 2:
            raise ValueError(
                "Heirarchical_Loss_Handler requires exactly two data split classes."
                )
        if len(config["LW_alg_classes"]) != 1:
            raise ValueError(
                "Heirarchical_Loss_Handler requires exactly one loss weighting algorithm class."
                )
        # Instantiate loss functions, data splits, and loss weighting algorithms
        self.loss_functions = nn.ModuleList(
            [LOSS_REGISTRY[loss_fn_class](**config) for loss_fn_class, config in zip(
                config["loss_function_classes"], config["loss_function_configs"]
            )]
        )
        self.data_splits = nn.ModuleList(
            [DATA_SPLIT_REGISTRY[data_split_class](**config) for data_split_class, config in zip(
                config["data_split_classes"], config["data_split_configs"]
            )]
        )
        # Extract number of splits from the first data split class
        if not hasattr(self.data_splits[0], "num_splits"):
            raise ValueError(
                "First data split class must have 'num_splits' attribute to determine number of splits."
            )