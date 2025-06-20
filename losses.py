"""
docstring: fill in later
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

class IS_loss(nn.Module):
    """
    Interval score loss. Assumes 95% CI. Requires network output to be dist. object.
    """

    def __init__(self, alpha=0.05):
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


class NLL_loss(nn.Module):
    """
    Interval score loss. Assumes 95% CI. Requires network output to be dist. object.
    """

    def __init__(self):
        super(NLL_loss, self).__init__()

    def forward(self, mu, y, var, sigma):
        loss = F.gaussian_nll_loss(mu, y, var, full=True, eps=1e-6) + 6
        return loss


class MSE_loss(nn.Module):
    """
    MSE loss. Works with either dist. object or raw tensor output.
    """

    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, mu, y, var, sigma):
        loss = F.mse_loss(mu, y)
        return loss
    
# Loss weighting algoithms

class Dymanic_Weights(nn.Module):
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


class Fixed_Weights(nn.Module):
    """
    TODO: UPDATE DOCSTRING AT LATER DATE
    """

    def __init__(self, num_loss_terms, ref_idx=0, alpha1=0.9, alpha2=0.999, eps=1e-8):
        """
        TODO: Add docstring for __init__, remove extraneous inputs
        """
        super(Fixed_Weights, self).__init__()
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


class No_Weights(nn.Module):
    """
    TODO: UPDATE DOCSTRING AT LATER DATE
    This class is used when no loss weighting is desired.
    """
    # TODO: Implement
    pass

