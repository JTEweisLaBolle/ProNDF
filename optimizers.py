"""
optimizers.py
This module contains optimizer classes and a registry to store them.
Additional or custom optimizers can be registered by the user if desired, 
either in this file or inline.
Registries are used to enable serialization by PyTorch Lightning's checkpointing 
system and automatic hyperparameter saving, as class objects can not be serialized.
Example usage:
    # Importing and using the registry and adding a custom optimizer
    from optimizers import OPTIMIZER_REGISTRY, register_optimizer

    # TODO: Finish docstring
"""

import torch


# Optimizer registry for storing different types of optimizers
OPTIMIZER_REGISTRY = {}

def register_optimizer(name):
    """
    Decorator to register a optimizer type with the given name.
    
    Args:
        name (str): The name of the optimizer type to register.
    
    Returns:
        function: The decorator function that registers the optimizer.
    """
    def decorator(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator


@register_optimizer("Adam")
class Adam(torch.optim.Adam):
    pass