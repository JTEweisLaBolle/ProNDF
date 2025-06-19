"""
docstring to be filled in later
This module provides a registry for block types and loss computation classes
"""

BLOCK_REGISTRY = {}

def register_block(name):
    """
    Decorator to register a block type with the given name.
    
    Args:
        name (str): The name of the block type to register.
    
    Returns:
        function: The decorator function that registers the block.
    """
    def decorator(cls):
        BLOCK_REGISTRY[name] = cls
        return cls
    return decorator


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