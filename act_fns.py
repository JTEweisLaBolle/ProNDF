"""
Docstring to be added later.
This module constains act fn classes
"""

# Activation function registry for storing different types of activation functions

ACT_FN_REGISTRY = {}

def register_act_fn(name):
    """
    Decorator to register a act_fn type with the given name.
    
    Args:
        name (str): The name of the act_fn type to register.
    
    Returns:
        function: The decorator function that registers the act_fn.
    """
    def decorator(cls):
        ACT_FN_REGISTRY[name] = cls
        return cls
    return decorator