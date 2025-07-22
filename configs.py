"""
Contains default configurations for various model components. User-defined 
configurations can also be saved here.
# TODO: Do we need this file? Maybe remove
"""

# Config registry for storing different types of configs
CONFIG_REGISTRY = {}

def register_config(name):
    """
    Decorator to register a config type with the given name.
    
    Args:
        name (str): The name of the config type to register.
    
    Returns:
        function: The decorator function that registers the config.
    """
    def decorator(cls):
        CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

# Default configs
