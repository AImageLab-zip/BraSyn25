def adapt_state_dict_proj_to_conv(state_dict):
    """
    Adapter function to convert state_dict from proj layers to conv layers.
    
    Args:
        state_dict (dict): The original state dictionary with proj layers
    
    Returns:
        dict: Adapted state dictionary with conv layers
    """
    adapted_dict = {}
    
    for key, value in state_dict.items():
        if '.proj.' in key:
            # Replace proj with conv
            new_key = key.replace('.proj.', '.conv.')
            adapted_dict[new_key] = value
        else:
            # Keep the key as-is
            adapted_dict[key] = value
    
    return adapted_dict