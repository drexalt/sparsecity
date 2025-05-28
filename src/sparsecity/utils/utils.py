from omegaconf import OmegaConf, DictConfig


def flatten_dict(d, parent_key="", sep="/"):
    """
    Flatten a nested dictionary, using separator to join keys.

    Args:
        d: Dictionary to flatten
        parent_key: Key of parent (for recursion)
        sep: Separator to use between nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            v = OmegaConf.to_container(v, resolve=True)
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
