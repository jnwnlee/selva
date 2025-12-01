import importlib
from typing import Optional

from omegaconf import DictConfig, OmegaConf



def instantiate_from_config(target: str, params: Optional[dict] = None):
    """
    Instantiate an object from a dotted path `target` and keyword `params`.
    Common name: instantiate_from_config
    """
    if not target or not isinstance(target, str):
        raise ValueError(f"Invalid target: {target!r}")
    params = {} if params is None else params

    # Convert OmegaConf DictConfig to plain dict if needed
    try:
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
    except Exception:
        pass

    try:
        module_path, attr_name = target.rsplit('.', 1)
    except ValueError as e:
        raise ValueError(f"Target must be like 'pkg.mod.Class', got {target!r}") from e

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Could not import module '{module_path}' for target '{target}'.") from e

    try:
        obj = getattr(module, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}' (from '{target}').") from e

    if not callable(obj):
        raise TypeError(f"Resolved target '{target}' is not callable.")
    return obj(**dict(params))