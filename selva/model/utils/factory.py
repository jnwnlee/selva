import torch

from selva.utils.misc import instantiate_from_config
from selva.model.networks_video_enc import TextSynch as TextSynchVideoEnc
from selva.model.networks_generator import MMAudio


_MODEL_ZOO = (TextSynchVideoEnc, MMAudio)


def create_model_from_factory(factory_path: str, name: str, **kwargs) -> torch.nn.Module:
    """
    Dynamically imports and calls a model factory function.
    """
    params = {'name': name, **kwargs}
    model = instantiate_from_config(factory_path, params)
    assert isinstance(model, _MODEL_ZOO), f"Model {type(model)} is not a valid model type."
    return model