""" Embedding Mixup
Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py
"""
from typing import Literal, Tuple, Union, List, Optional
from functools import partial
import gc

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2
from einops import rearrange
from omegaconf import DictConfig

from selva.data.vgg_sound import _SYNC_SIZE


class MixupBase:
    """ Base class for mixup on either data or feature domain.
    Applies different params to each element or whole batch.

    Args:
        generator (Optional[torch.Generator]): Random number generator for reproducibility
        modality (Literal['video', 'audio', 'both']): Modality to apply mixup on.
        mixup_lambda (float): Mixup lambda value, mixup is active if in [0., 1.].
        mixup_alpha (float): Mixup alpha value, mixup is active if > 0.
        prob (float): Probability of applying mixup per batch or element
        mode (Literal['elem','pair','batch', 'half']): How to apply mixup params (per 'batch', 'pair' (pair of elements), 'elem' (element), 'half' (half batch))
        eps (float): Small epsilon value to avoid zero lambda
    """
    def __init__(self, generator:torch.Generator, 
                 *, 
                 modality:Literal['video', 'audio', 'both'],
                 mixup_lambda:float=0.5, mixup_alpha:float=1., prob:float=1.0, 
                 mode:Literal['elem','pair','batch', 'half']='batch',
                 eps:float=0.05
                 ):
        self.modality = modality
        self.mixup_lambda:float = mixup_lambda
        self.mixup_alpha:float = mixup_alpha
        self.mix_prob:float = prob
        self.mode:str = mode
        self.eps:float = eps
        self.mixup_enabled:bool = True  # set to false to disable mixing (intended to be set by train loop)
        if generator.device.type == 'cuda':
            self.generator_cuda = generator
            generator_seed = generator.initial_seed()
            self.generator = torch.Generator(device='cpu')
            self.generator.manual_seed(generator_seed)
        else:
            self.generator = generator

        if not (self.mixup_lambda >= 0. and self.mixup_lambda <= 1.):
            raise ValueError(f"mixup_lambda {self.mixup_lambda} should be in [0., 1.].")
        if not self.mixup_alpha >= 0.:
            raise ValueError(f"mixup_alpha {self.mixup_alpha} >= 0. should be true.")
        if (self.mixup_alpha > 0. and self.mixup_lambda < 1.) or (self.mixup_alpha == 0. and self.mixup_lambda == 1.):
            raise ValueError(f"One of mixup_alpha {self.mixup_alpha} > 0., mixup_lambda {self.mixup_lambda} < 1. should be true.")
    
    def _params_per_elem(self, batch_size:int) -> np.ndarray:
        lam:np.ndarray = np.ones(batch_size, dtype=np.float32)
        if self.mixup_enabled:
            if self.mixup_lambda < 1.: # constant lambda
                lam_mix = np.full(batch_size, self.mixup_lambda, dtype=np.float32)
            elif self.mixup_alpha > 0.: # sampled lambda
                # Use torch's beta distribution with generator
                lam_mix = torch.distributions.Beta(
                    torch.tensor([self.mixup_alpha]), 
                    torch.tensor([self.mixup_alpha]),
                ).sample([batch_size]).numpy().astype(np.float32).reshape(-1)
            else:
                assert False, f"One of mixup_alpha {self.mixup_alpha} > 0., mixup_lambda {self.mixup_lambda} < 1. should be true."
            lam_mix[lam_mix < self.eps] = self.eps
            
            # Use torch's random with generator for the random comparison
            rand_vals = torch.rand(batch_size, generator=self.generator).numpy()
            lam = np.where(rand_vals < self.mix_prob, lam_mix, lam)
        return lam

    def _params_per_batch(self) -> float:
        lam:float = 1.
        if self.mixup_enabled:
            if self.mixup_lambda < 1.: # constant lambda
                lam = self.mixup_lambda
            elif self.mixup_alpha > 0.: # sampled lambda
                lam = torch.distributions.Beta(
                    torch.tensor([self.mixup_alpha]), 
                    torch.tensor([self.mixup_alpha]),
                ).sample().item()
            else:
                assert False, f"mixup_alpha {self.mixup_alpha} > 0., mixup_lambda {self.mixup_lambda} < 1. should be true."
            if lam < self.eps: lam = self.eps
            lam = float(lam)
        return lam


class DataMixupCollate(MixupBase):
    """ Mixup video in data domain.
    Applies different params to each element or whole batch.

    Args:
        generator (Optional[torch.Generator]): Random number generator for reproducibility
        modality (Literal['video', 'audio', 'both']): Modality to apply mixup on.
        mixup_lambda (float): Mixup lambda value, mixup is active if in [0., 1.].
        mixup_alpha (float): Mixup alpha value, mixup is active if > 0.
        prob (float): Probability of applying mixup per batch or element
        mode (Literal['elem','pair','batch', 'half']): How to apply mixup params (per 'batch', 'pair' (pair of elements), 'elem' (element), 'half' (half batch))
        eps (float): Small epsilon value to avoid zero lambda
    """
    def __init__(self, generator:torch.Generator, 
                 *, 
                 modality:Literal['video', 'audio', 'both']='video',
                 mixup_lambda:float=0.5, mixup_alpha:float=1., prob:float=1.0, 
                 mode:Literal['elem','pair','batch', 'half']='batch',
                 eps:float=0.05
                 ):
        super().__init__(generator, modality=modality,
                         mixup_lambda=mixup_lambda, mixup_alpha=mixup_alpha, prob=prob, 
                         mode=mode, eps=eps)
        
        self.source_video_key= 'sync_video'
        self.source_audio_key = 'audio'
        self.target_video_key = 'sync_video_mixed'
        self.target_audio_key = 'audio_mixed'
        
        if not mode == 'batch':
            raise ValueError(f"Mode {mode} is not supported for data domain.")
        self.sync_transform = v2.Compose([
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _concat_video_frames(self, batch:list, target_key:str='sync_video_mixed', source_key:str='sync_video') -> float:
        # only batch mode supported
        batch_size:int = len(batch)
        lam:float = self._params_per_batch()
        
        if lam == 1.:
            # no mixup, just return
            for i in range(batch_size):
                batch[i][target_key] = batch[i][source_key]
            return lam
        
        # Randomly choose between horizontal and vertical resizing using
        orig_size = int(lam * _SYNC_SIZE)
        is_horizontal = True # torch.rand(1, generator=self.generator).item() < 0.5
        if is_horizontal:
            # Horizontal resize
            resize_shape_orig = (_SYNC_SIZE, orig_size)
            resize_shape_pair = (_SYNC_SIZE, _SYNC_SIZE-orig_size)    
        else:
            # Vertical resize
            resize_shape_orig = (orig_size, _SYNC_SIZE)
            resize_shape_pair = (_SYNC_SIZE-orig_size, _SYNC_SIZE)
        sync_resize_orig = v2.Compose([
            v2.Resize(resize_shape_orig, interpolation=v2.InterpolationMode.BICUBIC),
        ])
        sync_resize_pair = v2.Compose([
            v2.Resize(resize_shape_pair, interpolation=v2.InterpolationMode.BICUBIC),
        ])
        
        batch_videos_orig = torch.stack([batch[i][source_key] for i in range(batch_size)], dim=0)
        batch_videos_pair = torch.stack([batch[batch_size - i - 1][source_key] for i in range(batch_size)], dim=0)
        # (B, T, C, H, W)
        # pass through resize, transform and concat
        batch_videos_orig = sync_resize_orig(batch_videos_orig)
        batch_videos_pair = sync_resize_pair(batch_videos_pair)
        batch_videos_concat = torch.cat((batch_videos_orig, batch_videos_pair), dim=-1 if is_horizontal else -2)
        batch_videos_concat = self.sync_transform(batch_videos_concat)

        num_mixup = int(self.mix_prob * batch_size)
        for i in range(num_mixup):
            batch[i][target_key] = batch_videos_concat[i]
        for i in range(num_mixup, batch_size):
            batch[i][target_key] = batch[i][source_key] # no mixup
            
        del batch_videos_orig, batch_videos_pair, sync_resize_orig, sync_resize_pair
        gc.collect()
            
        return lam 
    
    def _mix_audio_samples(self, batch:list, target_key:str='audio_mixed', source_key:str='audio',
                           normalize:bool = True) -> float:
        # assume source_key audios are normalized
        batch_size:int = len(batch)
        lam:float = self._params_per_batch()
        
        if lam == 1.:
            # no mixup, just return
            for i in range(batch_size):
                batch[i][target_key] = batch[i][source_key]
            return lam
        
        num_mixup = int(self.mix_prob * batch_size)
        for i in range(num_mixup):
            batch[i][target_key] = batch[i][source_key] * lam + batch[batch_size - i - 1][source_key] * (1 - lam)
            if normalize:
                source_abs_max = batch[i][source_key].abs().max()
                target_abs_max = batch[i][target_key].abs().max()
                batch[i][target_key] = batch[i][target_key] * (source_abs_max / target_abs_max)
        for i in range(num_mixup, batch_size):
            batch[i][target_key] = batch[i][source_key] # no mixup

        return lam
    
    def __call__(self, batch:list, _=None) -> torch.tensor:
        batch_size:int = len(batch)
        assert batch_size % 2 == 0, f'Batch size {batch_size} should be even when using mixup'
        half = 'half' in self.mode
        if half:
            batch_size //= 2
        
        if self.modality == 'video' or self.modality == 'both':
            lam = self._concat_video_frames(batch, target_key=self.target_video_key, source_key=self.source_video_key)
        if self.modality == 'audio' or self.modality == 'both':
            # raise NotImplementedError('Audio mixup is not implemented yet.')
            lam = self._mix_audio_samples(batch, target_key=self.target_audio_key, source_key=self.source_audio_key)
        
        return default_collate(batch)


class FeatureMixup(MixupBase):
    """ Mixup video in feature domain.
    Applies different params to each element or whole batch.

    Args:
        generator (Optional[torch.Generator]): Random number generator for reproducibility
        modality (Literal['video', 'audio', 'both']): Modality to apply mixup on.
        mixup_lambda (float): Mixup lambda value, mixup is active if in [0., 1.].
        mixup_alpha (float): Mixup alpha value, mixup is active if > 0.
        prob (float): Probability of applying mixup per batch or element
        mode (Literal['elem','pair','batch', 'half']): How to apply mixup params (per 'batch', 'pair' (pair of elements), 'elem' (element), 'half' (half batch))
        eps (float): Small epsilon value to avoid zero lambda
    """
    def __init__(self, generator:torch.Generator, 
                 *, 
                 modality:Literal['video', 'audio', 'both']='video',
                 mixup_lambda:float=0.5, mixup_alpha:float=1., prob:float=1.0, 
                 mode:Literal['elem','pair','batch', 'half']='batch',
                 eps:float=0.05
                 ):
        super().__init__(generator, modality=modality,
                         mixup_lambda=mixup_lambda, mixup_alpha=mixup_alpha, prob=prob, 
                         mode=mode, eps=eps)
        self.source_video_key= 'sync_f_vid_orig'
        self.source_audio_key = 'sync_f_aud_orig'
        self.target_video_key = 'sync_f_vid_mixed'
        self.target_audio_key = 'sync_f_aud_mixed'

    def _mix_elem_collate(self, batch:dict, 
                          target_keys:List[str]=['sync_features_mixed'], source_keys:List[str]=['sync_features_orig'],
                          half:bool=False) -> torch.tensor:
        assert len(target_keys) == len(source_keys), f"Length of target_keys {len(target_keys)} and source_keys {len(source_keys)} should be equal."
        batch_size:int = len(batch['id'])
        num_elem:int = batch_size // 2 if half else batch_size
        lam_batch:torch.tensor = torch.from_numpy(self._params_per_elem(num_elem))
        
        indices = torch.arange(num_elem)
        mix_indices = batch_size - indices - 1
        mix_mask = lam_batch < 1
        active_indices = indices[mix_mask]
        active_mix_indices = mix_indices[mix_mask]
        active_lambdas = lam_batch[mix_mask].unsqueeze(1)
        for target_key, source_key in zip(target_keys, source_keys):
            batch[target_key][active_indices] = (
                batch[source_key][active_indices] * active_lambdas + 
                batch[source_key][active_mix_indices] * (1 - active_lambdas)
            )
            batch[target_key][~indices[mix_mask]] = batch[source_key][~indices[mix_mask]]
        if half:
            lam_batch = torch.cat((lam_batch, torch.ones(num_elem, dtype=lam_batch.dtype)))
        return lam_batch.unsqueeze(1)

    def _mix_pair_collate(self, batch:dict,
                          target_keys:List[str]=['sync_features_mixed'], source_keys:List[str]=['sync_features_orig']) -> torch.tensor:
        assert len(target_keys) == len(source_keys), f"Length of target_keys {len(target_keys)} and source_keys {len(source_keys)} should be equal."
        batch_size:int = len(batch['id'])
        lam_batch:torch.tensor = torch.from_numpy(self._params_per_elem(batch_size // 2))
        
        indices = torch.arange(batch_size // 2)
        mix_indices = batch_size - indices - 1
        mix_mask = lam_batch < 1
        active_indices = indices[mix_mask]
        active_mix_indices = mix_indices[mix_mask]
        active_lambdas = lam_batch[mix_mask].unsqueeze(1)
        for target_key, source_key in zip(target_keys, source_keys):
            batch[target_key][active_indices] = (
                batch[source_key][active_indices] * active_lambdas + 
                batch[source_key][active_mix_indices] * (1 - active_lambdas)
            )
            batch[target_key][active_mix_indices] = (
                batch[source_key][active_mix_indices] * active_lambdas + 
                batch[source_key][active_indices] * (1 - active_lambdas)
            )
            batch[target_key][~indices[mix_mask]] = batch[source_key][~indices[mix_mask]]
            batch[target_key][~mix_indices[mix_mask]] = batch[source_key][~mix_indices[mix_mask]]
        lam_batch = torch.cat((lam_batch, lam_batch.flip(0)))
        return lam_batch.unsqueeze(1)

    def _mix_batch_collate(self, batch:dict,
                           target_keys:List[str]=['sync_features_mixed'], source_keys:List[str]=['sync_features_orig']) -> float:
        assert len(target_keys) == len(source_keys), f"Length of target_keys {len(target_keys)} and source_keys {len(source_keys)} should be equal."
        lam:float = self._params_per_batch()
        
        for target_key, source_key in zip(target_keys, source_keys):
            num_mixup = int(self.mix_prob * batch[source_key].shape[0])
            flipped_source = torch.flip(batch[source_key], dims=[0])
            batch[target_key] = batch[source_key] * lam + flipped_source * (1 - lam)
            batch[target_key][num_mixup:] = batch[source_key][num_mixup:] # no mixup
        return lam
    
    def __call__(self, batch:dict, _=None) -> None:
        batch_size:int = len(batch['id'])
        assert batch_size % 2 == 0, f'Batch size(={batch_size}) should be even when using this'
        half = 'half' in self.mode
        if half:
            batch_size //= 2
        
        # Mixup
        if self.mode == 'elem' or self.mode == 'half':
            collate_fn = partial(self._mix_elem_collate, half=half)
        elif self.mode == 'pair':
            collate_fn = self._mix_pair_collate
        else:
            collate_fn = self._mix_batch_collate
        
        if self.modality == 'both':
            target_keys, source_keys = [self.target_video_key, self.target_audio_key], [self.source_video_key, self.source_audio_key]
        elif self.modality == 'video':
            target_keys, source_keys = [self.target_video_key], [self.source_video_key]
        elif self.modality == 'audio':
            target_keys, source_keys = [self.target_audio_key], [self.source_audio_key]
        lam = collate_fn(batch, target_keys=target_keys, source_keys=source_keys)
        
        # return batch

