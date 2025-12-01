import gc
from typing import Optional, Union

import torch
from omegaconf import DictConfig

from selva.model.utils.features_utils import FeatureUtils
from selva.model.networks_video_enc import TextSynch
from selva.data.mixup import FeatureMixup


@torch.no_grad()
def preprocess_batch_with_tsynch(
    batch: dict,
    mixup_config: DictConfig,
    feature_extractor: FeatureUtils,
    net_video_enc: TextSynch,
    feature_mixup: Optional[FeatureMixup] = None,
    training: bool = False,
) -> None:
    if mixup_config.domain == 'embedding' and feature_mixup is None:
        raise ValueError('Mixup function is required for embedding domain.')

    bs: int = len(batch['id'])
    device = feature_extractor.device
    dtype = feature_extractor.dtype
    
    video_exist = batch.get('sync_video', None) is not None
    text_exist = batch.get('caption', None) is not None
    batch['video_exist'] = torch.tensor(video_exist, device=device, dtype=torch.bool, requires_grad=False)
    batch['text_exist'] = torch.tensor(text_exist, device=device, dtype=torch.bool, requires_grad=False)
    
    batch['a_mean'] = batch.get('a_mean').to(device, dtype)
    batch['a_std'] = batch['a_std'].to(device, dtype)
    
    batch['clip_features'] = None
    batch['text_clip_features'] = batch['text_clip_features'].to(device, dtype)

    if video_exist:
        tsynch_text_features = batch['text_features'].to(device, dtype)
        tsynch_text_mask = batch['text_masks'].to(device, dtype)
        
        if mixup_config.enabled and mixup_config.domain == 'data':
            tsynch_text_features, tsynch_text_mask = net_video_enc.prepend_silence_text_tokens(tsynch_text_features, tsynch_text_mask)
            batch['sync_video_mixed'] = batch['sync_video_mixed'].to(device, dtype, non_blocking=True)
            batch['sync_features'] = net_video_enc.encode_video_with_sync(
                batch['sync_video_mixed'], text_f=tsynch_text_features, text_mask=tsynch_text_mask
            )
        elif mixup_config.enabled and mixup_config.domain == 'embedding':
            assert feature_mixup.modality == mixup_config.params.modality, \
                f"Mixup class modality {feature_mixup.modality} should be same as config {mixup_config.params.modality}."
            feature_mixup.target_video_key = 'sync_features'
            feature_mixup(batch)
        else:
            batch['sync_video'] = batch['sync_video'].to(device, dtype)
            tsynch_text_features, tsynch_text_mask = net_video_enc.prepend_silence_text_tokens(tsynch_text_features, tsynch_text_mask)
            batch['sync_features'] = net_video_enc.encode_video_with_sync(
                batch['sync_video'], text_f=tsynch_text_features, text_mask=tsynch_text_mask
            )
        
    if training:
        for k, v in batch.items():
            if k in ['video_exist', 'text_exist', 'sync_features']:
                batch[k] = v.clone()

    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def preprocess_batch_with_mixup(
    batch: dict,
    mixup_config: DictConfig,
    feature_extractor: FeatureUtils = None,
    sync_batch_size_multiplier: int = 40,
    feature_mixup: Optional[FeatureMixup] = None,
    training: bool = False,
) -> None:
    if mixup_config.domain == 'embedding' and feature_mixup is None:
        raise ValueError('Mixup function is required for embedding domain.')
    
    bs: int = len(batch['id'])
    batch['text_features'], batch['text_mask'] = feature_extractor.encode_text(batch['caption'])
    if mixup_config.params.modality in ['video', 'both']:
        batch['sync_f_vid_orig'] = feature_extractor.encode_video_with_sync(batch['sync_video'],
                                                                            batch_size=bs *
                                                                            sync_batch_size_multiplier)
    if mixup_config.params.modality in ['audio', 'both']:
        batch['sync_f_aud_orig'] = feature_extractor.encode_audio_with_sync(batch['audio'],
                                                                            batch_size=bs *
                                                                            sync_batch_size_multiplier)
    if mixup_config.domain == 'data':
        if mixup_config.params.modality in ['video', 'both']:
            batch['sync_f_vid_mixed'] = feature_extractor.encode_video_with_sync(batch['sync_video_mixed'],
                                                                            batch_size=bs *
                                                                            sync_batch_size_multiplier)
        if mixup_config.params.modality in ['audio', 'both']:
            batch['sync_f_aud_mixed'] = feature_extractor.encode_audio_with_sync(batch['audio_mixed'],
                                                                            batch_size=bs *
                                                                            sync_batch_size_multiplier)
    if mixup_config.domain == 'embedding':
        assert feature_mixup.modality == mixup_config.params.modality, \
            f"Mixup class modality {feature_mixup.modality} should be same as config {mixup_config.params.modality}."
        feature_mixup(batch)
    if training:
        for k, v in batch.items():
            if k in ['text_features', 'text_mask',
                'sync_f_vid_orig', 'sync_f_aud_orig', 'sync_f_vid_mixed', 'sync_f_aud_mixed']:
                batch[k] = v.clone()

    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def preprocess_batch(
    batch: dict,
    mixup_config: DictConfig,
    feature_extractor: FeatureUtils = None,
    sync_batch_size_multiplier: Union[int, float] = 40,
    training: bool = False,
) -> None:
    bs: int = len(batch['id'])
    batch['text_features'], batch['text_mask'] = feature_extractor.encode_text(batch['caption'])
    sync_batch_size = int(bs * sync_batch_size_multiplier) if sync_batch_size_multiplier > 0 else bs
    if mixup_config.params.modality in ['video', 'both']:
        batch['sync_f_vid_orig'] = feature_extractor.encode_video_with_sync(batch['sync_video'],
                                                                            batch_size=sync_batch_size)
    if mixup_config.params.modality in ['audio', 'both']:
        batch['sync_f_aud_orig'] = feature_extractor.encode_audio_with_sync(batch['audio'],
                                                                            batch_size=sync_batch_size)

    if training:
        for k, v in batch.items():
            if k in ['text_features', 'text_mask',
                'sync_f_vid_orig', 'sync_f_aud_orig']:
                batch[k] = v.clone()

    gc.collect()
    torch.cuda.empty_cache()