import logging
import random
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from selva.data.vgg_sound import VGGSound
from selva.data.eval.eval_video_dataset import VGGSound as VGGSoundEval
from selva.data.eval.eval_video_dataset import InferenceVideoData, VGGMonoAudioBench
from selva.data.eval.audiocaps import AudioCapsData
from selva.data.mm_dataset import MultiModalDataset
from selva.data.mixup import DataMixupCollate
from selva.utils.dist_utils import local_rank

log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(f'Worker {worker_id} re-seeded with seed {worker_seed} in rank {local_rank}')


def load_video_data(cfg: DictConfig, data_cfg: DictConfig, normalize_audio: bool = False,
                    ) -> Dataset:
    dataset = VGGSound(root=data_cfg.root,
                    tsv_path=data_cfg.subset_name,
                    sample_rate=16_000,
                    duration_sec=8.0,
                    normalize_audio=normalize_audio,
                    mmap_dir=data_cfg.memmap_dir,
                    tsv_tsynch_path=data_cfg.tsv_tsynch,
                    mmap_tsync_dir=data_cfg.memmap_dir_tsynch,
                    data_dim=cfg.data_dim
                    )

    return dataset


def load_audio_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset:
    raise NotImplementedError('Audio data loading is not implemented yet')


def setup_training_datasets(cfg: DictConfig, 
                            generator: torch.Generator,
                            ) -> tuple[Dataset, DistributedSampler, DataLoader]:
    if cfg.mini_train:
        vgg = load_video_data(cfg, cfg.data.VGGSound_val, normalize_audio=True)
        dataset = MultiModalDataset([vgg], [])
    if cfg.example_train:
        video = load_video_data(cfg, cfg.data.Example_video, normalize_audio=True)
        dataset = MultiModalDataset([video], [])
    else:
        vgg = load_video_data(cfg, cfg.data.VGGSound, normalize_audio=True)
        # load the largest one first
        # you can add more video/audio data upon demand, such as
        # clotho = load_audio_data(cfg, cfg.data.Clotho)
        dataset = MultiModalDataset([vgg], [])

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory

    if cfg.mixup.domain == 'data':
        mixup_params = cfg.mixup.params
        collate_fn = DataMixupCollate(generator=generator, 
                                      **mixup_params)
    else:
        collate_fn = None

    sampler, loader = construct_loader(dataset,
                                       batch_size,
                                       num_workers,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=pin_memory,
                                       collate_fn=collate_fn)

    return dataset, sampler, loader


def setup_test_datasets(cfg: DictConfig,
                        generator: torch.Generator,
                        ) -> tuple[Dataset, DistributedSampler, DataLoader]:
    if cfg.example_train:
        dataset = load_video_data(cfg, cfg.data.Example_video, normalize_audio=False, split='test')
    elif cfg.dataset.startswith('vggsound'):
        dataset = load_video_data(cfg, cfg.data.VGGSound_test, normalize_audio=False, split='test')
    else:
        raise NotImplementedError(f'Unknown dataset for test: {cfg.dataset}')

    batch_size = cfg.batch_size
    num_workers = cfg.get('num_workers_val', cfg.num_workers)
    pin_memory = cfg.pin_memory
    
    if cfg.mixup.domain == 'data':
        mixup_config = cfg.mixup.params
        collate_fn = DataMixupCollate(generator=generator, 
                                      **mixup_config)
    else:
        collate_fn = None
    
    sampler, loader = construct_loader(dataset,
                                       batch_size,
                                       num_workers,
                                       shuffle=False,
                                       drop_last=False,
                                       pin_memory=pin_memory,
                                       collate_fn=collate_fn)

    return dataset, sampler, loader


def setup_val_datasets(cfg: DictConfig,
                       generator: torch.Generator,
                       ) -> tuple[Dataset, DataLoader, DataLoader]:
    if cfg.example_train:
        dataset = load_video_data(cfg, cfg.data.Example_video, normalize_audio=False)
    else:
        dataset = load_video_data(cfg, cfg.data.VGGSound_val, normalize_audio=False)

    val_batch_size = cfg.batch_size
    val_eval_batch_size = cfg.eval_batch_size
    num_workers = cfg.get('num_workers_val', cfg.num_workers)
    pin_memory = cfg.pin_memory
    
    if cfg.mixup.domain == 'data':
        mixup_config = cfg.mixup.params
        collate_fn = DataMixupCollate(generator=generator, 
                                      **mixup_config)
    else:
        collate_fn = None
    
    _, val_loader = construct_loader(dataset,
                                     val_batch_size,
                                     num_workers,
                                     shuffle=False,
                                     drop_last=False,
                                     pin_memory=pin_memory,
                                     collate_fn=collate_fn)
    _, eval_loader = construct_loader(dataset,
                                      val_eval_batch_size,
                                      num_workers,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=pin_memory,
                                      collate_fn=collate_fn)

    return dataset, val_loader, eval_loader


def setup_eval_dataset(dataset_name: str, cfg: DictConfig) -> tuple[Dataset, DataLoader]:
    if dataset_name.startswith('audiocaps_full'):
        dataset = AudioCapsData(cfg.eval_data.audiocaps_full.audio_path,
                                cfg.eval_data.audiocaps_full.csv_path)
    elif dataset_name.startswith('audiocaps'):
        dataset = AudioCapsData(cfg.eval_data.audiocaps.audio_path,
                                cfg.eval_data.audiocaps.csv_path)
    elif dataset_name.startswith('vggsound'):
        dataset = VGGSound(cfg.eval_data.vggsound.video_path,
                           cfg.eval_data.vggsound.csv_path,
                           duration_sec=cfg.duration_s)
    elif dataset_name.startswith('infer_video'):
        dataset = InferenceVideoData(cfg.eval_data.infer_video.video_path,
                                     cfg.eval_data.infer_video.jsonl_path,
                                     duration_sec=cfg.duration_s)
        cfg.batch_size = 1
    elif dataset_name.startswith('example_video'):
        dataset = VGGSoundEval(cfg.eval_data.example_video.video_path,
                               cfg.eval_data.example_video.csv_path,
                               duration_sec=cfg.duration_s)
    elif dataset_name.startswith('example_bench'):
        dataset = VGGMonoAudioBench(cfg.eval_data.example_bench.video_path,
                                    cfg.eval_data.example_bench.jsonl_path,
                                    duration_sec=cfg.duration_s)
    elif dataset_name in ['vgg_monoaudio_intra', 'vgg_monoaudio_inter']:
        dataset = VGGMonoAudioBench(cfg.eval_data[dataset_name].video_path,
                                    cfg.eval_data[dataset_name].csv_path,
                                    duration_sec=cfg.duration_s)
        
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, loader = construct_loader(dataset,
                                 batch_size,
                                 num_workers,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=pin_memory,
                                 error_avoidance=True)
    return dataset, loader


def error_avoidance_collate(batch):
    # Filter our None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def construct_loader(dataset: Dataset,
                     batch_size: int,
                     num_workers: int,
                     *,
                     shuffle: bool = True,
                     drop_last: bool = True,
                     pin_memory: bool = False,
                     error_avoidance: bool = False,
                     collate_fn = None) -> tuple[DistributedSampler, DataLoader]:
    train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
    train_loader = DataLoader(dataset,
                              batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              drop_last=drop_last,
                              persistent_workers=num_workers > 0,
                              pin_memory=pin_memory,
                              collate_fn=error_avoidance_collate if error_avoidance else collate_fn)
    return train_sampler, train_loader
