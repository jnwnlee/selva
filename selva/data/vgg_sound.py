import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from tensordict import TensorDict

from selva.data.av_utils import normalize_video_chunk
from selva.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VGGSound(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        *,
        tsv_path: Union[str, Path] = 'sets/vgg3-train.tsv',
        for_generator: bool = True,
        audio_required: bool = False,
        sample_rate: int = 16_000,
        duration_sec: float = 8.0,
        audio_samples: Optional[int] = None,
        normalize_audio: bool = False,
        clip_video_required: bool = False,
        mmap_dir: Union[str, Path] = None,
        tsv_tsynch_path: Union[str, Path] = None,
        mmap_tsync_dir: Union[str, Path] = None,
        data_dim: dict[str, int] = None,
    ):
        self.root = Path(root)
        self.audio_required = audio_required
        if audio_required:
            self.normalize_audio = normalize_audio
            if audio_samples is None:
                self.audio_samples = int(sample_rate * duration_sec)
            else:
                self.audio_samples = audio_samples
                effective_duration = audio_samples / sample_rate
                # make sure the duration is close enough, within 15ms
                assert abs(effective_duration - duration_sec) < 0.015, \
                    f'audio_samples {audio_samples} does not match duration_sec {duration_sec}'
        self.clip_video_required = clip_video_required
        self.for_generator = for_generator

        videos = sorted(os.listdir(self.root))
        videos = set([Path(v).stem for v in videos])  # remove extensions
        self.labels = {}
        self.videos = []
        missing_videos = []

        # read the tsv for subset information
        df_list = pd.read_csv(tsv_path, sep='\t', dtype={'id': str}).to_dict('records')
        for record in df_list:
            id = record['id']
            label = record['label']
            if id in videos:
                self.labels[id] = label
                self.videos.append(id)
            else:
                missing_videos.append(id)

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {root}')
            log.info(f'{len(self.videos)} videos found in {tsv_path}')
            log.info(f'{len(missing_videos)} videos missing in {root}')

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        if audio_required:
            self.expected_audio_length = self.audio_samples
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)
        if clip_video_required:
            self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)

        self.sync_transform = v2.Compose([
            v2.Resize((_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            # v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        if clip_video_required:
            self.clip_transform = v2.Compose([
                v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])    
        if audio_required:
            self.resampler = {}
        
        # mmap
        log.info(f'Loading precomputed mmap from {mmap_dir}')
        mmap_dir = Path(mmap_dir)
        td = TensorDict.load_memmap(mmap_dir)
        log.info(f'Loaded precomputed mmap from {mmap_dir}')
        self.sync_features = td['sync_features']
        if for_generator:
            self.mean = td['mean']
            self.std = td['std']
            self.text_clip_features = td['text_features']
        if clip_video_required:
            self.clip_features = td['clip_features']
        else:
            self.clip_features = None
        self.id2idx_mmap = {d['id']: i for i, d in enumerate(df_list)}
        
        mmap_tsync_dir = Path(mmap_tsync_dir)
        td_tsync = TensorDict.load_memmap(mmap_tsync_dir)
        log.info(f'Loaded precomputed tsync mmap from {mmap_tsync_dir}')
        self.text_features = td_tsync['text_features']
        self.text_masks = td_tsync['text_masks']
        df_list_tsync = pd.read_csv(tsv_tsynch_path, sep='\t').to_dict('records')
        self.id2idx_mmap_tsync = {d['id']: i for i, d in enumerate(df_list_tsync)}

        if local_rank == 0:
            log.info(f'Loaded {len(self)} samples.')
            log.info(f'Loaded sync_features: {self.sync_features.shape}.')
            log.info(f'Loaded text_features: {self.text_features.shape}.')
            log.info(f'Loaded text_masks: {self.text_masks.shape}.')
            if for_generator:
                log.info(f'Loaded mean: {self.mean.shape}.')
                log.info(f'Loaded std: {self.std.shape}.')
                log.info(f'Loaded text_clip_features: {self.text_clip_features.shape}.')
            if clip_video_required:
                log.info(f'Loaded clip_features: {self.clip_features.shape}.')
        
        assert self.sync_features.shape[1] == data_dim['sync_seq_len'], \
            f'{self.sync_features.shape[1]} != {data_dim["sync_seq_len"]}'
        assert self.text_features.shape[1] <= data_dim['text_flant5_max_seq_len'], \
            f'{self.text_features.shape[1]} > {data_dim["text_flant5_max_seq_len"]}'
        assert self.text_masks.shape[1] <= data_dim['text_flant5_max_seq_len'], \
            f'{self.text_masks.shape[1]} > {data_dim["text_flant5_max_seq_len"]}'
        assert self.sync_features.shape[-1] == data_dim['sync_dim'], \
            f'{self.sync_features.shape[-1]} != {data_dim["sync_dim"]}'
        assert self.text_features.shape[-1] == data_dim['text_flant5_dim'], \
            f'{self.text_features.shape[-1]} != {data_dim["text_flant5_dim"]}'
        if for_generator:
            assert self.mean.shape[1] == data_dim['latent_seq_len'], \
                f'{self.mean.shape[1]} != {data_dim["latent_seq_len"]}'
            assert self.std.shape[1] == data_dim['latent_seq_len'], \
                f'{self.std.shape[1]} != {data_dim["latent_seq_len"]}'
            assert self.text_clip_features.shape[1] == data_dim['text_clip_seq_len'], \
                f'{self.text_clip_features.shape[1]} != {data_dim["text_clip_seq_len"]}'
            assert self.text_clip_features.shape[-1] == data_dim['text_clip_dim'], \
                f'{self.text_clip_features.shape[-1]} != {data_dim["text_clip_dim"]}'
        if clip_video_required:
            assert self.clip_features.shape[1] == data_dim['clip_seq_len'], \
                f'{self.clip_features.shape[1]} != {data_dim["clip_seq_len"]}'
            assert self.clip_features.shape[-1] == data_dim['clip_dim'], \
                f'{self.clip_features.shape[-1]} != {data_dim["clip_dim"]}'
            
        self.video_exist = torch.tensor(1, dtype=torch.bool)
        self.text_exist = torch.tensor(1, dtype=torch.bool)


    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]: # mmap
        latents = self.mean
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))
    
    def get_memory_mapped_tensor(self) -> TensorDict:
        td = TensorDict({
            'sync_features': self.sync_features,
            'text_features': self.text_features,
            'text_masks': self.text_masks,
        })
        if self.for_generator:
            td['mean'] = self.mean
            td['std'] = self.std
            td['text_clip_features'] = self.text_clip_features
        if self.clip_video_required:
            td['clip_features'] = self.clip_features
        return td

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        
        if video_id in self.captions and torch.rand(1).item() < self.autoacd_sample_prob:
            label = self.captions[video_id]
        else:
            label = self.labels[video_id]
            
        reader = StreamingMediaDecoder(self.root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )
        if self.audio_required:
            reader.add_basic_audio_stream(frames_per_chunk=2**30, )
        if self.clip_video_required:
            reader.add_basic_video_stream(
                frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
                frame_rate=_CLIP_FPS,
                format='rgb24',
            )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        
        sync_chunk = data_chunk[0]
        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')
        sync_chunk = normalize_video_chunk(sync_chunk, self.sync_expected_length, 
                                           n_tolerance_frame=3, desc=video_id)
        sync_chunk = self.sync_transform(sync_chunk)
        
        if self.audio_required:
            audio_chunk = data_chunk[1]
            
        if self.clip_video_required:
            clip_chunk = data_chunk[2 if self.audio_required else 1]
            if clip_chunk is None:
                raise RuntimeError(f'CLIP video returned None {video_id}')
            clip_chunk = normalize_video_chunk(clip_chunk, self.clip_expected_length,
                                               n_tolerance_frame=1, desc=video_id)
            clip_chunk = self.clip_transform(clip_chunk)
        
        # process audio
        if self.audio_required:
            sample_rate = int(reader.get_out_stream_info(1).sample_rate)
            audio_chunk = audio_chunk.transpose(0, 1)
            audio_chunk = audio_chunk.mean(dim=0)  # mono
            if self.normalize_audio:
                abs_max = audio_chunk.abs().max()
                audio_chunk = audio_chunk * (0.95 / abs_max)
                if abs_max <= 1e-6:
                    raise RuntimeError(f'Audio is silent {video_id}')

            # resample
            if sample_rate == self.sample_rate:
                audio_chunk = audio_chunk
            else:
                if sample_rate not in self.resampler:
                    # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                    self.resampler[sample_rate] = torchaudio.transforms.Resample(
                        sample_rate,
                        self.sample_rate,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method='sinc_interp_kaiser',
                        beta=14.769656459379492,
                    )
                audio_chunk = self.resampler[sample_rate](audio_chunk)

            if audio_chunk.shape[0] < self.expected_audio_length:
                raise RuntimeError(f'Audio too short {video_id}')
            audio_chunk = audio_chunk[:self.expected_audio_length]

        data = {
            'id': video_id,
            'caption': label,
            'sync_video': sync_chunk,
            'sync_f_vid_orig': self.sync_features[self.id2idx_mmap[video_id]],
            'text_features': self.text_features[self.id2idx_mmap_tsync[video_id]],
            'text_masks': self.text_masks[self.id2idx_mmap_tsync[video_id]],
            'video_exist': self.video_exist,
            'text_exist': self.text_exist,
        }
        
        if self.for_generator:
            data['a_mean'] = self.mean[self.id2idx_mmap[video_id]]
            data['a_std'] = self.std[self.id2idx_mmap[video_id]]
            data['text_clip_features'] = self.text_clip_features[self.id2idx_mmap[video_id]]
        
        if self.audio_required:
            data['audio'] = audio_chunk
            
        if self.clip_video_required:
            data['clip_video'] = clip_chunk
            data['clip_features'] = self.clip_features[self.id2idx_mmap[video_id]],

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.videos[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.labels)
