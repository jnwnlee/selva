import json
import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder

from selva.data.av_utils import normalize_video_chunk
from selva.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VideoDataset(Dataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        *,
        duration_sec: float = 8.0,
        clip_video_required: bool = False,
    ):
        self.video_root = Path(video_root)
        self.duration_sec = duration_sec
        self.clip_video_required = clip_video_required

        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)
        self.sync_transform = v2.Compose([
            v2.Resize((_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            # v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        if self.clip_video_required:
            self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
            self.clip_transform = v2.Compose([
                v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])

        # to be implemented by subclasses
        self.captions = {}
        self.negative_captions = {}
        self.videos = sorted(list(self.captions.keys()))

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        caption = self.captions[video_id]
        negative_caption = self.negative_captions.get(video_id, None)

        reader = StreamingMediaDecoder(self.video_root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )
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
        
        if self.clip_video_required:
            clip_chunk = data_chunk[1]
            if clip_chunk is None:
                raise RuntimeError(f'CLIP video returned None {video_id}')
            clip_chunk = normalize_video_chunk(clip_chunk, self.clip_expected_length, 
                                               n_tolerance_frame=1, desc=video_id)
            clip_chunk = self.clip_transform(clip_chunk)

        data = {
            'name': video_id,
            'caption': caption,
            'sync_video': sync_chunk,
        }
        if self.clip_video_required:
            data['clip_video'] = clip_chunk
        if negative_caption is not None:
            data['negative_caption'] = negative_caption

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.videos[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.captions)


class VGGSound(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        *,
        duration_sec: float = 8.0,
        clip_video_required: bool = False,
    ):
        super().__init__(video_root, duration_sec=duration_sec, 
                         clip_video_required=clip_video_required)
        self.video_root = Path(video_root)
        self.csv_path = Path(csv_path)

        videos = sorted(os.listdir(self.video_root))
        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
        self.captions = {}

        df = pd.read_csv(csv_path, header=None, names=['id', 'sec', 'caption',
                                                       'split']).to_dict(orient='records')

        videos_no_found = []
        for row in df:
            if row['split'] == 'test':
                start_sec = int(row['sec'])
                video_id = str(row['id'])
                # this is how our videos are named
                video_name = f'{video_id}_{start_sec:06d}'
                if video_name + '.mp4' not in videos:
                    videos_no_found.append(video_name)
                    continue

                self.captions[video_name] = row['caption']

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
            log.info(f'{len(self.captions)} useable videos found')
            if videos_no_found:
                log.info(f'{len(videos_no_found)} found in {csv_path} but not in {video_root}')
                log.info(
                    'A small amount is expected, as not all videos are still available on YouTube')

        self.videos = sorted(list(self.captions.keys()))


class InferenceVideoData(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        jsonl_root: Union[str, Path],
        *,
        duration_sec: float = 10.0,
        clip_video_required: bool = False,
    ):
        super().__init__(video_root, duration_sec=duration_sec, 
                         clip_video_required=clip_video_required)
        self.video_root = Path(video_root)
        self.jsonl_root = Path(jsonl_root)

        videos = sorted(os.listdir(self.video_root))
        videos = [v[:-4] for v in videos]  # remove extensions
        self.captions = {}

        for v in videos:
            with open(self.jsonl_root / (v + '.jsonl')) as f:
                data = json.load(f)
                self.captions[v] = data['audio_prompt']
                self.negative_captions[v] = data.get('negative_audio_prompt', None)

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')

        self.videos = videos


class VGGMonoAudioBench(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        *,
        duration_sec: float = 8.0,
        clip_video_required: bool = False,
    ):
        super().__init__(video_root, duration_sec=duration_sec, 
                         clip_video_required=clip_video_required)
        self.video_root = Path(video_root)
        self.csv_path = Path(csv_path)

        videos = sorted(os.listdir(self.video_root))
        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
        self.captions = {}
        self.negative_captions = {}

        df = pd.read_csv(csv_path, header=0, usecols=['filename', 'label', 'paired_label']
                         ).to_dict(orient='records')

        videos_no_found = []
        for row in df:
            video_name = row['filename']
            if video_name + '.mp4' not in videos:
                videos_no_found.append(video_name)
                continue

            self.captions[video_name] = row['label']
            self.negative_captions[video_name] = row['paired_label']

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
            log.info(f'{len(self.captions)} useable videos found')
            if videos_no_found:
                log.info(f'{len(videos_no_found)} found in {csv_path} but not in {video_root}!')

        self.videos = sorted(list(self.captions.keys()))