import subprocess
from pathlib import Path
import shutil
import sys
import os
from argparse import Namespace, ArgumentParser

from torio.io import StreamingMediaDecoder
import torchaudio
from tqdm.auto import tqdm
import pandas as pd


def extract_audio_from_video(
    input_video_dir: Path,
    tsv_path: str,
    video_dir: Path,
    audio_dir: Path,
    sampling_rate: int = 16000,
    duration: int = 8,
):
    # for every .mp4 video in input_video_dir, extract audio as wav and save it in target_dir/audio, using ffmpeg
    resampler = {}
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tsv_path, sep='\t')
    for video_id in tqdm(df['id'], desc="Extracting audio from video", total=len(df)):
        video_name = video_id + ".mp4"
        video_path = input_video_dir / video_name
        audio_name = video_id + ".wav"
        audio_path = audio_dir / audio_name
        video_target_path = video_dir / video_name

        if video_path.exists():
            if audio_path.exists() and video_target_path.exists():
                continue
            reader = StreamingMediaDecoder(str(video_path))
            reader.add_basic_audio_stream(frames_per_chunk=2**30)
            reader.fill_buffer()
            
            sample_rate = int(reader.get_out_stream_info(0).sample_rate)
            audio_chunk = reader.pop_chunks()[0]
            audio_chunk = audio_chunk.transpose(0, 1).mean(dim=0) # stereo to mono
            
            if not sample_rate == sampling_rate:
                if sample_rate not in resampler:
                    resampler[sample_rate] = torchaudio.transforms.Resample(
                        sample_rate,
                        sampling_rate,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method='sinc_interp_kaiser',
                        beta=14.769656459379492,
                    )
                audio_chunk = resampler[sample_rate](audio_chunk)
            
            audio_chunk = audio_chunk[:sampling_rate * duration]
            # save audio
            torchaudio.save(str(audio_path), audio_chunk.unsqueeze(0), sampling_rate)

            # Copy video to target directory
            shutil.copy(video_path, video_target_path) # video could be longer than duration, but av_benchmark will crop it
        else:
            print(f"Video {video_name} not found in {input_video_dir}. Skipping...")

def extract_video_features(video_dir: Path, audio_dir: Path, target_dir: Path, batch_size: int, num_workers: int):
    from extract_video import extract as video_extract
    args = Namespace(
        gt_cache=target_dir,
        video_path=video_dir,
        gt_audio=audio_dir,
        audio_length=8.,
        num_workers=num_workers,
        gt_batch_size=batch_size
    )
    video_extract(args)

def extract_text_features(target_dir: Path, tsv_path: str):
    from extract_text import extract as text_extract
    # make temporary csv file
    temp_csv_path = Path(tsv_path).parent / "text_annotation_for_av_benchmark.csv"
    if temp_csv_path.exists():
        raise FileExistsError(f"{temp_csv_path} already exists")
    if Path(tsv_path).suffix == '.tsv':
        df = pd.read_csv(tsv_path, sep='\t')
        # change column name: id -> name, label -> caption
        df.columns = ['name', 'caption']
    else:
        df = pd.read_csv(tsv_path)
        # only get ['file_name', 'label'], and rename them to 'name' and 'caption'
        df = df[['file_name', 'label']]
        df.columns = ['name', 'caption']
    df.to_csv(temp_csv_path, index=False)

    args = Namespace(
        text_csv=temp_csv_path,
        output_cache_path=target_dir
    )
    text_extract(args)

def extract_audio_features(audio_dir: Path, target_dir: Path, batch_size: int, num_workers: int):
    from av_bench.extract import extract as audio_extract
    audio_extract(
        audio_path=audio_dir,
        output_path=target_dir,
        audio_length=8.,
        device='cuda',
        batch_size=batch_size,
        num_workers=num_workers,
        skip_video_related=True,
        skip_clap=True,
    )

def main():
    parser = ArgumentParser()
    parser.add_argument("--av_benchmark_dir", type=str, default="/home/junwon/av-benchmark")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=8)
    args = parser.parse_args()

    # Setup options/paths
    av_benchmark_dir = args.av_benchmark_dir
    sys.path.append(av_benchmark_dir)
    os.environ["OMP_NUM_THREADS"] = f"{args.num_workers}"
    metadata = {
        # 'vgg_train': {
        #     'source_video_dir': './data/vggsound/video',
        #     'tsv_path': './sets/vgg-train.tsv',
        #     'video_dir': './data/vggsound/train/video',
        #     'audio_dir': './data/vggsound/train/audio',
        #     'target_dir': './data/vggsound/train/gt_cache',
        # },
        # 'vgg_val': {
        #     'source_video_dir': './data/vggsound/video',
        #     'tsv_path': './sets/vgg-val.tsv',
        #     'video_dir': './data/vggsound/val/video',
        #     'audio_dir': './data/vggsound/val/audio',
        #     'target_dir': './data/vggsound/val/gt_cache',
        # },
        # 'vgg_test': {
        #     'source_video_dir': './data/vggsound/video',
        #     'tsv_path': './sets/vgg-test.tsv',
        #     'video_dir': './data/vggsound/test/video',
        #     'audio_dir': './data/vggsound/test/audio',
        #     'target_dir': './data/vggsound/test/gt_cache',
        # },
        'vgg_monoaudio_intra': {
            'source_video_dir': './data/vgg_monoaudio/intra_class/mixed',
            'video_dir': './data/vgg_monoaudio/intra_class/mixed',
            'audio_dir': './data/vgg_monoaudio/intra_class/target_audio',
            'target_dir': './data/vgg_monoaudio/intra_class/gt_cache_test',
            'tsv_path': './data/vgg_monoaudio/intra_class/metadata.csv',
        },
        'vgg_monoaudio_inter': {
            'source_video_dir': './data/vgg_monoaudio/inter_class/mixed',
            'video_dir': './data/vgg_monoaudio/inter_class/mixed',
            'audio_dir': './data/vgg_monoaudio/inter_class/target_audio',
            'target_dir': './data/vgg_monoaudio/inter_class/gt_cache_test',
            'tsv_path': './data/vgg_monoaudio/inter_class/metadata.csv',
        },
    }

    for data_name, data_info in tqdm(metadata.items(), desc="Feature caching for each dataset", total=len(metadata)):
        input_video_dir = Path(data_info['source_video_dir'])
        video_dir = Path(data_info['video_dir'])
        audio_dir = Path(data_info['audio_dir'])
        target_dir = Path(data_info['target_dir'])
        
        extract_audio_from_video(
            input_video_dir=input_video_dir, 
            tsv_path=data_info['tsv_path'], 
            video_dir=video_dir, 
            audio_dir=audio_dir, 
            sampling_rate=args.sampling_rate, 
            duration=args.duration
        )

        # use av-benchmark
        extract_video_features(
            video_dir=video_dir, 
            audio_dir=audio_dir, 
            target_dir=target_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        extract_text_features(
            target_dir=target_dir, 
            tsv_path=data_info['tsv_path']
        )
        extract_audio_features(
            audio_dir=audio_dir,
            target_dir=target_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )

if __name__ == "__main__":
    main()