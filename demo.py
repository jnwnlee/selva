import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from selva.utils.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from selva.model.flow_matching import FlowMatching
from selva.model.networks_generator import MMAudio, get_my_mmaudio
from selva.model.networks_video_enc import TextSynch, get_my_textsynch
from selva.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='small_16k',
                        help='small_16k')
    parser.add_argument('--video', type=Path, required=True, help='Path to the video file')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model_cfg: ModelConfig = all_model_cfg[args.variant]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    video_path: Path = Path(args.video).expanduser()
    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: Path = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # load TextSynch
    net_video_enc: TextSynch = get_my_textsynch('depth1').to(device, dtype).eval()
    net_video_enc.load_weights(torch.load(model_cfg.model_video_enc_path, map_location=device, weights_only=True))
    log.info(f'TextSynch: Loaded weights from {model_cfg.model_video_enc_path}')
    
    # load MMAudio
    net_generator: MMAudio = get_my_mmaudio(model_cfg.model_name).to(device, dtype).eval()
    net_generator.load_weights(torch.load(model_cfg.model_generator_path, map_location=device, weights_only=True))
    log.info(f'MMAudio: Loaded weights from {model_cfg.model_generator_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    if '16k' in model_cfg.model_name:
        tod_vae_ckpt = model_cfg.vae_path
        bigvgan_vocoder_ckpt = model_cfg.bigvgan_16k_path
    elif '44k' in model_cfg.model_name:
        tod_vae_ckpt = model_cfg.vae_path
        bigvgan_vocoder_ckpt = None
    else:
        raise ValueError(f'Unknown model variant: {model_cfg.model_name}')
    synchformer_ckpt = model_cfg.synchformer_ckpt
    
    feature_utils = FeaturesUtils(tod_vae_ckpt=tod_vae_ckpt,
                                  synchformer_ckpt=synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model_cfg.mode,
                                  bigvgan_vocoder_ckpt=bigvgan_vocoder_ckpt,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    log.info(f'Using video {video_path}')
    video_info = load_video(video_path, duration)
    clip_frames = None
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    sync_frames = sync_frames.unsqueeze(0)

    seq_cfg.duration = duration
    net_generator.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    net_video_enc.update_seq_lengths(video_seq_len=seq_cfg.sync_seq_len)

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net_video_enc=net_video_enc,
                      net_generator=net_generator,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]
    if video_path is not None:
        save_path = output_dir / f'{video_path.stem}.flac'
    else:
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
        save_path = output_dir / f'{safe_filename}.flac'
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

    log.info(f'Audio saved to {save_path}')
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f'{video_path.stem}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        log.info(f'Video saved to {output_dir / video_save_path}')

    log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
