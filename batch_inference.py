import logging
import os
from pathlib import Path
from typing import Union
import gc
import json

import hydra
import torch
import torch.distributed as distributed
import torchaudio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from av_bench.evaluate import evaluate
from av_bench.extract import extract

from selva.data.data_setup import setup_eval_dataset
from selva.utils.eval_utils import ModelConfig, all_model_cfg, load_video, make_video
from selva.model.flow_matching import FlowMatching
from selva.model.networks_video_enc import TextSynch
from selva.model.networks_generator import MMAudio, get_my_mmaudio
from selva.model.utils.features_utils import FeaturesUtils
from selva.model.utils.factory import create_model_from_factory
from selva.utils.eval_utils import generate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
log = logging.getLogger()


@torch.inference_mode()
@hydra.main(version_base='1.3.2', config_path='config', config_name='infer_config.yaml')
def main(cfg: DictConfig):
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(local_rank)
    
    run_dir = Path(HydraConfig.get().run.dir)
    if cfg.output_name is None:
        output_dir = run_dir / cfg.dataset
    else:
        output_dir = run_dir / f'{cfg.dataset}-{cfg.output_name}'
    if cfg.use_negative_caption:
        output_dir = output_dir.with_name(f'{output_dir.name}_negative_caption')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'audio').mkdir(parents=True, exist_ok=True)
    (output_dir / 'video').mkdir(parents=True, exist_ok=True)
    
    # set config
    if cfg.generator.model.name not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {cfg.generator.model.name}')
    model_cfg: ModelConfig = all_model_cfg[cfg.generator.model.name]
    video_enc_ckpt_path = Path(HydraConfig.get().run.dir) / f'{cfg.exp_id}_ema_final.pth' # Example adjustment
    if not video_enc_ckpt_path.exists():
        video_enc_ckpt_path = Path(cfg.video_enc.get('weights', model_cfg.model_video_enc_path))
        if not video_enc_ckpt_path.exists():
            model_cfg.download_video_enc_if_needed()
    generator_ckpt_path = Path(cfg.generator.get('weights', model_cfg.model_generator_path))
    if not generator_ckpt_path.exists():
        model_cfg.download_generator_if_needed()
    seq_cfg = model_cfg.seq_cfg
    seq_cfg.duration = cfg.duration_s

    # load TextSynch
    net_video_enc: TextSynch = create_model_from_factory(
        cfg.video_enc.model.factory_path,
        cfg.video_enc.model.name,
        **cfg.video_enc.model.get('params', {}),
    ).to(device).eval()
    if video_enc_ckpt_path.exists():
        net_video_enc.load_weights(torch.load(video_enc_ckpt_path, map_location=device, weights_only=True))    
        log.info(f'TextSynch: Loaded weights from trained ckpt {video_enc_ckpt_path}')
    else:
        net_video_enc.load_weights(torch.load(model_cfg.model_video_enc_path, map_location=device, weights_only=True))
        log.info(f'TextSynch: Loaded weights from {model_cfg.model_video_enc_path}')
    net_video_enc.update_seq_lengths(video_seq_len=model_cfg.seq_cfg.sync_seq_len)
    # load MMAudio
    net_generator: MMAudio = get_my_mmaudio(cfg.generator.model.name).to(device).eval()
    if generator_ckpt_path.exists():
        net_generator.load_weights(torch.load(generator_ckpt_path, map_location=device, weights_only=True))    
        log.info(f'MMAudio: Loaded weights from trained ckpt {generator_ckpt_path}')
    else:
        net_generator.load_weights(torch.load(model_cfg.model_generator_path, map_location=device, weights_only=True))
        log.info(f'MMAudio: Loaded weights from {model_cfg.model_generator_path}')
    net_generator.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    log.info(f'Latent seq len: {seq_cfg.latent_seq_len}')
    log.info(f'Clip seq len: {seq_cfg.clip_seq_len}')
    log.info(f'Sync seq len: {seq_cfg.sync_seq_len}')
    
    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    fm = FlowMatching(cfg.sampling.min_sigma,
                      inference_mode=cfg.sampling.method,
                      num_steps=cfg.sampling.num_steps)

    if '16k' in cfg.generator.model.name:
        tod_vae_ckpt = cfg.get('vae_16k_ckpt', model_cfg.vae_path)
        bigvgan_vocoder_ckpt = cfg.get('bigvgan_vocoder_ckpt', model_cfg.bigvgan_16k_path)
    elif '44k' in cfg.generator.model.name:
        tod_vae_ckpt = cfg.get('vae_44k_ckpt', model_cfg.vae_path)
        bigvgan_vocoder_ckpt = None
    else:
        raise ValueError(f'Unknown model variant: {cfg.generator.model.name}')
    synchformer_ckpt = cfg.get('synchformer_ckpt', model_cfg.synchformer_ckpt)
    if not (Path(tod_vae_ckpt).exists() and Path(synchformer_ckpt).exists() and \
        (Path(bigvgan_vocoder_ckpt).exists() if bigvgan_vocoder_ckpt is not None else True)):
        model_cfg.download_external_modules_if_needed()
    feature_utils = FeaturesUtils(tod_vae_ckpt=tod_vae_ckpt,
                                  synchformer_ckpt=synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model_cfg.mode,
                                  bigvgan_vocoder_ckpt=bigvgan_vocoder_ckpt,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device).eval()
    
    if cfg.compile:
        net_generator.preprocess_conditions = torch.compile(net_generator.preprocess_conditions)
        net_generator.predict_flow = torch.compile(net_generator.predict_flow)
        feature_utils.compile()
        net_video_enc.forward = torch.compile(net_video_enc.forward)
        net_video_enc.encode_video_with_sync = torch.compile(net_video_enc.encode_video_with_sync)

    dataset, loader = setup_eval_dataset(cfg.dataset, cfg)

    if cfg.dataset not in ['infer_video']:
        pbar = tqdm(total=(len(dataset)+world_size-1)//world_size) if local_rank == 0 else None
        with torch.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16, device_type=device):
            for batch in tqdm(loader):
                if all(os.path.isfile(output_dir / 'audio' / f"{name}.wav") for name in batch['name']) and \
                     all(os.path.isfile(output_dir / 'video' / f"{name}.mp4") for name in batch['name']):
                    continue
                audios = generate(None, # batch.get('clip_video', None),
                                  batch.get('sync_video', None),
                                  batch.get('caption', None),
                                  negative_text=batch.get('negative_caption', None) \
                                      if cfg.use_negative_caption else None,
                                  feature_utils=feature_utils,
                                  net_video_enc=net_video_enc,
                                  net_generator=net_generator,
                                  fm=fm,
                                  rng=rng,
                                  cfg_strength=cfg.cfg_strength)
                audios = audios.float().cpu()
                names = batch['name']
                for audio, name in zip(audios, names):
                    torchaudio.save(output_dir / 'audio' / f'{name}.wav', audio, seq_cfg.sampling_rate)
                    video_info = load_video(dataset.video_root / (name + '.mp4'),
                                            cfg.get('duration_s', model_cfg.seq_cfg.duration))
                    make_video(video_info, output_dir / 'video' / f'{name}.mp4', 
                           audio, sampling_rate=seq_cfg.sampling_rate)
                if pbar:
                    pbar.update(len(audios))
    else:
        # force batch size to 1 for videos with diverse lengths
        log.info('Forcing batch size to 1 for video inference')
        cfg.batch_size = 1
        pbar = tqdm(total=(len(dataset)+world_size-1)//world_size) if local_rank == 0 else None
        with torch.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16, device_type=device):
            for index_factor in range(len(dataset)//world_size+1):
                index = index_factor * world_size + local_rank
                if index >= len(dataset):
                    break
                if os.path.isfile(output_dir / 'audio' / f"{dataset.videos[index]}.flac") and \
                   os.path.isfile(output_dir / 'video' / f"{dataset.videos[index]}.mp4"):
                    print(f'Skipping {dataset.videos[index]} as output files already exist.')
                    if pbar:
                        pbar.update(1)
                    continue
                video_info = load_video(dataset.video_root / (dataset.videos[index] + '.mp4'), 
                                        cfg.get('duration_s', model_cfg.seq_cfg.duration))
                clip_frames = video_info.clip_frames
                sync_frames = video_info.sync_frames
                duration = video_info.duration_sec
                clip_frames = clip_frames.unsqueeze(0)
                sync_frames = sync_frames.unsqueeze(0)
                seq_cfg.duration = duration
                net_generator.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
                model_cfg.seq_cfg.duration = duration
                net_video_enc.update_seq_lengths(video_seq_len=model_cfg.seq_cfg.sync_seq_len)
                negative_text = None
                if cfg.use_negative_caption:
                    negative_text = dataset.negative_captions.get(dataset.videos[index], None)
                if negative_text is not None:
                    negative_text = [negative_text]
                
                audios = generate(None,
                                  sync_frames,
                                  [dataset.captions[ dataset.videos[index] ]],
                                  negative_text=negative_text,
                                  feature_utils=feature_utils,
                                  net_video_enc=net_video_enc,
                                  net_generator=net_generator,
                                  fm=fm,
                                  rng=rng,
                                  cfg_strength=cfg.cfg_strength)
                audio = audios.float().cpu()[0]
                torchaudio.save(output_dir / 'audio' / f'{dataset.videos[index]}.flac', audio, seq_cfg.sampling_rate)
                make_video(video_info, output_dir / 'video' / f'{dataset.videos[index]}.mp4', 
                           audio, sampling_rate=seq_cfg.sampling_rate)
                
                if pbar:
                    pbar.update(1)
    
    del net_video_enc, net_generator, feature_utils, fm
    gc.collect()
    torch.cuda.empty_cache()
    distributed.barrier()
                    
    if cfg.eval_data[cfg.dataset].get('gt_cache', None) is not None:
        extract(
            audio_path=output_dir / "audio",
            output_path=output_dir / "cache",
            audio_length=8.,
            device='cuda',
            batch_size=cfg.extract_batch_size,
            num_workers=cfg.num_workers,
            skip_video_related=False,
            skip_clap=False,
        )
        
        output_metrics = evaluate(gt_audio_cache=Path(cfg.eval_data[cfg.dataset].gt_cache),
                                pred_audio_cache=output_dir / 'cache')
        log.info(f'Evaluation metrics: {output_metrics}')
        # write dict as json
        output_metrics_path = output_dir / 'metrics.json'
        with open(output_metrics_path, 'w') as f:
                json.dump(output_metrics, f, indent=4)


def distributed_setup():
    distributed.init_process_group(backend="nccl")
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


if __name__ == '__main__':
    distributed_setup()

    main()

    # clean-up
    distributed.destroy_process_group()
