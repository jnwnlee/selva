import logging
from typing import Any, Mapping

import torch
from torch import nn

from selva.ext.synchformer.motionformer import MotionFormer
from selva.ext.synchformer.astransformer import AST


class Synchformer(nn.Module):

    def __init__(self, video: bool = True, audio: bool = False):
        super().__init__()
        
        self.video = video
        self.audio = audio
        
        if not video and not audio:
            raise ValueError('At least one of vis or audio should be True.')

        if self.video:
            self.vfeat_extractor = MotionFormer(extract_features=True,
                                                factorize_space_time=True,
                                                agg_space_module='TransformerEncoderLayer',
                                                agg_time_module='torch.nn.Identity',
                                                add_global_repr=False)
        if self.audio:
            self.afeat_extractor = AST(extract_features=True,
                                    max_spec_t=66,
                                    factorize_freq_time=True,
                                    agg_freq_module='TransformerEncoderLayer',
                                    agg_time_module='torch.nn.Identity',
                                    add_global_repr=False)

        # self.vfeat_extractor = instantiate_from_config(vfeat_extractor)
        # self.afeat_extractor = instantiate_from_config(afeat_extractor)
        # # bridging the s3d latent dim (1024) into what is specified in the config
        # # to match e.g. the transformer dim
        # self.vproj = instantiate_from_config(vproj)
        # self.aproj = instantiate_from_config(aproj)
        # self.transformer = instantiate_from_config(transformer)

    def forward(self, data):
        video, audio = None, None
        
        if self.video and self.audio:
            video, audio = data
        elif self.video:
            video = data
        elif self.audio:
            audio = data
            
        if self.video and video is not None:
            video = self.forward_vfeat(video)
        if self.audio and audio is not None:
            audio = self.forward_afeat(audio)
        
        if self.video and self.audio:
            return video, audio
        elif self.video:
            return video
        else:
            return audio
    
    def forward_vfeat(self, vis):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis = self.vfeat_extractor(vis)
        return vis
    
    def forward_afeat(self, aud):
        B, S, F, Ta = aud.shape
        aud = aud.permute(0, 1, 3, 2) # (B, S, Ta, F)
        aud, _ = self.afeat_extractor(aud)
        return aud
        

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        target_keys = (['vfeat_extractor'] if self.video else []) \
                    + (['afeat_extractor'] if self.audio else [])
        # discard all entries except vfeat_extractor / afeat_extractor
        sd = {k: v for k, v in sd.items() if any(k.startswith(tk) 
                                                 for tk in target_keys)}
        

        return super().load_state_dict(sd, strict)


if __name__ == "__main__":
    model = Synchformer(video=True, audio=True).cuda().eval()
    sd = torch.load('/mnt/hdd3/junwon/mmaudio/ext_weights/synchformer_state_dict.pth', weights_only=True)
    model.load_state_dict(sd)

    vid = torch.randn(2, 7, 16, 3, 224, 224).cuda()
    features = model.forward_vfeat(vid).detach().cpu()
    print(features.shape)
    
    aud = torch.randn(2, 16000*8).cuda()
    segment_size = 10_240 # 16000 * (16/25) = 16000 * 0.64
    step_size = 5_120 # segment_size // 2
    num_segments = (128000 - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(aud[:, i * step_size:i * step_size + segment_size])
    aud = torch.stack(segments, dim=1) # (B, S, T)
    print(aud.shape)
    import torchaudio
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
    )
    spec = spec.cuda()
    aud = spec(aud) # (B, S, F, T)
    aud = torch.log(aud + 1e-6)
    max_spec_t = 66
    if max_spec_t - aud.shape[-1] > 0:
        # pad the last dim (time) -> (..., n_mels, 0+time+difference)  # safe for batched input
        pad_dims = (0, max_spec_t - aud.shape[-1])
        aud = torch.nn.functional.pad(aud, pad_dims, 
                                        'constant', 0.0)
    aud = aud[..., :max_spec_t]  # (B, S, F, T=66)
    MEAN = -4.2677393
    STD = 4.5689974
    aud = (aud - MEAN) / (2 * STD)
    print(aud.shape)
    
    from einops import rearrange
    aud = rearrange(aud, 'b s f t -> (b s) 1 f t')
    print(aud.shape)
    aud = model.forward_afeat(aud).detach().cpu()
    print(aud.shape)
    aud = rearrange(aud, '(b s) 1 t d -> b (s t) d', b=2)
    print(aud.shape)
    

    # extract and save the state dict only
    # sd = torch.load('./ext_weights/sync_model_audioset.pt')['model']
    # torch.save(sd, './ext_weights/synchformer_state_dict.pth')
