from typing import Optional, Union, List, Tuple, Any, Mapping
from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from selva.model.text_synchformer import TextSynchformer
from selva.utils.transforms import generate_multiple_segments


@dataclass
class PreprocessedConditions:
    sync_f: torch.Tensor
    sync_f_c: torch.Tensor
    text_f: torch.Tensor
    text_f_c: torch.Tensor
    text_mask: torch.Tensor


class TextSynch(TextSynchformer):

    def __init__(self,
                 *,
                 text_dim: int,
                 video_seq_len: int = 192, 
                 max_text_seq_len: int = 512,
                 empty_string_feat: torch.Tensor = None,
                 num_sup_text_tokens: int = 5,
                 sync_batch_size_multiplier: Union[int, float] = -1,
                 xattn_depth: int = 1,
                 ) -> None:
        super().__init__(
            text_dim=text_dim, 
            max_text_seq_len=max_text_seq_len,
            xattn_depth=xattn_depth,
        )

        self._video_seq_len = video_seq_len
        self.num_sup_text_tokens = num_sup_text_tokens
        self.sync_batch_size_multiplier = sync_batch_size_multiplier
        
        if num_sup_text_tokens > 0:
            self.sup_text_feat = nn.Parameter(torch.zeros(num_sup_text_tokens, self.text_dim),
                                                  requires_grad=True)
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((1, text_dim))
        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        
        self.initialize_weights()
    
    def update_seq_lengths(self, video_seq_len: int) -> None:
        self._video_seq_len = video_seq_len
    
    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)
    
    def get_sup_text_sequence(self, bs: int) -> torch.Tensor:
        if self.num_sup_text_tokens <= 0:
            raise ValueError(f'supplementary text tokens not enabled as {self.num_sup_text_tokens=}')
        return self.sup_text_feat.expand(bs, -1, -1)
    
    def prepend_sup_text_tokens(self, text_f: torch.Tensor, text_mask: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_sup_text_tokens <= 0:
            return text_f, text_mask
        bs = text_f.shape[0]
        sup_text_f = self.get_sup_text_sequence(bs) # (B, S, D)
        sup_text_mask = torch.ones(bs, sup_text_f.shape[1], 
                                       device=text_mask.device, dtype=text_mask.dtype) # (B, S)
        text_f = torch.cat([sup_text_f, text_f], dim=1)
        text_mask = torch.cat([sup_text_mask, text_mask], dim=1)
        return text_f, text_mask

    def encode_video_with_sync(self, x: torch.Tensor, text_f: torch.Tensor,
                               text_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) H/W: 384

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        x = generate_multiple_segments(x, segment_size, step_size) # (B, S, T, C, H, W)
        num_segments = x.shape[1]
        
        outputs = []
        if self.sync_batch_size_multiplier <= 0:
            batch_size = b
        else:
            batch_size = int(b * self.sync_batch_size_multiplier)
        x = einops.rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            start_idx = i // num_segments
            end_idx = min((i + batch_size - 1) // num_segments + 1, b)
            text_f_batch = text_f[start_idx:end_idx]
            text_mask_batch = text_mask[start_idx:end_idx]
            current_total_batch_size = min(batch_size, b * num_segments - i)
            
            repeats = torch.zeros(end_idx - start_idx, dtype=torch.long, device=x.device)
            for j in range(current_total_batch_size):
                original_batch_idx = (i + j) // num_segments
                repeats[original_batch_idx - start_idx] += 1

            text_f_batch_repeated = torch.repeat_interleave(text_f_batch, repeats, dim=0)
            text_mask_batch_repeated = torch.repeat_interleave(text_mask_batch, repeats, dim=0)

            outputs.append(self.forward_vfeat(
                x[i:i + batch_size], 
                text_f=text_f_batch_repeated, 
                text_mask=text_mask_batch_repeated
            ))
        x = torch.cat(outputs, dim=0)
        x = einops.rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x
    
    def encode_audio_with_sync(self, x: torch.Tensor, text_f: torch.Tensor,
                               text_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_afeat(
            x, text_f=text_f, text_mask=text_mask
        )
    
    def load_synchformer_state_dict(self, src_dict: dict):
        self.load_state_dict(src_dict, strict=True)
    
    def load_weights(self, src_dict) -> None:
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.empty_string_feat.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.empty_string_feat.dtype

    @property
    def video_seq_len(self) -> int:
        return self._video_seq_len

    @property
    def audio_seq_len(self) -> int:
        return self._audio_seq_len
    
    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        target_keys = (['vfeat_extractor'] if self.video else []) \
                    + (['afeat_extractor'] if self.audio else []) \
                    + ['text_proj', 'synch_text_cross_blocks', 
                       'sup_text_feat', 'empty_string_feat']
        # discard all entries except vfeat_extractor / afeat_extractor
        sd = {k: v for k, v in sd.items() if any(k.startswith(tk) 
                                                 for tk in target_keys)}
        

        return nn.Module.load_state_dict(self, sd, strict=strict)


def depth1(**kwargs) -> TextSynch:
    return TextSynch(text_dim=768, # 2048 for xl
                     video_seq_len=192,
                     max_text_seq_len=512,
                     xattn_depth=1,
                     **kwargs)
    

def get_my_textsynch(name: str, **kwargs) -> TextSynch:
    if name.startswith('depth1'):
        return depth1(**kwargs)
    else:
        raise ValueError(f'Unknown model name: {name}')


if __name__ == '__main__':
    network = get_my_textsynch('depth1')
    # print the number of parameters in terms of millions
    num_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')
    
    torch.compile(network.encode_video_with_sync)
    print(f"Compiled encode_video_with_sync")
    torch.compile(network.predict_flow)
    print(f"Compiled predict_flow")
    torch.compile(network.preprocess_conditions)
    print(f"Compiled preprocess_conditions:")
    torch.compile(network.forward)
    print(f"Compiled forward:")
