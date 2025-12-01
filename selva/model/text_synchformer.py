import logging
from typing import Any, Mapping

import einops
import torch
from torch import nn

from selva.ext.synchformer.motionformer import MotionFormer, SpatialTransformerEncoderLayer, BaseEncoderLayer
from selva.ext.synchformer.astransformer import AST
from selva.model.transformer_layers import (MMCrossAttentionBlock)
from selva.model.low_level import MLP


class ExtendedMotionFormer(MotionFormer):
    """Extended MotionFormer with additional methods for text synchronization."""
    
    def forward_segments_without_aggregation(self, x, orig_shape: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features without spatial-temporal aggregation.
        Args:
            x: Input tensor of shape (BS, C, T, H, W) where S is the number of segments
            orig_shape: Original shape tuple (B, S, C, T, H, W)
        Returns:
            Tuple of (features, mask) where features are of shape (B*S, D, t, h, w)
        """
        x, x_mask = self.forward_features(x)
        
        assert self.extract_features

        # (BS, T, D) where T = 1 + (224 // 16) * (224 // 16) * 8
        x = x[:, 1:, :]  # without the CLS token for efficiency
        x = self.norm(x)
        x = self.pre_logits(x)
        
        if self.factorize_space_time:
            x = self.restore_spatio_temp_dims(x, orig_shape)  # (B*S, D, t, h, w) <- (B*S, t*h*w, D)
        return x, x_mask
    
    def spatiotemporal_aggregation(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial-temporal aggregation to features.
        Args:
            x: Features tensor of shape (B*S, D, t, h, w)
            x_mask: Mask tensor
        Returns:
            Aggregated features of shape (B*S, D) or (B*S, t, D)
        """
        if self.factorize_space_time:
            x = self.spatial_attn_agg(x, x_mask)  # (B*S, t, D)
            x = self.temp_attn_agg(x)  # (B*S, D) or (BS, t, D) if `self.temp_attn_agg` is `Identity`
        return x



class TextSynchformer(nn.Module):

    def __init__(self, video: bool = True, audio: bool = False,
                 text_dim: int = 1024, max_text_seq_len: int = 512, xattn_depth: int = 1):
        super().__init__()
        
        self.video = video
        self.audio = audio
        self.text_dim = text_dim
        self.max_text_seq_len = max_text_seq_len
        
        if not video and not audio:
            raise ValueError('At least one of video or audio should be True.')

        # Use ExtendedMotionFormer directly instead of inheriting from Synchformer
        if self.video:
            self.vfeat_extractor = ExtendedMotionFormer(
                extract_features=True,
                factorize_space_time=True,
                agg_space_module='TransformerEncoderLayer',
                agg_time_module='torch.nn.Identity',
                add_global_repr=False
            )
            
        if self.audio:
            self.afeat_extractor = AST(
                extract_features=True,
                max_spec_t=66,
                factorize_freq_time=True,
                agg_freq_module='TransformerEncoderLayer',
                agg_time_module='torch.nn.Identity',
                add_global_repr=False
            )
        
        # Get embedding dimensions from the video feature extractor
        if self.video:
            self.embed_dim = self.vfeat_extractor.embed_dim
            self.num_heads = self.vfeat_extractor.num_heads
            self.mlp_ratio = self.vfeat_extractor.mlp_ratio
        else:
            # Default values if no video
            self.embed_dim = 768
            self.num_heads = 12
            self.mlp_ratio = 4
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.embed_dim),
            nn.SiLU(),
            MLP(self.embed_dim, self.embed_dim * 4)
        )
        self.synch_text_cross_blocks = nn.ModuleList([
            MMCrossAttentionBlock(self.embed_dim, self.num_heads, 
                                  mlp_ratio=self.mlp_ratio, 
                                  kernel_size=1, padding=0,
                                  residual=True)
            for _ in range(xattn_depth)
        ])
    
    def initialize_weights(self):
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        self.text_proj.apply(_basic_init)
        self.synch_text_cross_blocks.apply(_basic_init)
        for block in self.synch_text_cross_blocks:
            nn.init.constant_(block.norm1.weight, 0.0)
            nn.init.constant_(block.norm1.bias, 0.0)
            nn.init.constant_(block.ffn.w2.weight, 0.0)
            
    
    def forward(self, data, text_features):
        video, audio = None, None
        
        if self.video and self.audio:
            video, audio = data
        elif self.video:
            video = data
        elif self.audio:
            audio = data
            
        if self.video and video is not None:
            video = self.forward_vfeat(video, text_features)
        if self.audio and audio is not None:
            audio = self.forward_afeat(audio, text_features)
        
        if self.video and self.audio:
            return video, audio
        elif self.video:
            return video
        else:
            return audio
    
    def forward_vfeat(self, vis, text_f, text_mask):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        
        # Flatten for processing
        orig_shape = (B, S, C, Tv, H, W)
        vis = einops.rearrange(vis, 'B S C Tv H W -> (B S) C Tv H W') # vis.view(B * S, C, Tv, H, W)
        vis, vis_mask = self.vfeat_extractor.forward_segments_without_aggregation(
            vis, orig_shape # B*S D t h w , BS t h w
        )
        
        text_f = self.text_proj(text_f)  # (B, text_dim) -> (B, embed_dim)
        
        BS, D, t, h, w = vis.shape
        vis = einops.rearrange(vis, '(B S) D t h w -> B (S t h w) D', B=B, S=S)
        vis_mask = einops.rearrange(vis_mask, '(B S) t h w -> B (S t h w)', B=B, S=S) \
            if vis_mask is not None else None
        for block in self.synch_text_cross_blocks:
            vis = block(vis, text_f, rot=None, x_mask=vis_mask, context_mask=text_mask)
        
        vis = einops.rearrange(vis, 'B (S t h w) D -> (B S) D t h w', B=B, S=S, D=D, t=t, h=h, w=w)
        vis_mask = einops.rearrange(vis_mask, 'B (S t h w) -> (B S) t h w', B=B, S=S, t=t, h=h, w=w) \
            if vis_mask is not None else None
        vis = self.vfeat_extractor.spatiotemporal_aggregation(
            vis, vis_mask
        )
        vis = vis.view(B, S, *vis.shape[1:])
        
        return vis
    
    def forward_afeat(self, aud):
        """Forward audio features."""
        raise NotImplementedError("Audio feature extraction is not implemented in TextSynchformer.")
        # B, S, F, Ta = aud.shape
        # aud = aud.permute(0, 1, 3, 2)  # (B, S, Ta, F)
        # aud, _ = self.afeat_extractor(aud)
        # return aud
    
    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        target_keys = (['vfeat_extractor'] if self.video else []) \
                    + (['afeat_extractor'] if self.audio else []) \
                    + ['text_proj', 'synch_text_cross_blocks']
        # discard all entries except vfeat_extractor / afeat_extractor
        sd = {k: v for k, v in sd.items() if any(k.startswith(tk) 
                                                 for tk in target_keys)}
        

        return super().load_state_dict(sd, strict)
    