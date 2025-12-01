from typing import Optional
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from selva.ext.rotary_embeddings import apply_rope
from selva.model.low_level import MLP, ChannelLastConv1d, ConvMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if attn_mask is not None:
        attn_mask = attn_mask.contiguous()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    b, h, n, d_head = out.shape
    out = out.permute(0, 2, 1, 3)  # Shape: (b, n, h, d_head)
    # Using reshape, which can handle non-contiguous tensors by copying if necessary
    out = out.reshape(b, n, h * d_head) # Shape: (b, n, h * d_head)
    # Ensure the final output is contiguous, similar to the original code's intent
    out = out.contiguous()
    return out

def attention_debug(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              attn_mask: Optional[torch.Tensor] = None,
              layer_idx: int = -1) -> None:
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if attn_mask is not None:
        attn_mask = attn_mask.contiguous()
    # out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    
    # debug attn map
    import math
    scale_factor = 1 / math.sqrt(q.size(-1))
    L, S = q.size(-2), k.size(-2)
    attn_bias = torch.zeros(q.shape[0], q.shape[1], L, S, dtype=q.dtype, device=q.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    torch.save(attn_weight.clone().cpu(), f'./debug_attn_weight_layer{layer_idx}_unnorm.pt')
    # normalize
    attn_weight = torch.softmax(attn_weight, dim=-1)
    torch.save(attn_weight.clone().cpu(), f'./debug_attn_weight_layer{layer_idx}.pt')


def create_mask(q_shape, k_shape, device, q_mask=None, k_mask=None):
    def default(val, d):
        return val if val is not None else (d() if isfunction(d) else d)
    b, i, j, device = q_shape[0], q_shape[-2], k_shape[-2], device
    q_mask = default(q_mask, torch.ones((b, i), device=device, dtype=torch.bool))
    k_mask = default(k_mask, torch.ones((b, j), device=device, dtype=torch.bool))
    attn_mask = rearrange(q_mask, 'b i -> b 1 i 1') * rearrange(k_mask, 'b j -> b 1 1 j')
    return attn_mask


class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)

    def pre_attention(
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
            q_mask: Optional[torch.Tensor] = None,
            k_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, v, k = self.pre_attention(x)
        if q_mask is not None or k_mask is not None:
            attn_mask = create_mask(q.shape, k.shape, q.device, 
                                    q_mask=q_mask, k_mask=k_mask)
        else:
            attn_mask = None
        out = attention(q, k, v, attn_mask)
        return out


class CrossAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        """
        Args:
            dim (int): Input dimension.
            nheads (int): Number of attention heads.
            
        Attributes:
            q_proj (Linear): Linear transformation for the query.
            kv_proj (Linear): Linear transformation for the key and value.
            q_norm (RMSNorm): Layer normalization for the query.
            k_norm (RMSNorm): Layer normalization for the key.
            split_into_heads (Rearrange): Rearrange layer to split the input into heads.
        """
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_q_into_heads = Rearrange('b n (h d) -> b h n d',
                          h=nheads,
                          d=dim // nheads)
        self.split_kv_into_heads = Rearrange('b n (h d j) -> b h n d j',
                          h=nheads,
                          d=dim // nheads,
                          j=2)

    def pre_attention(
            self, x: torch.Tensor, c: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        # c: batch_size * n_cond_tokens * n_channels
        q = self.q_proj(x)
        kv = self.kv_proj(c)
        q = self.split_q_into_heads(q)
        k, v = self.split_kv_into_heads(kv).chunk(2, dim=-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if rot is not None:
            q = apply_rope(q, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
            c: torch.Tensor,  # batch_size * n_cond_tokens * n_channels
            context_mask: Optional[torch.Tensor] = None,
            rot: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        q, k, v = self.pre_attention(x, c, rot)
        if context_mask is not None:
            attn_mask = create_mask(q.shape, k.shape, q.device, k_mask=context_mask)
        else:
            attn_mask = None
        out = attention(q, k, v, attn_mask)
        return out


class MMCrossAttentionBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                #  pre_only: bool = False,
                 kernel_size: int = 7,
                 padding: int = 3,
                 residual: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn = CrossAttention(dim, nhead)

        if kernel_size == 1:
            self.linear1 = nn.Linear(dim, dim)
        else:
            self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)

        if kernel_size == 1:
            self.ffn = MLP(dim, int(dim * mlp_ratio))
        else:
            self.ffn = ConvMLP(dim,
                                int(dim * mlp_ratio),
                                kernel_size=kernel_size,
                                padding=padding)

        self.residual = residual

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        # x: BS * N * D
        # cond: BS * D
        
        # if self.pre_only:
        #     (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
        #     gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        # else:
        #     (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
        #      gate_mlp) = modulation.chunk(6, dim=-1)

        # x = self.norm1(x)
        q, k, v = self.attn.pre_attention(x, c, rot)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor):
        # if self.pre_only:
        #     return x

        # (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        if self.residual:
            x = x + self.norm1(self.linear1(attn_out)) # * gate_msa
            # https://github.com/haidog-yaqub/EzAudio/blob/2eb0bd90013584c6e28a6c14ec28b935f1e78de5/src/models/blocks.py#L158
            # https://github.com/huggingface/diffusers/blob/07dd6f8c0e267662f62c39cd8334c2b5d157ab39/src/diffusers/models/transformers/transformer_flux.py#L170
            # https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/attention.py#L274
        else:
            x = self.norm1(self.linear1(attn_out))
        r = self.norm2(x)
        x = x + self.ffn(r)

        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor], 
                x_mask: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        q, k, v = self.pre_attention(x, cond, rot)
        if x_mask is not None or context_mask is not None:
            attn_mask = create_mask(q.shape, k.shape, q.device, q_mask=x_mask, k_mask=context_mask)
        else:
            attn_mask = None
        attn_out = attention(q, k, v, attn_mask=attn_mask)
        x = self.post_attention(x, attn_out)

        return x


class MMDitSingleBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 pre_only: bool = False,
                 kernel_size: int = 7,
                 padding: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)

        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(dim,
                                   int(dim * mlp_ratio),
                                   kernel_size=kernel_size,
                                   padding=padding)

            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp

        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor]) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)

        return x


class JointBlock(nn.Module):

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False):
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(dim,
                                             nhead,
                                             mlp_ratio,
                                             pre_only=False,
                                             kernel_size=3,
                                             padding=1)
        self.clip_block = MMDitSingleBlock(dim,
                                           nhead,
                                           mlp_ratio,
                                           pre_only=pre_only,
                                           kernel_size=3,
                                           padding=1)
        self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)

    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, text_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, latent_rot: torch.Tensor,
                clip_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # latent: BS * N1 * D
        # clip_f: BS * N2 * D
        # c: BS * (1/N) * D
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

        latent_len = latent.shape[1]
        clip_len = clip_f.shape[1]
        text_len = text_f.shape[1]

        joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]

        attn_out = attention(*joint_qkv)
        x_attn_out = attn_out[:, :latent_len]
        c_attn_out = attn_out[:, latent_len:latent_len + clip_len]
        t_attn_out = attn_out[:, latent_len + clip_len:]

        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
        if not self.pre_only:
            clip_f = self.clip_block.post_attention(clip_f, c_attn_out, c_mod)
            text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)

        return latent, clip_f, text_f
    
    def forward_debug(self, latent: torch.Tensor, clip_f: torch.Tensor, text_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, latent_rot: torch.Tensor,
                clip_rot: torch.Tensor,
                layer_idx: int = -1,
                ) -> None:
        # latent: BS * N1 * D
        # clip_f: BS * N2 * D
        # c: BS * (1/N) * D
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

        latent_len = latent.shape[1]
        clip_len = clip_f.shape[1]
        text_len = text_f.shape[1]

        joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]

        attn_out = attention_debug(*joint_qkv, layer_idx=layer_idx)
        return None


class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent
