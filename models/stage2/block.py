# References:
#   Mamba:  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
#   VAR:    https://github.com/FoundationVision/VAR/blob/main/models/var.py

from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
 

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
        residual_in_fp32=False, adaln_group=False, mixer_drop=0.0, mlp_drop=0.0
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)

        # modify
        self.mixer_dropout = nn.Dropout(mixer_drop)
        self.adaln_group = adaln_group
        self.adaln_factor = 3   # alpha, beta, gamma

        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
            self.adaln_factor += 3
            self.mlp_dropout = nn.Dropout(0.0)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        
        # adaLN
        if adaln_group:
            self.scale_shift_table = nn.Parameter(torch.randn(1, self.adaln_factor, dim) / dim**0.5)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, self.adaln_factor * dim, bias=True)
            )
            # zero-out
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, cls_embed=None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        # adaLN
        if self.adaln_group:
            scale_shift_params = (self.scale_shift_table + cls_embed).unbind(1)
        else:
            scale_shift_params = self.adaLN_modulation(cls_embed).chunk(self.adaln_factor, dim=1)

        if self.adaln_factor == 3:
            shift_mixer, scale_mixer, gate_mixer = scale_shift_params
        elif self.adaln_factor == 6:
            shift_mixer, shift_mlp, scale_mixer, scale_mlp, gate_mixer, gate_mlp = scale_shift_params
        else:
            raise ValueError("Unsupported adaln_factor value")
        
        # hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        hidden_states = self.mixer_dropout(
            gate_mixer.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_mixer, scale_mixer),
                inference_params=inference_params,
                **mixer_kwargs
            )
        )

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            # hidden_states = self.mlp(hidden_states)
            hidden_states = self.mlp_dropout(
                gate_mlp.unsqueeze(1) * self.mlp(
                    modulate(hidden_states, shift_mlp, scale_mlp)
                )
            )

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)