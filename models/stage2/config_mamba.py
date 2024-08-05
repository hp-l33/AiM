from dataclasses import dataclass, field


@dataclass
class MambaConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    n_layer: int = 48
    vocab_size: int = 16384
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    num_classes: int = 1000
    num_img_tokens: int = 256
    adaln_single: bool = False
    token_drop: float = 0.0
    mixer_drop: float = 0.0
    mlp_drop: float = 0.0