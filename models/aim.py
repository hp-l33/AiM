import torch
import torch.nn as nn
from util.helper import instantiate_from_config
from .stage2.config_mamba import MambaConfig
from .stage2.mixer_seq_simple import MambaLMHeadModel


class AiM(nn.Module):
    def __init__(self, stage1_config, stage2_config):
        super().__init__()
        self.num_classes = stage1_config.params.num_classes
        self.num_img_tokens = stage1_config.params.num_img_tokens
        self.num_embed_dim = stage1_config.params.embed_dim
        
        # init all models
        self.vqvae = self.init_1st_stage_model(stage1_config) if stage1_config.target is not None else None
        self.mamba = self.init_2nd_stage_model(stage2_config)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def init_1st_stage_model(self, config):
        model = instantiate_from_config(config)
        model.eval()
        [p.requires_grad_(False) for p in model.parameters()]
        return model

    def init_2nd_stage_model(self, config):
        mamba_config = instantiate_from_config(config)
        model = MambaLMHeadModel(mamba_config)
        return model

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.mamba.parameters())
        if non_embedding:
            n_params -= self.mamba.backbone.embeddings.word_embeddings.weight.numel()
        return n_params

    def forward(self, x, c):
        code = self.encode_to_z(x)[1] if len(x.shape) == 4 else x.squeeze(1)
        cond = self.encode_to_c(c)
        
        target = code

        logits = self.mamba(code[:, :-1], cond=cond).logits
        logits = logits[:, cond.shape[1]-1:]
        
        return logits, target

    @torch.no_grad()
    def sample_cfg(self, sos_token, temperature=1.0, top_k=0, top_p=1.0, fast=True):
        # classifier free guidance
        sos_token = torch.cat([sos_token, torch.full_like(sos_token, self.num_classes)])

        max_length = self.num_img_tokens + sos_token.shape[1]
        x = self.mamba.generate(input_ids=sos_token,
                                cond=sos_token,
                                max_length=max_length,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                cg=fast)
        
        self.mamba._decoding_cache = None
        return x[:, 1:]
    
    @torch.no_grad()
    def generate(self, c=None, batch=4, temperature=1.0, top_k=0, top_p=1.0, fast=True):
        if c is None:
            c = torch.randint(self.num_classes, (batch, 1), device=self.mamba.lm_head.weight.device)
        else:
            batch = c.shape[0]

        sos_tokens = self.encode_to_c(c)
        
        tokens = self.sample_cfg(sos_tokens, temperature=temperature, top_k=top_k, top_p=top_p, fast=fast)[:batch]
        
        shape = (batch, self.num_embed_dim, int(tokens.shape[-1]**0.5), int(tokens.shape[-1]**0.5))
        imgs = self.decode_to_img(tokens, shape)

        return imgs
    
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, log = self.vqvae.encode(x)
        indices = log[-1].view(quant_z.shape[0], -1)
        return quant_z, indices
    
    @torch.no_grad()
    def encode_to_c(self, c):
        sos_tokens = c.view(-1, 1).contiguous()
        return sos_tokens.long()
    
    @torch.no_grad()
    def decode_to_img(self, index, z_shape):
        x = self.vqvae.decode_code(index, shape=z_shape)
        return x
    
    def from_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        for key in list(ckpt):
            if 'vq_vae' in key:
                new_key = 'vqvae.' + '.'.join(key.split('.')[1:])
                ckpt[new_key] = ckpt.pop(key)
            if self.vqvae is None and 'vqvae' in key:
                ckpt.pop(key)
            if 'shared' in key:
                new_key = key.replace("shared_adaln", "adaln_single")
                ckpt[new_key] = ckpt.pop(key)
            if 'adaln_single' in key:
                new_key = key.replace("adaln_single", "adaln_group")
                ckpt[new_key] = ckpt.pop(key)
            if 'para' in key:
                new_key = key.replace("adaLN_parameters", "scale_shift_table")
                ckpt[new_key] = ckpt.pop(key)
            if 'position_embeddings' in key:
                ckpt[key] = (ckpt.pop(key))[:self.num_img_tokens+1:]

        self.load_state_dict(ckpt, strict=False)
        print(f"Restored from {ckpt_path}")
    