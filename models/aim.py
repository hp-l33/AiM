import torch
import torch.nn as nn
from util.helper import instantiate_from_config
from .stage2.config_mamba import MambaConfig
from .stage2.mixer_seq_simple import MambaLMHeadModel
from .stage1.vq_model import VQ_models

from huggingface_hub import PyTorchModelHubMixin


class AiM(nn.Module, PyTorchModelHubMixin, repo_url="https://github.com/hp-l33/AiM", pipeline_tag="unconditional-image-generation", license="mit"):
    def __init__(self, config: MambaConfig):
        super().__init__()
        # init all models
        self.vqvae = self.init_1st_stage_model()
        self.mamba = self.init_2nd_stage_model(config)
        
        self.num_classes = config.num_classes
        self.num_tokens = config.num_tokens
        self.num_embed_dim = self.vqvae.config.embed_dim

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def init_1st_stage_model(self):
        model = VQ_models['VQ-f16']()
        model.eval()
        [p.requires_grad_(False) for p in model.parameters()]
        return model

    def init_2nd_stage_model(self, config):
        model = MambaLMHeadModel(config)
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

        max_length = self.num_tokens + sos_token.shape[1]
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
    def generate(self, c=None, batch=4, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=5.0, fast=True):
        if c is None:
            c = torch.randint(self.num_classes, (batch, 1), device=self.mamba.lm_head.weight.device)
        else:
            batch = c.shape[0]

        self.mamba.cfg_scale = cfg_scale
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
     
    
def AiM_B(**kwargs):
    return AiM(MambaConfig(d_model=768, n_layer=24, adaln_group=False, **kwargs))


def AiM_L(**kwargs):
    return AiM(MambaConfig(d_model=1024, n_layer=48, adaln_group=True, num_groups=4, **kwargs))


def AiM_XL(**kwargs):
    return AiM(MambaConfig(d_model=1536, n_layer=48, adaln_group=True, num_groups=4, **kwargs))


AiM_models = {'AiM-B': AiM_B, 'AiM-L': AiM_L, 'AiM-XL': AiM_XL}