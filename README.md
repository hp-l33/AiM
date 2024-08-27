# AiM: Scalable Autoregressive Image Generation with Mamba🐍

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2408.12245-b31b1b.svg)](https://arxiv.org/abs/2408.12245)&nbsp;
[![weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-hp--l33/aim-yellow)](https://huggingface.co/collections/hp-l33/aim-66cd87744764acddd30ce80a)&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rrulJmMDTi3dJrgHGhzeEjbS4pOnZ5vV?usp=sharing)

</div>

<p align="center">
<img src="figure/title.png" width=95%>
<p>

<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2408.12245">Scalable Autoregressive Image Generation with Mamba</a>
</p>

## 💡 What is AiM
The first (as far as we know) Mamba 🐍 based autoregressive image generation model, offering competitive generation quality 💪 with diffusion models and faster inference speed ⚡️.

We also propose a more general form of adaLN, called **adaLN-group**, which balances parameter count and performance ⚖️. Notably, adaLN-group can be flexibly converted to adaLN and adaLN-single equivalently.


## 🔔 Update
* [2024-08-27] Improved HF integration, now supports `from_pretrained` for direct model loading.
* [2024-08-23] A minor bug in ``train_stage2.py`` has been fixed.
* [2024-08-23] Code and Model Release.


## 🚀 Getting Started
### Train


```
accelerate launch --num_processes=32 --num_machines=... --main_process_ip=... --main_process_port=... --machine_rank=... train_stage2.py --aim-model AiM-XL --dataset /your/data/path/ --vq-ckpt /your/ckpt/path/vq_f16.pt --batch-size 64 --lr 8e-4 --epochs 350
```

### Inference
You can play with AiM in the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rrulJmMDTi3dJrgHGhzeEjbS4pOnZ5vV?usp=sharing) or:
```
from aim import AiM

model = AiM.from_pretrained("hp-l33/aim-xlarge").cuda()
model.eval()

imgs = model.generate(batch=8, temperature=1, top_p=0.98, top_k=600, cfg_scale=5)
```
The first time Mamba runs, it will invoke the triton compiler and autotune, so it may be slow. From the second run onwards, the inference speed will be very fast. See:
> https://github.com/state-spaces/mamba/issues/389#issuecomment-2171755306

## 🤗 Model Zoo
The model weights can be downloaded from the [![weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-hp--l33/aim-yellow)](https://huggingface.co/collections/hp-l33/aim-66cd87744764acddd30ce80a).


Model | params | FID | weight 
--- |:---:|:---:|:---:|
AiM-B   | 148M | 3.52 | [aim-base](https://huggingface.co/hp-l33/aim-base)
AiM-L   | 350M | 2.83 | [aim-large](https://huggingface.co/hp-l33/aim-large)
AiM-XL  | 763M | 2.56 | [aim-xlarge](https://huggingface.co/hp-l33/aim-xlarge)


## 🌹 Acknowledgments
This project would not have been possible without the computational resources provided by Professor [Guoqi Li](https://casialiguoqi.github.io) and his team. We would also like to thank the following repositories and papers for their inspiration:
* [VQGAN](https://github.com/CompVis/taming-transformers)
* [Mamba](https://github.com/state-spaces/mamba)
* [LlamaGen](https://github.com/FoundationVision/LlamaGen)
* [VAR](https://github.com/FoundationVision/VAR)
* [DiT](https://github.com/facebookresearch/DiT)



## 📖 BibTeX
```
@misc{li2024scalableautoregressiveimagegeneration,
      title={Scalable Autoregressive Image Generation with Mamba}, 
      author={Haopeng Li and Jinyue Yang and Kexin Wang and Xuerui Qiu and Yuhong Chou and Xin Li and Guoqi Li},
      year={2024},
      eprint={2408.12245},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.12245}, 
}
```