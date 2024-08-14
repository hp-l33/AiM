# AiM: Scalable Autoregressive Image Generation with Mamba🐍

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2024.0814-b31b1b.svg)](https://arxiv.org/abs/)&nbsp;
[![project page](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-hp--l33/aim-yellow)](https://huggingface.co/hp-l33/aim)&nbsp;

</div>

<p align="center">
<img src="figure/title.png" width=95%>
<p>
Autoregressive Image Generation via Mamba🐍

## Update
* [2040-08-20] Code and Model Release


## Getting Started
### Training Scripts
```
accelerate launch \
--num_processes=32 \
--num_machines=... \
--main_process_ip=... \
--main_process_port=... \
--machine_rank=... \
train_stage2.py \
--config ./config/aim-xl-imagenet256.yaml \
--run-name ... \
--batch-size 64 \
--lr 8e-4 \
--epochs 350
```

## Model Zoo
We will release the model weights soon