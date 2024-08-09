# AiM: Scalable Autoregressive Image Generation with Mamba🐍
<p align="center">
<img src="figure/title.png" width=95%>
<p>
Autoregressive Image Generation via Mamba🐍

## Update
* [2040-08-20] release code


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