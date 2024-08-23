import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import argparse
from datetime import datetime
from transformers import TrainingArguments
from trainer import Stage2Trainer, collate_fn
from models.aim import AiM_models
from util.data import build_dataset


def create_training_arguments(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.decay,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        evaluation_strategy=args.eval_strategy,
        bf16=True,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        lr_scheduler_kwargs={'min_lr': args.min_lr} if args.scheduler != 'linear' else None,
    )
    return training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aim-model", type=str, choices=["AiM-B", "AiM-L", "AiM-XL", "AiM-1B"], help="AiM models")
    parser.add_argument("--aim-ckpt", type=str, default=None, help="checkpoint path")
    parser.add_argument("--vq-model", type=str, default="VQ-f16", choices=["VQ-f16"], help="VQ models")
    parser.add_argument("--vq-ckpt", type=str, help="checkpoint path")
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument("--output-dir", type=str, default='./checkpoints', help="output root directory")
    parser.add_argument("--resume-dir", type=str, help="resume directory")
    
    parser.add_argument("--epochs", type=int, default=300, help="total epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--num-workers", type=int, default=24, help="dataloader num workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--min-lr", type=float, default=None, help="min learning rate")
    parser.add_argument("--decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="adam beta2")
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="warmup ratio")
    parser.add_argument("--scheduler", type=str, default='linear', choices=['linear', 'cosine_with_min_lr'], help="lr scheduler")

    parser.add_argument("--save-total-limit", type=int, default=1, help="save total limit")
    parser.add_argument("--save-strategy", type=str, default='steps', choices=['steps', 'epochs'], help="save strategy")
    parser.add_argument("--eval-strategy", type=str, default='no', choices=['no', 'steps', 'epochs'], help="eval strategy")

    args = parser.parse_args()

    # setting output path
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.output_dir = os.path.join(args.output_dir, f"{args.aim_model}_{current_time}")
    
    # create model and dataset
    model = AiM_models[args.aim_model]()
    model.vqvae.load_state_dict(torch.load(args.vq_ckpt))
    if args.aim_ckpt is not None:
        model.mamba.load_state_dict(torch.load(args.aim_ckpt))
        
    train_data, eval_data = build_dataset(args.dataset, norm=True)

    # create trainer and run
    trainer = Stage2Trainer(
        model,
        create_training_arguments(args),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=args.resume_dir)