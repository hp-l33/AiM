import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
from omegaconf import OmegaConf
from transformers import TrainingArguments
from trainer import Stage2Trainer, collate_fn
from util.helper import naming_experiment_with_time, instantiate_from_config


def create_model(config):
    model_config = config['model'] if 'model' in config else config
    model = instantiate_from_config(model_config)
    return model


def create_datasets(config):
    data_config = config['data'] if 'data' in config else config
    train_data, eval_data = instantiate_from_config(data_config)
    return train_data, eval_data


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
    parser.add_argument("--run-name", type=str, help="experiment name.")
    parser.add_argument("--output-dir", type=str, default='./checkpoints', help="output root directory. The full directory format is <output-dir/run-name>.")
    parser.add_argument("--result-dir", type=str, default='./results', help="result root directory. The full directory format is <results-dir/run-name>.")
    parser.add_argument("--resume-dir", type=str, help="resume directory")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path")
    # train parameters
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
    # save strategy
    parser.add_argument("--save-total-limit", type=int, default=1, help="save total limit")
    parser.add_argument("--save-strategy", type=str, default='steps', choices=['steps', 'epochs'], help="save strategy")
    parser.add_argument("--eval-strategy", type=str, default='no', choices=['no', 'steps', 'epochs'], help="eval strategy")
    # config
    parser.add_argument("--config", type=str, help="config path")
    args = parser.parse_args()

    # setting output path
    if args.run_name is None:
        args.run_name = naming_experiment_with_time()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.result_dir = os.path.join(args.result_dir, args.run_name)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # create model and dataset from config
    config = OmegaConf.load(args.config)
    model = create_model(config)
    train_data, eval_data = create_datasets(config)

    # create trainer and run
    trainer = Stage2Trainer(
        model,
        create_training_arguments(args),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=args.resume_dir)