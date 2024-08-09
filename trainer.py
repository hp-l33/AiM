import os
import torch
from torchvision import utils as vutils
from transformers import Trainer, TrainerCallback


def collate_fn(examples):
    inputs = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"inputs":inputs, "labels":labels}


class Stage2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        logits, target = model(inputs['inputs'], inputs['labels'])
        logits = logits.view(-1, logits.shape[-1])
        target = target.view(-1)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, target)

        return (loss, None) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        loss_accumulator = 0.0

        self.model.eval()
        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)
            with torch.no_grad():
                loss = self.compute_loss(self.model, batch)
                loss_accumulator += loss.item()

        avg_loss = loss_accumulator / len(eval_dataloader)
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        self.log(metrics)

        return metrics

    def get_decay_parameter_names(self, model):
        if hasattr(model, 'mamba'):
            param_dict = {pn: p for pn, p in model.mamba.named_parameters()}    
        else:
            # decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
            # decay_parameters = [name for name in decay_parameters if "bias" not in name]
            param_dict = {pn: p for pn, p in model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_parameters = [n for n, p in param_dict.items() if p.dim() >= 2]
        return decay_parameters