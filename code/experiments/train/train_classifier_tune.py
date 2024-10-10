import os
import wandb
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from peft import PeftModel
from transformers.trainer import (
    TRAINING_ARGS_NAME,
    logger,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from torch.distributions import Categorical, kl_divergence
from transformers.modeling_utils import unwrap_model
import torch
from datasets import Dataset
from src import (AcceleratorState,
                 create_model, 
                 create_tokenizer,
                 get_lora_model,
                 get_classifier_head,
                 LabeledStringDataCollator,
                 CT_PROMPT,
                 LING_PROMPT,
                 NUMBER_PROMPT,
                 entrypoint,
                 get_token_vec)

from datasets import load_dataset

import pandas as pd 
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes.autograd._functions")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WandbConfigUpdateCallback(TrainerCallback):
    def __init__(self, **config):
        self._config = config

    def on_train_begin(self, _args, state, _control, **_):
        if state.is_world_process_zero:
            wandb.config.update(self._config, allow_val_change=True)

            del self._config
            
            
class ClassificationTuner(Trainer):
    WEIGHTS_NAME = "classifier_model.bin"

    @dataclass
    class Args(TrainingArguments):
        fp16: bool = field(default=not torch.cuda.is_bf16_supported())
        bf16: bool = field(default=torch.cuda.is_bf16_supported())
        ddp_find_unused_parameters: bool = field(default=False)
        log_on_each_node: bool = field(default=False)
        evaluation_strategy: str = field(default="steps")
        dataloader_num_workers: int = field(default=4)
        optim: str = field(default="adamw_torch")
        lr: float = field(default=1e-4)
        lr_scheduler_type: str = field(default="cosine")
        weight_decay: float = field(default=0.0)
        warmup_ratio: float = field(default=0.0)
        gradient_accumulation_steps: int = field(default=1)
        report_to: str = field(default="wandb")
        ## Custom Args.
        target_layer: int = field(default=-1)
        with_lora: bool = field(default=False)

    def __init__(
        self,
        args=None,
        train_dataset=None,
        tokenizer=None,
        classifier_model=None,
        **kwargs,
    ):
        args.label_names = train_dataset.column_names

        self._collate_fn = LabeledStringDataCollator(tokenizer)

        self.classifier_model = classifier_model

        super().__init__(
            **kwargs,
            args=args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=default_collate,
        )

    def _wrap_model(self, *args, **kwargs):
        if unwrap_model(self.classifier_model) is self.classifier_model:
            self.classifier_model = self.accelerator.prepare(self.classifier_model)

        return super()._wrap_model(*args, **kwargs)

    def prepare_inputs(self, model, inputs):
        prompts = inputs['prompt']
        targets = inputs['y']
        predictions = inputs['y_pred']
        q_labels = torch.tensor(inputs['c']).long()
        q_labels = q_labels.to(self.accelerator.device)

        return prompts, targets, predictions, q_labels

    def prepare_class_inputs(
        self, model, inputs, targets, predictions, class_labels, eval_mode=False
    ):
        
        class_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(inputs, predictions).items()
        }
        #import ipdb; ipdb.set_trace()
        inference_mode = (not self.args.with_lora) or eval_mode

        with torch.inference_mode(inference_mode):
            class_inputs = model(**class_inputs, output_hidden_states=True)
            class_inputs = class_inputs.hidden_states[self.args.target_layer]
            class_inputs = class_inputs[..., -1, :]
        if inference_mode:
            class_inputs = class_inputs.clone()

        return class_inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs, targets, predictions, class_labels = self.prepare_inputs(model, inputs)

        class_inputs = self.prepare_class_inputs(
            model, inputs, targets, predictions, class_labels
        )

        class_logits = self.classifier_model(class_inputs)

        loss = F.cross_entropy(class_logits, class_labels)

        loss_metrics = {
            "loss": loss.detach().item(),
        }

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        return (loss, None) if return_outputs else loss

    @torch.inference_mode
    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_labels, all_logits = [], []

        for inputs in tqdm(eval_dataloader, leave=False):
            inputs, targets, predictions, class_labels = self.prepare_inputs(
                self.model, inputs
            )

            class_inputs = self.prepare_class_inputs(
                self.model, inputs, targets, predictions, class_labels, eval_mode=True
            )

            class_logits = self.classifier_model(class_inputs)

            [
                l.append(v)
                for l, v in zip(
                    (all_labels, all_logits),
                    self.accelerator.gather_for_metrics((class_labels, class_logits)),
                )
            ]

        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        metrics = {
            f"{metric_key_prefix}_N": all_labels.size(0),
            f"{metric_key_prefix}_acc": (all_logits.argmax(dim=-1) == all_labels)
            .float()
            .mean()
            .item(),
            f"{metric_key_prefix}_loss": F.cross_entropy(all_logits, all_labels).item(),
        }

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            selected_adapters=["default"],
            save_embedding_layers=False,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        torch.save(
            unwrap_model(self.classifier_model).state_dict(),
            os.path.join(output_dir, self.WEIGHTS_NAME),
        )


@entrypoint
def main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style=None,
    max_token_length=None,
    num_workers=4,
    use_dataset_cache=True,
    model_name=None,
    int8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    peft_dir=None,
    with_lora=False,
    batch_size=4,
    warmup_ratio=0.0,
    lr=1e-4,
    max_steps=5000,
    gradient_accumulation_steps=1,
):
    accelerator = AcceleratorState()

    trainer_args = ClassificationTuner.Args(
        seed=seed,
        output_dir=log_dir,
        max_steps=max_steps,
        eval_steps=max_steps // 10,
        save_steps=max_steps // 10,
        logging_steps=max(1, max_steps // 200),
        dataloader_num_workers=num_workers,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        with_lora=with_lora,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    with accelerator.main_process_first():
        train_data = pd.read_csv("./final_data/dev/ct_train.csv")
        valid_data = pd.read_csv("./final_data/dev/ct_valid.csv")

        train_data = Dataset.from_pandas(train_data)
        valid_data = Dataset.from_pandas(valid_data)
        
    tokenizer = create_tokenizer(model_name)
    
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map={"": accelerator.local_process_index}
        )

    model = get_lora_model(
        model,
        peft_id_or_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=with_lora,
        adapter_name="default",
    )

    classifier_model = get_classifier_head(
        input_size=model.config.hidden_size,
        checkpoint_dir=peft_dir,
        is_trainable=True,
        weights_name=ClassificationTuner.WEIGHTS_NAME,
    )

    model.classifier_model = classifier_model.to(model.dtype)

    trainer = ClassificationTuner(
        model=model,
        classifier_model=classifier_model,
        args=trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        callbacks=[
            WandbConfigUpdateCallback(
                dataset=dataset,
                prompt_style=prompt_style,
                max_token_length=max_token_length,
                model_name=model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                peft_dir=peft_dir,
            ),
        ],
    )
    trainer.train()


if __name__ == "__main__":
    import fire

    fire.Fire(main)