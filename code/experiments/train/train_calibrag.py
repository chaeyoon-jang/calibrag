import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd 
from tqdm.auto import tqdm
from datasets import Dataset
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from transformers import set_seed 

from transformers.trainer import (
    TRAINING_ARGS_NAME,
    logger,
    Trainer,
    TrainingArguments
)
from transformers.modeling_utils import unwrap_model

from src.logging import (
    entrypoint,
    WandbConfigUpdateCallback
)
from src.distributed import AcceleratorState
from src.llm_model_utils import (
    create_model,
    create_tokenizer
)
from src.peft_utils import (
    get_lora_model,
    get_classifier_head
)
from src.generate_utils import (
    LabeledStringDataCollator
)
 
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
        target_layer: int = field(default=-2)
        with_lora: bool = field(default=False)

    def __init__(
        self,
        args=None,
        train_dataset=None,
        tokenizer=None,
        classifier_model=None,
        input_prediction=False,
        **kwargs,
    ):
        args.label_names = train_dataset.column_names
        
        self.input_prediction = input_prediction
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

    def prepare_class_inputs(
        self, model, inputs, eval_mode=False
    ):
        class_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(inputs).items()}
            
        inference_mode = (not self.args.with_lora) or eval_mode

        with torch.inference_mode(inference_mode):
            class_inputs = model(**class_inputs, output_hidden_states=True)
            class_inputs = class_inputs.hidden_states[self.args.target_layer]
            class_inputs = class_inputs[..., -1, :]
        if inference_mode:
            class_inputs = class_inputs.clone()

        return class_inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        
        class_labels = inputs['correct']
        inputs = inputs['z_prompt']     
           
        class_inputs = self.prepare_class_inputs(model, inputs)

        class_logits = self.classifier_model(class_inputs)
        loss = F.cross_entropy(class_logits, class_labels)

        loss_metrics = {
            "class_loss": loss.detach().item(),
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
            
            class_labels = inputs['correct']
            inputs = inputs['z_prompt']
            
            class_inputs = self.prepare_class_inputs(
                self.model, inputs, eval_mode=True
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

        all_probs = torch.softmax(all_logits, dim=-1)

        num_classes = all_logits.size(-1)
        one_hot_labels = F.one_hot(all_labels, num_classes=num_classes).float()
        brier_score = torch.mean(torch.sum((all_probs - one_hot_labels) ** 2, dim=1)).item()

        def compute_ece(probs, labels, n_bins=15):
            confidences, predictions = torch.max(probs, 1)
            accuracies = predictions.eq(labels)
            bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
            ece = torch.zeros(1, device=probs.device)
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            return ece.item()

        ece_score = compute_ece(all_probs, all_labels)

        metrics = {
            f"{metric_key_prefix}_N": all_labels.size(0),
            f"{metric_key_prefix}_acc": (all_logits.argmax(dim=-1) == all_labels)
            .float()
            .mean()
            .item(),
            f"{metric_key_prefix}_loss": F.cross_entropy(all_logits, all_labels).item(),
            f"{metric_key_prefix}_brier_score": brier_score,
            f"{metric_key_prefix}_ece": ece_score,
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
    data_dir="./data/processed/calibrag/bm25",
    input_prediction=False,
    max_token_length=None,
    num_workers=4,
    model_name="Meta-Llama-3.1-8B-Instruct",
    int8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    peft_dir=None,
    with_lora=False,
    batch_size=4,
    warmup_ratio=0.0,
    lr=1e-4,
    max_steps=10000,
    gradient_accumulation_steps=1,
):
    
    set_seed(seed)
    
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
        train_data = pd.read_csv(os.path.join(f"{data_dir}", "train.csv"))
        train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        valid_data = pd.read_csv(os.path.join(f"{data_dir}", "valid.csv"))

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
        input_prediction=input_prediction,
        callbacks=[
            WandbConfigUpdateCallback(
                dataset=dataset,
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