import transformers
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split

IGNORE_LABEL = -100

def get_token_vec(tokenizer, c_type="number"):
    vocab = tokenizer.get_vocab()

    def _create_vec(raw_list):
        for t in raw_list:
            assert t in vocab, f"Cannot handle {t} as a single token."

        return torch.tensor([tokenizer(t).input_ids[-1] for t in raw_list])

    if c_type == "ct":
        raw_strings = ["i", "ii"]
        
    elif c_type == "number":
        raw_strings=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    
    elif c_type == "ling":
        raw_strings = ["Unlikely", "Doubtful", "Uncertain", "Ambiguous", 
                       "Probable", "Likely", "Possible", "Specified",
                       "Confirmed", "Certain", "Inevitable"]
    else:
        raise NotImplementedError
    return _create_vec(raw_strings)


@dataclass
class LabeledStringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    base_prompt: str = "You are a helpful assistant."

    @staticmethod
    def get_tokenizer_args(tokenizer):
        return dict(
            padding=True,
            truncation=True,
            max_length=(
                tokenizer.model_max_length
                if hasattr(tokenizer, "model_max_length")
                else None
            ),
            return_tensors="pt",
            return_length=True,
        )

    def __call__(self, prompts, targets=None):
        tokenizer_args = self.get_tokenizer_args(self.tokenizer)
        
        if (
            self.tokenizer.name_or_path
            and ("Llama-3" in self.tokenizer.name_or_path)
            and ("Instruct" in self.tokenizer.name_or_path)
        ):
            msgs = [
                [
                    {"role": "system", "content": self.base_prompt},
                    {"role": "user", "content": p},
                ]
                for p in prompts
            ]

            prompts = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in msgs
            ]
        
        if targets:
            all_prompts = [p + t for p, t in zip(prompts, targets)]
        else:
            all_prompts = prompts
        
        inputs = self.tokenizer(all_prompts, **tokenizer_args)
        input_lengths = inputs.pop("length")

        if targets:
            un_inputs = self.tokenizer(prompts, **tokenizer_args)
            un_input_lengths = un_inputs.pop("length")

            labels = inputs.get("input_ids").clone()
            for i, l in enumerate(input_lengths - un_input_lengths):
                labels[i, :-l] = IGNORE_LABEL
            inputs["labels"] = labels
        return inputs


def train_test_split(dataset, test_size=0.2, seed=None):
    N = len(dataset)
    N_test = int(test_size * N)
    N -= N_test

    if seed is not None:
        train, test = random_split(
            dataset, [N, N_test], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train, test = random_split(dataset, [N, N_test])

    return train, test


def get_num_workers(num_workers=4):
    num_gpus_per_host = torch.cuda.device_count()
    if num_gpus_per_host == 0:
        return num_workers
    return (num_workers + num_gpus_per_host - 1) // num_gpus_per_host


def get_loader(dataset, batch_size=128, num_workers=4, accelerator=None, **kwargs):
    num_workers = get_num_workers(num_workers=num_workers)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
    )
    if accelerator is not None:
        loader = accelerator.prepare(loader)

    return loader
