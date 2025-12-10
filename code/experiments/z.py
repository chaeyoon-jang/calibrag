import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import GenerationConfig, set_seed
from accelerate import Accelerator

from src.logging import entrypoint
from src.llm_model_utils import create_model, create_tokenizer
from src.generate_utils import generate_outputs
from src.llm_data_utils import get_loader
from src.prompt_hub import Z_PROMPT


@entrypoint(with_accelerator=True, with_wandb=False)
def main(
    seed=0,
    dataset_dir="logs/golden/raw",
    train_file="train.csv",
    valid_file="valid.csv",
    model_name="Meta-Llama-3.1-8B-Instruct",
    int8=True,
    batch_size=8,
    max_new_tokens=50,
    temperature=1.0,
    top_p=1.0,
    log_dir=None,
    accelerator=None,
):

    set_seed(seed)
    if accelerator is None:
        accelerator = Accelerator()

    tokenizer = create_tokenizer(model_name)
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map="auto",
    )
    model.eval()

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=temperature != 1.0,
        temperature=temperature,
        top_p=top_p,
    )

    base_prompt = Z_PROMPT

    for split in [train_file, valid_file]:
        if split is None:
            continue
        print(f"Processing {split}...")
        path = os.path.join(dataset_dir, split)
        df = pd.read_csv(path)

        df["z_prompt"] = [
            base_prompt.replace("<question>", q)
                       .replace("<title>", d_t)
                       .replace("<context>", d)
            for q, d, d_t in zip(df["q"], df["d"], df["d_t"])
        ]

        loader = get_loader(Dataset.from_pandas(df), batch_size=batch_size,
                            pin_memory=True, accelerator=accelerator)

        outputs = generate_outputs(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            generation_config=generation_config,
            input_col_name="z_prompt"
        )

        df["z"] = outputs

        save_path = os.path.join(dataset_dir, split.replace(".csv", "_z.csv"))
        df.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)