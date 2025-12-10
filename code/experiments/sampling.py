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
from src.prompt_hub import XZ_PRED_Y_PROMPT 


@entrypoint(with_accelerator=True, with_wandb=False)
def main(
    seed=0,
    dataset_dir="logs/golden",
    raw_subdir="raw",
    temp_subdir="temperature",
    model_name="Meta-Llama-3.1-8B-Instruct",
    int8=True,
    batch_size=16,
    max_new_tokens=50,
    num_samples_train="0,1,2",
    num_samples_valid="0,1,2",
    log_dir=None,
    accelerator=None,
):
    set_seed(seed)
    if accelerator is None:
        accelerator = Accelerator()

    if isinstance(num_samples_train, str):
        num_samples_train = (
            [int(x.strip()) for x in num_samples_train.split(",") if x.strip()]
            if num_samples_train.strip() else []
        )
    elif isinstance(num_samples_train, int):
        num_samples_train = [num_samples_train]

    if isinstance(num_samples_valid, str):
        num_samples_valid = (
            [int(x.strip()) for x in num_samples_valid.split(",") if x.strip()]
            if num_samples_valid.strip() else []
        )
    elif isinstance(num_samples_valid, int):
        num_samples_valid = [num_samples_valid]

    tokenizer = create_tokenizer(model_name)
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map="auto",
    )
    model.eval()

    for split in ["valid", "train"]:
        print(f"Processing split: {split}")

        sample_indices = num_samples_train if split == "train" else num_samples_valid
        if not sample_indices:
            continue

        z_path = os.path.join(dataset_dir, raw_subdir, f"{split}_z.csv")
        temp_path = os.path.join(dataset_dir, temp_subdir, f"{split}_t.csv")

        df_z = pd.read_csv(z_path)
        temperatures = pd.read_csv(temp_path)["temperature"].tolist()

        assert len(df_z) % batch_size == 0
        assert len(df_z) // batch_size == len(temperatures)

        for sample_idx in sample_indices:
            print(f"Sampling {sample_idx} for {split}")

            df = df_z.copy()

            df["y_pred_prompt"] = [
                XZ_PRED_Y_PROMPT.replace("<question>", q).replace("<context>", z)
                for q, z in zip(df["q"], df["z"])
            ]

            y_preds = []
            temp_values = []

            for i in tqdm(range(0, len(df), batch_size), desc=f"Generating y_pred for {split} sample {sample_idx}"):
                df_batch = df.iloc[i:i+batch_size].copy()
                temp = float(temperatures[i // batch_size])

                generation_config = GenerationConfig(
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=1.0,
                )

                loader = get_loader(Dataset.from_pandas(df_batch), batch_size=batch_size,
                                    pin_memory=True, accelerator=accelerator)

                outputs = generate_outputs(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    loader=loader,
                    generation_config=generation_config,
                    input_col_name="y_pred_prompt"
                )

                y_preds.extend(outputs)
                temp_values.extend([temp] * len(df_batch))

            df["y_pred"] = y_preds
            df["temp"] = temp_values

            save_path = os.path.join(dataset_dir, f"{split}_{sample_idx}.csv")
            df = df[["q", "x", "y", "d", "d_t", "z_prompt", "z", "y_pred_prompt", "y_pred", "temp"]]
            df.to_csv(save_path, index=False)
            print(f"Saved to {save_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
