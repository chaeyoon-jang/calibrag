import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import set_seed

from src.logging import entrypoint
from src.llm_model_utils import create_model, create_tokenizer
from src.distributed import AcceleratorState
from src.prompt_hub import Z_PROMPT_D_Q

@entrypoint
def main(
    seed=0,
    log_dir: str = None,
    dataset: str = "test",
    model_name="Meta-Llama-3.1-8B-Instruct",
    batch_size=4,
    int8=True,
    retrieval_method="bm25",
    flush_every=2000,
):
    set_seed(seed)
    accelerator = AcceleratorState()

    dataset_dir = f'./data/{dataset}'
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"No folder found: {dataset_dir}")

    with accelerator.main_process_first():
        rag_files = os.listdir(os.path.join(dataset_dir, "rag"))
        rag_data_path = [f for f in rag_files if retrieval_method in f]

        def extract_key(filename):
            return filename.split('_')[1].split('.')[0]

        rag_data_path = sorted(rag_data_path, key=extract_key)
        print(f"=====> Loaded RAG datasets: {rag_data_path}")

    tokenizer = create_tokenizer(model_name)
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map="auto",
    )
    model.eval()
    model.requires_grad_(False)

    os.makedirs(log_dir, exist_ok=True)

    for rag_file in rag_data_path:
        task_name = extract_key(rag_file)
        rag_data = json.load(open(os.path.join(dataset_dir, "rag", rag_file)))

        # ... (중략: import 및 설정 동일)

        context_texts = []
        id_to_title = {}

        for ex in rag_data:
            for ctx in ex["contexts"]:
                context_prompt = Z_PROMPT_D_Q.replace("<context>", ctx["text"]).split("Question:")[0].strip()
                context_texts.append(context_prompt)
                id_to_title[context_prompt] = ctx.get("title", "")

        context_dataset = Dataset.from_dict({"text": context_texts})
        id_to_cache = {}
        all_part_paths = []
        total_cached = 0

        for i in tqdm(range(0, len(context_dataset), batch_size), desc=f"Embedding+Cache ({task_name})"):
            batch_texts = context_dataset[i: i + batch_size]["text"]
            batch = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            with torch.inference_mode():
                outputs = model(**batch, use_cache=True, return_dict=True)
                pkv = outputs.past_key_values

            for b_idx, doc_text in enumerate(batch_texts):
                id_to_cache[doc_text] = {
                    "input_ids": batch["input_ids"][b_idx].cpu(),
                    "attention_mask": batch["attention_mask"][b_idx].cpu(),
                    "past_key_values": [
                        (layer_k[b_idx].cpu(), layer_v[b_idx].cpu())
                        for layer_k, layer_v in pkv
                    ],
                    "title": id_to_title[doc_text],
                }
                total_cached += 1

            if total_cached % flush_every == 0:
                part_path = os.path.join(log_dir, f"{task_name}_{retrieval_method}_past_part_{total_cached}.pt")
                torch.save(id_to_cache, part_path)
                all_part_paths.append(part_path)
                id_to_cache.clear()
                torch.cuda.empty_cache()
                print(f"[Flush] Saved {flush_every} items to {part_path}")

        if id_to_cache:
            part_path = os.path.join(log_dir, f"{task_name}_{retrieval_method}_past_part_{total_cached}.pt")
            torch.save(id_to_cache, part_path)
            all_part_paths.append(part_path)
            print(f"[Flush] Saved remaining to {part_path}")

        # Merge parts
        merged_cache = {}
        for path in all_part_paths:
            part_cache = torch.load(path)
            merged_cache.update(part_cache)
        final_path = os.path.join(log_dir, f"{task_name}_{retrieval_method}_past.pt")
        torch.save(merged_cache, final_path)
        print(f"Final merged cache saved to: {final_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
