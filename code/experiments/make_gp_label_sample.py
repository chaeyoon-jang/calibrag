import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import pandas as pd
from datasets import Dataset
from functools import partial 
from transformers import GenerationConfig, set_seed

from src.logging import entrypoint
from src.llm_model_utils import (
    create_model,
    create_tokenizer
    )
from src.llm_data_utils import get_loader, LabeledStringDataCollator 
from src.prompt_hub import XZ_PRED_Y_PROMPT

import torch 
from tqdm import tqdm   
from peft import PeftModel

def wrapped_generate_output(
    model, tokenizer, generation_inputs, generation_config,
    num_samples=10, std_dev=0.02):#std_dev=0.01):
    
    input_ids = generation_inputs["input_ids"]
    attention_mask = generation_inputs.get("attention_mask")

    input_embeds = model.get_input_embeddings()(input_ids)

    all_outputs = []
    for _ in range(num_samples):
        
        torch.manual_seed(_)
        noise = torch.randn_like(input_embeds) * std_dev
        perturbed_embeds = input_embeds + noise

        perturbed_inputs = {
            "inputs_embeds": perturbed_embeds,
            "attention_mask": attention_mask
        }
        
        outputs = model.generate(
            **perturbed_inputs,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            generation_config=generation_config
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(outputs)

    return all_outputs


def generate_outputs(
    accelerator,
    model,
    tokenizer,
    loader,
    generation_config,
    input_col_name="prompt"):
    
    collate_fn = LabeledStringDataCollator(tokenizer)
    
    results = []
    for inputs in tqdm(loader):
        inputs = inputs[input_col_name]
        
        generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
            }
            
        generation_outputs = wrapped_generate_output(model,
                                                     tokenizer,
                                                     generation_inputs,
                                                     generation_config)
        results.extend(generation_outputs)
        del generation_outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    return results

def generate_samples(
    accelerator,
    model,
    tokenizer,
    dataset,
    batch_size,
    generation_config):
        
    base_prompt = XZ_PRED_Y_PROMPT
    temp_df = dataset.copy()[['q', 'x', 'y', 'd', 'd_t', 'z_prompt', 'z']]
    dataset['y_pred_prompt'] = [base_prompt
                                .replace("<question>", x)
                                .replace("<context>", z)
                                for x, z in zip(list(dataset['x']), 
                                                list(dataset['z']))]
        
    print(f"=====> y prompt:\n{dataset['y_pred_prompt'][0]}\n")
    
    loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                pin_memory=True, accelerator=accelerator)
    
    outputs = generate_outputs(
        accelerator,
        model,
        tokenizer,
        loader,
        generation_config,
        "y_pred_prompt")
    
    print(f"=====> y pred:\n{outputs[0]}\n")

    temp_df['y_pred_prompt'] = dataset['y_pred_prompt']
    
    all_data = []
    for i in range(10):
        temp_data = temp_df.copy()
        temp_data['y_pred'] = outputs[i::10] 
        all_data.append(temp_data)
    all_data = pd.concat(all_data, axis=0)
    return all_data

@entrypoint(with_accelerator=True, with_wandb=False)
def main(
    seed: int=0,
    accelerator = None,
    log_dir: str = "./logs",
    model_name: str = "Meta-Llama-3.1-8B-Instruct",
    int8: bool = True,
    max_new_tokens: int = 50,
    batch_size: int = 1,
    retrieval_method: str = "bm25"
):    
    log_dir = os.path.join(log_dir, f"{retrieval_method}_gp_user_samples")
    
    ############################# Loading datasets #############################
    all_data_path = [f"data/dev/processed/calibrag/{retrieval_method}/train.csv", 
                        f"data/dev/processed/calibrag/{retrieval_method}/valid.csv"]
    
    ######################## Loading tokenizer & model #########################
    tokenizer = create_tokenizer(model_name)
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        use_int8=int8,
        device_map="auto",
    )
    
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False, 
        temperature=1.0,
        top_p=1.0,
    )
    
    ############################# Loading PEFT model ###########################
    ############################# and classifier head ##########################
    gen_func = partial(generate_samples,
                       accelerator=accelerator,
                       model=model,
                       tokenizer=tokenizer,
                       batch_size=batch_size,
                       generation_config=generation_config)
    
    ########################### Generating outputs #############################
    
    #for seed in range(10):
    model.eval()        
    for data_path in all_data_path:
        log_data_name = data_path.split("/")[-1]
        os.makedirs(f"{log_dir}", exist_ok=True)
        
        set_seed(0)
        data = pd.read_csv(data_path).sample(frac=1).reset_index(drop=True)
        data = data[:1000]
        
        set_seed(seed)
        
        df = gen_func(dataset=data)   
        df.to_csv(f"{log_dir}/gp_sample_seed_{seed}_{log_data_name}", index=False) 

if __name__ == "__main__":
    import fire
    fire.Fire(main)