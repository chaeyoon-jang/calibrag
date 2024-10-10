# base packages
import os
import json
import wandb
import torch
import logging
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import GenerationConfig, set_seed

# custum packages
from src import (
    entrypoint, 
    generate_outputs,
    create_model,
    create_tokenizer,
    get_loader,
    XZ_PRED_Y_INSTRUCTION,
    XZ_PRED_Y_PROMPT,
    XZ_PRED_Y_INSTRUCTION_EVAL,
    XZ_PRED_Y_PROMPT_EVAL
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

number_to_uc = ["Unlikely", "Doubtful", "Uncertain", 
                "Ambiguous", "Probable", "Likely", "Possible", 
                "Specified","Confirmed", "Certain", "Inevitable"]

uc_to_number = {"Unlikely": 0, "Doubtful": 1, "Uncertain": 2, 
            "Ambiguous": 3, "Probable": 4, "Likely": 5, "Possible":6, 
            "Specified":7,"Confirmed":8, "Certain":9, "Inevitable":10}

def decision_generate(
    accelerator,
    model,
    tokenizer,
    dataset,
    batch_size,
    generation_config,
    inference=False,
    uc_type="ct"):
    
    if inference:
        
        base_instruction = XZ_PRED_Y_INSTRUCTION_EVAL
        base_prompt = XZ_PRED_Y_PROMPT_EVAL
        
        if uc_type == "ling":
            dataset['uc'] = [f'{number_to_uc[int(uc)]}' for uc in dataset['uc']]
        
        elif uc_type == "number":
            dataset['uc'] = [f'{uc} (0-10)' for uc in dataset['uc']]
                
        else:
            try:
                dataset['uc'] = [f"{round(n*100,2)} (0-100)" for n in list(dataset['uc'])]
            except:
                dataset['uc'] = [n for n in list(dataset['uc'])]
                
        dataset['decision_prompt'] = [base_prompt.replace("<question>", str(x)).replace("<context>", str(z_pred)).replace("<confidence>", str(uc))\
                for x, z_pred, uc in zip(list(dataset['x']), list(dataset['z_pred']), list(dataset['uc']))]
        
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        outputs = generate_outputs(accelerator,
                                   model,
                                   tokenizer,
                                   loader,
                                   generation_config,
                                   base_instruction,
                                   "decision_prompt")
        
        dataset['y_pred'] = outputs 
    
    else:
        
        base_instruction = XZ_PRED_Y_INSTRUCTION
        base_prompt = XZ_PRED_Y_PROMPT 
        
        dataset['decision_prompt'] = [base_prompt.replace("<question>", x).replace("<context>", z_pred)\
            for x, z_pred in zip(list(dataset['x']), list(dataset['z_pred']))]
 
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        outputs = generate_outputs(accelerator,
                                   model,
                                   tokenizer,
                                   loader,
                                   generation_config,
                                   base_instruction,
                                   "decision_prompt")
        
        dataset['y_pred'] = outputs 
        
    return dataset


@entrypoint(with_accelerator=True)
def main(
    seed: int = 0,
    accelerator = None,
    log_dir: str = None,
    data_dir: str = None,
    use_dataset_cache: bool = True,
    model_name: str = "Meta-Llama-3.1-8B-Instruct",
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    inference: bool = False,
    uc_type: str = "ct",
):
    
    set_seed(seed)
    
    config = dict(
        seed=seed,
        log_dir=log_dir,
        data_dir=data_dir,
        use_dataset_cache=use_dataset_cache,
        model_name=model_name,
        batch_size=batch_size,
        inference=inference
    )
    
    if accelerator.is_main_process:
        wandb.config.update(config)
        
    # loading datasets
    if os.path.exists(data_dir):
        with accelerator.main_process_first():
            all_data_path = os.listdir(data_dir)
        
        print(all_data_path)
        all_data = [pd.read_csv(os.path.join(data_dir, p)) for p in all_data_path] 
        
    else:
        raise FileNotFoundError(f"No files found in the folder: {data_dir}")  
    
    # loading tokenizer & model
    tokenizer = create_tokenizer(
        model_name
    )
    model = create_model(
        model_name,
        tokenizer=tokenizer,
        device_map="auto",
    )
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature != 1.0 else False, 
        temperature=temperature,
        top_p=top_p
    )
    
    # start generation
    model.eval()        
    for data, data_path in zip(all_data, all_data_path):
        
        df = decision_generate(
            accelerator,
            model,
            tokenizer,
            data,
            batch_size,
            generation_config,
            inference,
            uc_type)
        
        os.makedirs(f"{log_dir}/decision", exist_ok=True)
        df.to_csv(os.path.join(f"{log_dir}/decision", data_path))
        

if __name__ == "__main__":
    import fire
    fire.Fire(main)