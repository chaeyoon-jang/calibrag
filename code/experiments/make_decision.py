################################################################################
## This file covers two main cases:
# 1. Generating 'y_pred' form 'x' and generated context 'z'. (Infernece=False)
# 2. Generating 'y_pred' form 'x', 'z' and confidence 'c'. (Infernece=True)
################################################################################
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
import pandas as pd
from datasets import Dataset
from transformers import GenerationConfig, set_seed

from src.logging import entrypoint
from src.generate_utils import generate_outputs
from src.llm_model_utils import (
    create_model,
    create_tokenizer
)
from src.llm_data_utils import get_loader
from src.prompt_hub import (
    XZ_PRED_Y_PROMPT,
    XZ_PRED_Y_PROMPT_EVAL,
    number_to_uc
)


def decision_generate(
    accelerator,
    model,
    tokenizer,
    dataset,
    batch_size,
    generation_config,
    inference=False,
    c_type="ct"):
    
    if inference:
        
        base_prompt = XZ_PRED_Y_PROMPT_EVAL
        
        if c_type == "ling":
            dataset['c'] = [f'{number_to_uc[int(uc)]}' for uc in dataset['c']]
        
        elif c_type == "number":
            dataset['c'] = [f'{uc} (0-10)' for uc in dataset['c']]
                
        else:
            try:
                dataset['c'] = [f"{round(n*100,2)} (0-100)" 
                                 for n in list(dataset['c'])]
            except:
                dataset['c'] = [n for n in list(dataset['c'])]
                
        dataset['y_pred_prompt'] = [base_prompt
                                    .replace("<question>", str(x))
                                    .replace("<context>", str(z))
                                    .replace("<confidence>", str(uc)) 
                                    for x, z, uc in zip(list(dataset['x']),
                                                             list(dataset['z']),
                                                             list(dataset['c']))]
        
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        outputs = generate_outputs(accelerator,
                                   model,
                                   tokenizer,
                                   loader,
                                   generation_config,
                                   "y_pred_prompt")
        
        dataset['y_pred'] = outputs 
    
    else:
        
        base_prompt = XZ_PRED_Y_PROMPT 
        
        dataset['y_pred_prompt'] = [base_prompt
                                    .replace("<question>", x)
                                    .replace("<context>", z)
                                    for x, z in zip(list(dataset['x']), 
                                                    list(dataset['z']))]
 
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        outputs = generate_outputs(accelerator,
                                   model,
                                   tokenizer,
                                   loader,
                                   generation_config,
                                   "y_pred_prompt")
        
        dataset['y_pred'] = outputs 
        
    return dataset


@entrypoint(with_accelerator=True, with_wandb=False)
def main(
    seed: int = 0,
    accelerator = None,
    log_dir: str = None,
    data_dir: str = None,
    use_dataset_cache: bool = True,
    model_name: str = "Meta-Llama-3.1-8B-Instruct",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    inference: bool = False,
    c_type: str = "ct",
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
        
    ############################# Loading datasets #############################
    if os.path.exists(data_dir):
        with accelerator.main_process_first():
            all_data_path = os.listdir(data_dir)
        
        print(all_data_path)
        all_data = [pd.read_csv(os.path.join(data_dir, p)) for p in all_data_path] 
        
    else:
        raise FileNotFoundError(f"No files found in the folder: {data_dir}")  
    
    ######################## Loading tokenizer & model #########################
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
    
    ########################### Generating outputs #############################
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
            c_type)
        
        os.makedirs(f"{log_dir}/decision", exist_ok=True)
        df.to_csv(os.path.join(f"{log_dir}/decision", data_path), index=False)
        

if __name__ == "__main__":
    import fire
    fire.Fire(main)