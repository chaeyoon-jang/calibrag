################################################################################
## This file covers two main cases:
# 1. Generating 'y_pred' form 'x' and generated context 'z'. (Infernece=False)
# 2. Generating 'y_pred' form 'x', 'z' and confidence 'c'. (Infernece=True)
################################################################################
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
import pandas as pd
from tqdm.autonotebook import tqdm 
from transformers import set_seed, AutoTokenizer
from vllm import LLM, SamplingParams 

from src.logging import entrypoint
from src.llm_model_utils import create_tokenizer
from src.prompt_hub import (
    XZ_PRED_Y_PROMPT,
    XZ_PRED_Y_PROMPT_MED,
    XZ_PRED_Y_PROMPT_EVAL,
    XZ_PRED_Y_PROMPT_EVAL_MED,
    number_to_uc
)


def decision_generate(
    model,
    tokenizer,
    dataset,
    batch_size,
    sampling_params,
    inference=False,
    c_type="ct",
    medical=False):
    
    # drop nan values
    dataset = dataset.dropna().reset_index(drop=True)
    
    if inference:
        
        base_prompt = XZ_PRED_Y_PROMPT_EVAL_MED if medical else XZ_PRED_Y_PROMPT_EVAL
        
        if (
            tokenizer.name_or_path
            and ("Llama-3" in tokenizer.name_or_path)
            and ("Instruct" in tokenizer.name_or_path)
        ):
            msgs = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": base_prompt}]
            
            base_prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            
        if c_type == "ling":
            dataset['c'] = [f'{number_to_uc[int(uc)]}' for uc in dataset['c']]
        
        elif c_type == "number":
            dataset['c'] = [f'{round(uc, 2)} (0-10)' for uc in dataset['c']]
                
        else:
            try:
                dataset['c'] = [f"{round(n*100,2)} (0-100)" 
                                 for n in list(dataset['c'])]
            except:
                dataset['c'] = [n for n in list(dataset['c'])]
         
        if medical:
            try:
                dataset['y_pred_prompt'] = [base_prompt
                                            .replace("<question>", str(x))
                                            .replace("<context>", str(z))
                                            .replace("<confidence>", str(uc))
                                            .replace("<option>", o)
                                            for x, z, uc, o in zip(list(dataset['x']),
                                                                list(dataset['z']),
                                                                list(dataset['c']),
                                                                list(dataset['option']))]
            except:
                import ipdb; ipdb.set_trace()
        else:
            dataset['y_pred_prompt'] = [base_prompt
                                        .replace("<question>", str(x))
                                        .replace("<context>", str(z))
                                        .replace("<confidence>", str(uc))
                                        for x, z, uc in zip(list(dataset['x']),
                                                            list(dataset['z']),
                                                            list(dataset['c']))]
        
    else:
        
        base_prompt =  XZ_PRED_Y_PROMPT_MED  if medical else XZ_PRED_Y_PROMPT 
        
        if (
            tokenizer.name_or_path
            and ("Llama-3" in tokenizer.name_or_path)
            and ("Instruct" in tokenizer.name_or_path)
        ):
            msgs = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": base_prompt}]
            
            base_prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        
        if medical:
            dataset['y_pred_prompt'] = [base_prompt.replace("<question>", x)
                                        .replace("<context>", z)
                                        .replace("<option>", o)
                                        for x, z, o in zip(list(dataset['x']), 
                                                        list(dataset['z']),
                                                        list(dataset['option']))]
        else:
            dataset['y_pred_prompt'] = [base_prompt.replace("<question>", x)
                                        .replace("<context>", z)
                                        for x, z in zip(list(dataset['x']), 
                                                        list(dataset['z']))]
    
    ## Generate outputs.
    all_outputs = []
    for idx in tqdm(range(0, len(dataset['y_pred_prompt']), batch_size)):
        
        if idx == 0:
            print(f"Input prompt:\n {dataset['y_pred_prompt'][idx]}")
            
        if idx + batch_size > len(dataset['y_pred_prompt']):
            batch_prompt = dataset['y_pred_prompt'][idx:]
        else:
            batch_prompt = dataset['y_pred_prompt'][idx:idx+batch_size]
        
        outputs = model.generate(batch_prompt, sampling_params, use_tqdm=False)
        outputs = [outputs[i].outputs[0].text for i in range(len(outputs))]
        all_outputs.extend(outputs)
    
    dataset['y_pred'] = all_outputs
        
    return dataset


@entrypoint(with_wandb=False)
def main(
    seed: int = 0,
    log_dir: str = None,
    data_dir: str = None,
    use_dataset_cache: bool = True,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    inference: bool = False,
    c_type: str = "ct",
    medical: bool = False,
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
    #wandb.config.update(config)
        
    ############################# Loading datasets #############################
    if os.path.exists(data_dir):
        all_data_path = os.listdir(data_dir)
        
        print(all_data_path)
        all_data = [pd.read_csv(os.path.join(data_dir, p)) for p in all_data_path] 
        
    else:
        raise FileNotFoundError(f"No files found in the folder: {data_dir}")  
    
    ######################## Loading tokenizer & model #########################
    tokenizer = AutoTokenizer.from_pretrained(model_name)#create_tokenizer(model_name.split("/")[-1])
    model = LLM(model_name)
    
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0 if not do_sample else temperature,
        top_p=top_p,
        seed=seed
    )
    
    ########################### Generating outputs #############################     
    for data, data_path in zip(all_data, all_data_path):
        
        df = decision_generate(
            model,
            tokenizer,
            data,
            batch_size,
            sampling_params,
            inference,
            c_type,
            medical)
        
        os.makedirs(f"{log_dir}/decision", exist_ok=True)
        df.to_csv(os.path.join(f"{log_dir}/decision", data_path), index=False)
        

if __name__ == "__main__":
    import fire
    fire.Fire(main)