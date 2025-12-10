import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random 
import wandb
import pandas as pd
from functools import partial 
from transformers import set_seed

from src.logging import entrypoint
from tqdm import tqdm 
from vllm import LLM, SamplingParams 
from src.prompt_hub import XZ_PRED_Y_PROMPT, XZ_PRED_Y_PROMPT_2

def calibrag_case(
    model,
    tokenizer,
    batch_size,
    #generation_config,
    temp,
    max_new_tokens=50,
    top_p=1.0,
    seed=0,
    dataset=None):
    
    base_prompt = XZ_PRED_Y_PROMPT_2
    temp_df = dataset.copy()[['q', 'x', 'y', 'd', 'd_t', 'z_prompt', 'z']]
    #loader = get_loader(Dataset.from_pandas(temp_df), 
    #                    batch_size=batch_size,
    #                    pin_memory=True,
    #                    accelerator=accelerator)
    
    #print(temp_df['y_pred_prompt'][0])
    #print("")
    
    all_outputs, new_temp = [], []
    temp_idx = 0
    for idx in tqdm(range(0, len(dataset['z']), batch_size)):
        
        dataset['y_pred_prompt'] = [base_prompt
                                    .replace("<question>", x)
                                    .replace("<context>", z)
                                    for x, z in zip(list(dataset['x']), 
                                                    list(dataset['z']))]
        
        dataset['y_pred_prompt'] = [tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) 
                        for text in dataset['y_pred_prompt']]
        
        if idx + batch_size > len(dataset['y_pred_prompt']):
            batch_prompt = dataset['y_pred_prompt'][idx:]
        else:
            batch_prompt = dataset['y_pred_prompt'][idx:idx+batch_size]
        
        if idx == 0:
            print(f"Input prompt:\n {dataset['y_pred_prompt'][idx]}")

        generation_config = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=top_p,
            seed=seed)
        
        pre_outputs = model.generate(batch_prompt, 
                                 generation_config,
                                 use_tqdm=False)
        pre_outputs = [pre_outputs[i].outputs[0].text.split("<answer>")[0] + "<answer>" for i in range(len(pre_outputs))]
        re_batch_prompt = [b+p for b, p in zip(list(batch_prompt), pre_outputs)]
        
        #all_outputs.extend(outputs)
        #new_temp.extend([temp[temp_idx] for _ in range(len(outputs))])
        
        generation_config = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temp[temp_idx],
            top_p=top_p,
            seed=seed)

        outputs = model.generate(re_batch_prompt, 
                                 generation_config,
                                 use_tqdm=False)
        outputs = [outputs[i].outputs[0].text for i in range(len(outputs))]
        outputs = [p + c for p, c in zip(pre_outputs, outputs)]
        all_outputs.extend(outputs)
        new_temp.extend([temp[temp_idx] for _ in range(len(outputs))])
        
        temp_idx += 1

    temp_df['y_pred'] = all_outputs 
    temp_df['temperature'] = new_temp
    
    print(temp_df['y_pred'][0])
    print(temp_df['temperature'][0])
    
    
    return temp_df
    

@entrypoint(with_accelerator=False, with_wandb=False, with_logging=False)
def main(
    log_dir: str = "./logs",
    model_name: str = "Meta-Llama-3.1-8B-Instruct",
    max_new_tokens: int = 50,
    #temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    retrieval_method="bm25",
    turn=0, # 0, 1, 2, 3, 4, 5
):
    log_dir = os.path.join(log_dir, f"{retrieval_method}_user_samples")
    
    ############################# Loading datasets #############################
    all_data_path = [f"data/dev/processed/calibrag/{retrieval_method}/train.csv", 
                        f"data/dev/processed/calibrag/{retrieval_method}/valid.csv"]
    
    ######################## Loading tokenizer & model #########################
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = create_tokenizer(model_name)
    #model = create_model(
    #    model_name,
    #    tokenizer=tokenizer,
    #    use_int8=int8,
    #    device_map="auto",
    #)
    model = LLM(model_name)
    #generation_config = GenerationConfig(
    #    pad_token_id=tokenizer.pad_token_id,
    #    bos_token_id=tokenizer.bos_token_id,
    #    eos_token_id=tokenizer.eos_token_id,
    #    max_new_tokens=max_new_tokens,
    #    do_sa mple=do_sample, 
    #    temperature=temperature,
    #    top_p=top_p
    #)
    
    ############################# Loading PEFT model ###########################
    ############################# and classifier head ##########################
    gen_func = partial(calibrag_case,
                        model=model,
                        tokenizer=tokenizer,
                        batch_size=batch_size)
    
    ########################### Generating outputs #############################  
    ''''''    
    for data_path in all_data_path:
        log_data_name = data_path.split("/")[-1]
        os.makedirs(f"{log_dir}", exist_ok=True)
        
        data = pd.read_csv(data_path).sample(frac=1).reset_index(drop=True)
        type = 'train' if 'train' in data_path else 'valid'
        temp_data = pd.read_csv(f'/mnt/home/chaeyun-jang/calibrag/code/logs/{type}_temperatures.csv').reset_index(drop=True)[f'{type}_t']
        
        data = data[:1000]
        temp_data = temp_data[:1000]
        
        for i in range(10):
            df = gen_func(dataset=data,
                            seed=i,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens,
                            temp=list(temp_data))#, generation_config=generation_config)
            df.to_csv(f"{log_dir}/sample_seed_{i}_{log_data_name}", index=False) 
            
        '''
        if turn == 0:
            #generation_config = SamplingParams(
            #    max_tokens=max_new_tokens,
            #    temperature=0.0,
            #    top_p=top_p,
            #    seed=0)
            #df = gen_func(dataset=data, generation_config=generation_config)
            #df.to_csv(f"{log_dir}/greedy_{log_data_name}", index=False) 

            generation_config = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=0)
            df = gen_func(dataset=data,
                          seed=0,
                          top_p=top_p,
                          max_new_tokens=max_new_tokens,
                          temp=list(temp_data))#, generation_config=generation_config)
            df.to_csv(f"{log_dir}/t_seed_0_{log_data_name}", index=False) 
        
        else:
            for i in range(2):
                set_seed(turn * 2 - 1 + i)
                print(f"Seed {turn * 2 - 1 + i} Generating...")
                generation_config = SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=turn * 2 - 1 + i)
                df = gen_func(dataset=data,
                              seed=turn * 2 - 1 + i,
                              top_p=top_p,
                              max_new_tokens=max_new_tokens,
                              temp=list(temp_data))#, generation_config=generation_config)
                df.to_csv(f"{log_dir}/t_seed_{turn * 2 - 1 + i}_{log_data_name}", index=False) 
        ''' 
        
if __name__ == "__main__":
    import fire
    fire.Fire(main)