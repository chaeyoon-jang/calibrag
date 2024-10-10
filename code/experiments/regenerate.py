import os
import json
import wandb
import torch
import logging
import warnings 
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import GenerationConfig, set_seed

from src import (
    entrypoint,
    generate_outputs, 
    generate_uc_outputs,
    generate_rag_outputs,
    create_model,
    create_tokenizer,
    get_lora_model,
    get_classifier_head,
    get_loader,
    CT_PROMPT,
    LING_PROMPT,
    NUMBER_PROMPT,
    QZ_PRED_INSTRUCTION,
    QZ_PRED_PROMPT,
    X_PRED_Y_INSTRUCTION,
    X_PRED_Y_PROMPT,
    BASE_INSTRUCTION,
    QUERY_REGENERATION_INSTRUCTION,
    QUERY_REGENERATION_PROMPT
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from")

uc_prompts = {
    'ct': CT_PROMPT,
    'ling': LING_PROMPT, 
    'number': NUMBER_PROMPT
}


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} could not be decoded as valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

def resize_uc_token_embeddings(tokenizer, model, uc_tokens):
    extra_token_count = len(tokenizer) - model.get_input_embeddings().weight.data.size(0)

    if extra_token_count:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        for i, uc_token in enumerate(uc_tokens):
            uc_token_ids = tokenizer.convert_tokens_to_ids(uc_token)
            uc_token_embeddings = input_embeddings[uc_token_ids]
            uc_mean_embedding = uc_token_embeddings.mean(dim=0, keepdim=True)
            input_embeddings[-extra_token_count+i] = uc_mean_embedding
            output_embeddings[-extra_token_count+i] = uc_mean_embedding


def direct_case(
    accelerator,
    model,
    tokenizer,
    dataset,
    batch_size,
    generation_config,
    inference=False,
    rag_dataset=None,
    uc_type=None,
    with_classifier=False):
    
    if inference:
        
        # generate model outputs
        base_instruction = QZ_PRED_INSTRUCTION
        base_prompt = QZ_PRED_PROMPT
        
        if 'z' in dataset.columns:
            dataset['output_prompt'] = [base_prompt.replace("<question>", q).replace("<title>", z_t).replace("context", z)\
                for q, z, z_t in zip(list(dataset['q']), list(dataset['z']), list(dataset['z_t']))]
                        
        else: 
            dataset['top1_z'] = [data[0]['text'] for data in rag_dataset]
            dataset['top1_z_t'] = [data[0]['title'] for data in rag_dataset]
            
            dataset['output_prompt'] = [base_prompt.replace("<question>", q).replace("<title>", z_t).replace("context", z)\
                for q, z, z_t in zip(list(dataset['q']), list(dataset['top1_z']), list(dataset['top1_z_t']))]
        
        dataset = dataset.dropna().reset_index(drop=True)
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        outputs = generate_outputs(
            accelerator,
            model,
            tokenizer,
            loader,
            generation_config,
            base_instruction,
            "output_prompt")
        
        dataset['z_pred'] = outputs 
        
        if 'z' not in dataset.columns:
            # generate model certainties
            base_instruction = BASE_INSTRUCTION
            uc_prompt = uc_prompts[uc_type]
            
            dataset['uc_prompt'] = [p+z_pred+uc_prompt for p, z_pred in zip(list(dataset['output_prompt']), list(dataset['z_pred']))]
            
            loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                        pin_memory=True, accelerator=accelerator)
            
            uc_outputs = generate_uc_outputs(
                accelerator,
                model,
                tokenizer,
                loader,
                base_instruction,
                uc_type,
                with_classifier)
            dataset['uc'] = uc_outputs 
        
    else:
        # generate model outputs
        base_instruction = X_PRED_Y_INSTRUCTION
        base_prompt = X_PRED_Y_PROMPT 
        
        dataset['output_prompt'] = [base_prompt.replace("<question>", q) for q in list(dataset['x'])]
        
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        outputs = generate_outputs(
            accelerator,
            model,
            tokenizer,
            loader,
            generation_config,
            base_instruction,
            "output_prompt")
        
        dataset['y_pred'] = outputs 
        
    return dataset  


def calibrag_case(
    accelerator,
    model,
    tokenizer,
    dataset,
    batch_size,
    generation_config,
    inference=False,
    rag_dataset=None,
    threshold=1.0,
    top_k=20):
    
    total_data = len(dataset)
    base_instruction = QZ_PRED_INSTRUCTION
    base_prompt = QZ_PRED_PROMPT
    
    if inference:
        
        dataset['output_prompt'] = [
            base_prompt.replace("<question>", q).replace("<title>", z_t).replace("context", z)
            for q, z, z_t in zip(list(dataset['q']), dataset['z_t'], dataset['z'])
        ]
        
        loader = get_loader(
            Dataset.from_pandas(dataset),
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator
        )

        uc_outputs = generate_rag_outputs(
            accelerator,
            model,
            tokenizer,
            loader,
            base_instruction
        )

        dataset['uc'] = uc_outputs
   
    else:
        # generate all z_preds 
        all_result = []
        for i in range(len(rag_dataset[0])):
            
            temp_dataset = dataset.copy()
            
            temp_dataset['z'] = [data[i]['text'] for data in rag_dataset]
            temp_dataset['z_t'] = [data[i]['title'] for data in rag_dataset]
            
            temp_dataset['output_prompt'] = [base_prompt.replace("<question>", q).replace("<title>", z_t).replace("context", z)\
                for q, z, z_t in zip(temp_dataset['q'], temp_dataset['z'], temp_dataset[f'z_t'])]
    
            loader = get_loader(Dataset.from_pandas(temp_dataset), batch_size=batch_size,
                        pin_memory=True, accelerator=accelerator)
            dataset = dataset.dropna().reset_index(drop=True)
            
            outputs = generate_outputs(
                accelerator,
                model, 
                tokenizer,
                loader,
                generation_config,
                base_instruction,
                "output_prompt")
            
            temp_dataset['z_pred'] = outputs 
            
            all_result.append(temp_dataset)
        
        dataset = pd.concat(all_result)
        
    return dataset


def regenerate_query(
    accelerator,
    model,
    tokenizer,
    dataset,
    batch_size,
    generation_config,
):
    base_instruction = QUERY_REGENERATION_INSTRUCTION
    base_prompt = QUERY_REGENERATION_PROMPT
    
    dataset['query_prompt'] = [
        base_prompt.replace("<query>", q)
        for q in list(dataset['q'])
    ]    
    
    loader = get_loader(
        Dataset.from_pandas(dataset),
        batch_size=batch_size,
        pin_memory=True,
        accelerator=accelerator
    )    
    
    outputs = generate_outputs(
        accelerator,
        model,
        tokenizer,
        loader,
        generation_config,
        base_instruction,
        input_col_name="query_prompt")
    
    dataset["q"] = outputs 
    
    return dataset 
    

@entrypoint(with_accelerator=True)
def main(
    seed: int=0,
    accelerator = None,
    log_dir: str = None,
    dataset: str = "dev",
    use_dataset_cache: bool = True,
    model_name: str = "Meta-Llama-3.1-8B-Instruct",
    query_peft_dir: str = None,
    weights_name: str = "classifier_model.bin",
    with_classifier: bool = False,
    int8: bool = True,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    inference: bool = False,
    uc_type="ling",  
):
    
    set_seed(seed)
    
    config = dict(
        seed=seed,
        log_dir=log_dir,
        dataset=dataset,
        use_dataset_cache=use_dataset_cache,
        model_name=model_name,
        query_peft_dir=query_peft_dir,
        with_classifier=with_classifier,
        batch_size=batch_size,
        uc_type=uc_type
    )
    
    if accelerator.is_main_process:
        wandb.config.update(config)
        
    # loading datasets 
    if os.path.exists(f'./data/{dataset}'):
        with accelerator.main_process_first():
            data_path = os.listdir(f'./data/{dataset}')
            new_data_path = []
            for p in data_path:
                if p.split('.')[-1] == 'csv':
                    new_data_path.append(p)
                else:
                    rag_folder = os.path.join(f'./data/{dataset}', p)
                    
        print(new_data_path)
        all_data = [pd.read_csv(os.path.join(f"./data/{dataset}", p)) for p in new_data_path] 
        all_rag_data = ["" for _ in range(len(all_data))]
    
    else:
        raise FileNotFoundError(f"No files found in the folder: ./data/{dataset}")  
    
    # loading tokenizer & model
    tokenizer = create_tokenizer(
        model_name
    )
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
        do_sample=True if temperature != 1.0 else False, 
        temperature=temperature,
        top_p=top_p
    )
    
    if uc_type == "ling":
        
        new_words = ["Unlikely", "Doubtful", "Uncertain", 
                     "Ambiguous", "Probable", "Likely", "Possible", 
                    "Specified","Confirmed", "Certain", "Inevitable"]
        
        uc_tokens = []
        for word in new_words:
            tokens = tokenizer.tokenize(word, add_special_tokens=False)
            if len(tokens) > 1:
                uc_tokens.append(tokens)
            
        tokenizer.add_tokens(new_words)
        resize_uc_token_embeddings(tokenizer, model, uc_tokens)
    
    if query_peft_dir:
        model = get_lora_model(
            model,
            peft_id_or_dir=query_peft_dir,
            is_trainable=False,
            adapter_name="query",
        )

        if with_classifier:
            classifier_model = get_classifier_head(
                input_size=model.config.hidden_size,
                checkpoint_dir=query_peft_dir,
                is_trainable=False,
                weights_name=weights_name,
            )

            model.classifier_model = classifier_model.to(model.dtype)
            model.classifier_model = model.classifier_model.to(accelerator.device)
            model.classifier_model.target_layer = -1
        
    # start generation
    model.eval()        
    for data_path, rag_data, data in zip(new_data_path, all_rag_data, all_data):
        
        if uc_type == "calibrag":
            df = calibrag_case(
                accelerator,
                model,
                tokenizer,
                data, 
                batch_size,
                generation_config,
                inference,
                rag_data)

            os.makedirs(f"{log_dir}/{dataset}-calibrag", exist_ok=True)
            df.to_csv(os.path.join(f"{log_dir}/{dataset}-calibrag", data_path)) 
        
        if uc_type == "regenerate":
            df = regenerate_query(
                accelerator,
                model,
                tokenizer,
                data, 
                batch_size,
                generation_config)
            
            os.makedirs(f"{log_dir}/{dataset}-regenerate", exist_ok=True)
            df.to_csv(os.path.join(f"{log_dir}/{dataset}-regenerate", data_path))
            
        else:
            df = direct_case(
                accelerator,
                model,
                tokenizer,
                data,
                batch_size,
                generation_config,
                inference,
                rag_data,
                uc_type,
                with_classifier)

            os.makedirs(f"{log_dir}/{dataset}-{uc_type}", exist_ok=True)
            df.to_csv(os.path.join(f"{log_dir}/{dataset}-{uc_type}", data_path)) 


if __name__ == "__main__":
    import fire

    fire.Fire(main)