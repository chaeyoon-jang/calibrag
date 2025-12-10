################################################################################
## This file covers two main cases:
# 1. Generating 'y_pred' from 'x' without using a retrieved document 'd' (Baselines).
# 2. Generating 'z' from query q with a retrieved document 'd' (inference=True):
#    - Training: Generate 'z' using all top-k retrieved documents 'd' for 'q'.
#    - Inference: 
#       1) Generate 'z' using the top-1 (reranked by f) document 'd' for 'q'.
#       2) In baselines, Generate 'c' from 'z' and 'q'.
# 3. Generated new query 'q' from a original 'q'.
################################################################################
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import wandb
import pandas as pd
import random
import numpy as np
from datasets import Dataset
from functools import partial 
from transformers import GenerationConfig, set_seed

from src.logging import entrypoint
from src.generate_utils import (
    generate_outputs,
    generate_c_outputs,
    generate_rag_outputs,
    generate_rag_temp_outputs,
    generate_rag_outputs_dq,
    )
from src.llm_model_utils import (
    create_model,
    create_tokenizer,
    resize_c_token_embeddings
    )
from src.peft_utils import (
    get_lora_model,
    get_classifier_head
    ) 
from src.llm_data_utils import get_loader 
from src.prompt_hub import (
    c_prompts,
    Z_PROMPT,
    Z_PROMPT_MED,
    Z_PROMPT_D_Q,
    X_PRED_Y_PROMPT,
    QUERY_REGENERATION_PROMPT,
    X_PRED_Y_PROMPT_MED
    )


def direct_case(
    accelerator,
    model,
    tokenizer,
    dataset,
    batch_size,
    generation_config,
    inference=False,
    rag_dataset=None,
    c_type=None,
    with_classifier=False,
    medical=False,
    temp=False,
    temp_value=[1.0],
    base_reranking=False,
    best_k=1):
    
    if inference:

        dataset = pd.DataFrame()
        dataset['q'] = [data['q'] for data in rag_dataset]
        dataset['x'] = [data['x'] for data in rag_dataset]
        dataset['y'] = [data['y'] for data in rag_dataset]
        
        base_prompt = Z_PROMPT_MED if medical else Z_PROMPT
        
        if 'd' in dataset.columns:
            ## This case is for <ablation studies>
            ## where we have to generate z given q with reranked d.
            
            dataset['z_prompt'] = [
                base_prompt.replace("<question>", q)
                .replace("<title>", z_t)
                .replace("<context>", z)
                for q, z, z_t in zip(dataset['q'], dataset['d'], dataset['d_t'])
                ]
            
        elif base_reranking:
            print("=====> Running base reranking ...")
            attempts = {}

            for k in range(len(rag_dataset[0]["contexts"])):
                print(f"=====> base reranking: {k}-th document")
                ds = [data["contexts"][k]["text"] for data in rag_dataset]
                d_ts = [data["contexts"][k]["title"] for data in rag_dataset]

                z_prompts = [
                    base_prompt.replace("<question>", q)
                    .replace("<title>", z_t)
                    .replace("<context>", d)
                    for q, d, z_t in zip(dataset["q"], ds, d_ts)
                ]

                row_df = pd.DataFrame({
                    "q": dataset["q"],
                    "x": dataset["x"],
                    "y": dataset["y"],
                    "z_prompt": z_prompts
                })

                loader = get_loader(
                    Dataset.from_pandas(row_df),
                    batch_size=batch_size,
                    pin_memory=True,
                    accelerator=accelerator
                )

                z_outputs = generate_outputs(
                    accelerator,
                    model,
                    tokenizer,
                    loader,
                    generation_config,
                    "z_prompt"
                )

                row_df["z"] = z_outputs
                c_prompt = c_prompts[f"{c_type}_temp"].replace("<temp_value>", str(temp_value[0])) if temp else c_prompts[c_type]
                row_df["c_prompt"] = [p + z + c_prompt for p, z in zip(row_df["z_prompt"], row_df["z"])]

                loader = get_loader(
                    Dataset.from_pandas(row_df),
                    batch_size=4,
                    pin_memory=True,
                    accelerator=accelerator
                )

                c_outputs = generate_c_outputs(
                    accelerator,
                    model,
                    tokenizer,
                    loader,
                    c_type,
                    with_classifier,
                    "c_prompt"
                )

                for i in range(len(row_df)):
                    data_id = i  # same as dataset['id'].iloc[i]
                    attempt = {
                        "x": row_df["x"].iloc[i],
                        "y": row_df["y"].iloc[i],
                        "q": row_df["q"].iloc[i],
                        "z": row_df["z"].iloc[i],
                        "z_prompt": row_df["z_prompt"].iloc[i],
                        "c": c_outputs[i],
                    }
                    if data_id not in attempts:
                        attempts[data_id] = []
                    attempts[data_id].append(attempt)

            # Pick best for each sample
            final_results = []
            for data_id in attempts:
                best_attempt = max(attempts[data_id], key=lambda x: x["c"])
                final_results.append(best_attempt)

            dataset = pd.DataFrame(final_results)

        else: 
            ## This case is for <main experiments-baselines>
            ## where we have to generate z given q with original top-k d.
            
            if best_k == 1:
                dataset['d'] = [data['contexts'][0]['text'] for data in rag_dataset]
                dataset['d_t'] = [data['contexts'][0]['title'] for data in rag_dataset]
                
                dataset['z_prompt'] = [
                    base_prompt.replace("<question>", q)
                    .replace("<title>", z_t)
                    .replace("<context>", z)
                    for q, z, z_t in zip(dataset['q'], dataset['d'], dataset['d_t'])
                    ]
                
            elif best_k > 1:
                k = min(best_k, len(rag_dataset[0]["contexts"]))
                contexts = []
                titles = []
                
                for i in range(k):
                    contexts.append([data["contexts"][i]["text"] for data in rag_dataset])
                    titles.append([data["contexts"][i]["title"] for data in rag_dataset])

                combined_contexts = []
                for idx in range(len(rag_dataset)):
                    combined = "\n" + "\n".join(
                        f"{titles[i][idx]} - {contexts[i][idx]}" for i in range(k)
                    )
                    combined_contexts.append(combined)

                dataset['z_prompt'] = [
                    base_prompt.replace("<question>", q)
                    .replace("<title> - <context>", cxt)
                    for q, cxt in zip(dataset["q"], combined_contexts)
                ]
        
        print(f"=====> z prompt:\n{dataset['z_prompt'][0]}\n")
        
        dataset = dataset.dropna().reset_index(drop=True)
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        ## Generate z from q using retrieved document d.
        outputs = generate_outputs(
            accelerator,
            model,
            tokenizer,
            loader,
            generation_config,
            "z_prompt")
        
        dataset['z'] = outputs 
        
        print(f"=====> z pred:\n{dataset['z'][0]}\n")
        
        ## Generate confidence-c from z and q.
        ## ct: Is the proposed answer corect? (Yes/No)
        ## number: provide the certainty level of answer (0-10)
        ## ling: provide the certainty level of answer (Unlikely, ..., Inevitable)
        
        c_prompt = c_prompts[f"{c_type}_temp"].replace('<temp_value>', str(temp_value[0])) if temp else c_prompts[c_type]
        
        dataset['c_prompt'] = [p + z_pred + c_prompt for p, z_pred 
                               in zip(list(dataset['z_prompt']),
                                      list(dataset['z']))]
        
        print(f"=====> c prompt:\n{dataset['c_prompt'][0]}\n")
        
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=2, #batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        c_outputs = generate_c_outputs(
            accelerator,
            model,
            tokenizer,
            loader,
            c_type,
            with_classifier,
            "c_prompt")
        
        print(f"=====> c pred: {c_outputs[0]}\n")
        
        dataset['c'] = c_outputs 
        
    else:
        ## This case for training where we have to generate y from x. 
        ## For ling, numer -> we have to sample 10 times.
        
        base_prompt = X_PRED_Y_PROMPT_MED if medical else X_PRED_Y_PROMPT

        if medical:
            dataset['y_prompt'] = [base_prompt.replace("<question>", q).replace("<option>", o) 
                                   for q, o in zip(dataset['x'], dataset['option'])]
        else:
            dataset['y_prompt'] = [base_prompt.replace("<question>", q) 
                                for q in list(dataset['x'])]
            
        print(f"=====> y prompt:\n{dataset['y_prompt'][0]}\n")
        
        loader = get_loader(Dataset.from_pandas(dataset), batch_size=batch_size,
                    pin_memory=True, accelerator=accelerator)
        
        outputs = generate_outputs(
            accelerator,
            model,
            tokenizer,
            loader,
            generation_config,
            "y_prompt")
        
        print(f"=====> y pred:\n{outputs[0]}\n")

        dataset['y_pred'] = outputs 
        
    return dataset  


def calibrag_case(
    accelerator,
    model,
    tokenizer,
    batch_size,
    generation_config,
    inference=False,
    dataset=None,
    rag_dataset=None,
    threshold=1.0,
    top_k=20,
    best_k=1,
    medical=False,
    multi=False,
    multi_reranking=False,
    doc_query=False,
    temp=False,
    temp_value=[1.0]):
    
    total_data = len(dataset)
    base_prompt = Z_PROMPT_MED if medical else Z_PROMPT
    base_prompt = Z_PROMPT_D_Q if doc_query else Z_PROMPT
    
    # It is possible that len(rag_dataset) < len(dataset)
    dataset = pd.DataFrame()
    dataset['q'] = [data['q'] for data in rag_dataset]
    dataset['x'] = [data['x'] for data in rag_dataset]
    dataset['y'] = [data['y'] for data in rag_dataset]
    
    if inference:
        ## Generate z for each Top-k documents.
        ## Rerank the documents based on the confidence score.
        ## If the confidence score is greater than the threshold, then stop.
        
        final_results = []
        attempts = {}  
        dataset['id'] = range(len(dataset))  

        def process_dataset(
            dataset,
            rag_dataset,
            accelerator,
            model,
            tokenizer,
            base_prompt,
            batch_size,
            doc_query,
            current_idx,
            attempts):

            if current_idx >= top_k:
                if best_k == 1:
                    for data_id in attempts:
                        best_attempt = max(attempts[data_id], key=lambda x: x['c'])
                        final_results.append(best_attempt)
                else:
                    for data_id in attempts:
                        sorted_attempts = sorted(attempts[data_id], key=lambda x: x["c"], reverse=True)
                        topk_attempts = sorted_attempts[:best_k]

                        context_block = "\n" + "\n".join(
                            a["z_prompt"].split("Retrieved Context:")[1].split("Answer:")[0].strip()
                            for a in topk_attempts
                        )

                        q = topk_attempts[0]["q"]
                        x = topk_attempts[0]["x"]
                        y = topk_attempts[0]["y"]

                        new_z_prompt = base_prompt.replace("<question>", q).replace("<title> - <context>", context_block)

                        final_results.append({
                            "x": x,
                            "y": y,
                            "q": q,
                            "z_prompt": new_z_prompt,
                        })
                return

            else:
                print(f"=====> Current index is {current_idx}")
                print(f'=====> The remaining data is {len(dataset)}/{total_data}')

            ds = [data['contexts'][current_idx]['text'] for data in rag_dataset]
            d_ts = [data['contexts'][current_idx]['title'] for data in rag_dataset]
            
            dataset['z_prompt'] = [
                base_prompt.replace("<question>", q)
                .replace("<title>", z_t)
                .replace("<context>", z)
                for q, z, z_t in zip(dataset['q'], ds, d_ts)
            ]
            
            loader = get_loader(
                Dataset.from_pandas(dataset),
                batch_size=batch_size,
                pin_memory=True,
                accelerator=accelerator
            )
            
            if doc_query:
                c_outputs = generate_rag_outputs_dq(
                    accelerator,
                    model,
                    tokenizer,
                    loader,
                    multi,
                    multi_reranking,
                )
            
            elif temp and (multi or multi_reranking): 
                if isinstance(temp_value, list) and len(temp_value) > 1:
                    all_preds = []
                    all_probs = []
                    for t_value in temp_value:
                        temp_c_outputs = generate_rag_temp_outputs(
                            accelerator,
                            model,
                            tokenizer,
                            loader,
                            multi,
                            multi_reranking,
                            temperature=t_value,
                        )
                        for i, row in enumerate(temp_c_outputs):
                            pred = int(np.argmax(row))
                            prob = float(np.max(row))
                            if len(all_preds) <= i:
                                all_preds.append([])
                                all_probs.append([])
                            all_preds[i].append(pred)
                            all_probs[i].append(prob)

                    c_outputs = []
                    for preds, probs in zip(all_preds, all_probs):
                        avg_class = float(np.mean(preds))
                        avg_prob = float(np.mean([p for p in probs])) 
                        c_outputs.append({"c": avg_class, "prob": avg_prob})
                            
                else:
                    c_outputs = generate_rag_temp_outputs(
                        accelerator,
                        model,
                        tokenizer,
                        loader,
                        multi,
                        multi_reranking,
                        temperature=temp_value[0],
                    )
            
            elif temp:
                if isinstance(temp_value, list):
                    all_c_outputs = []
                    for t_value in temp_value:
                        temp_c_outputs = generate_rag_temp_outputs(
                            accelerator,
                            model,
                            tokenizer,
                            loader,
                            multi,
                            multi_reranking,
                            temperature=t_value,
                        )
                        all_c_outputs.append(temp_c_outputs)
                    c_outputs = np.mean(all_c_outputs, axis=0)
                    
            else:
                c_outputs = generate_rag_outputs(
                    accelerator,
                    model,
                    tokenizer,
                    loader,
                    multi,
                    multi_reranking,
                )

            new_rag_dataset = []
            new_x = []
            new_y = []
            new_q = []
            new_id = []

            for i in range(len(dataset)):
                c_output = c_outputs[i]
                data_id = dataset['id'].iloc[i]
                pred_class = c_output["c"] if multi_reranking else c_output
                prob = c_output["prob"] if multi_reranking else c_output
                attempt = {
                    'x': dataset['x'].iloc[i],
                    'y': dataset['y'].iloc[i],
                    'q': dataset['q'].iloc[i],
                    'z_prompt': dataset['z_prompt'].iloc[i],
                    'c': pred_class,
                    'prob': prob,
                }

                if data_id not in attempts:
                    attempts[data_id] = []
                attempts[data_id].append(attempt)
                
                if multi:
                    continue

                if prob >= threshold:
                    continue  
                else:
                    new_rag_dataset.append(rag_dataset[i])
                    new_x.append(dataset['x'].iloc[i])
                    new_y.append(dataset['y'].iloc[i])
                    new_q.append(dataset['q'].iloc[i])
                    new_id.append(data_id)  

            if len(new_q) > 0:
                dataset_retry = pd.DataFrame()
                dataset_retry['q'] = new_q
                dataset_retry['x'] = new_x
                dataset_retry['y'] = new_y
                dataset_retry['id'] = new_id  

                process_dataset(
                    dataset_retry,
                    new_rag_dataset,
                    accelerator,
                    model,
                    tokenizer,
                    base_prompt,
                    batch_size,
                    doc_query,
                    current_idx + 1,
                    attempts
                )
            else:
                for data_id in attempts:
                    if multi_reranking:
                        attempt_list = attempts[data_id]
                        class_list = [a["c"] for a in attempt_list]
                        max_class = max(class_list)
                        best_attempts = [a for a in attempt_list if a["c"] == max_class]
                        best_attempt = max(best_attempts, key=lambda x: x['prob'])
                        
                    else:
                        best_attempt = max(attempts[data_id], key=lambda x: x['c'])
                        final_results.append(best_attempt)

        process_dataset(
            dataset,
            rag_dataset,
            accelerator,
            model,
            tokenizer,
            base_prompt,
            batch_size,
            doc_query,
            current_idx=0,
            attempts=attempts
        )

        dataset = pd.DataFrame(final_results)
        dataset = dataset.dropna().reset_index(drop=True)
        
        loader = get_loader(
            Dataset.from_pandas(dataset),
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator
        )
        
        if best_k > 1:
            print("=====> Generating final confidence scores...")
            
            if isinstance(temp_value, list) and temp:
                all_c_outputs = []

                for t_value in temp_value:
                    final_c_outputs = generate_rag_temp_outputs(
                        accelerator,
                        model,
                        tokenizer,
                        loader,
                        multi,
                        multi_reranking,
                        temperature=t_value,
                    )
                    all_c_outputs.append(final_c_outputs)

                if multi_reranking:
                    dataset['c'] = np.mean(
                        [[c_output['prob'] for c_output in single_temp_output] 
                        for single_temp_output in all_c_outputs], 
                        axis=0
                    ).tolist()
                else:
                    dataset['c'] = np.mean(all_c_outputs, axis=0).tolist()

            else:
                final_c_outputs = generate_rag_outputs(
                    accelerator,
                    model,
                    tokenizer,
                    loader,
                    multi,
                    multi_reranking
                )

                if multi_reranking:
                    dataset['c'] = [c_output['prob'] for c_output in final_c_outputs]
                else:
                    dataset['c'] = final_c_outputs
        
        outputs = generate_outputs(
            accelerator,
            model,
            tokenizer,
            loader,
            generation_config,
            "z_prompt"
        )

        dataset['z'] = outputs
   
    else:
        ## Generate all z for each Top-k documents.
        
        all_result = []
        for i in range(len(rag_dataset[0]['contexts'])):
            
            temp_dataset = dataset.copy()
            
            temp_dataset['d'] = [data['contexts'][i]['text'] 
                                 for data in rag_dataset]
            temp_dataset['d_t'] = [data['contexts'][i]['title'] 
                                   for data in rag_dataset]
            
            temp_dataset['z_prompt'] = [
                base_prompt.replace("<question>", q)
                .replace("<title>", z_t)
                .replace("<context>", z)
                for q, z, z_t in zip(temp_dataset['q'], 
                                     temp_dataset['d'], 
                                     temp_dataset[f'd_t'])
                ]
            
            print(f"=====> {i}th z prompt:\n{temp_dataset['z_prompt'][0]}\n")
            
            temp_dataset = temp_dataset.dropna().reset_index(drop=True)
            loader = get_loader(Dataset.from_pandas(temp_dataset), 
                                batch_size=batch_size,
                                pin_memory=True,
                                accelerator=accelerator)
            
            outputs = generate_outputs(
                accelerator,
                model, 
                tokenizer,
                loader,
                generation_config,
                "z_prompt")
            
            temp_dataset['z'] = outputs 
            
            print(f"=====> {i}th z pred:\n{temp_dataset['z'][0]}\n")
            
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
        input_col_name="query_prompt")
    
    dataset["new_q"] = outputs 
    
    return dataset 
    

@entrypoint(with_accelerator=True, with_wandb=False)
def main(
    seed: int=0,
    accelerator = None,
    log_dir: str = None,
    dataset: str = "dev",
    model_name: str = "Meta-Llama-3.1-8B-Instruct",
    query_peft_dir: str = None,
    weights_name: str = "classifier_model.bin",
    with_classifier: bool = False,
    int8: bool = True,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    inference: bool = False,
    retrieval_method: str = "bm25",
    c_type= "calibrag",  
    medical=False,
    top_k=20,
    best_k=1,
    target_layer=-1,
    multi=False,
    multi_reranking=False,
    base_reranking=False,
    doc_query=False,
    temp=False,
    temp_output=11,
    temp_value: str ="1.0, 1.1, 1.2, 1.3, 1.4, 1.5"):
    
    set_seed(seed) # for sampling method in ling and number.
 
    if isinstance(temp_value, str):
        temp_value = list([float(t.strip()) for t in temp_value.split(",") if t.strip()])
    elif isinstance(temp_value, (float, int)):
        temp_value = list([float(temp_value)])
    temp_value = list(temp_value)
    ############################# Loading datasets #############################
    if os.path.exists(f'./data/{dataset}'):
        with accelerator.main_process_first():
            data_path = os.listdir(f'./data/{dataset}')
            raw_path = os.listdir(f'./data/{dataset}/raw')
            rag_path = os.listdir(f'./data/{dataset}/rag')
            new_data_path = [p for p in raw_path if p.endswith('.csv') and 'oe' in p]
            rag_data_path = [p for p in rag_path] if medical else [p for p in rag_path if retrieval_method in p]
            #new_data_path = []; rag_data_path = []
            #for p in data_path:
            #    if (p.split('.')[-1] == 'csv') & ('oe' in p):
            #        new_data_path.append(p)
            #    else:
            #        if retrieval_method in p:
            #            rag_data_path.append(p)
        
        all_data = [pd.read_csv(os.path.join(f"./data/{dataset}/raw", p)) 
                    for p in new_data_path] 
        print(f"=====> Loaded datasets: {new_data_path}")
        
        if (c_type == 'calibrag') or ("test" in dataset):
            
            if len(all_data) > 1:
                def extract_key(filename):
                    return filename.split('_')[1].split('.')[0]  # 'bioasq', 'mmlu', 'pubmedqa'

                sort_order = [extract_key(name) for name in new_data_path]
                rag_data_path = sorted(rag_data_path, key=lambda x: sort_order.index(extract_key(x)))

            all_rag_data = [json.load(open(os.path.join(f"./data/{dataset}/rag", p))) 
                            for p in rag_data_path]
            
            print(f"=====> Loaded RAG datasets: {rag_data_path}")
            
        else:
            all_rag_data = ["" for _ in range(len(all_data))]
    
    else:
        raise FileNotFoundError(f"No files found in the folder: ./data/{dataset}")  
    
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
        do_sample=False, #True if temperature != 1.0 else False, 
        temperature=temperature,
        top_p=top_p
    )
    
    ############################# Loading PEFT model ###########################
    ############################# and classifier head ##########################
    if c_type == "ling":
        
        new_words = ["Unlikely", "Doubtful", "Uncertain", 
                     "Ambiguous", "Probable", "Likely", "Possible", 
                    "Specified","Confirmed", "Certain", "Inevitable"]
        
        c_tokens = []
        for word in new_words:
            tokens = tokenizer.tokenize(word, add_special_tokens=False)
            if len(tokens) > 1:
                c_tokens.append(tokens)
            
        tokenizer.add_tokens(new_words)
        resize_c_token_embeddings(tokenizer, model, c_tokens)
    
    if query_peft_dir:
        model = get_lora_model(
            model,
            peft_id_or_dir=query_peft_dir,
            is_trainable=False,
            adapter_name="query",
        )

        if with_classifier:
            if temp and c_type=="calibrag":
                classifier_model = get_classifier_head(
                    input_size=model.config.hidden_size,
                    classifier_model_name="fourier",
                    checkpoint_dir=query_peft_dir,
                    is_trainable=False,
                    weights_name=weights_name,
                    output_size= temp_output,
                )
            else:
                classifier_model = get_classifier_head(
                    input_size=model.config.hidden_size,
                    checkpoint_dir=query_peft_dir,
                    is_trainable=False,
                    weights_name=weights_name,
                    output_size=11 if (multi or multi_reranking) else 2,
                )

            model.classifier_model = classifier_model.to(model.dtype)
            model.classifier_model = model.classifier_model.to(accelerator.device)
            model.classifier_model.target_layer = target_layer
        
    if c_type == "calibrag":
        gen_func = partial(calibrag_case,
                           accelerator=accelerator,
                           model=model,
                           tokenizer=tokenizer,
                           batch_size=batch_size,
                           generation_config=generation_config,
                           inference=inference,
                           top_k=top_k,
                           best_k=best_k,
                           medical=medical,
                           multi=multi,
                           multi_reranking=multi_reranking,
                           doc_query=doc_query,
                           temp=temp,
                           temp_value=temp_value)
        
    elif c_type == "regenerate":
        gen_func = partial(regenerate_query,
                           accelerator=accelerator,
                           model=model,
                           tokenizer=tokenizer,
                           batch_size=batch_size,
                           generation_config=generation_config)
    else:
        gen_func = partial(direct_case,
                           accelerator=accelerator,
                           model=model,
                           tokenizer=tokenizer,
                           batch_size=batch_size,
                           generation_config=generation_config,
                           inference=inference,
                           c_type=c_type,
                           with_classifier=with_classifier,
                           best_k=best_k,
                           medical=medical,
                           temp=temp,
                           temp_value=temp_value,
                           base_reranking=base_reranking)
    
    ########################### Generating outputs #############################
    model.eval()        
    for data_path, rag_data, data in zip(new_data_path, all_rag_data, all_data):
        df = gen_func(dataset=data, rag_dataset=rag_data)   
        path = f"{log_dir}/{c_type}/{dataset}-{c_type}"
        if multi or multi_reranking:
            path = os.path.join(path, "multi")
        elif doc_query:
            path = os.path.join(path, "dq")
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, data_path), index=False) 


if __name__ == "__main__":
    import fire
    fire.Fire(main)