import os
import time
import wandb
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from g4f.client import Client
from openai import OpenAI, APIError
from concurrent.futures import ThreadPoolExecutor

from src.logging import entrypoint, logging
from src.evaluate_fn import (
    compute_auroc,
    compute_brier_score,
    compute_ece,
    compute_nll
)
from src.llm_data_utils import get_loader
from src.prompt_hub import (
    EVALUATION_INSTRUCTION,
    EVALUATION_PROMPT,
    EVALUATION_PROMPT_10,
    EVALUATION_PROMPT_20,
    Q_X_INSTRUCTION,
    Q_X_PROMPT,
    uc_to_number
)


def openai_query(
    system_prompt,
    prompt,
    openai_model_name="gpt-3.5-turbo",
    max_tokens=40):
    
    if openai_model_name == "gpt-3.5-turbo":
        client = Client()
    else:
        client = OpenAI()
       
    sampled_response = None
    while sampled_response is None:
        try:
            response = client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            sampled_response = response.choices[0].message.content
        except APIError:
            logging.exception("OpenAI API Error.", exc_info=True)
            time.sleep(1)
    return sampled_response


def evaluate_equivalency_with_oracle(
    ground_truth,
    prediction,
    question,
    oracle_fn,
    oracle_kwargs
):
    if isinstance(prediction, list):
        ## This is for the case where we have multiple predictions.
        prompt = EVALUATION_PROMPT_10 if len(prediction) == 10\
            else EVALUATION_PROMPT_20
            
        for i in range(len(prediction)):
            prompt = prompt.replace(f"<prediction{i}>", prediction[i])
            
        sampled_response = oracle_fn(
            system_prompt=EVALUATION_INSTRUCTION, prompt=prompt, **oracle_kwargs
        )
        return sampled_response.strip().lower()
    
    else:
        ## This is for the case where we have a single prediction.
        prompt = (
            EVALUATION_PROMPT.replace("<ground-truth>", ground_truth)
            .replace("<prediction>", prediction)
            .replace("<question>", question)
            )
        
        sampled_response = oracle_fn(
            system_prompt=EVALUATION_INSTRUCTION, prompt=prompt, **oracle_kwargs
        )
        
        return "yes" in sampled_response.strip().lower()


def generate_oe_query_with_oracle(
    question,
    oracle_fn,
    oracle_kwargs
):
    prompt = Q_X_PROMPT.replace("<question>", question)
    
    sampled_response = oracle_fn(
        system_prompt=Q_X_INSTRUCTION, prompt=prompt, **oracle_kwargs
    )
    
    return sampled_response


def grade_oe_preds(
    true,
    pred,
    questions,
    strategy="substring",
    max_threads=50,
    max_tokens=40,
):
    if strategy == "substring":
        comparison_fn = lambda t, p, q: t in p
        
    elif "gpt" in strategy:
        comparison_fn = lambda t, p, q: evaluate_equivalency_with_oracle(
            t,
            p,
            q,
            oracle_fn=openai_query,
            oracle_kwargs={"openai_model_name": strategy,
                           "max_tokens": max_tokens})  
        
    else:
        raise ValueError(f"Invalid comparison strategy {strategy}")
    
    with ThreadPoolExecutor(min(max_threads, len(true))) as p:
        results = list(p.map(comparison_fn, true, pred, questions))
        
    return results


def generate_oe_queries(
    questions,
    max_threads=50,
    max_tokens=40,
    strategy="gpt-4o-mini"
):
    comparison_fn = lambda q: generate_oe_query_with_oracle(
        q,
        oracle_fn=openai_query,
        oracle_kwargs={"openai_model_name": strategy,
                       "max_tokens": max_tokens})  

    with ThreadPoolExecutor(min(max_threads, len(questions))) as p:
        results = list(p.map(comparison_fn, questions))
        
    return results


def evaluate_all_metrics(data, c_type):
    
    idxs = []
    for i in range(len(data)):
        if 'no answer' in data['y_pred'][i].lower():
            idxs.append(i)
            
    no_answer = len(idxs)/len(data)
    new = data.drop(idxs, axis=0).reset_index(drop=True)
    
    y_true = list(new['correct'])

    if c_type == "ling":
        y_prob  = [uc_to_number[n]/10 for n in list(new['c'])]
    
    elif c_type == "number":
        y_prob  = [float(n.split(' (')[0])/10 for n in list(new['c'])] 
        
    else:
        try:
            y_prob = [float(n.split(' (')[0])/100 for n in list(new['c'])]
        except:
            y_prob = new['c']
            
    acc = sum(y_true)/len(data)
    ece = compute_ece(y_true, y_prob, n_bins=15)
    brier_score = compute_brier_score(y_true, y_prob)
    auroc = compute_auroc(y_true, y_prob)
    nll = compute_nll(y_true, y_prob)

    print("AUROC:", auroc)
    print("ACC: ", acc)
    print("ECE:", ece)
    print("Brier Score:", brier_score)
    print("NLL:", nll)
    print("NA", no_answer)
    return auroc, acc, ece, brier_score, nll
        

@entrypoint(with_accelerator=True)
def main(
    type="eval",
    data_dir=None,
    strategy="gpt-4o-mini",
    log_dir=None,
    accelerator=None,
    batch_size=4,
    multiple=False,
    max_tokens=40,
    c_type=None
    ):
    
    print(log_dir)
    
    if os.path.exists(data_dir):
        with accelerator.main_process_first():
            all_data_path = os.listdir(data_dir)
        print(all_data_path)       
        all_data = [pd.read_csv(os.path.join(data_dir, p))
                    .dropna().reset_index(drop=True) for p in all_data_path] 
    
    else:
        raise FileNotFoundError(f"No files found in the folder: {data_dir}")
    
    if type == "eval":
        for data_path, data in zip(all_data_path, all_data):
            
            if 'oc' in data.columns:
                continue 
            
            elif 'y_pred' in data.columns:
                                
                loader = get_loader(Dataset.from_pandas(data),
                                    batch_size=batch_size,
                                    pin_memory=True,
                                    accelerator=accelerator)
                
                results = []
                for inputs in tqdm(loader):
                    inputs = [dict(zip(inputs.keys(), vals)) 
                              for vals in zip(*inputs.values())]
                    targets = [inp.pop("y") for inp in inputs]
                    
                    if multiple:
                        outputs = [eval(inp.pop("y_pred")) for inp in inputs]
                    else:
                        outputs = [inp.pop("y_pred") for inp in inputs] 
                        
                    questions = [inp.pop("x") for inp in inputs]
        
                    result = grade_oe_preds(targets,
                                            outputs,
                                            questions,
                                            strategy,
                                            max_tokens)
                    results.extend(result)
                                
                data['correct'] = results
                data['correct'] = data['correct'].astype(int)
                
                os.makedirs(f"{log_dir}", exist_ok=True)
                data.to_csv(f"{log_dir}/test_{data_path}", index=False) 
                
                if c_type:
                    auroc, acc, ece, brier_score, nll = evaluate_all_metrics(data, c_type=c_type)
                    
                    metrics = {
                        'auroc': auroc,
                        'acc': acc,
                        'ece': ece,
                        'brier_score': brier_score,
                        'nll': nll
                    }
                    results = pd.DataFrame([metrics], index=[data_path])
                    table = wandb.Table(dataframe=results)
                    wandb.log({"evaluation_metrics": table})

            else:
                #TODO: exception error
                exit()
    
    elif type == "oe":
        for data_path, data in zip(all_data_path, all_data):
            if 'q' in data.columns:
                continue 
            elif 'x' in data.columns:
                loader = get_loader(Dataset.from_pandas(data),
                                    batch_size=batch_size,
                                    pin_memory=True,
                                    accelerator=accelerator)
                
                results = []
                for inputs in tqdm(loader):
                    inputs = [dict(zip(inputs.keys(), vals)) 
                              for vals in zip(*inputs.values())] 
                    questions = [inp.pop("x") for inp in inputs]
                    result = generate_oe_queries(questions,
                                                 strategy=strategy,
                                                 max_tokens=max_tokens)
                    results.extend(result)
                    
                data['q'] = results
                
                os.makedirs(f"{log_dir}", exist_ok=True)
                data.to_csv(f"{log_dir}/oe_{data_path}", index=False) 
                
            else:
                #TODO: exception error
                exit() 
                

if __name__ == "__main__":
    import fire 
    fire.Fire(main)