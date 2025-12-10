import os
import time
import wandb
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from src.evaluate_fn import (
    compute_auroc,
    compute_brier_score,
    compute_ece,
    compute_nll
)
from src.llm_data_utils import get_loader
from src.prompt_hub import (
    EVALUATION_PROMPT,
    uc_to_number
)

def eval_using_vllm(
    true,
    pred,
    questions,
    tokenizer, 
    model,
    sampling_params):

    inputs = [EVALUATION_PROMPT.replace("<ground-truth>", ground_truth)
              .replace("<prediction>", prediction)
              .replace("<question>", question)
              for ground_truth, prediction, question in zip(true, pred, questions)]
    inputs = [tokenizer.apply_chat_template(
        [{"role": "user", "content": inp}],
        tokenize=False,
        add_generation_prompt=True) for inp in inputs]
    
    outputs = model.generate(
        inputs,
        sampling_params,
        use_tqdm=False)
    outputs = [outputs[i].outputs[0].text for i in range(len(outputs))]
    outputs = ["yes" in sampled_response.strip().lower() for sampled_response in outputs]
    return outputs
    
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
        import numpy as np 
        bins = np.linspace(0, 1, 12)
        avg_bins = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        y_prob = []
        for n in list(new['c']):
            val = float(n.split(' (')[0])
            if val.is_integer():
                y_prob.append(avg_bins[int(val)])
            else:
                y_prob.append(val / 10)
        
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
           
def main(
    type="eval",
    data_dir=None,
    strategy="gpt-4o-mini",
    log_dir=None,
    accelerator=None,
    batch_size=4,
    multiple=False,
    max_tokens=5,
    c_type=None
    ):
    
    print(log_dir)
    
    if os.path.exists(data_dir):
        all_data_path = os.listdir(data_dir)
        print(all_data_path)       
        all_data = [pd.read_csv(os.path.join(data_dir, p))
                    .dropna().reset_index(drop=True) for p in all_data_path] 
    
    else:
        raise FileNotFoundError(f"No files found in the folder: {data_dir}")
    
    if strategy == "vllm":
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    
        sampling_params = SamplingParams(
            max_tokens = max_tokens,
            temperature = 0.0,
            top_p = 1.0,
            seed = 0,
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        model = LLM(model="meta-llama/Llama-3.2-3B-Instruct")
        
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
                    
                    result = eval_using_vllm(targets,
                                            outputs,
                                            questions,
                                            tokenizer,
                                            model,
                                            sampling_params)
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
            else:
                #TODO: exception error
                exit()
    

if __name__ == "__main__":
    import fire 
    fire.Fire(main)