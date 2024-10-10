import gc
import torch
from tqdm.auto import tqdm
from peft import PeftModel
import torch.nn.functional as F
from .llm_data_utils import LabeledStringDataCollator, get_token_vec

def wrapped_generate_output(
    model,
    tokenizer,
    generation_inputs,
    generation_config):
    
    while True:
        try:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            generation_outputs = model.generate(
                **generation_inputs, 
                eos_token_id=terminators,
                generation_config=generation_config
            )
            return generation_outputs
        except Exception as e:
            generation_outputs = []
            new_bs = max(1, generation_inputs["input_ids"].size(0) // 2)
            for i in range(0, generation_inputs["input_ids"].size(0), new_bs):
                inputs = {k: v[i : i + new_bs] for k, v in generation_inputs.items()}
                _outputs = wrapped_generate_output(model, inputs, generation_config)
                generation_outputs.append(_outputs)
            return torch.cat(generation_outputs, dim=0)


def generate_outputs(
    accelerator,
    model,
    tokenizer,
    loader,
    generation_config,
    base_instruction,
    input_col_name="prompt"):
    
    collate_fn = LabeledStringDataCollator(tokenizer, base_instruction)
    
    results = []
    for inputs in tqdm(loader):
        inputs = inputs[input_col_name]
        
        generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
            }

        if isinstance(model, PeftModel):
            model.set_adapter("query")
            
        generation_outputs = wrapped_generate_output(model,
                                          tokenizer,
                                          generation_inputs,
                                          generation_config)
        
        generations = tokenizer.batch_decode(
            generation_outputs[:, generation_inputs.get("input_ids").size(-1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            )
        
        results.extend(generations)
        del generation_outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def generate_uc_outputs(
    accelerator,
    model,
    tokenizer,
    loader,
    base_instruction,
    uc_type,
    with_classifier=False):
    
    collate_fn = LabeledStringDataCollator(tokenizer, base_instruction)
    uc_token_vec = get_token_vec(tokenizer, uc_type)
    
    results = []
    for inputs in tqdm(loader):
        inputs = inputs['uc_prompt']
        inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
            }

        if isinstance(model, PeftModel):
            model.set_adapter("query")
        
        outputs = model(**inputs, output_hidden_states=True)
        
        if with_classifier:
            class_inputs = outputs.hidden_states[-1]
            class_inputs = class_inputs[..., -1, :]
            
            outputs = model.classifier_model(class_inputs)
            prob = F.softmax(outputs, dim=1)[:,1].detach().float().cpu().numpy()
            results.extend(prob.tolist())
            
            del inputs 
            del class_inputs
            del outputs
            
            gc.collect()
            torch.cuda.empty_cache()
        
        else:
            uc_logits = outputs.logits[..., -1, uc_token_vec]
            
            if uc_type == "ct":
                outputs_uc = F.softmax(uc_logits, dim=1)[:,1]
            else:
                outputs_uc = torch.argmax(uc_logits, dim=-1) 
                
            results.extend(outputs_uc.detach().cpu().numpy())
            
            del outputs
            del uc_logits
            del outputs_uc 
            
            gc.collect()
            torch.cuda.empty_cache()
    
    return results


def generate_rag_outputs(
    accelerator,
    model,
    tokenizer,
    loader,
    base_instruction,
    temperature=1.0):
    
    collate_fn = LabeledStringDataCollator(tokenizer, base_instruction)
    
    results = []
    for inputs in tqdm(loader):
        
        inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs['output_prompt']).items()
            }

        if isinstance(model, PeftModel):
            model.set_adapter("query")

        with torch.inference_mode():
            class_inputs = model(**inputs, output_hidden_states=True)
            class_inputs = class_inputs.hidden_states[-1]
            class_inputs = class_inputs[..., -1, :]

        outputs = model.classifier_model(class_inputs)
        if temperature != 1.0:
            prob = F.softmax(outputs / temperature, dim=1)[:, 1].detach().float().cpu().numpy()
        else:
            prob = F.softmax(outputs, dim=1)[:,1].detach().float().cpu().numpy()
        results.extend(prob.tolist())
        
        del inputs 
        del class_inputs
        del outputs
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return results