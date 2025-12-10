import gc
import torch
from tqdm.auto import tqdm
from peft import PeftModel
import torch.nn.functional as F
import numpy as np
import os
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
    input_col_name="prompt"):
    
    collate_fn = LabeledStringDataCollator(tokenizer)
    
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


def generate_c_outputs(
    accelerator,
    model,
    tokenizer,
    loader,
    c_type,
    with_classifier=False,
    input_col_name="prompt"):
    
    collate_fn = LabeledStringDataCollator(tokenizer)
    uc_token_vec = get_token_vec(tokenizer, c_type)
    
    results = []
    for inputs in tqdm(loader):
        inputs = inputs[input_col_name]
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
            
            if c_type == "ct":
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
    multi,
    multi_reranking,
    temperature=1.0):
    
    collate_fn = LabeledStringDataCollator(tokenizer)
    
    results = []
    for inputs in tqdm(loader):
        
        inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs['z_prompt']).items()
            }

        if isinstance(model, PeftModel):
            model.set_adapter("query")
        
        try:
            with torch.inference_mode():
                class_inputs = model(**inputs, output_hidden_states=True)
                class_inputs = class_inputs.hidden_states[-2]
                class_inputs = class_inputs[..., -1, :]
            
        except:
            import ipdb; ipdb.set_trace()
            
        outputs = model.classifier_model(class_inputs)
        if temperature != 1.0:
            probs = F.softmax(outputs / temperature, dim=1).detach().float().cpu().numpy()
        else:
            probs = F.softmax(outputs, dim=1).detach().float().cpu().numpy()
        
        if multi:
            preds = np.argmax(probs, axis=1)
            results.extend(preds.tolist())
        elif multi_reranking:
            results.extend(probs.tolist())
        else:
            results.extend(probs[:, 1].tolist())
        
        del inputs 
        del class_inputs
        del outputs
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return results

def generate_rag_temp_outputs(
    accelerator,
    model,
    tokenizer,
    loader,
    multi,
    multi_reranking,
    temperature,
    ):
    
    collate_fn = LabeledStringDataCollator(tokenizer)
    
    results = []
    for inputs in tqdm(loader):
        
        inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs['z_prompt']).items()
            }

        if isinstance(model, PeftModel):
            model.set_adapter("query")
        
        try:
            with torch.inference_mode():
                class_inputs = model(**inputs, output_hidden_states=True)
                class_inputs = class_inputs.hidden_states[-2]
                class_inputs = class_inputs[..., -1, :]
            
        except:
            import ipdb; ipdb.set_trace()
            
        # outputs = model.classifier_model(class_inputs)
        # if temperature != 1.0:
        #     probs = F.softmax(outputs / temperature, dim=1).detach().float().cpu().numpy()
        # else:
        #     probs = F.softmax(outputs, dim=1).detach().float().cpu().numpy()
        
        B = class_inputs.size(0)
        temp_tensor = torch.full((B,), temperature, dtype=torch.float32, device=class_inputs.device)

        outputs = model.classifier_model(class_inputs, temp_tensor)
        probs = F.softmax(outputs, dim=-1).detach().cpu().numpy()
        
        if multi:
            preds = np.argmax(probs, axis=1)
            results.extend(preds.tolist())
        elif multi_reranking:
            results.extend(probs.tolist())
        else:
            results.extend(probs[:, 0].tolist())
        
        del inputs 
        del class_inputs
        del outputs
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def generate_rag_outputs_dq(
    accelerator,
    model,
    tokenizer,
    loader,
    multi,
    multi_reranking,
    temperature=1.0,
    #embedding_cache_dir,
    #retrieval_method="bm25",
):
    '''
    cache_path = os.path.join(embedding_cache_dir, f"{retrieval_method}_past.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    print(f"=====> Loading past_key_value cache from: {cache_path}")
    cached_pkv_dict = torch.load(cache_path)

    collate_fn = LabeledStringDataCollator(tokenizer)
    results = []

    for inputs in tqdm(loader):
        prompts = inputs["z_prompt"]
        doc_prompts = []
        questions = []
        past_key_values_list = []
        doc_attn_masks = []

        for p in prompts:
            if "Question:" not in p:
                raise ValueError(f"Prompt missing 'Question:': {p}")
            doc_prompt, question = p.split("Question:", 1)
            doc_prompt = doc_prompt.strip()
            question = question.strip()

            if doc_prompt not in cached_pkv_dict:
                raise KeyError(f"Document prompt not found in cache: {doc_prompt[:100]}...")

            cache = cached_pkv_dict[doc_prompt]
            doc_prompts.append(doc_prompt)
            questions.append(question)

            past_key_values_list.append([
                (k.to(accelerator.device), v.to(accelerator.device))
                for k, v in cache["past_key_values"]
            ])
            doc_attn_masks.append(cache["attention_mask"].to(accelerator.device))  # shape: (L_doc,)

        # Prepare question input tensors
        q_inputs = collate_fn(questions)
        q_inputs = {k: v.to(accelerator.device) for k, v in q_inputs.items()}

        # Merge attention masks: [doc_attn | q_attn]
        doc_attn_masks = torch.stack(doc_attn_masks, dim=0)  # shape: (B, L_doc)
        q_attn_masks = q_inputs["attention_mask"]             # shape: (B, L_q)
        attention_mask = torch.cat([doc_attn_masks, q_attn_masks], dim=1)  # shape: (B, L_doc + L_q)

        # Reformat past_key_values for batching
        pkv_stacked = list(zip(*past_key_values_list))  # num_layers * batch
        past_key_values = [
            (
                torch.stack([k for k, _ in layer], dim=0),
                torch.stack([v for _, v in layer], dim=0)
            )
            for layer in pkv_stacked
        ]

        with torch.inference_mode():
            out = model(
                input_ids=q_inputs["input_ids"],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            class_input = out.hidden_states[-2][..., -1, :]

        if isinstance(model, torch.nn.Module) and hasattr(model, "set_adapter"):
            model.set_adapter("query")

        logits = model.classifier_model(class_input)
        probs = F.softmax(logits / temperature if temperature != 1.0 else logits, dim=1).detach().cpu().numpy()

        if multi:
            results.extend(probs.argmax(axis=1).tolist())
        elif multi_reranking:
            results.extend(probs.tolist())
        else:
            results.extend(probs[:, 1].tolist())

        del q_inputs, attention_mask, past_key_values, out, class_input
        torch.cuda.empty_cache()

    return results
    '''        
    collate_fn = LabeledStringDataCollator(tokenizer)
    results = []

    for inputs in tqdm(loader):
        prompts = inputs["z_prompt"]
        doc_prompts = []
        questions = []

        for p in prompts:
            if "Question:" not in p:
                raise ValueError(f"Prompt missing 'Question:': {p}")
            doc_part, q_part = p.split("Question:", 1)
            doc_prompts.append(doc_part.strip())
            questions.append(q_part.strip())

        doc_inputs = collate_fn(doc_prompts)
        doc_inputs = {k: v.to(accelerator.device) for k, v in doc_inputs.items()}

        with torch.inference_mode():
            doc_out = model(**doc_inputs, use_cache=True, return_dict=True)
            past_key_values = doc_out.past_key_values

        question_inputs = collate_fn(questions)
        question_inputs = {k: v.to(accelerator.device) for k, v in question_inputs.items()}

        doc_len = doc_inputs["input_ids"].shape[1]
        q_attn = question_inputs["attention_mask"]
        doc_attn = torch.ones((q_attn.shape[0], doc_len), dtype=q_attn.dtype, device=accelerator.device)
        attention_mask = torch.cat([doc_attn, q_attn], dim=1)

        with torch.inference_mode():
            out = model(
                input_ids=question_inputs["input_ids"],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            class_input = out.hidden_states[-2][..., -1, :]

        if isinstance(model, PeftModel):
            model.set_adapter("query")

        logits = model.classifier_model(class_input)
        probs = F.softmax(logits / temperature if temperature != 1.0 else logits, dim=1).detach().cpu().numpy()

        if multi:
            results.extend(probs.argmax(axis=1).tolist())
        elif multi_reranking:
            results.extend(probs.tolist())
        else:
            results.extend(probs[:, 1].tolist())

        del inputs, doc_inputs, question_inputs, attention_mask, doc_out, out, class_input, past_key_values
        gc.collect()
        torch.cuda.empty_cache()

    return results