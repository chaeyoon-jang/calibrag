import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import transformers

from src import load_retriever, load_data, load_passages, normalize, Indexer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def embed_queries(per_gpu_batch_size,
                  lowercase,
                  normalize_text,
                  question_maxlength,
                  queries,
                  model, 
                  tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if lowercase:
                q = q.lower()
            if normalize_text:
                q = normalize(q)
            batch_question.append(q)

            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    
    for i, file_path in enumerate(embedding_files):
        
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def add_passages(queries, passages, top_passages_and_scores):
    assert len(queries) == len(top_passages_and_scores)
    merged_data = []
    for i, q in enumerate(queries):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        merged_data.append([
            {
                "q_id": i,
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ])
    return merged_data


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main(model_name_or_path="facebook/contriever-msmarco",
         no_fp16=False,
         projection_size=768,
         n_subquantizers=0,
         n_bits=8,
         passages_embeddings="wikipedia_embeddings/*",
         passages_path="psgs_w100.tsv",
         indexing_batch_size=1000000,
         n_docs=20,
         dataset="dev",
         lowercase=True,
         normalize_text=True,
         question_maxlength=512,
         per_gpu_batch_size=128,
         ):
    
    # load the quries
    if os.path.exists(f'./data/{dataset}'):
        data_path = os.listdir(f'./data/{dataset}')
        new_data_path = []
        for p in data_path:
            if 'oe' in p:
                new_data_path.append(p)
        print(new_data_path)
        all_data = [pd.read_csv(os.path.join(f"./data/{dataset}", p)) for p in new_data_path] 
    else:
        raise FileNotFoundError(f"No files found in the folder: ./data/{dataset}")
    
    # load the retrieval model
    print(f"Loading model from: {model_name_or_path}")
    model, tokenizer, _ = load_retriever(model_name_or_path)
    model.eval()
    model = model.cuda()
    if not no_fp16:
        model = model.half()

    index = Indexer(projection_size, n_subquantizers, n_bits)

    # index all passages
    input_paths = glob.glob(passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    
    if os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()

        index_encoded_data(index, input_paths, indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        index.serialize(embeddings_dir)

    # load passages
    passages = load_passages(passages_path)
    passage_id_map = {x["id"]: x for x in passages}
    
    # embed queries
    for idx, data in enumerate(all_data):
        queries = list(data['q'])

        print(f"Search {len(queries)} queries...")
        questions_embedding = embed_queries(per_gpu_batch_size,
                                            lowercase,
                                            normalize_text,
                                            question_maxlength,
                                            queries,
                                            model,
                                            tokenizer)
        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        
        merged_data = add_passages(queries, passage_id_map, top_ids_and_scores) 
        
        output_path = os.path.join(f"./data/{dataset}/rag_data", f"{new_data_path[idx].split('/')[-1]}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    import fire 
    fire.Fire(main)