################################################################################
## This file covers the retrieval process using open-ended query 'q'.
## The retrieval process is conducted using BM25, CDE, and Contriever.
################################################################################
import json
import os
import torch
import random
import numpy as np
import pandas as pd 
import argparse

from transformers import set_seed
from tqdm.autonotebook import tqdm 
from transformers import AutoTokenizer, AutoModel
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from src.rag_utils import (
    load_retriever,
    DenseEncoderModel,
    CDEModel,
    load_passages
)


def generate_dataset_embeddings(minicorpus_docs,
                                model,
                                tokenizer,
                                batch_size):

    ## Stage 1: Gather dataset embeddings.
    minicorpus_docs = tokenizer(
        ['search_document: ' + minicorpus_docs[doc_id]['text']\
            for doc_id in minicorpus_docs.keys()],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    dataset_embeddings = []
    model.eval()
    for i in tqdm(range(0, len(minicorpus_docs["input_ids"]), batch_size)):
        batch = {k: v[i:i+batch_size].cuda() for k, v in minicorpus_docs.items()}
        with torch.no_grad():
            embeddings = model.first_stage_model(**batch)
            dataset_embeddings.append(embeddings.cpu())
            
    return torch.cat(dataset_embeddings)


def parse_args():
    
    parser = argparse.ArgumentParser(
        description='Retrieval for open-ended query.')
    parser.add_argument('--num_retrieval',
                        type=int,
                        default=40)
    parser.add_argument('--data_name',
                        type=str,
                        default='oe_base_train')
    parser.add_argument('--qa_data',
                        type=str,
                        default='./data/dev/oe_base_train.csv')
    parser.add_argument('--documents_pool',
                        type=str,
                        default='./data/retrieval_database/psgs_w100.tsv')
    parser.add_argument('--retrieval_method',
                        type=str,
                        default='bm25',
                        choices=['bm25', 'cde', 'contriever'])
    parser.add_argument('--elastic_search_server',
                        type=str,
                        default='http://localhost:9200')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--output_folder',
                        type=str,
                        default='./data/dev')
    parser.add_argument('--seed',
                        type=int,
                        default=0)
    return parser.parse_args()


def main(args):
    
    set_seed(args.seed)
    # Load QA data.
    print("=====> Loading QA Dataset")

    dataset = pd.read_csv(args.qa_data)
    questions = dataset['x']
    oe_questions = dataset['q']
    answers = dataset['y']
    
    retrieval_queries = {}
    for i in range(len(oe_questions)):
        question = oe_questions[i]
        qa_id = str(args.data_name) + "_" + str(i)
        retrieval_queries[qa_id] = question
        
    # Load pool of documents.
    print("=====> Loading Pool of Documents")
    
    raw_passages = load_passages(args.documents_pool)
    #raw_passages = torch.load(args.documents_pool)
    titles = [item['title'] for item in raw_passages]
    texts = [item['text'] for item in raw_passages]
    
    retrieval_corpus = {}
    for i in range(len(titles)):
        json_obj = {}
        json_obj["title"] = titles[i]
        json_obj["text"] = texts[i]
        retrieval_corpus[str(i)] = json_obj

    # Conducting retrieval.
    print("=====> Starting Retrieval")
    
    if args.retrieval_method == 'bm25':
        model = BM25(hostname=args.elastic_search_server,
                     index_name=args.data_name + "_bm25", initialize=True)
        retriever = EvaluateRetrieval(model)
        
    elif args.retrieval_method == 'contriever':
        model, tokenizer, _ = load_retriever('facebook/contriever-msmarco')
        model = model.cuda()
        model.eval()
        query_encoder = model
        doc_encoder = model

        model = DRES(DenseEncoderModel(query_encoder=query_encoder,
                                       doc_encoder=doc_encoder,
                                       tokenizer=tokenizer),
                     batch_size=args.batch_size)
        retriever = EvaluateRetrieval(model, score_function="dot")
        
    elif args.retrieval_method == 'cde':
        print("=====> Generating Dataset Embeddings")
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("jxm/cde-small-v1", 
                                          trust_remote_code=True).cuda()
        
        ## For generating dataset embeddings, 
        ## we need to sample a subset of the corpus.
        if len(retrieval_corpus) >= 512:
            selected_keys = random.sample(list(retrieval_corpus.keys()), k=512)
            minicorpus_docs = {key: retrieval_corpus[key] 
                               for key in selected_keys}
        else:
            minicorpus_docs = retrieval_corpus 
        
        dataset_embeddings = generate_dataset_embeddings(minicorpus_docs, 
                                                         model, 
                                                         tokenizer,
                                                         args.batch_size)
        
        print("=====> Retrieving with CDE")
        model.eval()
        query_encoder = model 
        doc_encoder = model
        
        model = DRES(CDEModel(query_encoder=query_encoder,
                              doc_encoder=doc_encoder,
                              tokenizer=tokenizer,
                              dataset_embeddings=dataset_embeddings),
                     batch_size=args.batch_size)
        retriever = EvaluateRetrieval(model, score_function="dot")
        
    else:
        raise ValueError("Wrong retrieval method is inserted.")
    
    #if args.retrieval_method != 'cde':
    #try:
    retrieval_scores = retriever.retrieve(retrieval_corpus,
                                          retrieval_queries)
    print("retrieval done") 
    #except Exception as e:
    #    print("retrieval exception: " + str(e))
        
    # Construct dataset using retrieved scores.
    print("=====> Starting Construction of Dataset")
    sorted_idxs = []
    sorted_scores = []

    retrieval_scores_idxs = list([int(n.split("_")[-1])
                                  for n in retrieval_scores.keys()])
    
    for i in retrieval_scores_idxs:
        scores_i = np.array(list(
            retrieval_scores['{}_{}'.format(args.data_name, i)].values()))
        sorted_idx = np.argsort(scores_i)[::-1]
        keys = list(retrieval_scores['{}_{}'.format(args.data_name, i)].keys())

        sorted_idxs_i = []
        sorted_scores_i = []
        for j in range(min(len(scores_i), args.num_retrieval)):
            sorted_idxs_i.append(int(keys[sorted_idx[j]]))
            sorted_scores_i.append(scores_i[sorted_idx[j]])

        sorted_idxs.append(sorted_idxs_i)
        sorted_scores.append(sorted_scores_i)
        
    res = []
    for idx, i in enumerate(retrieval_scores_idxs):
        new_item = {}
        new_item['q'] = oe_questions[i]
        new_item['y'] = answers[i]
        new_item['x'] = questions[i]

        ctxs = []
        for j in range(len(sorted_idxs[idx])):
            ctx = {}
            ctx['id'] = sorted_idxs[idx][j]
            ctx['title'] = titles[sorted_idxs[idx][j]]
            ctx['text'] = texts[sorted_idxs[idx][j]]
            ctx['score'] = sorted_scores[idx][j]
            ctxs.append(ctx)
        new_item['contexts'] = ctxs
        res.append(new_item)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    print("=====> All Procedure is finished!")
    with open(f'./{args.output_folder}/{args.data_name}_{args.retrieval_method}.json',
              "w", encoding='utf-8') as writer:
        writer.write(json.dumps(res, indent=4, ensure_ascii=False) + "\n")
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)