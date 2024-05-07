import pprint
import json
import copy
import os
import numpy as np
from tqdm import tqdm
import time
import argparse

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from beir.reranking import Rerank

import src_con.contriever
from src_con.dense_model import DenseEncoderModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retrieval for Open-domain QA.')
    parser.add_argument('--num_retrieval', type=int, default=100)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--qa_data', type=str, default=None)
    parser.add_argument('--documents_pool', type=str, default=None)
    parser.add_argument('--retrieval_method', type=str, default='contriever', choices=['bm25', 'dpr', 'contriever'])
    parser.add_argument('--elastic_search_server', type=str, default=None, help='required to use bm25')
    parser.add_argument('--batch_size', type=int, default=64, help='required to use dpr or contriever')
    parser.add_argument('--output_folder', type=str, default=None)

    args = parser.parse_args()

    # Load QA data
    print("=====> Loading QA Dataset")

    dataset = json.load(open(args.qa_data))
    questions = dataset['question']
    answers = dataset['answer']
    
    retrieval_queries = {}
    for i in range(len(questions)):
        question = questions[i]
        qa_id = str(args.data_name) + "_" + str(i)
        retrieval_queries[qa_id] = question

    # Load pool of documents
    print("=====> Loading Pool of Documents")
    
    raw_passages = json.load(open(args.documents_pool))
    titles = raw_passages['title']
    texts = raw_passages['text']

    retrieval_corpus = {}
    for i in range(len(titles)):
        json_obj = {}
        json_obj["title"] = titles[i]
        json_obj["text"] = texts[i]
        retrieval_corpus[str(i)] = json_obj

    # Conducting retrieval
    print("=====> Starting Retrieval")
    
    if args.retrieval_method == 'bm25':
        model = BM25(hostname=args.elastic_search_server, index_name=args.data_name + "_bm25", initialize=True)
        retriever = EvaluateRetrieval(model)
    elif args.retrieval_method == 'contriever':
        model, tokenizer, _ = src_con.contriever.load_retriever('facebook/contriever-msmarco')
        model = model.cuda()
        model.eval()
        query_encoder = model
        doc_encoder = model

        model = DRES(DenseEncoderModel(query_encoder=query_encoder, doc_encoder=doc_encoder, tokenizer=tokenizer),batch_size=args.batch_size)
        retriever = EvaluateRetrieval(model, score_function="dot")
    elif args.retrieval_method == 'dpr':
        model = DRES(models.SentenceBERT(("facebook-dpr-question_encoder-multiset-base",
                                  "facebook-dpr-ctx_encoder-multiset-base",
                                  " [SEP] "), batch_size=64))
        retriever = EvaluateRetrieval(model, score_function="dot")
    else:
        raise ValueError("Wrong retrieval method is inserted.")

    try:
        retrieval_scores = retriever.retrieve(retrieval_corpus, retrieval_queries)
        print("retrieval done") 
    except Exception as e:
        print("retrieval exception: " + str(e))

    # Construct dataset using retrieved scores 
    print("=====> Starting Construction of Dataset")
    
    sorted_idxs = []
    sorted_scores = []

    for i in range(len(raw_passages)):
        scores_i = np.array(list(retrieval_scores['{}_{}'.format(args.data_name, i)].values()))
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
    for i in range(len(dataset)):
        new_item = {}
        new_item['question'] = questions[i]
        new_item['answer'] = answers[i]

        ctxs = []
        for j in range(len(sorted_idxs[i])):
            ctx = {}
            ctx['id'] = sorted_idxs[i][j]
            ctx['title'] = titles[sorted_idxs[i][j]]
            ctx['text'] = texts[sorted_idxs[i][j]]
            ctx['score'] = sorted_scores[i][j]
            ctxs.append(ctx)
        new_item['contexts'] = ctxs
        res.append(new_item)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    print("=====> All Procedure is finished!")
    with open(f'./{args.output_folder}/{args.data_name}_{args.retrieval_method}.json', "w", encoding='utf-8') as writer:
        writer.write(json.dumps(res, indent=4, ensure_ascii=False) + "\n")