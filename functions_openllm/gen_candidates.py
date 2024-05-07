import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai

from datetime import timedelta, datetime
from functions_openllm.common import api_query

################## Functions for Candidate Generation ##################

# Note. Without constraints on output words, LLMs are too verbose as we consider zero-shot setup. 
# Therefore, we commonly apply constraints on number of words for both baseline and ours.

def get_query_candidate(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text = f"Below are {n_articles} passages related to the question at the end. After reading the passages, provide two correct candidates for the answer to the question at the end. Each answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."
    for i in range(start_idx, start_idx + n_articles):
        idx_ctx = (i % len_ctxs)
        text += f"\n\nPassage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']}"
    text += f"\n\nQuestion:\n{data['question']}"
    
    ## Even with LLaMa2-70B, openLLM does not follow the instruciton. Therefore, we remind LLM the instruction at the end.
    text += f"\n\nEach answer should be in the form: (a) xx, (b) yy, and should not exceed 3 words for each candidate."
    text += f"\n\nAnswer: "
    return text

def use_api_candidate(model, model_type, tokenizer, dataset, n_articles=10, start_idx=0):
    res = []
    queries = []
    
    for i, example in enumerate(dataset):
        query = get_query_candidate(dataset, i, n_articles, start_idx)
        queries.append(query)
    
    for query in tqdm(queries):
        answer = api_query(model, model_type, tokenizer, query)
        res.extend([[ans] for ans in answer])

    return res

def normalize_answer(s):
    chrs = [' ', ',', '.']
    while s[0] in chrs:
        s = s[1:]

    while s[-1] in chrs:
        s = s[:-1]
    
    return s

def divide_candidates(raw_candidates):
    res = []
    for item in raw_candidates:
        res_item = []
        raw_candidate = item[0]
        for i in range(4):
            try:
                target_symbol = chr(i + ord('a'))
                idx = raw_candidate.index(f'({target_symbol})')
                if i < 3: 
                    try:
                        next_symbol = chr(i + 1 + ord('a'))
                        idx_next = raw_candidate.index(f'({next_symbol})')
                        res_item.append(normalize_answer(raw_candidate[idx + len(target_symbol) + 2:idx_next]))
                    except:
                        res_item.append(normalize_answer(raw_candidate[idx + len(target_symbol) + 2:]))
                        break
                else:
                    res_item.append(normalize_answer(raw_candidate[idx + len(target_symbol) + 2:]))
            except:
                res_item.append(normalize_answer(raw_candidate))
                break
        res.append(res_item)
    return res

def handle_except(res_candidates, raw_candidates):
    for i, item in enumerate(res_candidates):
        if len(item) == 1:
            if len(item[0]) == 0:
                idx = raw_candidates[i][0].index(f'(a)')
                res_candidates[i] = raw_candidates[i][0][:idx]
            elif len(item[0].split(',')) > 4:
                print(i)
                print(item)
                res_candidates[i] = raw_candidates[i]
            else:
                new_res_candidate = []
                for split in res_candidates[i][0].split(','):
                    new_res_candidate.append(normalize_answer(split))
                res_candidates[i] = new_res_candidate
    return res_candidates

def get_choices_sampling(preds):
    choices = []
    avg_len = 0
    
    for pred in preds:
        choices_i = []

        for pred_i in pred:
            if pred_i not in choices_i:
                choices_i.append(pred_i)
        choices.append(choices_i)
        avg_len += len(choices_i)

    return choices

def post_process_candidate(raw_candidates):
    divided_candidates = divide_candidates(raw_candidates)
    res_candidates = handle_except(divided_candidates, raw_candidates)
    choices_candidates = get_choices_sampling(res_candidates)
    return choices_candidates

def separation(choices, n_choice=2):
    res = [] 
    for i in range(n_choice):
        res_i = []
        for choice in choices:
            if len(choice) > i:
                res_i.append([choice[i]])
            else:
                res_i.append(['N/A'])
        res.append(res_i)
    return res    

