import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai

from datetime import timedelta, datetime
from functions_openllm.common import api_query

################## Functions for Conditional Summarization ##################

def convert_choices_to_texts(choices):
    res = ''
    
    for i, item in enumerate(choices):
        order_txt = '({})'.format(chr(ord('a') + i))
        ith_txt = order_txt + ' ' + item + ' '
        res += ith_txt
    
    return res[:-1]

def gen_summary_mcq(dataset, choices, idx, pred_idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    
    text = ''
    for i in range(start_idx, start_idx + n_articles):
        idx_ctx = (i % len_ctxs)
        text += f"\n\nPassage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']}"

    text += f"\n\nYour job is to act as a professional writer. You will write a good-quality passage that can support the given prediction about the question only based on the information in the provided supporting passages.\n\nNow, let's start. After you write, please write [DONE] to indicate you are done. Do not write a prefix (e.g., \"Response:\") while writing a passage."
    text += f"\n\nQuestion: {data['question']}\nChoices:{convert_choices_to_texts(choices[idx])}\nPrediction:({chr(ord('a')+pred_idx)}) {choices[idx][pred_idx]}"
    text += f"\nPassage: "
    
    return text

def use_api_summary(model, model_type, tokenizer, dataset, choices, pred, n_articles=10, start_idx=0):
    queries = []
    answers = []
    for i, example in enumerate(dataset):
        if len(choices[i]) == 1 or choices[i][pred] == 'N/A':
            pass
        else:
            query = gen_summary_mcq(dataset, choices, i, pred, n_articles)
            queries.append(query)
    
    for query in tqdm(queries):
        answer = api_query(model, model_type, tokenizer, query)
        answers.extend(answer)

    res = ['N/A' for _ in range(len(dataset))]
    answer_idx = 0
    for i, example in enumerate(dataset):
        if len(choices[i]) == 1 or choices[i][pred] == 'N/A':
            pass
        else:
            res[i] = answers[answer_idx]
            answer_idx += 1
    
    return res