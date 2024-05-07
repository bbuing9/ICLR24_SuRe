import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai

from datetime import timedelta, datetime

################## Basic Functions ##################

def api_query(model, query, temp, iters):
    waiting_time = 0.5
            
    response = None
    while response is None:
        try:
            messages = [
                    {"role": "system", "content": query},
            ]
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temp,
                n=iters
            )
        except:
            time.sleep(waiting_time)
            if waiting_time < 5:
                waiting_time += 0.5
            else:
                break
    
    res_iter = []    
    if response is not None:
        for iter in range(iters):    
            try:
                answer = response['choices'][iter]['message']['content']
            except:
                answer = 'N/A'
            res_iter.append(answer)
    else:
        for iter in range(iters):
            res_iter.append('N/A')

    return res_iter

# Note. Without constraints on output words, LLMs are too verbose as we consider zero-shot setup. 
# Therefore, we commonly apply constraints on number of words for both baseline and ours.

def get_query_baseline(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    if n_articles > 0:
        text = ""
        for i in range(start_idx, start_idx+n_articles):
            idx_ctx = (i % len_ctxs)
            text += f"Passage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']} \n\n"
        text += f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "
    else:
        text = f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "
        
    return text

def use_api_base(model, dataset, iters=1, n_articles=10, temp=0.0, start_idx=0):
    res = []
    
    for i, example in tqdm(enumerate(dataset)):
        query = get_query_baseline(dataset, i, n_articles, start_idx)
        res_i = api_query(model, query, temp, iters)
        res.append(res_i)
    return res