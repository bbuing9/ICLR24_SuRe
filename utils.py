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
                engine=model,
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