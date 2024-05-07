import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai

from datetime import timedelta, datetime
from functions.common import api_query

################## Functions for Selection (Self-verification & Ranking) ##################

def verification(data, pred, passage):
    query = f"Question: {data['question']}"
    len_ctxs = len(data['contexts'])
    
    text = ''
    text += f"Question:\n{data['question']}"
    text += f"\n\nPrediction:\n{pred}"
    text += f"\n\nPassage:\n{passage}"
    text += f"\n\nDoes the passage correctly support the prediction? Choices: [True, False]. Answer:"
    
    return text

def use_api_verif(model, dataset, choices, passages, pred_idx):
    res = []
    for i, _ in tqdm(enumerate(dataset)):
        if len(choices[i]) == 1 or passages[i] == 'N/A':
            res.append('False')
        else:
            query = verification(dataset[i], choices[i][pred_idx], passages[i])
            answer = api_query(model, query, 0, 1)[0]
            res.append(answer)
    return res

def ranking_summary(data, passages1, passages2):
    query = f"Question: {data['question']}"
    
    text = "Question: Given the following passages, determine which one provides a more informative answer to the subsequent question."

    text += f"\n\nPassage 1:\n{passages1}"
    
    text += f"\n\nPassage 2:\n{passages2}"
    
    text += f"\n\nTarget Question:\n{query}"
    
    text += f"\n\nYour Task:\nIdentify which passage (Passage 1 or Passage 2) is more relevant and informative to answer the question at hand. Choices: [Passage 1, Passage 2].\n\nAnswer:"
    
    return text

def error_check(res):
    corner_cases = ['passage 1, passage 2', 'choice (a) summary 2', '(b) the wood (summary 2)', '(b) summary 2', '[summary 1, summary 2]', '[summary 1]', '[summary 2]', 'passage 1.', 'passage 2.', 'brady seals', 'passage 1 or passage 2', 'inconclusive', 'no articles', 'sorry', '3', 'uncertain', 'n', 'both', 'either', 'passage 3', 'passage 4', 'passage 5', 'passage 6', 'passage 7', 'passage 8', 'not possible', 'insufficient']

    for item in corner_cases:
        if item in res:
            return True
    return False

def postprocess_rank(pred1, pred2):
    if pred1.lower() == 'passage 1' and pred2.lower() == 'passage 2':
        res = 0
    elif pred1.lower() == 'passage 2' and pred2.lower() == 'passage 1':
        res = 1
    elif pred1.lower() == pred2.lower():
        res = 0.5
    elif error_check(pred1.lower()):
        if error_check(pred2.lower()):
            res = 0.5
        elif pred2.lower() == 'passage 2':
            res = 0
        else:
            res = 1
    elif error_check(pred2.lower()):
        if error_check(pred1.lower()):
            res = 0.5
        elif pred1.lower() == 'passage 1':
            res = 0
        else:
            res = 1
    else:
        res = 0.5 
    return res

def use_api_rank(model, dataset, choices, passages1, passages2):
    res = []
    
    for i, example in tqdm(enumerate(dataset)):
        rank = np.zeros(2)

        # To mitigate order bias of LLMs, we use the averaged results 
        query1 = ranking_summary(example, passages1[i], passages2[i])
        answer1 = api_query(model, query1, 0, 1)[0]

        query2 = ranking_summary(example, passages2[i], passages1[i])
        answer2 = api_query(model, query2, 0, 1)[0]

        res_rank = postprocess_rank(answer1, answer2)
        if res_rank == 0:
            rank[0] += 1
        elif res_rank == 1:
            rank[1] += 1
        else:
            rank += 0.5 
        res.append(rank)
    return res