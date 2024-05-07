import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai
import os 

from datetime import timedelta, datetime
from functions_openllm.gen_candidates import post_process_candidate, separation, use_api_candidate
from functions_openllm.gen_summary import use_api_summary
from functions_openllm.verification import use_api_verif, use_api_rank

################## Unified Implementation of SuRe ##################

def sure_infer(model, model_type, tokenizer, dataset, n_articles=10, output_path='./'):
    print("=====> SuRe Step #1. Candidate Generation")
    
    path_to_candidate = output_path + '/{}'.format('candidates.json')
    if not os.path.exists(path_to_candidate):
        sure_candidate = use_api_candidate(model, model_type, tokenizer, dataset, n_articles=n_articles)
        with open(path_to_candidate, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(sure_candidate, indent=4, ensure_ascii=False) + "\n")
    else:
        sure_candidate = json.load(open(path_to_candidate))
    sure_candidate_post = post_process_candidate(sure_candidate)
    sure_candidate1, sure_candidate2 = separation(sure_candidate_post)

    print("=====> SuRe Step #2. Conditional Summarization")
    path_to_summary1 = output_path + '/{}'.format('summary1.json')
    if not os.path.exists(path_to_summary1):
        summary_candidate1 = use_api_summary(model, model_type, tokenizer, dataset, sure_candidate_post, pred=0, n_articles=n_articles)
        with open(path_to_summary1, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(summary_candidate1, indent=4, ensure_ascii=False) + "\n")
    else:
        summary_candidate1 = json.load(open(path_to_summary1))
    path_to_summary2 = output_path + '/{}'.format('summary2.json')
    if not os.path.exists(path_to_summary2):
        summary_candidate2 = use_api_summary(model, model_type, tokenizer, dataset, sure_candidate_post, pred=1, n_articles=n_articles)
        with open(path_to_summary2, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(summary_candidate2, indent=4, ensure_ascii=False) + "\n")
    else:
        summary_candidate2 = json.load(open(path_to_summary2))

    print("=====> SuRe Step #3. Self-Verification and Ranking")
    path_to_verfi1 = output_path + '/{}'.format('verif1.json')
    if not os.path.exists(path_to_verfi1):
        correctness_summary1 = use_api_verif(model, model_type, tokenizer, dataset, sure_candidate_post, summary_candidate1, pred_idx=0)
        with open(path_to_verfi1, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(correctness_summary1, indent=4, ensure_ascii=False) + "\n")
    else:
        correctness_summary1 = json.load(open(path_to_verfi1))

    path_to_verfi2 = output_path + '/{}'.format('verif2.json')
    if not os.path.exists(path_to_verfi2):
        correctness_summary2 = use_api_verif(model, model_type, tokenizer, dataset, sure_candidate_post, summary_candidate2, pred_idx=1)
        with open(path_to_verfi2, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(correctness_summary2, indent=4, ensure_ascii=False) + "\n")
    else:
        correctness_summary2 = json.load(open(path_to_verfi2))
    
    path_to_rank = output_path + '/{}'.format('ranking.npy')
    if not os.path.exists(path_to_rank):
        ranking_summary12 = use_api_rank(model, model_type, tokenizer, dataset, sure_candidate_post, summary_candidate1, summary_candidate2)
        np.save(path_to_rank, ranking_summary12)
    else:
        ranking_summary12 = np.load(path_to_rank)
    
    sure_fin_preds, sure_fin_summary, all_indices = get_final_pred_sure(sure_candidate1, sure_candidate2, summary_candidate1, summary_candidate2, correctness_summary1, correctness_summary2, ranking_summary12)
    path_to_index = output_path + '/{}'.format('indices.json')
    with open(path_to_index, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_indices, indent=4, ensure_ascii=False) + "\n")
    path_to_summary = output_path + '/{}'.format('results_summary.json')
    with open(path_to_summary, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(sure_fin_summary, indent=4, ensure_ascii=False) + "\n")

    return sure_fin_preds

################## Functions to Get Prediction ##################

def error_check_fin(res):
    corner_cases = ['Cannot', 'False', 'Unknown', 'N/A']
    
    for item in corner_cases:
        if item in res:
            return True
    return False

def get_final_pred_sure(candidate1, candidate2, summary1, summary2, correct1, correct2, ranking):
    n_sample = len(candidate1)
    n_choices = np.zeros(3)
    res, res_summary = [], []
    cand1_indices, cand2_indices, tie_indices = [], [], []
    for i in range(n_sample):
        rank_i = ranking[i]

        if ('True' in correct1[i] and 'True' in correct2[i]) or ('False' in correct1[i] and 'False' in correct2[i]):
            rank_i = rank_i
        elif 'True' in correct1[i] and error_check_fin(correct2[i]):
            rank_i = 0.5 * rank_i + 0.5 * np.array([1,0])
        elif error_check_fin(correct1[i]) and 'True' in correct2[i]:
            rank_i = 0.5 * rank_i + 0.5 * np.array([0,1])
        else:
            rank_i = rank_i
        
        max_vote = np.max(rank_i)

        # If Tie, then select first candidate as answer 
        if (rank_i == max_vote).sum() > 1:
            res.append([candidate1[i][0]])
            res_summary.append(summary1[i])
            n_choices[1] += 1
            tie_indices.append(i)
        else:
            select_idx = np.argmax(rank_i)
            if select_idx == 0:
                res.append([candidate1[i][0]])
                res_summary.append(summary1[i])
                n_choices[0] += 1
                cand1_indices.append(i)
            else:
                res.append([candidate2[i][0]])
                res_summary.append(summary2[i])
                n_choices[2] += 1
                cand2_indices.append(i)
    return res, res_summary, [cand1_indices, tie_indices, cand2_indices]

