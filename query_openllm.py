import os
# import pprint
import json
import copy
import time
import argparse

import numpy as np

from tqdm import tqdm

from functions_openllm import use_api_base, sure_infer
from data_utils import get_em_f1

from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query QA Data to GPT API.')
    parser.add_argument('--data_name', type=str, default=None, help='Name of QA Dataset')
    parser.add_argument('--qa_data', type=str, default=None, help='Path to QA Dataset')
    parser.add_argument('--start', type=int, default=None, help='Start index of QA Dataset')
    parser.add_argument('--end', type=int, default=None, help='End index of QA Dataset')
    parser.add_argument('--lm_type', type=str, default='llama2', help='Type of LLM (llama2, gemma, mistral)')
    parser.add_argument('--n_retrieval', type=int, default=10, help='Number of retrieval-augmented passages')
    parser.add_argument('--infer_type', type=str, default='sure', help='Inference Method (base or sure)', choices=['base', 'sure'])
    parser.add_argument('--output_folder', type=str, default=None, help='Path for save output files')
    
    args = parser.parse_args()

    # Load QA Dataset
    print("=====> Data Load...")
    dataset = json.load(open(args.qa_data))
    start_idx, end_idx = args.start, args.end
    if start_idx is None:
        start_idx = 0
    elif end_idx is None:
        end_idx = len(dataset)
    else:
        if start_idx >= end_idx:
            raise ValueError
    dataset = dataset[start_idx:end_idx]
    print("Number of QA Samples: {}".format(len(dataset)))

    # Load LLM (Caution. In the paper, we only validate the generalization of SuRe with LLaMA2-chat-70B. Below LLMs are just included for the convenience) 
    if args.lm_type == "gemma":
        model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    elif args.lm_type == "mistral":
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    elif args.lm_type == "llama":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    else:
        raise ValueError
    model = model.cuda()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    method = f'{args.data_name}_start{start_idx}_end{end_idx}_{args.infer_type}_ret{str(args.n_retrieval)}'
    method_folder = args.output_folder + '/{}'.format(method)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    print("=====> Begin Inference (type: {})".format(args.infer_type))
    if args.infer_type == 'base':
        results = use_api_base(model, args.lm_type, tokenizer, dataset, n_articles=args.n_retrieval)
    else:
        results = sure_infer(model, args.lm_type, tokenizer, dataset, n_articles=args.n_retrieval, output_path=method_folder)

    print("=====> All Procedure is finished!")
    with open(f'./{method_folder}/results.json', "w", encoding='utf-8') as writer:
        writer.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")

    print("=====> Results of {}".format(method))
    em, f1 = get_em_f1(dataset, results)
    print("EM: {} F1: {}".format(em.mean(), f1.mean()))
    
    # To compare sure's summarization with generic one
    ans_idx = np.where(em == 1)[0]
    np.save(f'./{method_folder}/{args.infer_type}_ans_idx.npy', ans_idx)
