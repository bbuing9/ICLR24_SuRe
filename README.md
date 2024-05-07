# SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs

This repository provides datasets, and code for the following paper (will be released soon!):

> [SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs](https://arxiv.org/abs/2404.13081) <br>
> [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), Jaehyun Nam, Sangwoo Mo, Jongjin Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha, Jinwoo Shin <br>
> [ICLR 2024](https://iclr.cc/) <be>

## Preliminary

The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using `Python 3.7`.

# Initial Process: Preparation of Dataset with Retrieved Passages 

Before using our framework, one needs to prepare the QA dataset by retrieving relevant documents for each question. For the experiments, we retrieved passages using [BEIR](https://github.com/beir-cellar/beir) framework. Specifically, one can generate own dataset with `retrieve.py`, by (1) preparing a target QA dataset and a pool of retrieved documents and (2) inserting the desired retrieval method among `[bm25, dpr, contriever]`. 
- We assume that both the QA dataset and the pool of retrieved documents have `json` format, where the former has `question` and  `answer` fields, and the latter has `title` and  `text` fields.
- To use BM25, one needs to be able to use [ElasticSearch]([https://github.com/beir-cellar/beir](https://elasticsearch-py.readthedocs.io/en/v8.13.0/)) in local server and designate the location through `--elastic_search_server`.
- For contriever, we adopted the original code base from the [official github](https://github.com/facebookresearch/contriever). All the licenses or rights follow its policy. 

Below is an example of how to run the retrieval to construct a dataset (`xx` should be filled by user). 
```
python retrieve.py --data_name xx --qa_data xx --documents_pool xx --output_folder xx --retrieval_method xx
```

On the other hand, one could directly download the used dataset in the paper (including QA and retrieved passages) from [Google Drive](https://drive.google.com/drive/folders/1trPfSK37CJIFRY3ef-YNb6VKGdRKZm74?usp=sharing). 

# Main Experiments: SuRe for improved Open-domain QA with LLMs 

After preparation of QA dataset with retrieved passages, one can get the answer from LLM through `OPEN-AI API` by running `query_gpt.py`. Specifically, with `--infer_type` as `base`, one can get answers by simply appending `--n_retrieval` retrieved passages. With `--infer_type` as `sure`, one can use the proposed SuRE framework to get the answer. Below is an example of how to run the codes to get the answers. 

```
python query_gpt.py --data_name xx --qa_data xx --lm_type xx --api_key xx --n_retrieval xx --infer_type xx --output_folder xx 
```

On the other hand, one can check the step-by-step process via notebook file (`sure_notebook.ipynb`). We released the result `json` files of main tables (Tables 1 and 2) in the following [Google Drive link](https://drive.google.com/drive/folders/1N69GmLhjlO2cEE1534xWUq94bVURxhk_?usp=sharing).

## SuRe with Open-source LLMs

While our experiments were mainly conducted focused on the recent API-based LLMs, SuRe can be applied to open-source LLMs (e.g., LLaMA, Mistral, and Gemma) as shown by our results on LLaMA2-70B-chat. To ease the usage of SuRe with open-source LLMs, we also released the code base for open-source LLMs. Currently, we include 3 representative LLMs ([LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), and [Gemma](https://huggingface.co/google/gemma-1.1-7b-it)), and one can designate the desired one by inserting the proper name into `--lm_type`. We believe that one can easily modify our code base for their own LLMs. Lastly, we remark that we have only demonstrated the proposed framework with strong LLM (LLaMA2-70B-chat). Below is an example of running the code with open-source LLMs.

```
python query_openllm.py --data_name xx --qa_data xx --lm_type xx --n_retrieval xx --infer_type xx --output_folder xx 
```

## Verification of the effectiveness of SuRe's summarization as Explicit Rationale

In addition to providing more accurate answers, another unique advantage of SuRe is to provide conditional summarization that supports given prediction, which can be viewed as an explicit rationale for the answer. To demonstrate its effectiveness, we conducted two different experiments: (1) Reranking and (2) Preference. All the experiments can be conducted by notebook files (`rerank.ipynb` and `pref_eval.ipynb`). To ease the experiments, we released the results of generic summarization in the following [Google Drive link](https://drive.google.com/drive/folders/1NlOarNKcLM8PLrtBwVm612wLrOrU1odu?usp=sharing).  

## Citation
If you find this work useful for your research, please cite our papers:

```
@article{kim2024sure,
  title={SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs},
  author={Kim, Jaehyung and Nam, Jaehyun and Mo, Sangwoo and Park, Jongjin and Lee, Sang-Woo and Seo, Minjoon and Ha, Jung-Woo and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2404.13081},
  year={2024}
}
```
