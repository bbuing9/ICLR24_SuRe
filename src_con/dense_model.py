from typing import List, Dict
import torch
import numpy as np

class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        idx = range(len(queries))
        queries = [queries[i] for i in idx]
        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.cuda() for key, value in qencode.items()}
                emb = self.query_encoder(**qencode, normalize=self.norm_query)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]
        corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
        if self.normalize_text:
            corpus = [normalize_text.normalize(c) for c in corpus]
        if self.lower_case:
            corpus = [c.lower() for c in corpus]

        allemb = []
        nbatch = (len(corpus) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus))

                cencode = self.tokenizer.batch_encode_plus(
                    corpus[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                cencode = {key: value.cuda() for key, value in cencode.items()}
                emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        allemb = allemb.cpu().numpy()
        return allemb