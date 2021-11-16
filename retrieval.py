import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from rank_bm25 import BM25Plus
from preprocessor import Preprocessor

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "/opt/ml/project/odqa/data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.tokenizer = tokenize_fn
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로

        print('Preprocessing Wiki Data')
        preprocessor = Preprocessor()
        self.contexts = []
        for con in tqdm(contexts) : 
            doc = preprocessor.doc_preprocess(con)
            if doc not in self.contexts :
                self.contexts.append(doc)

        self.ids = list(range(len(self.contexts)))
        print('Wikipedia data size : %d' %len(self.contexts))

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path) :
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            tokenized_contexts= [self.tokenizer(i) for i in tqdm(self.contexts)]
            self.p_embedding = BM25Plus(tokenized_contexts)    
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_embedding is not None and isinstance(dataset, Dataset) 
        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            
        total = []
        with timer("query exhaustive search"):
            doc_scores, doc_indices = self.get_relevant_doc(
                dataset["question"], k=topk
            )
        for idx, example in enumerate(
            tqdm(dataset, desc="Sparse retrieval: ")
        ):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices[idx],
                "context": " ".join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def get_relevant_doc(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        tokenized_queries= [self.tokenizer(i) for i in queries] 
        doc_scores = []
        doc_indices = []
        for i in tqdm(tokenized_queries):
            scores = self.p_embedding.get_scores(i)
            sorted_score = np.sort(scores)[::-1]
            sorted_id = np.argsort(scores)[::-1]

            doc_scores.append(sorted_score[:k])
            doc_indices.append(sorted_id[:k])
        return doc_scores, doc_indices
