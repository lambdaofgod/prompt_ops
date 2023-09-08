import math
import os
from functools import lru_cache
from typing import Any, Dict, List

import fastapi
import fire
import uvicorn
from colbert import Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field

app = fastapi.FastAPI()


class SearchConfig(BaseModel):
    index_name: str = "lifestyle.dev.2bits"

    index_path: str = "ColBERT/experiments/default/indexes"
    hf_dataset: str = "colbertv2/lotte_passages"
    dataset_part: str = "lifestyle"
    datasplit: str = "dev"


class SearchAppState(BaseModel):
    searcher: Searcher
    counter: Dict[str, int]

    @staticmethod
    def setup(search_config: SearchConfig):
        collection_dataset = load_dataset(
            search_config.hf_dataset, search_config.dataset_part
        )
        collection = [
            x["text"]
            for x in collection_dataset[search_config.datasplit + "_collection"]
        ]

        index_root = os.path.join(os.getcwd(), search_config.index_path)
        searcher = Searcher(
            index=f"{index_root}/{search_config.index_name}",
            collection=collection,
        )
        counter = {"api": 0}
        app = fastapi.FastAPI()

        return SearchAppState(searcher=searcher, counter=counter)

    @lru_cache(maxsize=1000000)
    def api_search_query(self, query, k):
        print(f"Query={query}")
        if k == None:
            k = 10
        k = min(int(k), 100)
        pids, ranks, scores = self.searcher.search(query, k=100)

        pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
        passages = [self.searcher.collection[pid] for pid in pids]
        probs = [math.exp(score) for score in scores]
        probs = [prob / sum(probs) for prob in probs]
        topk = []
        for pid, rank, score, prob in zip(pids, ranks, scores, probs):
            text = self.searcher.collection[pid]
            d = {"text": text, "pid": pid, "rank": rank, "score": score, "prob": prob}
            topk.append(d)
        topk = list(sorted(topk, key=lambda p: (-1 * p["score"], p["pid"])))
        return {"query": query, "topk": topk}

    class Config:
        arbitrary_types_allowed = True


def setup_app(app):
    search_app_state = SearchAppState.setup(SearchConfig())
    app.state.search_app_state = search_app_state


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=10)


class SearchResponse(BaseModel):
    query: str
    topk: List[Dict[str, Any]]


@app.post("/api/search", response_model=SearchResponse)
def api_search(search_request: SearchRequest):
    state = app.state.search_app_state
    state.counter["api"] += 1
    print("API request count:", state.counter["api"])
    return state.api_search_query(search_request.query, search_request.k)


def main(port=8989):
    setup_app(app)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    fire.Fire(main)
