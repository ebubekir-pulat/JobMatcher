from src.embedding import build_index
from src.retrieval import search
from src.ranking import rerank
import json
import config


"""
Initialisation
Load embedding model
Load FAISS index
Load ID mapping
Load job cache
Load cross-encoder

Note: Ensure src/embedding/build_index.py and data/db.py have already been run.
"""

emb_model = build_index.load_embedding_model()
faiss_index = search.load_index(config.INDEX_PATH)
mapping = search.load_mapping(config.MAPPING_PATH)
cross_encoder = rerank.load_cross_encoder()

with open(config.CACHE_PATH, "r") as f:
    job_cache = json.load(f)

"""
Per Query
Get user resume
Encode resume
Retrieve top-K jobs using FAISS
Rerank those jobs using cross-encoder
Return top-N results
"""
while True:
    resume = input("Enter a Resume: ")
    recommended_jobs = search.retrieve_jobs(resume, 10, emb_model, faiss_index, mapping)
    reranked_jobs = rerank.rerank_jobs(resume, recommended_jobs, job_cache, cross_encoder)
    print(reranked_jobs, end="\n\n")