from sentence_transformers import CrossEncoder
import json

def load_cross_encoder():
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return model

def load_job_cache(path: str):
    with open(path, "r") as f:
        jobs = json.load(f)
    return jobs

def score_pair(model, resume_text, job_description):
    score = model.predict([(resume_text, job_description)])
    return score[0]

def rerank_jobs(resume_text: str, retrieved_jobs, job_cache, model):
    scored_jobs = []

    for job in retrieved_jobs:
        job_descr = job_cache[job["job_id"]]["description"]
        score = score_pair(model, resume_text, job_descr)
        scored_jobs.append({"job_id": job["job_id"], "final_score": score})

    reranked_jobs = sorted(scored_jobs, key=lambda x: x["final_score"], reverse=True)
    return reranked_jobs