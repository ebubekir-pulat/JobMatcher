import faiss
import pickle

def load_index(path: str):
    return faiss.read_index(path)

def load_mapping(path: str):
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    return mapping

def encode_resume(model, text: str):
    res_emb = model.encode([text])
    return res_emb.reshape(1, -1).astype("float32")

def retrieve_jobs(resume_text: str, top_k: int, model, index, mapping):
    top_k = min(top_k, index.ntotal)
    res_emb = encode_resume(model, resume_text)
    distances, indices = index.search(res_emb, top_k)
    rec_jobs = []
    
    for idx, dist in zip(indices[0], distances[0]):
        rec_jobs.append({
            "job_id": mapping[idx], 
            "score": 1 / (1 + float(dist))
        })

    return rec_jobs