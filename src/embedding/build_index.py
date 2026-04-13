import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

def load_jobs(db_path: str):
    """
    Returns:
        job_ids: List[str]
        descriptions: List[str]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    res = cursor.execute("SELECT job_id, description FROM jobs")
    rows = res.fetchall()
    job_ids = [row[0] for row in rows]
    descriptions = [row[1] for row in rows]
    conn.close()
    return job_ids, descriptions

def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def encode_jobs(model, descriptions: list):
    embeddings = model.encode(descriptions)
    embeddings = np.array(embeddings).astype("float32")
    print(f"Embeddings Shape: {embeddings.shape}")
    return embeddings

# Below From: https://www.pinecone.io/learn/series/faiss/faiss-tutorial/
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    
    # Init Index
    index = faiss.IndexFlatL2(dim) # IndexFlatL2 == Eucliedan distance
    print(index.is_trained) # Should return False, because IndexFlatL2 does not need training

    index.add(embeddings)
    print(index.ntotal) # Should print out number of embeddings

    return index

def save_index(index, path: str):
    faiss.write_index(index, path)

def save_mapping(job_ids: list, path: str):
    with open(path, "wb") as f:
        pickle.dump(job_ids, f)

def main():
    job_ids, descriptions = load_jobs("jobs.db")
    model = load_embedding_model()
    embeddings = encode_jobs(model, descriptions)
    index = build_faiss_index(embeddings)
    save_index(index, "embeddings/faiss_index.bin")
    save_mapping(job_ids, "embeddings/id_mapping.pkl")