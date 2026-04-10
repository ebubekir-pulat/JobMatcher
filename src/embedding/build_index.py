import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_jobs(db_path: str):
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    res = cursor.execute("SELECT job_id, description FROM jobs")
    return res.fetchone()
    
    """
    Returns:
        job_ids: List[str]
        descriptions: List[str]
    """

def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def encode_jobs(model, descriptions: list):
    embeddings = model.encode(descriptions)
    print(f"Embeddings Shape: {embeddings.shape}")
    return embeddings

# Read: https://www.pinecone.io/learn/series/faiss/faiss-tutorial/
def build_faiss_index(embeddings: np.ndarray):