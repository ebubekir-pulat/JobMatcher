from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "embeddings" / "faiss_index.bin"
MAPPING_PATH = BASE_DIR / "embeddings" / "id_mapping.pkl"
CACHE_PATH = BASE_DIR / "embeddings" / "job_cache.json"
DB_PATH = BASE_DIR / "data" / "jobs.db"