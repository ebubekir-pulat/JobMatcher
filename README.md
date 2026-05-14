# Job Matcher

Job recommendation system using dense retrieval and cross-encoder reranking.

## Overview

This project matches resumes to relevant job postings using a two-stage retrieval pipeline:

1. Semantic retrieval using Sentence Transformers embeddings + FAISS
2. Cross-encoder reranking for more precise relevance scoring

The system is designed to efficiently retrieve relevant jobs from a large collection of postings while balancing retrieval speed and ranking quality.

---

## How To Start

###  Install Dependencies
pip install -r requirements.txt

### Build Jobs Database
python -m data.db

### Build Embeddings and FAISS Index
python -m src.embedding.build_index

### Run Application
python -m src.main


## API

The project includes a FastAPI service for querying job matches through HTTP requests.

### Run API
uvicorn src.api:app --reload

### Endpoint
POST /match_jobs

### Example Request
{
  "resume": "Machine learning engineer with NLP and Python experience",
  "top_k": 10,
  "top_n": 5
}

### Example Response
{
  "matches": [
    {
      "job_id": "3884923055",
      "final_score": 5.821333885192871
    },
    {
      "job_id": "3871631334",
      "final_score": 4.84067440032959
    },
    {
      "job_id": "3885845727",
      "final_score": 4.395063400268555
    },
    {
      "job_id": "3885828275",
      "final_score": 4.351306915283203
    },
    {
      "job_id": "3784120102",
      "final_score": 3.228658437728882
    }
  ]
}

### Interactive Docs

FastAPI automatically generates API documentation at: http://127.0.0.1:8000/docs

## Architecture

Resume
↓
Sentence Transformer Encoder
↓
FAISS Similarity Search
↓
Top-K Retrieved Jobs
↓
Cross-Encoder Reranker
↓
Final Ranked Jobs

---

## Technologies Used

- Python
- SQLite
- HuggingFace Datasets
- Sentence Transformers
- FAISS
- NumPy
- Cross-Encoders

---

## Features

- Dense semantic retrieval
- Approximate nearest neighbour search
- Cross-encoder reranking
- Precomputed job embeddings
- JSON job cache
- FastAPI inference endpoint

---

## Project Structure

```text
job_matcher/
├── data/
├── embeddings/
├── src/
│   ├── embedding/
│   ├── retrieval/
│   ├── ranking/
│   ├── config.py
│   └── main.py