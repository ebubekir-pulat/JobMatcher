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