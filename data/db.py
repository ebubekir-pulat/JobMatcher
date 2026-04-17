import sqlite3
from datasets import load_dataset

db_file = "jobs.db"
try:
    conn = sqlite3.connect(db_file)
    print("Jobs Database Built")
except:
    print("Error During Jobs Database Initialisation")

jobs_ds_whole = load_dataset("datastax/linkedin_job_listings", split="train")
dataset_size = 20
jobs_ds = jobs_ds_whole.select_columns(["job_id", "description"])[:dataset_size]

cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, description TEXT)")

for job in jobs_ds:
    cursor.execute(f"""
        INSERT OR IGNORE INTO jobs (job_id, description) VALUES (?, ?)
    """, (job["job_id"], job["description"]))

conn.commit()