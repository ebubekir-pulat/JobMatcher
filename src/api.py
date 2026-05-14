from fastapi import FastAPI
from pydantic import BaseModel
from src import service

class JobMatchRequest(BaseModel):
    resume: str
    top_k: int
    top_n: int

app = FastAPI()

@app.post("/match_jobs")
async def match_jobs(item: JobMatchRequest):
    res = service.get_job_matches(item.resume, item.top_k, item.top_n)
    return {"matches": res}