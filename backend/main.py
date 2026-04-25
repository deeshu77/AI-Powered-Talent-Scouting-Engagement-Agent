# main.py
# Run with:  uvicorn main:app --reload
# Test with: http://127.0.0.1:8000/docs

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embeddings import build_vectorstore
from agent import analyze_candidate, calculate_match_score, calculate_final_score

vectorstore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    print("🚀 Starting up — building candidate index...")
    vectorstore = build_vectorstore()
    yield
    print("🛑 Shutting down.")

app = FastAPI(
    title="AI Recruiting Agent",
    description="Send a job description, get back ranked candidates.",
    lifespan=lifespan
)

class JDRequest(BaseModel):
    jd: str

@app.post("/analyze_jd")
def analyze_jd(request: JDRequest):
    if not request.jd.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    top_matches = vectorstore.similarity_search_with_score(request.jd, k=5)

    results = []

    for document, faiss_distance in top_matches:
        candidate = document.metadata

        # FIX: convert numpy.float32 → plain Python float so FastAPI can serialize it
        faiss_distance = float(faiss_distance)

        match_score = calculate_match_score(faiss_distance)
        conversation, interest_score = analyze_candidate(candidate, request.jd)
        final_score = calculate_final_score(match_score, interest_score)

        results.append({
            "name": str(candidate.get("name", "Unknown")),
            "match_score": float(match_score),
            "interest_score": int(interest_score),
            "final_score": float(final_score),
            "conversation": str(conversation)
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    return {"candidates": results}