# embeddings.py
# Builds a searchable vector index from candidates.json.
#
# Stack:
#   fastembed  → converts text to vectors (no torch, no transformers, ~50MB)
#   chromadb   → stores and searches vectors (pure Python, Streamlit Cloud safe)
#
# No API key needed for embeddings — runs 100% locally.

import json
import os
from dotenv import load_dotenv
from fastembed import TextEmbedding
import chromadb

load_dotenv()

# Embedding model — small, fast, accurate. Downloads ~50MB on first run, cached after.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_candidates():
    """Read all candidates from candidates.json."""
    if not os.path.exists("candidates.json"):
        raise RuntimeError("candidates.json not found in the project folder.")
    with open("candidates.json") as f:
        data = json.load(f)
    if not data:
        raise RuntimeError("candidates.json is empty. Please add at least one candidate.")
    return data


def build_vectorstore():
    """
    Convert each candidate profile into a vector and store in ChromaDB.
    Returns a (chroma_collection, embedding_model) tuple used for searching.
    """
    candidates = load_candidates()

    print("Loading embedding model...")
    embed_model = TextEmbedding(model_name=EMBED_MODEL)
    print("✅ Embedding model ready.")

    # Build text summaries for each candidate
    texts = []
    ids = []
    metadatas = []

    for i, candidate in enumerate(candidates):
        text = (
            f"Name: {candidate['name']}. "
            f"Skills: {', '.join(candidate['skills'])}. "
            f"Experience: {candidate['experience']}. "
            f"Projects: {', '.join(candidate['projects'])}."
        )
        texts.append(text)
        ids.append(str(i))
        metadatas.append(candidate)

    # Generate embeddings for all candidates
    embeddings = [vec.tolist() for vec in embed_model.embed(texts)]

    # Store in ChromaDB (in-memory — rebuilds fresh each server start)
    client = chromadb.Client()

    # Delete existing collection if it exists (safe restart)
    try:
        client.delete_collection("candidates")
    except Exception:
        pass

    collection = client.create_collection(
        name="candidates",
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Vector index built with {len(candidates)} candidates.")
    return collection, embed_model


def search_candidates(collection, embed_model, jd_text, top_k=5):
    """
    Find the top_k most similar candidates for a given job description.
    Returns a list of (candidate_dict, similarity_score) tuples.
    """
    # Embed the job description
    jd_vector = list(embed_model.embed([jd_text]))[0].tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[jd_vector],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    candidates_found = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to 0-100 similarity score
        similarity = (1 - distance / 2) * 100
        similarity = round(max(0.0, min(similarity, 100.0)), 2)
        candidates_found.append((metadata, similarity))

    return candidates_found