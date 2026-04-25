# embeddings.py
# This file builds a searchable index of candidates using vector embeddings.
# Think of it like a "smart search engine" that finds candidates by meaning, not just keywords.

import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the GEMINI_API_KEY from the .env file
load_dotenv()


def load_candidates():
    """Read candidates from the candidates.json file."""
    BASE_DIR = os.path.dirname(__file__)  # path of embeddings.py
    file_path = os.path.join(BASE_DIR, "candidates.json")

    print("Loading candidates from:", file_path)  # debug line

    with open(file_path, "r") as f:
        return json.load(f)


def build_vectorstore():
    """
    Convert each candidate's profile into a vector (list of numbers)
    and store them in FAISS so we can search by similarity later.
    """

    candidates = load_candidates()

    # Step 1: Turn each candidate into a plain text description
    texts = []
    metadatas = []

    for candidate in candidates:
        # Create a single text summary of this candidate
        text = (
            f"Name: {candidate['name']}. "
            f"Skills: {', '.join(candidate['skills'])}. "
            f"Experience: {candidate['experience']}. "
            f"Projects: {', '.join(candidate['projects'])}."
        )
        texts.append(text)
        metadatas.append(candidate)  # Keep the original data attached

    # Step 2: Load a free local embedding model (downloads ~90MB once, then cached)
    # This model converts text into numbers that capture meaning
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 3: Build the FAISS vector store from the texts
    # FAISS stores all candidate vectors so we can quickly find the closest ones
    print("Building vector index...")
    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)

    print("✅ Vector index ready.")
    return vectorstore