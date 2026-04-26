# embeddings.py

import json
import os
from fastembed import TextEmbedding
import chromadb

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def find_candidates_file():
    """
    Try every possible location for candidates.json.
    Prints each path tried so you can see in Streamlit logs exactly what's happening.
    """
    # All possible locations to check
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    possible_paths = [
        os.path.join(script_dir, "candidates.json"),          # same folder as this script
        os.path.join(cwd, "candidates.json"),                  # current working directory
        os.path.join(cwd, "backend", "candidates.json"),       # cwd/backend/
        "/mount/src/ai-agent/backend/candidates.json",         # Streamlit Cloud common path
        "/mount/src/backend/candidates.json",                  # Streamlit Cloud variant
    ]

    print(f"🔍 Script location: {script_dir}")
    print(f"🔍 Working directory: {cwd}")

    for path in possible_paths:
        print(f"🔍 Trying: {path} → exists: {os.path.exists(path)}")
        if os.path.exists(path):
            print(f"✅ Found candidates.json at: {path}")
            return path

    # Last resort: search entire /mount directory (Streamlit Cloud)
    for root, dirs, files in os.walk("/mount"):
        for file in files:
            if file == "candidates.json":
                full = os.path.join(root, file)
                print(f"✅ Found by search: {full}")
                return full

    raise RuntimeError(
        f"candidates.json not found anywhere.\n"
        f"Script dir: {script_dir}\n"
        f"Working dir: {cwd}\n"
        f"Files in script dir: {os.listdir(script_dir)}\n"
        f"Files in cwd: {os.listdir(cwd)}"
    )


def load_candidates():
    path = find_candidates_file()
    with open(path) as f:
        data = json.load(f)
    if not data:
        raise RuntimeError("candidates.json is empty.")
    return data


def build_vectorstore():
    candidates = load_candidates()
    print(f"📂 Loaded {len(candidates)} candidates.")

    embed_model = TextEmbedding(model_name=EMBED_MODEL)
    print("✅ Embedding model ready.")

    texts, ids, metadatas = [], [], []

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

    embeddings = [vec.tolist() for vec in embed_model.embed(texts)]

    client = chromadb.Client()
    try:
        client.delete_collection("candidates")
    except Exception:
        pass

    collection = client.create_collection(
        name="candidates",
        metadata={"hnsw:space": "cosine"}
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
    jd_vector = list(embed_model.embed([jd_text]))[0].tolist()

    results = collection.query(
        query_embeddings=[jd_vector],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    candidates_found = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        similarity = round((1 - distance / 2) * 100, 2)
        similarity = max(0.0, min(similarity, 100.0))
        candidates_found.append((metadata, similarity))

    return candidates_found