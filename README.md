# AI Recruiting Agent

Rank candidates against job descriptions using semantic search + Groq AI.

## Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| Embeddings | `fastembed` + `BAAI/bge-small-en-v1.5` | No torch/transformers — lightweight, clean terminal |
| Vector search | `chromadb` | Pure Python, works on Streamlit Cloud without issues |
| AI (conversation/scoring) | Groq API — `llama-3.3-70b-versatile` | Free, fast (~300 tok/s), 14,400 req/day |
| Frontend | `streamlit` | Simple, fast UI |

## Project Structure

```
backend/
├── app.py              # Streamlit frontend
├── agent.py            # Groq AI — conversation, scoring, skill analysis
├── embeddings.py       # fastembed + chromadb vector index
├── candidates.json     # Candidate database
├── requirements.txt    # All dependencies
├── .env                # API keys 
├── .gitignore

```

## Setup (Local)

### 1. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> On first run, fastembed downloads `BAAI/bge-small-en-v1.5` (~50MB).
> This happens once and is cached locally. No sign-up or API key needed.

### 3. Get a free Groq API key
1. Go to https://console.groq.com
2. Sign up (email only, no credit card)
3. Click **API Keys → Create API Key**

### 4. Create `.env` file
```
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### 5. Run
```bash
streamlit run app.py
```

## candidates.json Format

```json
[
  {
    "name": "Rahul Sharma",
    "skills": ["Python", "Machine Learning", "TensorFlow", "SQL"],
    "experience": "3 years",
    "projects": ["Fraud Detection System", "Customer Churn Prediction"]
  }
]
```

## Scoring Logic

```
Match Score    = cosine similarity (ChromaDB) × 100          → 0 to 100
Interest Score = Groq AI prediction from conversation        → 0 to 100
Final Score    = (match_weight × Match) + (interest_weight × Interest)
```

Default: 70% match, 30% interest. Adjustable in sidebar.

## Deploy to Streamlit Cloud (Free)

```bash
# Step 1: push to GitHub
git init
git add .
git commit -m "AI Recruiting Agent"
git remote add origin https://github.com/YOUR_USERNAME/ai-recruiting-agent.git
git push -u origin main
```

```
Step 2: deploy
  → Go to https://share.streamlit.io
  → New app → select repo → branch: main → file: app.py → Deploy

Step 3: add API key (replaces .env on cloud)
  → App dashboard → Settings → Secrets → add:
     GROQ_API_KEY = "gsk_xxxxxxxxxxxx"
```

Your `.env` is excluded by `.gitignore` — your key is never on GitHub.