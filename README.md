# AI Powered Talent Scouting & Engagement Agent

Rank candidates against job descriptions using semantic search + Groq AI.

## How It Works

1. Candidate profiles are loaded from `candidates.json`
2. Each profile is converted into a vector using `all-MiniLM-L6-v2` (a free local embedding model, ~90MB, downloads once automatically via `sentence-transformers`)
3. Vectors are stored in a FAISS index for fast similarity search
4. When a recruiter submits a JD, it is embedded the same way and FAISS finds the top matching candidates
5. For each candidate, Groq AI (free, fast) generates a simulated conversation, interest score, skill gap analysis, and recruiter recommendation
6. Results are ranked by a weighted final score and displayed in the UI

## Project Structure

```
backend/
├── app.py              # Streamlit frontend
├── agent.py            # Groq AI calls — conversation, scoring, skill analysis
├── embeddings.py       # Builds FAISS vector index from candidates.json
├── candidates.json     # Candidate database
├── requirements.txt    # All dependencies
├── .env                # Your API keys (never commit this)
├── .gitignore          # Excludes .env and venv from git

```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-recruiting-agent.git
cd ai-recruiting-agent/backend
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

> On first run, `sentence-transformers` will automatically download the
> `all-MiniLM-L6-v2` embedding model (~90MB). This happens once and is
> then cached locally. No API key or sign-up needed for this model.

### 4. Get a free Groq API key
- Go to https://console.groq.com
- Sign up (just email, no credit card)
- Click **API Keys → Create API Key**
- Copy the key

### 5. Create your `.env` file
Create a file named `.env` inside the `backend/` folder:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### 6. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Models Used

| Model | Purpose | Provider | Cost |
|-------|---------|----------|------|
| `all-MiniLM-L6-v2` | Text embeddings for candidate-JD similarity | HuggingFace (runs locally) | Free, no API key |
| `llama-3.3-70b-versatile` | Conversation, scoring, skill analysis, recommendation | Groq | Free tier (14,400 req/day) |

## Scoring Logic

```
Match Score   = 1 / (1 + FAISS_distance) × 100     # Semantic similarity, 0–100
Interest Score = AI-predicted enthusiasm × 10        # From simulated conversation, 0–100
Final Score   = (match_weight × Match) + ((100 - match_weight) × Interest)
```

Default weights: 70% match, 30% interest (adjustable in the sidebar).

## Deploy to Streamlit Cloud (Free)

```bash
# Step 1: Push to GitHub
git init
git add .
git commit -m "AI Recruiting Agent"
git remote add origin https://github.com/YOUR_USERNAME/ai-recruiting-agent.git
git push -u origin main
```

Then:
1. Go to https://share.streamlit.io
2. Click **New app**
3. Select your repo → branch: `main` → file: `app.py`
4. Click **Deploy**
5. Go to **Settings → Secrets** and add:
```
GROQ_API_KEY = "gsk_xxxxxxxxxxxx"
```

Your `.env` file is excluded by `.gitignore` — your key is never pushed to GitHub.
On Streamlit Cloud, secrets work exactly like `.env` locally — no code changes needed.

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

Add as many candidates as needed. The vector index rebuilds automatically on server restart.