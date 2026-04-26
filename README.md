# 🤖 AI-Powered Talent Scouting & Engagement Agent

An intelligent recruiting agent that analyzes job descriptions and ranks candidates using semantic search and generative AI.

Given a JD, the agent handles:
- **JD Parsing** — understands role requirements semantically, not just keywords
- **Candidate Discovery & Matching** — vector similarity search with explainability (matched skills, missing skills, transferable skills)
- **Conversational Outreach** — simulates a realistic recruiter-candidate dialogue for each candidate
- **Ranked Output** — combined score with AI recommendation (Contact / Maybe / Pass) the recruiter can act on immediately

🔗 **Live Demo:** https://hirenex.streamlit.app/

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RECRUITER (User)                         │
│              Enters JD via text box or uploads file             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                          │
│  • File upload (PDF / DOCX / TXT, max 20MB)                    │
│  • Text extraction (pdfplumber / python-docx)                  │
│  • Sidebar: adjustable match weight, number of results          │
│  • Results: ranked cards, skill badges, CSV export             │
└───────────────────────────┬─────────────────────────────────────┘
                            │  JD text
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EMBEDDING LAYER                             │
│  fastembed — BAAI/bge-small-en-v1.5 (runs locally, ~50MB)      │
│  • JD text → 384-dimensional vector                            │
│  • Candidate profiles → vectors (built once at startup)        │
└───────────────┬─────────────────────────────────┬──────────────┘
                │ JD vector                       │ Candidate vectors
                ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VECTOR STORE (ChromaDB)                     │
│  • Stores all candidate vectors in-memory                      │
│  • Cosine similarity search → returns top-K candidates         │
│  • Distance → Match Score (0–100)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Top-K candidates + match scores
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI AGENT (Groq API)                         │
│  Model: llama-3.3-70b-versatile                                │
│  One call per candidate produces:                              │
│  ├── Simulated recruiter-candidate conversation (4 lines)      │
│  ├── Interest score (1–10)                                     │
│  ├── Skill analysis: Matched / Missing / Transferable          │
│  └── Recommendation: Contact / Maybe / Pass                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SCORING & RANKING                           │
│  Final Score = (0.70 × Match Score) + (0.30 × Interest Score)  │
│  Candidates sorted by Final Score descending                   │
│  Recruiter sees ranked list + can export as CSV                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scoring Logic

### Match Score (0–100)
Computed from ChromaDB cosine distance between the JD vector and each candidate vector:
```
similarity = 1 - (cosine_distance / 2)
Match Score = similarity × 100
```
A score of 90+ means near-identical semantic profile. A score of 50 means moderate overlap.

### Interest Score (0–100)
Groq AI reads the candidate profile against the JD and simulates a conversation. It then predicts how interested the candidate would realistically be, rated 1–10, scaled to 0–100. A candidate who is overqualified or in a different domain scores low; a well-matched candidate scores high.

### Final Score (0–100)
```
Final Score = (match_weight / 100) × Match Score
            + (interest_weight / 100) × Interest Score
```
Default weights: **70% Match, 30% Interest** — adjustable in the sidebar. Skills match is weighted higher because technical fit is harder to develop than enthusiasm.

### Recommendation Logic
The AI gives one of three verdicts based on combined evidence:
| Verdict | Meaning |
|---------|---------|
| ✅ Contact | Strong match on required skills, high interest predicted |
| 🟡 Maybe | Partial match with transferable skills worth exploring |
| ❌ Pass | Significant skill gaps, low predicted interest |

---

## How It Works

```
Job Description (text or file upload)
        │
        ▼
  fastembed converts JD text → 384-dim vector
        │
        ▼
  ChromaDB cosine search → top 5 candidate vectors
        │
        ▼
  For each candidate, Groq AI produces in one call:
    ├── Recruiter-candidate conversation (4 lines)
    ├── Interest score (1–10)
    ├── Skill analysis (Matched / Missing / Transferable)
    └── Recommendation (Contact / Maybe / Pass)
        │
        ▼
  Final Score = 70% Match + 30% Interest
        │
        ▼
  Ranked output displayed to recruiter
```

---

## Sample Input & Output

### Sample Input — Job Description
```
We are looking for a Machine Learning Engineer to build and deploy ML models
at scale. The role requires Python, TensorFlow or PyTorch, experience with
data pipelines, and familiarity with cloud platforms (AWS or GCP).
Experience: 2–5 years. Nice to have: Docker, MLflow, Kubernetes.
```

### Sample Output
```json
{
  "candidates": [
    {
      "name": "Meera Iyer",
      "match_score": 87.8,
      "interest_score": 80,
      "final_score": 85.5,
      "conversation": "Recruiter: Meera, your recommendation engine project looks very relevant — can you tell me about the ML stack you used?\nMeera: Absolutely, I built it using TensorFlow and deployed it on GCP with automated data pipelines.\nRecruiter: Great. The role also involves Docker for containerisation — do you have hands-on experience?\nMeera: I have used Docker in side projects and am actively building on it — comfortable picking it up quickly.",
      "skill_analysis": "Matched: Python, TensorFlow, GCP, data pipelines\nMissing: Kubernetes\nTransferable: GCP experience covers cloud requirement; pipeline work transfers directly",
      "recommendation": "Contact — strong core ML skills and GCP experience directly satisfy 80% of JD requirements; Kubernetes gap is minor and learnable."
    },
    {
      "name": "Siddharth Bose",
      "match_score": 75.6,
      "interest_score": 70,
      "final_score": 74.0,
      "conversation": "Recruiter: Siddharth, your fraud detection project is interesting — what frameworks did you use?\nSiddharth: I used Scikit-learn and some TensorFlow for the classification models.\nRecruiter: The role leans heavily on cloud deployments — have you worked with AWS or GCP?\nSiddharth: Not extensively, mostly local deployments, but I am familiar with AWS basics.",
      "skill_analysis": "Matched: Python, TensorFlow, Machine Learning\nMissing: Cloud platforms (AWS/GCP hands-on), Docker, Kubernetes\nTransferable: Strong Python and ML fundamentals reduce ramp-up time",
      "recommendation": "Maybe — solid ML foundation but lacks cloud deployment experience which is central to this role; worth a screening call."
    }
  ]
}
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | `fastembed` + `BAAI/bge-small-en-v1.5` | Lightweight (~50MB), no torch/transformers dependency |
| Vector Search | `chromadb` | Pure Python, zero build issues on Streamlit Cloud |
| AI Agent | Groq API — `llama-3.3-70b-versatile` | Free tier, extremely fast (~300 tok/s) |
| Frontend | `streamlit` | Clean UI, fast to build and deploy |

---

## Project Structure

```
AI-Agent/
├── backend/
│   ├── app.py              # Streamlit frontend — UI, file upload, results display
│   ├── agent.py            # Groq AI — conversation, scoring, skill analysis, recommendation
│   ├── embeddings.py       # fastembed + ChromaDB — builds and searches vector index
│   ├── candidates.json     # Candidate database
│   └── requirements.txt    # All dependencies
├── .gitignore              # Excludes .env, venv, __pycache__
└── README.md
```

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/deeshu77/AI-Powered-Talent-Scouting-Engagement-Agent.git
cd AI-Powered-Talent-Scouting-Engagement-Agent/backend
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

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> On first run, `fastembed` downloads `BAAI/bge-small-en-v1.5` (~50MB) and caches it locally.
> No sign-up or API key required for embeddings.

### 4. Get a free Groq API key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with email — no credit card required
3. Click **API Keys → Create API Key** and copy it

### 5. Create a `.env` file inside `backend/`
```
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

> ⚠️ Never commit this file. It is already excluded by `.gitignore`.

### 6. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Adding Candidates

Edit `backend/candidates.json`:

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

The vector index rebuilds automatically on every server restart.

---

## Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Step 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select repo → Branch: `main` → **Main file path: `backend/app.py`**
4. Click **Deploy**

### Step 3 — Add API key
1. App dashboard → **Settings → Secrets**
2. Add:
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxx"
```
3. Click **Save** — app reboots automatically

---

## Free Tier Limits

| Service | Free Limit |
|---------|-----------|
| Groq API | 14,400 requests/day, 30 req/minute |
| Streamlit Cloud | 1 app, always free |
| fastembed model | Downloaded once, runs locally forever |