# agent.py
# Uses Groq API (free, fast) for AI-powered candidate analysis.
# Get your free key at: https://console.groq.com
# Add to .env: GROQ_API_KEY=gsk_xxxxxxxxxxxx

import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def ask_groq(prompt):
    """Send a prompt to Groq and return the text response. Retries on rate limit."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    for attempt in range(3):
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code == 429:
            print(f"⏳ Rate limited. Waiting 10s... (attempt {attempt + 1}/3)")
            time.sleep(10)
            continue
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")
    raise RuntimeError("Groq API failed after 3 retries.")


def analyze_candidate(candidate, jd):
    """
    Core agent function.
    Single Groq call that returns:
      - A simulated recruiter conversation
      - Interest score (1-10)
      - Skill match explanation (which skills match, which are missing)
      - A one-line recruiter recommendation

    Returns: (conversation, interest_score, skill_analysis, recommendation)
    """
    prompt = f"""
You are an expert AI recruiter. Analyze this candidate against the job description.

Candidate:
- Name: {candidate['name']}
- Skills: {', '.join(candidate['skills'])}
- Experience: {candidate['experience']}
- Projects: {', '.join(candidate['projects'])}

Job Description:
{jd}

Do all four things below. Reply in EXACTLY this format:

CONVERSATION:
[Write a 2-3 line recruiter-candidate dialogue showing interest level]

SCORE: [single integer 1-10]

SKILL_ANALYSIS:
Matched: [list skills from candidate that match the JD]
Missing: [list important JD skills the candidate lacks]

RECOMMENDATION:
[One sentence: should the recruiter contact this candidate, and why?]
"""
    try:
        raw = ask_groq(prompt)

        # Parse conversation
        conv = re.search(r'CONVERSATION:\s*(.*?)\s*SCORE:', raw, re.DOTALL)
        conversation = conv.group(1).strip() if conv else "Not available."

        # Parse interest score
        score_match = re.search(r'SCORE:\s*(\d+)', raw)
        interest_score = min(int(score_match.group(1)), 10) * 10 if score_match else 50

        # Parse skill analysis
        skill_match = re.search(r'SKILL_ANALYSIS:\s*(.*?)\s*RECOMMENDATION:', raw, re.DOTALL)
        skill_analysis = skill_match.group(1).strip() if skill_match else "Not available."

        # Parse recommendation
        rec_match = re.search(r'RECOMMENDATION:\s*(.*?)$', raw, re.DOTALL)
        recommendation = rec_match.group(1).strip() if rec_match else "Not available."

        return conversation, interest_score, skill_analysis, recommendation

    except Exception as e:
        return f"[Error: {e}]", 50, "Not available.", "Not available."


def calculate_match_score(faiss_distance):
    """Convert FAISS L2 distance to 0-100 score. Lower distance = better match."""
    similarity = 1 / (1 + faiss_distance)
    return round(similarity * 100, 2)


def calculate_final_score(match_score, interest_score, match_weight=70):
    """Weighted combination of match and interest scores."""
    interest_weight = 100 - match_weight
    return round((match_weight / 100) * match_score + (interest_weight / 100) * interest_score, 2)