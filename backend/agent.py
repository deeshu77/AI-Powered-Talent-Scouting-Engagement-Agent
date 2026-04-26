# agent.py
# Handles all Groq AI calls — conversation, interest score, skill analysis, recommendation.
#
# Get your free Groq API key at: https://console.groq.com
# Add to .env: GROQ_API_KEY=gsk_xxxxxxxxxxxx
# Free tier: 14,400 requests/day, very fast (~300 tokens/sec)

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
    """
    Send a prompt to Groq and return the response text.
    Retries automatically if rate limited (HTTP 429).
    """
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
            # Read the wait time Groq tells us, then retry
            wait = 15
            try:
                details = response.json()["error"]["details"]
                for d in details:
                    if "retryDelay" in d:
                        wait = int(re.search(r'\d+', d["retryDelay"]).group()) + 2
                        break
            except Exception:
                pass
            print(f"⏳ Rate limited. Waiting {wait}s... (attempt {attempt + 1}/3)")
            time.sleep(wait)
            continue

        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

    raise RuntimeError("Groq failed after 3 retries. Daily quota may be exhausted.")


def analyze_candidate(candidate, jd):
    """
    One Groq call per candidate that returns all four outputs:
      1. Simulated recruiter-candidate conversation
      2. Interest score (1-10)
      3. Skill gap analysis (matched vs missing)
      4. One-line recruiter recommendation

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

Reply in EXACTLY this format with no extra text:

CONVERSATION:
[2-3 line recruiter-candidate dialogue showing interest level]

SCORE: [single integer 1-10]

SKILL_ANALYSIS:
Matched: [skills from candidate that match JD requirements]
Missing: [important JD skills the candidate lacks]

RECOMMENDATION:
[One sentence: should recruiter contact this candidate and why?]
"""
    try:
        raw = ask_groq(prompt)

        # Parse conversation
        conv = re.search(r'CONVERSATION:\s*(.*?)\s*SCORE:', raw, re.DOTALL)
        conversation = conv.group(1).strip() if conv else "Not available."

        # Parse interest score → scale to 0-100
        score_m = re.search(r'SCORE:\s*(\d+)', raw)
        interest_score = min(int(score_m.group(1)), 10) * 10 if score_m else 50

        # Parse skill analysis
        skill_m = re.search(r'SKILL_ANALYSIS:\s*(.*?)\s*RECOMMENDATION:', raw, re.DOTALL)
        skill_analysis = skill_m.group(1).strip() if skill_m else "Not available."

        # Parse recommendation
        rec_m = re.search(r'RECOMMENDATION:\s*(.*?)$', raw, re.DOTALL)
        recommendation = rec_m.group(1).strip() if rec_m else "Not available."

        return conversation, interest_score, skill_analysis, recommendation

    except Exception as e:
        return f"[Error: {e}]", 50, "Not available.", "Not available."


def calculate_final_score(match_score, interest_score, match_weight=70):
    """
    Combine match score (skills similarity) and interest score (AI enthusiasm prediction).
    match_weight: percentage given to skills match (default 70%)
    """
    interest_weight = 100 - match_weight
    return round((match_weight / 100) * match_score + (interest_weight / 100) * interest_score, 2)