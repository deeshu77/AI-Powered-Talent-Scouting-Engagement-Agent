# agent.py
# Handles all Groq AI calls.
# Get free key: https://console.groq.com
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
    """Send prompt to Groq, retry on rate limit."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3  # slight creativity for natural conversation
    }
    for attempt in range(3):
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code == 429:
            wait = 15
            try:
                for d in response.json()["error"]["details"]:
                    if "retryDelay" in d:
                        wait = int(re.search(r'\d+', d["retryDelay"]).group()) + 2
                        break
            except Exception:
                pass
            print(f"⏳ Rate limited. Waiting {wait}s... (attempt {attempt+1}/3)")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")
    raise RuntimeError("Groq failed after 3 retries.")


def analyze_candidate(candidate, jd):
    """
    Full recruiter-grade analysis of one candidate against a JD.

    Covers:
      1. JD parsing     — extracts key requirements from the JD
      2. Match analysis — explains WHY the candidate matches or doesn't
      3. Simulated outreach conversation — realistic recruiter dialogue
      4. Interest score — predicted candidate enthusiasm
      5. Recommendation — clear action for the recruiter

    Returns: (conversation, interest_score, skill_analysis, recommendation)
    """

    prompt = f"""
You are a senior technical recruiter at a top tech company with 10+ years of experience.
Your job is to evaluate candidates fairly and give recruiters clear, actionable insights.

═══════════════════════════════════════
JOB DESCRIPTION:
{jd}
═══════════════════════════════════════
CANDIDATE PROFILE:
Name: {candidate['name']}
Skills: {', '.join(candidate['skills'])}
Experience: {candidate['experience']}
Projects: {', '.join(candidate['projects'])}
═══════════════════════════════════════

Perform a complete recruiter evaluation. Follow the EXACT format below.
Do not add extra text, headers, or explanations outside the format.

CONVERSATION:
Recruiter: [Open with a specific reference to one of their projects or skills that matches the JD]
{candidate['name']}: [Respond with genuine interest or hesitation based on how well they fit — be realistic]
Recruiter: [Ask one specific follow-up question about a key JD requirement]
{candidate['name']}: [Give a concrete answer referencing their actual experience]

SCORE: [Integer 1-10 reflecting realistic interest level based on fit]

SKILL_ANALYSIS:
Matched: [List specific skills/experience from candidate that directly satisfy JD requirements]
Missing: [List specific skills from JD the candidate lacks — write "None" if fully qualified]
Transferable: [Skills candidate has that could partially compensate for gaps]

RECOMMENDATION:
[One decisive sentence: Contact / Maybe / Pass — state the reason using specific evidence from their profile]
"""

    try:
        raw = ask_groq(prompt)

        # Parse conversation
        conv = re.search(r'CONVERSATION:\s*(.*?)\s*SCORE:', raw, re.DOTALL)
        conversation = conv.group(1).strip() if conv else "Not available."

        # Parse score → scale to 0-100
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
    """70% skills match + 30% predicted interest = final recruiter score."""
    interest_weight = 100 - match_weight
    return round((match_weight / 100) * match_score + (interest_weight / 100) * interest_score, 2)