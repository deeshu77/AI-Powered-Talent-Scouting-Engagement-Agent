# app.py
# Run with: streamlit run app.py

import json
import os
import csv
import io as _io
import streamlit as st

from embeddings import build_vectorstore, search_candidates
from agent import analyze_candidate, calculate_final_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Recruiting Agent", page_icon="🤖", layout="wide")

MAX_FILE_SIZE_MB = 20

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
h1,h2,h3 { color: #ffffff; }
.card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px 24px; margin-bottom:14px; }
.cname { font-size:1.15rem; font-weight:700; color:#58a6ff; }
.badge { display:inline-block; padding:2px 9px; border-radius:20px; font-size:0.76rem; font-weight:600; margin:2px; }
.bs { background:#1f3a5f; color:#79c0ff; }
.slabel { color:#8b949e; font-size:0.8rem; }
.sval { color:#fff; font-size:1.05rem; font-weight:700; }
.conv { background:#0d1117; border-left:3px solid #58a6ff; border-radius:0 8px 8px 0; padding:10px 14px; margin-top:10px; color:#c9d1d9; font-size:0.86rem; white-space:pre-wrap; }
.rec  { background:#0d1f12; border-left:3px solid #56d364; border-radius:0 8px 8px 0; padding:8px 14px; margin-top:8px; color:#aef1c0; font-size:0.86rem; }
.miss { background:#1f0d0d; border-left:3px solid #f47067; border-radius:0 8px 8px 0; padding:8px 14px; margin-top:8px; color:#f9a8a8; font-size:0.86rem; }
.stButton>button { background:#238636; color:#fff; border:none; border-radius:8px; padding:10px 28px; font-size:1rem; font-weight:600; width:100%; }
.stButton>button:hover { background:#2ea043; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_file_size(f):
    size_mb = f.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File is {size_mb:.1f} MB. Max allowed is {MAX_FILE_SIZE_MB} MB.")
        return False
    return True


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(_io.BytesIO(uploaded_file.read())) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if not text.strip():
                st.warning("⚠️ PDF appears scanned — no text found.")
            return text
        except ImportError:
            st.error("Run: pip install pdfplumber")
            return ""
    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(_io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            st.error("Run: pip install python-docx")
            return ""
    st.error("Unsupported format. Upload PDF, DOCX, or TXT.")
    return ""


@st.cache_resource(show_spinner="Building candidate index...")
def get_vectorstore():
    """Build once, cache for the session."""
    return build_vectorstore()


# Always resolve candidates.json relative to this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_PATH = os.path.join(BASE_DIR, "candidates.json")

def load_candidate_count():
    """Try all possible paths to count candidates."""
    try:
        from embeddings import find_candidates_file
        path = find_candidates_file()
        with open(path) as f:
            return len(json.load(f))
    except Exception:
        return 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Candidates to show", 3, 10, 5)
    match_weight = st.slider("Match score weight (%)", 50, 90, 70, 5)
    st.caption(f"Interest weight: {100 - match_weight}%")
    st.divider()

    st.markdown("### 📦 Database")
    # Get count from vectorstore if loaded, else show loading state
    try:
        col, _ = get_vectorstore()
        count = col.count()
    except Exception:
        count = 0
    st.metric("Total Candidates", count)

    st.divider()
    st.markdown("### ℹ️ Scoring")
    st.markdown("""
- **Match Score** — semantic similarity (vector search)
- **Interest Score** — AI-predicted enthusiasm
- **Final Score** — weighted combination
- **Skill Analysis** — matched & missing skills
- **Recommendation** — AI contact verdict
    """)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 AI-Powered Talent Scouting & Engagement Agent")
st.markdown("Rank candidates against any job description using semantic search + Groq AI.")
st.divider()

left, right = st.columns([1, 1.6], gap="large")

# ── Input ─────────────────────────────────────────────────────────────────────
with left:
    st.markdown("### 📋 Job Description")
    tab1, tab2 = st.tabs(["✏️ Type / Paste", "📁 Upload File"])
    jd_text = ""

    with tab1:
        jd_input = st.text_area(
            "JD", height=280, label_visibility="collapsed",
            placeholder="e.g. We need a Python ML engineer with TensorFlow experience..."
        )

    with tab2:
        uploaded = st.file_uploader("Upload", type=["pdf", "txt", "docx"], label_visibility="collapsed")
        jd_from_file = ""
        if uploaded:
            if check_file_size(uploaded):
                jd_from_file = extract_text(uploaded)
                if jd_from_file:
                    size_mb = uploaded.size / (1024 * 1024)
                    st.success(f"✅ {uploaded.name} ({size_mb:.1f} MB) — {len(jd_from_file)} chars")
                    st.text_area("Preview", jd_from_file[:400] + "...", height=100, disabled=True)

    jd_text = jd_from_file if jd_from_file else jd_input

    if jd_text.strip():
        wc = len(jd_text.split())
        color = "#56d364" if wc >= 30 else "#e3b341"
        st.markdown(
            f'<div style="color:{color};font-size:0.8rem;">📝 {wc} words — '
            f'{"Good length ✓" if wc >= 30 else "Add more detail for better results"}</div>',
            unsafe_allow_html=True
        )

    st.markdown("&nbsp;")
    go = st.button("🔍 Analyze Candidates", use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────
with right:
    st.markdown("### 🏆 Ranked Candidates")

    if not go:
        st.markdown(
            '<div style="color:#8b949e;margin-top:60px;text-align:center;">'
            '👈 Enter a JD and click Analyze</div>',
            unsafe_allow_html=True
        )
    else:
        if not jd_text.strip():
            st.error("⚠️ Please provide a job description.")
            st.stop()
        if len(jd_text.split()) < 10:
            st.warning("⚠️ JD is very short — results may be less accurate.")

        # Load vectorstore (cached)
        try:
            collection, embed_model = get_vectorstore()
        except Exception as e:
            import os
            base = os.path.dirname(os.path.abspath(__file__))
            cpath = os.path.join(base, "candidates.json")
            st.error(f"❌ Could not load candidate database: {e}")
            st.error(f"🔍 Debug — looking for candidates.json at: {cpath}")
            st.error(f"🔍 Debug — file exists: {os.path.exists(cpath)}")
            st.error(f"🔍 Debug — files in folder: {os.listdir(base)}")
            st.stop()

        # Search for top candidates
        with st.spinner("🔍 Finding best candidates..."):
            try:
                matches = search_candidates(collection, embed_model, jd_text, top_k)
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                st.stop()

        # Analyze each candidate with Groq
        results = []
        bar = st.progress(0, text="Analyzing candidates...")

        for i, (candidate, match_score) in enumerate(matches):
            name = candidate.get("name", "?")
            bar.progress((i + 1) / len(matches), text=f"Analyzing {name}... ({i+1}/{len(matches)})")

            try:
                conversation, interest_score, skill_analysis, recommendation = analyze_candidate(candidate, jd_text)
            except Exception as e:
                conversation = f"Error: {e}"
                interest_score, skill_analysis, recommendation = 50, "N/A", "N/A"

            final_score = calculate_final_score(match_score, interest_score, match_weight)

            results.append({
                "candidate":      candidate,
                "match_score":    round(float(match_score), 2),
                "interest_score": int(interest_score),
                "final_score":    float(final_score),
                "conversation":   str(conversation),
                "skill_analysis": str(skill_analysis),
                "recommendation": str(recommendation),
            })

        bar.empty()
        results.sort(key=lambda x: x["final_score"], reverse=True)

        # Best match banner
        st.markdown(
            f'<div style="background:#1a4731;border-radius:8px;padding:10px 16px;'
            f'color:#56d364;font-weight:600;margin-bottom:16px;">'
            f'🥇 Best match: {results[0]["candidate"].get("name","?")}</div>',
            unsafe_allow_html=True
        )

        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

        for rank, r in enumerate(results, 1):
            c = r["candidate"]
            medal = medals[rank - 1] if rank <= len(medals) else f"#{rank}"
            skills_html = "".join(
                f'<span class="badge bs">{s}</span>' for s in c.get("skills", [])
            )

            with st.expander(f"{medal} {c.get('name')} — Score: {r['final_score']}", expanded=(rank <= 2)):
                st.markdown(f"""
                <div class="card">
                  <div class="cname">{medal} {c.get('name', 'Unknown')}</div>
                  <div style="color:#8b949e;font-size:0.82rem;">{c.get('experience','?')} experience</div>
                  <div style="margin-top:8px;">{skills_html}</div>
                  <div style="display:flex;gap:28px;margin-top:12px;">
                    <div><div class="slabel">Match Score</div><div class="sval">{r['match_score']}</div></div>
                    <div><div class="slabel">Interest Score</div><div class="sval">{r['interest_score']}</div></div>
                    <div><div class="slabel">Final Score</div><div class="sval">{r['final_score']}</div></div>
                  </div>
                  <div class="conv">{r['conversation']}</div>
                  <div class="rec">💡 {r['recommendation']}</div>
                  <div class="miss">🔍 {r['skill_analysis'].replace("Matched:", "<b>✅ Matched:</b>").replace("Missing:", "<br><b>❌ Missing:</b>").replace("Transferable:", "<br><b>🔄 Transferable:</b>")}</div>
                </div>
                """, unsafe_allow_html=True)

        # Summary table
        st.markdown("#### 📊 Summary Table")
        table = [
            {
                "Rank":           medals[i] if i < len(medals) else f"#{i+1}",
                "Name":           r["candidate"].get("name"),
                "Experience":     r["candidate"].get("experience"),
                "Match Score":    r["match_score"],
                "Interest Score": r["interest_score"],
                "Final Score":    r["final_score"],
            }
            for i, r in enumerate(results)
        ]
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Match Score":    st.column_config.ProgressColumn("Match",    min_value=0, max_value=100, format="%.1f"),
                "Interest Score": st.column_config.ProgressColumn("Interest", min_value=0, max_value=100, format="%.0f"),
                "Final Score":    st.column_config.ProgressColumn("Final",    min_value=0, max_value=100, format="%.1f"),
            }
        )

        # CSV export
        buf = _io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
        st.download_button("⬇️ Download CSV", buf.getvalue(), "results.csv", "text/csv")