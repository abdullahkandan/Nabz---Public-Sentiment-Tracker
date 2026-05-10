import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from groq import Groq
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

SOURCES = [
    {"name": "Dawn", "url": "https://www.dawn.com/feeds/home"},
    {"name": "The News", "url": "https://www.thenews.com.pk/rss/1/1"},
    {"name": "ARY News", "url": "https://arynews.tv/feed"},
    {"name": "BOL News", "url": "https://www.bolnews.com/feed"},
]

STOPWORDS = set([
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","has","have","had","will","would",
    "he","she","it","they","we","you","i","his","her","its","their","our",
    "this","that","these","those","as","by","from","up","about","into",
    "after","says","say","said","over","new","two","one","also","amid",
    "after","before","during","against","between","pakistan","pakistani"
])

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_data(ttl=1800)
def fetch_and_analyze():
    analyzer = load_analyzer()
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []

    for source in SOURCES:
        try:
            response = requests.get(source["url"], headers=headers, timeout=10, verify=False)
            soup = BeautifulSoup(response.text, "xml")
            items = soup.find_all("item")
            for item in items:
                headline = item.title.text.strip() if item.title else None
                if not headline:
                    continue
                score = analyzer.polarity_scores(headline)
                compound = score["compound"]
                if compound >= 0.05:
                    sentiment = "positive"
                elif compound <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                articles.append({
                    "headline": headline,
                    "source": source["name"],
                    "date": item.pubDate.text.strip() if item.pubDate else None,
                    "url": item.link.text.strip() if item.link else None,
                    "sentiment": sentiment,
                    "sentiment_score": round(abs(compound), 3)
                })
        except Exception as e:
            st.warning(f"{source['name']} failed: {e}")

    df = pd.DataFrame(articles)
    if df.empty:
        return df, pd.DataFrame(columns=["topic", "count"]), []

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    all_words = []
    for headline in df["headline"]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', headline.lower())
        all_words.extend([w for w in words if w not in STOPWORDS])
    top_keywords = Counter(all_words).most_common(15)
    kw_df = pd.DataFrame(top_keywords, columns=["topic", "count"])

    stories = detect_stories(df)

    return df, kw_df, stories

def detect_stories(df):
    if len(df) < 2:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df["headline"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    threshold = 0.15
    visited = set()
    clusters = []

    for i in range(len(df)):
        if i in visited:
            continue
        similar = [i]
        for j in range(len(df)):
            if i != j and j not in visited and similarity_matrix[i][j] > threshold:
                similar.append(j)
        if len(similar) >= 2:
            cluster_df = df.iloc[similar]
            outlets = cluster_df["source"].unique().tolist()
            if len(outlets) >= 2:
                sentiments = cluster_df["sentiment"].value_counts().to_dict()
                dominant_sentiment = max(sentiments, key=sentiments.get)
                clusters.append({
                    "headlines": cluster_df["headline"].tolist(),
                    "outlets": outlets,
                    "outlet_count": len(outlets),
                    "dominant_sentiment": dominant_sentiment,
                    "label": cluster_df["headline"].iloc[0][:80]
                })
            visited.update(similar)

    clusters.sort(key=lambda x: x["outlet_count"], reverse=True)
    return clusters[:10]

def ask_nabz(question, df):
    client = Groq(api_key=GROQ_API_KEY)
    headlines_text = "\n".join([
        f"[{row['source']}] [{row['sentiment']}] {row['headline']}"
        for _, row in df.head(200).iterrows()
    ])
    sentiment_summary = df["sentiment"].value_counts().to_dict()
    prompt = f"""You are Nabz, an intelligence analyst for Pakistani public discourse.
You have access to {len(df)} headlines scraped from Dawn, ARY News, The News, and BOL News in the last 30 minutes.

Sentiment breakdown: {sentiment_summary}

Headlines sample:
{headlines_text}

User question: {question}

Answer in 3-5 sentences. Be specific, reference actual headlines where relevant. Be analytical, not generic."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

def get_bias_data(keyword, df):
    keyword = keyword.lower()
    filtered = df[df["headline"].str.lower().str.contains(keyword)]
    if filtered.empty:
        return None
    bias = filtered.groupby("source")["sentiment"].value_counts().unstack(fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in bias.columns:
            bias[col] = 0
    bias = bias[["positive", "neutral", "negative"]].reset_index()
    bias["total"] = bias["positive"] + bias["neutral"] + bias["negative"]
    bias = bias[bias["total"] > 0]
    return bias, filtered

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nabz", page_icon="📰", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-base:        #08090D;
    --bg-surface:     #0C0F18;
    --bg-card:        #101420;
    --bg-card-hover:  #141926;
    --border:         #1C2235;
    --border-subtle:  #12151F;
    --accent:         #00C9A7;
    --accent-dim:     rgba(0, 201, 167, 0.08);
    --accent-glow:    rgba(0, 201, 167, 0.18);
    --text-primary:   #E2E8F2;
    --text-secondary: #8895A8;
    --text-muted:     #3E4858;
    --positive:       #00C87A;
    --positive-bg:    rgba(0, 200, 122, 0.08);
    --positive-bd:    rgba(0, 200, 122, 0.24);
    --negative:       #E8412A;
    --negative-bg:    rgba(232, 65, 42, 0.08);
    --negative-bd:    rgba(232, 65, 42, 0.24);
    --neutral-col:    #6B7688;
    --neutral-bg:     rgba(107, 118, 136, 0.10);
    --neutral-bd:     rgba(107, 118, 136, 0.22);
    --live:           #FF3B3B;
    --r:              5px;
    --r-sm:           3px;
}

/* ── Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}
#MainMenu, footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2A3045; }

.main .block-container {
    padding: 1rem 2.5rem 3rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 1.4rem 1.6rem !important;
    transition: border-color 0.22s ease, background 0.22s ease;
}
[data-testid="metric-container"]:hover {
    border-color: var(--accent) !important;
    background: var(--bg-card-hover) !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    margin: 0 !important;
}
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 400 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.75rem 0 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    margin-bottom: 0.45rem !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.8rem 1rem !important;
    background: var(--bg-card) !important;
    border-radius: var(--r) var(--r) 0 0 !important;
    transition: background 0.18s ease;
    list-style: none !important;
}
[data-testid="stExpander"] summary:hover {
    background: var(--bg-card-hover) !important;
}
[data-testid="stExpander"] summary svg {
    fill: var(--text-muted) !important;
    flex-shrink: 0 !important;
}
[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid var(--border-subtle) !important;
    border-radius: var(--r) var(--r) 0 0 !important;
}
[data-testid="stExpander"] > div > div:last-child,
[data-testid="stExpander"] details > div {
    background: var(--bg-surface) !important;
    padding: 1rem 1.1rem !important;
}

/* ── Info / Alert boxes ── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-radius: var(--r) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] .stMarkdown p {
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
    line-height: 1.65 !important;
    margin: 0 !important;
}
[data-testid="stAlert"] > div > div:first-child svg { display: none !important; }

/* ── Warning boxes ── */
[data-testid="stNotification"],
div[data-baseweb="notification"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid #C8850A !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; }
[data-testid="stTextInput"] label {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    transition: border-color 0.2s ease;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: var(--accent-dim) !important;
    border: 1px solid rgba(0, 201, 167, 0.28) !important;
    border-radius: var(--r-sm) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: var(--accent) !important;
    font-size: 0.75rem !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMultiSelect"] label {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMultiSelect"] input { color: var(--text-primary) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
}

/* ── Plotly chart wrapper ── */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
}

/* ── Column gap ── */
[data-testid="stHorizontalBlock"] { gap: 1rem !important; }

/* ── Markdown text ── */
.stMarkdown p {
    color: var(--text-primary) !important;
    line-height: 1.65 !important;
}
.stMarkdown strong { color: var(--text-primary) !important; font-weight: 600 !important; }
.stMarkdown li {
    color: var(--text-secondary) !important;
    font-size: 0.855rem !important;
    line-height: 1.6 !important;
    margin-bottom: 0.25rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: var(--accent) !important;
}

/* ── Dropdown menu ── */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
[data-baseweb="menu"] li { color: var(--text-primary) !important; }
[data-baseweb="menu"] li:hover { background: var(--bg-card-hover) !important; }

/* ── Animations ── */
@keyframes pulse-live {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(255, 59, 59, 0.45); }
    60%       { opacity: 0.75; box-shadow: 0 0 0 5px rgba(255, 59, 59, 0); }
}
@keyframes fade-up {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1.5rem; animation: fade-up 0.45s ease forwards;">
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.55rem;">
        <h1 style="
            font-family:'DM Sans',sans-serif;
            font-size:2.8rem;
            font-weight:700;
            letter-spacing:-0.055em;
            color:#00C9A7;
            margin:0;
            line-height:1;
        ">NABZ</h1>
        <div style="
            display:flex;
            align-items:center;
            gap:0.4rem;
            background:rgba(255,59,59,0.07);
            border:1px solid rgba(255,59,59,0.2);
            border-radius:3px;
            padding:0.2rem 0.55rem;
            margin-top:0.15rem;
        ">
            <span style="
                display:inline-block;
                width:6px; height:6px;
                background:#FF3B3B;
                border-radius:50%;
                animation:pulse-live 1.9s ease-in-out infinite;
            "></span>
            <span style="
                font-family:'DM Mono',monospace;
                font-size:0.58rem;
                font-weight:500;
                letter-spacing:0.13em;
                color:#FF3B3B;
            ">LIVE</span>
        </div>
    </div>
    <p style="
        font-family:'DM Sans',sans-serif;
        font-size:0.82rem;
        color:#8895A8;
        letter-spacing:0.04em;
        margin:0 0 1.4rem;
        font-weight:400;
    ">Pakistan's Public Intelligence Tracker</p>
    <div style="height:1px; background:linear-gradient(to right, #00C9A7 0%, #1C2235 55%, transparent 100%);"></div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Fetching latest headlines..."):
    df, kw_df, stories = fetch_and_analyze()

col1, col2, col3 = st.columns(3)
col1.metric("Total Headlines", len(df))
col2.metric("Sources", df["source"].nunique() if not df.empty else 0)
col3.metric("Last Updated", datetime.now().strftime("%d %b %Y, %H:%M"))

st.divider()

# ── Section: Ask Nabz ─────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.7rem;">
    <div style="width:2px; height:1rem; background:#00C9A7; border-radius:1px; flex-shrink:0;"></div>
    <span style="
        font-family:'DM Sans',sans-serif;
        font-size:0.62rem;
        font-weight:600;
        letter-spacing:0.15em;
        text-transform:uppercase;
        color:#8895A8;
    ">Ask Nabz</span>
</div>
""", unsafe_allow_html=True)
question = st.text_input("Ask anything about today's news in Pakistan",
                          placeholder="e.g. What's the mood around petrol prices?")
if question and not df.empty:
    with st.spinner("Analyzing..."):
        answer = ask_nabz(question, df)
    st.info(answer)

st.divider()

# ── Section: Breaking Stories ─────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.6rem;">
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.3rem;">
        <div style="width:2px; height:1rem; background:#00C9A7; border-radius:1px; flex-shrink:0;"></div>
        <span style="
            font-family:'DM Sans',sans-serif;
            font-size:0.62rem;
            font-weight:600;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#8895A8;
        ">Breaking Stories</span>
    </div>
    <p style="
        font-family:'DM Sans',sans-serif;
        font-size:0.775rem;
        color:#3E4858;
        margin:0 0 0.5rem 1.7rem;
        line-height:1.5;
    ">Topics covered simultaneously across multiple outlets</p>
</div>
""", unsafe_allow_html=True)

_PILL = {
    "positive": ("#00C87A", "rgba(0,200,122,0.08)",  "rgba(0,200,122,0.24)"),
    "negative": ("#E8412A", "rgba(232,65,42,0.08)",  "rgba(232,65,42,0.24)"),
    "neutral":  ("#6B7688", "rgba(107,118,136,0.10)","rgba(107,118,136,0.22)"),
}

if stories:
    for story in stories:
        sent = story["dominant_sentiment"]
        pill_color, pill_bg, pill_bd = _PILL.get(sent, _PILL["neutral"])
        with st.expander(f"{story['label']}  ·  {story['outlet_count']} outlets"):
            st.markdown(
                f"""<div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.9rem;">
                    <span style="
                        font-family:'DM Mono',monospace;
                        font-size:0.6rem;
                        font-weight:500;
                        letter-spacing:0.1em;
                        background:rgba(0,201,167,0.07);
                        border:1px solid rgba(0,201,167,0.2);
                        color:#00C9A7;
                        border-radius:3px;
                        padding:0.15rem 0.5rem;
                        text-transform:uppercase;
                    ">{story['outlet_count']} outlets</span>
                    <span style="
                        font-family:'DM Mono',monospace;
                        font-size:0.6rem;
                        font-weight:500;
                        letter-spacing:0.1em;
                        background:{pill_bg};
                        border:1px solid {pill_bd};
                        color:{pill_color};
                        border-radius:3px;
                        padding:0.15rem 0.5rem;
                        text-transform:uppercase;
                    ">{sent}</span>
                    <span style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:#8895A8;">
                        {', '.join(story['outlets'])}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("**Related headlines:**")
            for h in story["headlines"]:
                st.markdown(f"- {h}")
else:
    st.info("No cross-outlet stories detected yet.")

st.divider()

# ── Section: Outlet Bias Tracker ──────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.6rem;">
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.3rem;">
        <div style="width:2px; height:1rem; background:#00C9A7; border-radius:1px; flex-shrink:0;"></div>
        <span style="
            font-family:'DM Sans',sans-serif;
            font-size:0.62rem;
            font-weight:600;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#8895A8;
        ">Outlet Bias Tracker</span>
    </div>
    <p style="
        font-family:'DM Sans',sans-serif;
        font-size:0.775rem;
        color:#3E4858;
        margin:0 0 0.5rem 1.7rem;
        line-height:1.5;
    ">Search a topic to see how each outlet covers it</p>
</div>
""", unsafe_allow_html=True)

keyword = st.text_input("Enter a keyword", placeholder="e.g. army, PTI, economy, India")
if keyword and not df.empty:
    result = get_bias_data(keyword, df)
    if result is None:
        st.warning(f"No headlines found containing '{keyword}'")
    else:
        bias_df, matched = result
        st.markdown(f"**{len(matched)} headlines** mention '{keyword}' across {len(bias_df)} outlets")

        bias_melted = bias_df.melt(id_vars=["source", "total"],
                                    value_vars=["positive", "neutral", "negative"],
                                    var_name="sentiment", value_name="count")
        colors = {"positive": "#00D9A3", "neutral": "#A0A0A0", "negative": "#FF6B4A"}
        fig = px.bar(bias_melted, x="source", y="count", color="sentiment",
                     color_discrete_map=colors, barmode="group")
        fig.update_layout(plot_bgcolor="#101420", paper_bgcolor="#08090D",
                          font_color="white", legend_title="Sentiment",
                          xaxis_title="Outlet", yaxis_title="Headlines")
        st.plotly_chart(fig, use_container_width=True)

        with st.spinner("Analyzing outlet bias..."):
            bias_headlines = "\n".join([
                f"[{row['source']}] [{row['sentiment']}] {row['headline']}"
                for _, row in matched.iterrows()
            ])
            bias_prompt = f"""You are Nabz, a media bias analyst for Pakistani news.

A user searched for the keyword: "{keyword}"

Here are all headlines mentioning this keyword, labeled by outlet and sentiment:
{bias_headlines}

In 4-6 sentences, analyze how different outlets are covering this topic. Are any outlets more negative or positive? Do they frame the story differently? Be specific, reference actual headlines. Be analytical and direct."""

            client = Groq(api_key=GROQ_API_KEY)
            bias_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": bias_prompt}],
                max_tokens=400
            )
            st.info(bias_response.choices[0].message.content)

        st.markdown("**Matching headlines:**")
        st.dataframe(matched[["headline", "source", "sentiment"]].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

st.divider()

# ── Section: Trending Topics ──────────────────────────────────────────────────
if not kw_df.empty:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.75rem;">
        <div style="width:2px; height:1rem; background:#00C9A7; border-radius:1px; flex-shrink:0;"></div>
        <span style="
            font-family:'DM Sans',sans-serif;
            font-size:0.62rem;
            font-weight:600;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#8895A8;
        ">Trending Topics</span>
    </div>
    """, unsafe_allow_html=True)
    fig3 = px.bar(kw_df.sort_values("count"), x="count", y="topic",
                  orientation="h", color="count",
                  color_continuous_scale=["#A0A0A0", "#00D9A3"])
    fig3.update_layout(showlegend=False, plot_bgcolor="#101420",
                       paper_bgcolor="#08090D", font_color="white",
                       coloraxis_showscale=False, yaxis_title=None,
                       xaxis_title="Mentions")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

col_left, col_right = st.columns(2)

# ── Section: Sentiment Distribution ──────────────────────────────────────────
with col_left:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.75rem;">
        <div style="width:2px; height:1rem; background:#00C9A7; border-radius:1px; flex-shrink:0;"></div>
        <span style="
            font-family:'DM Sans',sans-serif;
            font-size:0.62rem;
            font-weight:600;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#8895A8;
        ">Sentiment Distribution</span>
    </div>
    """, unsafe_allow_html=True)
    if not df.empty:
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        colors = {"positive": "#00D9A3", "neutral": "#A0A0A0", "negative": "#FF6B4A"}
        fig1 = px.bar(sentiment_counts, x="sentiment", y="count",
                      color="sentiment", color_discrete_map=colors)
        fig1.update_layout(showlegend=False, plot_bgcolor="#101420",
                           paper_bgcolor="#08090D", font_color="white")
        st.plotly_chart(fig1, use_container_width=True)

# ── Section: Headlines by Source ─────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.75rem;">
        <div style="width:2px; height:1rem; background:#00C9A7; border-radius:1px; flex-shrink:0;"></div>
        <span style="
            font-family:'DM Sans',sans-serif;
            font-size:0.62rem;
            font-weight:600;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#8895A8;
        ">Headlines by Source</span>
    </div>
    """, unsafe_allow_html=True)
    if not df.empty:
        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        fig2 = px.pie(source_counts, values="count", names="source",
                      color_discrete_sequence=["#00D9A3", "#FF6B4A", "#4A90D9", "#F5A623"])
        fig2.update_layout(plot_bgcolor="#101420", paper_bgcolor="#08090D", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Section: Latest Headlines ─────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.75rem;">
    <div style="width:2px; height:1rem; background:#00C9A7; border-radius:1px; flex-shrink:0;"></div>
    <span style="
        font-family:'DM Sans',sans-serif;
        font-size:0.62rem;
        font-weight:600;
        letter-spacing:0.15em;
        text-transform:uppercase;
        color:#8895A8;
    ">Latest Headlines</span>
</div>
""", unsafe_allow_html=True)
if not df.empty:
    source_filter = st.multiselect("Filter by source", options=df["source"].unique(),
                                    default=df["source"].unique())
    sentiment_filter = st.multiselect("Filter by sentiment",
                                       options=["positive", "neutral", "negative"],
                                       default=["positive", "neutral", "negative"])
    filtered = df[df["source"].isin(source_filter) & df["sentiment"].isin(sentiment_filter)]
    filtered_display = filtered[["headline", "source", "sentiment", "sentiment_score", "date", "url"]].sort_values("date", ascending=False)
    filtered_display["headline"] = filtered_display.apply(
        lambda row: f'<a href="{row["url"]}" target="_blank" style="color:#00C9A7; text-decoration:none;">{row["headline"]}</a>' 
        if pd.notna(row["url"]) else row["headline"], axis=1
    )
    st.write(
        filtered_display[["headline", "source", "sentiment", "sentiment_score", "date"]]
        .to_html(escape=False, index=False),
        unsafe_allow_html=True
    )