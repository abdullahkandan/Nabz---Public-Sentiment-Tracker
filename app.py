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

st.set_page_config(page_title="Nabz", page_icon="📰", layout="wide")

st.title("Nabz")
st.caption("Pakistan's Public Intelligence Tracker")

with st.spinner("Fetching latest headlines..."):
    df, kw_df, stories = fetch_and_analyze()

col1, col2, col3 = st.columns(3)
col1.metric("Total Headlines", len(df))
col2.metric("Sources", df["source"].nunique() if not df.empty else 0)
col3.metric("Last Updated", datetime.now().strftime("%d %b %Y, %H:%M"))

st.divider()

st.subheader("Ask Nabz")
question = st.text_input("Ask anything about today's news in Pakistan",
                          placeholder="e.g. What's the mood around petrol prices?")
if question and not df.empty:
    with st.spinner("Analyzing..."):
        answer = ask_nabz(question, df)
    st.info(answer)

st.divider()

st.subheader("Breaking Stories")
st.caption("Topics being covered simultaneously across multiple outlets")

if stories:
    for story in stories:
        sentiment_color = {"positive": "#00D9A3", "negative": "#FF6B4A", "neutral": "#A0A0A0"}
        color = sentiment_color.get(story["dominant_sentiment"], "#A0A0A0")
        with st.expander(f"📰 {story['label']}... — {story['outlet_count']} outlets"):
            st.markdown(f"**Outlets:** {', '.join(story['outlets'])}")
            st.markdown(f"**Dominant sentiment:** :{story['dominant_sentiment']}")
            st.markdown("**Related headlines:**")
            for h in story["headlines"]:
                st.markdown(f"- {h}")
else:
    st.info("No cross-outlet stories detected yet.")

st.divider()

st.subheader("Outlet Bias Tracker")
st.caption("Search a topic to see how each outlet covers it")

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
        fig.update_layout(plot_bgcolor="#1A1A2E", paper_bgcolor="#1A1A2E",
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

if not kw_df.empty:
    st.subheader("Trending Topics")
    fig3 = px.bar(kw_df.sort_values("count"), x="count", y="topic",
                  orientation="h", color="count",
                  color_continuous_scale=["#A0A0A0", "#00D9A3"])
    fig3.update_layout(showlegend=False, plot_bgcolor="#1A1A2E",
                       paper_bgcolor="#1A1A2E", font_color="white",
                       coloraxis_showscale=False, yaxis_title=None,
                       xaxis_title="Mentions")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    if not df.empty:
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        colors = {"positive": "#00D9A3", "neutral": "#A0A0A0", "negative": "#FF6B4A"}
        fig1 = px.bar(sentiment_counts, x="sentiment", y="count",
                      color="sentiment", color_discrete_map=colors)
        fig1.update_layout(showlegend=False, plot_bgcolor="#1A1A2E",
                           paper_bgcolor="#1A1A2E", font_color="white")
        st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("Headlines by Source")
    if not df.empty:
        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        fig2 = px.pie(source_counts, values="count", names="source",
                      color_discrete_sequence=["#00D9A3", "#FF6B4A", "#4A90D9", "#F5A623"])
        fig2.update_layout(plot_bgcolor="#1A1A2E", paper_bgcolor="#1A1A2E", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Latest Headlines")
if not df.empty:
    source_filter = st.multiselect("Filter by source", options=df["source"].unique(),
                                    default=df["source"].unique())
    sentiment_filter = st.multiselect("Filter by sentiment",
                                       options=["positive", "neutral", "negative"],
                                       default=["positive", "neutral", "negative"])
    filtered = df[df["source"].isin(source_filter) & df["sentiment"].isin(sentiment_filter)]
    filtered_display = filtered[["headline", "source", "sentiment", "sentiment_score", "date"]].sort_values("date", ascending=False)
    st.dataframe(filtered_display, use_container_width=True, hide_index=True)