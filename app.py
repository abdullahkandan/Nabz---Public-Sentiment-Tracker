import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

st.set_page_config(
    page_title="Nabz",
    page_icon="📰",
    layout="wide"
)

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
            response = requests.get(source["url"], headers=headers, timeout=10)
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
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    all_words = []
    for headline in df["headline"]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', headline.lower())
        all_words.extend([w for w in words if w not in STOPWORDS])

    top_keywords = Counter(all_words).most_common(15)
    kw_df = pd.DataFrame(top_keywords, columns=["topic", "count"])

    return df, kw_df

st.title("Nabz")
st.caption("Pakistan's Public Intelligence Tracker")

with st.spinner("Fetching latest headlines..."):
    df, kw_df = fetch_and_analyze()

col1, col2, col3 = st.columns(3)
col1.metric("Total Headlines", len(df))
col2.metric("Sources", df["source"].nunique())
col3.metric("Last Updated", datetime.now().strftime("%d %b %Y, %H:%M"))

st.divider()

st.subheader("Trending Topics")
fig3 = px.bar(kw_df.sort_values("count"), x="count", y="topic",
              orientation="h",
              color="count",
              color_continuous_scale=["#A0A0A0", "#00D9A3"])
fig3.update_layout(
    showlegend=False,
    plot_bgcolor="#1A1A2E",
    paper_bgcolor="#1A1A2E",
    font_color="white",
    coloraxis_showscale=False,
    yaxis_title=None,
    xaxis_title="Mentions"
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
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
    source_counts = df["source"].value_counts().reset_index()
    source_counts.columns = ["source", "count"]
    fig2 = px.pie(source_counts, values="count", names="source",
                  color_discrete_sequence=["#00D9A3", "#FF6B4A", "#4A90D9", "#F5A623"])
    fig2.update_layout(plot_bgcolor="#1A1A2E", paper_bgcolor="#1A1A2E", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Latest Headlines")
source_filter = st.multiselect("Filter by source", options=df["source"].unique(),
                                default=df["source"].unique())
sentiment_filter = st.multiselect("Filter by sentiment", options=["positive", "neutral", "negative"],
                                   default=["positive", "neutral", "negative"])

filtered = df[df["source"].isin(source_filter) & df["sentiment"].isin(sentiment_filter)]
filtered_display = filtered[["headline", "source", "sentiment", "sentiment_score", "date"]].sort_values("date", ascending=False)
st.dataframe(filtered_display, use_container_width=True, hide_index=True)