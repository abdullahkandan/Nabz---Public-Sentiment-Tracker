import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Nabz",
    page_icon="📰",
    layout="wide"
)

st.title("Nabz")
st.caption("Pakistan's Public Intelligence Tracker")

@st.cache_data
def load_data():
    df = pd.read_csv("nabz_headlines.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Total Headlines", len(df))
col2.metric("Sources", df["source"].nunique())
col3.metric("Last Updated", datetime.now().strftime("%d %b %Y"))

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