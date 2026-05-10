# Nabz — Pakistan's Public Intelligence Tracker

Nabz is a real-time news intelligence platform that scrapes, analyzes, and interprets Pakistani public discourse as it happens. It pulls live headlines from Dawn, ARY News, The News, and BOL News every 30 minutes, runs sentiment analysis on every headline, extracts trending topics from the aggregate, and powers a natural language query engine that lets you ask plain-English questions about what's happening in Pakistan right now.

## Live App
[nabzpk.streamlit.app](https://nabzpk.streamlit.app)

## What It Does

**Real-Time Scraping:** Automatically fetches headlines from four major Pakistani news outlets every 30 minutes via RSS. No manual updates, no static datasets.

**Sentiment Analysis:** Every headline is scored as positive, negative, or neutral using VADER sentiment analysis, calibrated for news text. The distribution across sources reveals the emotional temperature of Pakistani media at any given moment.

**Trending Topics:** Frequency-based keyword extraction surfaces what words and phrases are appearing most across all outlets right now, cutting through individual headlines to show the shape of the news cycle.

**Ask Nabz:** A natural language intelligence engine powered by Groq (Llama 3.3 70B). Ask anything — "What's the mood around petrol prices?", "What's happening in Karachi?", "How is Iran being covered?" — and Nabz queries the live headline dataset and returns a specific, analytical answer grounded in actual articles.

## Tech Stack

- **Frontend:** Streamlit
- **Scraping:** Requests + BeautifulSoup (RSS feeds)
- **Sentiment:** VADER (vaderSentiment)
- **LLM:** Groq API (Llama 3.3 70B)
- **Data:** pandas
- **Visualization:** Plotly Express
- **Deployment:** Streamlit Cloud

## Data Sources

| Outlet | Feed |
|--------|------|
| Dawn | dawn.com/feeds/home |
| ARY News | arynews.tv/feed |
| The News | thenews.com.pk/rss/1/1 |
| BOL News | bolnews.com/feed |

## Project Structure
Nabz/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── .gitignore              # Excludes .env and secrets
└── README.md

## Key Features at a Glance

- 300+ headlines analyzed per refresh cycle
- 4 Pakistani news sources aggregated in one place
- Sentiment breakdown per source and overall
- Top 15 trending keywords extracted from live headlines
- LLM query engine grounded in real-time data, not general knowledge

## About

Built by Abdullah Kandan, a data science student at a university in Islamabad. This project is part of a portfolio focused on NLP, real-time data pipelines, and applied AI for the Pakistani context.