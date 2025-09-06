# app.py
import streamlit as st
import pandas as pd
import re
from datetime import datetime
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="SocialGuard Prototype", layout="wide")

# -------------------------
# Helper functions
# -------------------------
def load_data(path="sample_data.csv"):
    # Build full file path relative to this script
    full_path = os.path.join(os.path.dirname(__file__), path)
    try:
        df = pd.read_csv(full_path, parse_dates=["date"])
        if 'label' in df.columns:
            df['label'] = df['label'].str.lower()
        return df
    except Exception as e:
        # Fallback to dummy data if CSV fails to load
        data = {
            "username": ["@bot1", "@suspect2", "@genuine3"],
            "date": [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02"), pd.Timestamp("2024-06-03")],
            "label": ["bot", "suspicious", "genuine"],
            "text": [
                "Get free money! Click here: http://spam.com",
                "Win a prize by subscribing! http://fake.com",
                "Enjoying the sunshine today with friends."
            ]
        }
        df = pd.DataFrame(data)
        df['label'] = df['label'].str.lower()
        return df

def heuristics_analyze(text):
    text_lower = text.lower()
    urls = len(re.findall(r"https?://", text_lower))
    exclam = text.count("!")
    tokens = re.findall(r"\w+", text_lower)
    common = max([tokens.count(t) for t in set(tokens)]) if tokens else 0
    spammy_words = sum(1 for w in ["free","win","prize","click","subscribe","buy","follow"] if w in text_lower)
    score = 0
    score += urls * 2
    score += max(0, common - 4)
    score += exclam
    score += spammy_words * 2
    reasons = []
    if urls > 0: reasons.append(f"{urls} URL(s)")
    if common > 5: reasons.append("repetitive text")
    if spammy_words: reasons.append("spammy keywords")
    if exclam > 2: reasons.append("excessive punctuation")
    if score >= 4:
        label = "Suspicious"
    elif score >= 2:
        label = "Bot"
    else:
        label = "Genuine"
    return {"label": label, "score": score, "reasons": reasons}

# -------------------------
# Load dataset
# -------------------------
df = load_data()

# session state for model and vectorizer
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("SocialGuard Prototype")
menu = st.sidebar.radio("Navigate", ["Dashboard","Analyze Text","Alerts","Account Detail","Train Model","Download Report"])

# -------------------------
# Dashboard
# -------------------------
if menu == "Dashboard":
    st.title("SocialGuard — Dashboard (Prototype)")
    col1, col2 = st.columns([2,1])

    # summary stats
    counts = df['label'].value_counts().reindex(["bot","suspicious","genuine"]).fillna(0)
    counts.index = counts.index.str.capitalize()
    with col1:
        st.subheader("Risk Distribution")
        fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detections over time")
        times = df.groupby(df['date'].dt.date).size().reset_index(name='count')
        fig2 = px.line(times, x='date', y='count', markers=True, title="Flagged accounts per day")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Quick Stats")
        st.metric("Total flagged", int(len(df)))
        st.metric("Bots", int(counts.get("Bot",0)))
        st.metric("Suspicious", int(counts.get("Suspicious",0)))
        st.metric("Genuine", int(counts.get("Genuine",0)))

    st.markdown("---")
    st.subheader("Recent Alerts")
    recent = df.sort_values(by="date", ascending=False).head(10)[["username","date","label","text"]]
    st.table(recent)

# -------------------------
# Analyze Text
# -------------------------
elif menu == "Analyze Text":
    st.title("Analyze Post Text")
    st.write("Paste a social media post and analyze it with heuristic rules or ML (if trained).")
    text_input = st.text_area("Post text", height=150)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Heuristic Analyze"):
            if not text_input.strip():
                st.error("Paste some text to analyze.")
            else:
                res = heuristics_analyze(text_input)
                st.success(f"Label: {res['label']}")
                st.write("Score:", res['score'])
                if res['reasons']:
                    st.write("Reasons:", ", ".join(res['reasons']))
    with colB:
        if st.session_state.model is not None and st.session_state.vectorizer is not None:
            if st.button("ML Predict"):
                X = st.session_state.vectorizer.transform([text_input])
                pred = st.session_state.model.predict(X)[0]
                st.info(f"ML Prediction: {pred.capitalize()}")
        else:
            st.info("Train the ML model under 'Train Model' to enable ML predictions.")

# -------------------------
# Alerts
# -------------------------
elif menu == "Alerts":
    st.title("Alerts / Flagged Accounts")
    st.write("This table uses sample data. Click a username in 'Account Detail' to view breakdown.")
    st.dataframe(df[["username","date","label","text"]].sort_values(by="date", ascending=False))

# -------------------------
# Account Detail
# -------------------------
elif menu == "Account Detail":
    st.title("Account Detail")
    user = st.text_input("Enter username (e.g., @free_money_bot)")
    if st.button("Lookup"):
        row = df[df['username'].str.lower() == user.lower()]
        if row.empty:
            st.warning("User not found in sample data.")
        else:
            row = row.iloc[0]
            st.subheader(row['username'])
            st.write("Date:", row['date'].date())
            st.write("Label:", row['label'])
            st.write("Text:", row['text'])
            # run heuristics too
            heur = heuristics_analyze(row['text'])
            st.write("Heuristic assessment:", heur)

# -------------------------
# Train Model
# -------------------------
elif menu == "Train Model":
    st.title("Train tiny ML model (on sample data)")
    st.write("This trains a simple TF-IDF + LogisticRegression model on sample_data.csv labels: bot/suspicious/genuine.")
    if st.button("Train Model Now"):
        df_train = df.copy()
        X = df_train['text'].fillna("").values
        y = df_train['label'].str.lower().values
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
        Xv = vectorizer.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(Xtr, ytr)
        acc = model.score(Xte, yte)
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.success(f"Trained small model. Validation accuracy: {acc:.2f}")
        st.info("Now you can use 'ML Predict' in Analyze Text.")

# -------------------------
# Download Report
# -------------------------
elif menu == "Download Report":
    st.title("Download flagged accounts")
    st.write("Download sample flagged accounts as CSV.")
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="flagged_accounts_sample.csv", mime="text/csv")