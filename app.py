# app.py
"""
AI Phishing Email Detector - Streamlit app
TF-IDF + Logistic Regression trained on Kaggle Phishing Emails dataset.
Author & Deployer: Umaima Qureshi
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# ----------------------------
# Page config & CSS
# ----------------------------
st.set_page_config(page_title="AI Phishing Email Detector — by Umaima Qureshi", layout="wide")
st.markdown(
    """
    <style>
    .title {font-size:28px; font-weight:700; margin-bottom: -10px;}
    .subtitle {font-size:14px; color: #6c757d; margin-top: 2px;}
    .footer {color:#6c757d; font-size:12px; text-align:center;}
    .small-muted {color:#6c757d; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Utility functions
# ----------------------------
@st.cache_data
def load_csv_from_bytes(uploaded_bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(uploaded_bytes))

def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="title">AI Phishing Email Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">By <strong>Umaima Qureshi</strong> — TF-IDF + Logistic Regression, trained on a Kaggle phishing dataset.</div>', unsafe_allow_html=True)
st.write("")

# ----------------------------
# Load dataset
# ----------------------------
st.header("📂 Dataset")
main_csv_path = "Phishing_Email.csv"
sample_csv_path = "Phishing_Email_Sample.csv"
uploaded_file = st.file_uploader("Upload Phishing_Email.csv (optional)", type=["csv"])

# Determine which CSV to use
if uploaded_file is not None:
    df = load_csv_from_bytes(uploaded_file.read())
elif os.path.exists(main_csv_path):
    df = safe_read_csv(main_csv_path)
elif os.path.exists(sample_csv_path):
    st.info("Full Kaggle dataset not found. Using sample CSV for demo/testing.")
    df = safe_read_csv(sample_csv_path)
else:
    st.warning("No dataset found — using tiny built-in demo dataset.")
    df = pd.DataFrame({
        "Email Text": [
            "Urgent! Your account has been suspended. Click http://fakebank.com to verify.",
            "Hi team, attached is the agenda for tomorrow's meeting. Regards.",
            "Dear user, update your password at http://phishingsite.com immediately to avoid suspension.",
            "Hello Omaima, congrats on your results. Let's celebrate this week!"
        ],
        "Email Type": ["Phishing Email", "Safe Email", "Phishing Email", "Safe Email"]
    })

# ----------------------------
# Clean & prepare dataset
# ----------------------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

text_col = "Email Text" if "Email Text" in df.columns else df.columns[0]
label_col = "Email Type" if "Email Type" in df.columns else df.columns[-1]

df[text_col] = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.strip() != ""].reset_index(drop=True)

# Remove first email if desired (currently hardcoded)
df = df.drop(index=0, errors="ignore").reset_index(drop=True)

label_map = {"Phishing Email": 1, "Safe Email": 0}
if df[label_col].dtype == object:
    df['label'] = df[label_col].map(label_map)
    df['label'] = df['label'].fillna(0).astype(int)
else:
    df['label'] = df[label_col].astype(int)

df['processed_text'] = df[text_col].apply(preprocess_text)

# Dataset preview
with st.expander("🔍 Dataset preview & label distribution", expanded=False):
    st.dataframe(df[[text_col, label_col]].head(6), height=160)
    counts = df['label'].value_counts().rename({0: "Safe (0)", 1: "Phishing (1)"}).to_frame("count")
    st.write("Label distribution:")
    st.dataframe(counts, height=120)

# ----------------------------
# Model training
# ----------------------------
@st.cache_resource
def train_model(processed_texts, labels, test_size=0.2, random_state=42):
    strat = labels if len(np.unique(labels)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=test_size, random_state=random_state, stratify=strat
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }

model_info = train_model(df['processed_text'].tolist(), df['label'].values)
vectorizer, model, accuracy = model_info["vectorizer"], model_info["model"], model_info["accuracy"]

# ----------------------------
# Model performance
# ----------------------------
st.header("📊 Model performance (held-out test set)")
st.write(f"**Accuracy:** {accuracy:.2%}")

with st.expander("Show confusion matrix", expanded=False):
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(model_info["confusion_matrix"], annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Safe (0)", "Phishing (1)"])
    ax.set_yticklabels(["Safe (0)", "Phishing (1)"])
    st.pyplot(fig)

with st.expander("Precision / Recall / F1", expanded=False):
    report_df = pd.DataFrame(model_info["report"]).transpose().round(3)
    st.dataframe(report_df, height=200)

# ----------------------------
# Inference UI
# ----------------------------
st.header("✉️ Try it yourself: Detect phishing from pasted email")
input_col, hint_col = st.columns([3,1])

with input_col:
    email_input = st.text_area("Paste full email text here", height=200,
                               placeholder="Example: Urgent — verify your account at http://fakebank.com ...")
    uploaded_txt = st.file_uploader("Or upload a .txt file with email content", type=["txt"])
    if uploaded_txt is not None and not email_input:
        try:
            email_input = uploaded_txt.read().decode("utf-8", errors="ignore")
        except Exception:
            email_input = str(uploaded_txt.getvalue())

    if st.button("🔍 Detect Phishing"):
        if not email_input.strip():
            st.warning("Please paste or upload some email text to analyze!")
        else:
            processed_input = preprocess_text(email_input)
            input_vec = vectorizer.transform([processed_input])
            try:
                proba = model.predict_proba(input_vec)[0][1]
            except Exception:
                try:
                    score = model.decision_function(input_vec)[0]
                    proba = 1/(1+np.exp(-score))
                except Exception:
                    proba = None

            pred = model.predict(input_vec)[0]

            if pred == 1:
                if proba is not None:
                    st.error(f"⚠️ **PHISHING** — Confidence: {proba:.2%}")
                else:
                    st.error("⚠️ **PHISHING** (confidence not available)")
                st.write("**Possible reasons (content-based):**")
                if "url" in processed_input:
                    st.write("- Contains URL tokens.")
                if re.search(r'\b(urgent|immediately|verify|password|suspended|click)\b', processed_input):
                    st.write("- Contains urgent or action words.")
                if re.search(r'\b(bank|account|verify|login|password|security)\b', processed_input):
                    st.write("- Contains financial/security keywords.")
            else:
                if proba is not None:
                    st.success(f"✅ **Likely SAFE** — Confidence (phishing prob): {proba:.2%}")
                else:
                    st.success("✅ **Likely SAFE**")
                st.write("No obvious red flags found.")

with hint_col:
    st.markdown("**Explainability hints**")
    st.markdown("- Words like **urgent**, **verify**, **click** raise suspicion.")
    st.markdown("- Presence of **URL** or **email** tokens is a red flag.")
    st.markdown("- This model is content-based only — it doesn't scan attachments or links.")
    st.markdown("- Treat >70% phishing probability emails carefully.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown('<div class="footer"> Developed and Deployed by <strong>Umaima Qureshi</strong> — for educational/demo purposes. '
            'For production use: add URL/attachment scanning, human review, and other security checks.</div>', unsafe_allow_html=True)

