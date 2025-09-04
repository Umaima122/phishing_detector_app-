# app.py
"""
AI Phishing Email Detector - Streamlit app
TF-IDF + Logistic Regression trained on Kaggle Phishing Emails dataset.
Place Phishing_Email.csv in the same folder, or upload within the app.
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
# Utility & preprocessing
# ----------------------------
st.set_page_config(page_title="AI Phishing Email Detector", layout="wide")

@st.cache_data
def load_csv_from_bytes(uploaded_bytes) -> pd.DataFrame:
    # load file from uploaded bytes (for streamlit file_uploader fallback)
    return pd.read_csv(io.BytesIO(uploaded_bytes))

def safe_read_csv(path: str) -> pd.DataFrame:
    # Safely read CSV, return empty df on error
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # replace urls and emails with tokens
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    # remove characters except letters and whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# Load dataset (from disk or upload)
# ----------------------------
st.title("AI Phishing Email Detector")
st.caption("TF-IDF + Logistic Regression — trained on a Kaggle phishing emails dataset. Beginner-friendly & ready to demo for your BSCS application.")

col1, col2 = st.columns([3,1])

with col2:
    st.markdown("### Dataset")
    csv_path = "Phishing_Email.csv"
    file_present = os.path.exists(csv_path)
    if file_present:
        st.success(f"Found `{csv_path}` in the folder ✅")
        use_local = st.checkbox("Use local Phishing_Email.csv", value=True)
    else:
        st.info("No local dataset found. Upload CSV or use a small example.")
        use_local = False

    uploaded_file = st.file_uploader("Upload Phishing_Email.csv (optional)", type=["csv"])

# determine dataframe source
df = pd.DataFrame()
if use_local:
    df = safe_read_csv(csv_path)
elif uploaded_file is not None:
    try:
        df = load_csv_from_bytes(uploaded_file.read())
    except Exception as e:
        st.error("Failed to read uploaded CSV: " + str(e))

# If still empty, provide a tiny sample so app can run
if df.empty:
    st.warning("No dataset loaded — using a tiny built-in sample so you can test the UI.")
    sample = {
        "Email Text": [
            "Urgent! Your account has been suspended. Click http://fakebank.com to verify.",
            "Hi team, attached is the agenda for tomorrow's meeting. Regards.",
            "Dear user, update your password at http://phishingsite.com immediately to avoid suspension.",
            "Hello Omaima, congrats on your results. Let's celebrate this week!"
        ],
        "Email Type": ["Phishing Email", "Safe Email", "Phishing Email", "Safe Email"]
    }
    df = pd.DataFrame(sample)

# ----------------------------
# Clean and prepare dataset
# ----------------------------
# Some kaggle csvs include an 'Unnamed: 0' column — drop if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# handle expected column names (support both "Email Text" & "Email Type")
possible_text_cols = [c for c in df.columns if "text" in c.lower() or "email" in c.lower()]
# prefer exact matches
if "Email Text" in df.columns:
    text_col = "Email Text"
else:
    # find a likely candidate
    text_col = None
    for c in df.columns:
        if "text" in c.lower() or "email" in c.lower():
            # avoid 'Email Type' or label columns
            if "type" not in c.lower() and "label" not in c.lower():
                text_col = c
                break
    if text_col is None:
        # as fallback, take first column
        text_col = df.columns[0]

if "Email Type" in df.columns:
    label_col = "Email Type"
else:
    # try to find a label column
    label_col = None
    for c in df.columns:
        if "type" in c.lower() or "label" in c.lower() or "class" in c.lower():
            label_col = c
            break

# If label column missing, we will create a fake sample label for demo
if label_col is None:
    st.warning("No label column detected. The app will still run but model training will use generated demo labels.")
    df["Email Type"] = df[text_col].apply(lambda s: "Phishing Email" if "http" in str(s).lower() or "urgent" in str(s).lower() or "click" in str(s).lower() else "Safe Email")
    label_col = "Email Type"

# Drop rows with empty or null text
df[text_col] = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.strip() != ""].reset_index(drop=True)

# Map labels to numeric
label_map = {"Phishing Email": 1, "Safe Email": 0}
# If labels are already numeric/bool, keep them
if df[label_col].dtype == object:
    df['label'] = df[label_col].map(label_map)
    # for any unmapped values, try to coerce common terms
    unmapped_mask = df['label'].isna()
    if unmapped_mask.any():
        df.loc[unmapped_mask, 'label'] = df.loc[unmapped_mask, label_col].apply(lambda s: 1 if "phish" in str(s).lower() or "spam" in str(s).lower() else 0)
    df['label'] = df['label'].astype(int)
else:
    # numeric already
    df['label'] = df[label_col].astype(int)

# Preprocess
df['processed_text'] = df[text_col].apply(preprocess_text)

# Show dataset snapshot and simple stats
with col1:
    st.subheader("Dataset preview & stats")
    st.dataframe(df[[text_col, label_col]].head(6))
    counts = df['label'].value_counts().rename({0: "Safe (0)", 1: "Phishing (1)"}).to_frame("count")
    st.write("Label distribution:")
    st.dataframe(counts)

# ----------------------------
# Model training (cached)
# ----------------------------
@st.cache_resource
def train_model(processed_texts, labels, test_size=0.2, random_state=42):
    """
    Train TF-IDF + LogisticRegression and return vectorizer, model, metrics, splits.
    This is cached so reruns do not retrain unnecessarily.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=test_size, random_state=random_state, stratify=labels if len(np.unique(labels))>1 else None
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
        "report": report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }

# Only train if we have at least 2 samples in each class (otherwise model will still train but stratify removed)
if df['label'].nunique() == 1:
    st.warning("Dataset contains only one label class; metrics like accuracy will be trivial. For best results use the full Kaggle CSV.")
model_info = train_model(df['processed_text'].tolist(), df['label'].values)

vectorizer = model_info["vectorizer"]
model = model_info["model"]
accuracy = model_info["accuracy"]

# Show metrics
st.markdown("---")
st.subheader("Model performance (on held-out test set)")
st.write(f"**Accuracy:** {accuracy:.2%}")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(4,3))
cm = model_info["confusion_matrix"]
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticklabels(["Safe (0)","Phishing (1)"])
ax.set_yticklabels(["Safe (0)","Phishing (1)"])
st.pyplot(fig)

# Show classification report summary
report = model_info["report"]
st.write("Precision / Recall / F1 for each class:")
report_df = pd.DataFrame(report).transpose().round(3)
st.dataframe(report_df)

# ----------------------------
# Inference UI
# ----------------------------
st.markdown("---")
st.subheader("Detect phishing from a pasted email (or upload .txt)")
colA, colB = st.columns([3,1])
with colA:
    email_input = st.text_area("Paste full email text here", height=240, placeholder="Example: Urgent — verify your account at http://fakebank.com ...")
    uploaded_txt = st.file_uploader("Or upload a .txt file with email content", type=["txt"])

with colB:
    st.markdown("**Explainability hints**")
    st.write("- Words like 'urgent', 'verify', 'click', or presence of URL tokens often raise phishing probability.")
    st.write("- This is a content-based model — it does not check link safety or attachments.")
    st.write("- Use confidence to decide: anything above ~70% should be treated carefully.")

if uploaded_txt is not None and not email_input:
    try:
        text_bytes = uploaded_txt.read()
        email_input = text_bytes.decode("utf-8", errors="ignore")
    except Exception:
        email_input = str(uploaded_txt.getvalue())

if st.button("Detect Phishing"):
    if not email_input or email_input.strip() == "":
        st.warning("Please paste or upload some email text to analyze!")
    else:
        processed_input = preprocess_text(email_input)
        input_vec = vectorizer.transform([processed_input])
        pred = model.predict(input_vec)[0]
        # If model supports predict_proba:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_vec)[0][1]  # prob of class 1 (Phishing)
        else:
            # fallback: use decision_function if available
            try:
                score = model.decision_function(input_vec)[0]
                proba = 1/(1+np.exp(-score))
            except Exception:
                proba = None

        if pred == 1:
            if proba is not None:
                st.error(f"⚠️ **PHISHING** — Confidence: {proba:.2%}")
            else:
                st.error("⚠️ **PHISHING** (confidence not available)")
            st.write("Possible reasons (content-based):")
            # show a few heuristic clues
            clues = []
            if "url" in processed_input:
                clues.append("- Contains URL tokens (possible link to phishing site).")
            if re.search(r'\b(urgent|immediately|verify|password|suspended|click)\b', processed_input):
                clues.append("- Contains urgent or action-oriented words often used in phishing.")
            if re.search(r'\b(bank|account|verify|login|password|security)\b', processed_input):
                clues.append("- Contains financial/security related keywords.")
            if not clues:
                clues.append("- Text matched patterns commonly seen in known phishing emails.")
            for c in clues:
                st.write(c)
        else:
            if proba is not None:
                st.success(f"✅ **Likely SAFE** — Confidence (phishing prob): {proba:.2%}")
            else:
                st.success("✅ **Likely SAFE**")
            st.write("No obvious content-based red flags found.")

# ----------------------------
# Optional: quick demo examples
# ----------------------------
st.markdown("---")
st.subheader("Quick test examples")
with st.expander("Show examples"):
    st.write("Click an example to populate the input box.")
    if st.button("Example: Phishing — account suspended"):
        st.session_state['example_text'] = "Urgent! Your account has been suspended. Click http://fakebank.com to verify your identity and avoid permanent suspension."
    if st.button("Example: Safe — meeting"):
        st.session_state['example_text'] = "Hi Omaima, reminder that we have our study group at 4 PM tomorrow. Bring your notebooks."
    # inject into the text area (client-side update)
    if 'example_text' in st.session_state:
        # Ugly but works: re-render the text area value by writing with st.text_area again
        # Note: Streamlit doesn't support programmatically setting text_area value directly without experimental features.
        st.write("Paste this into the input box manually or re-run with the example text shown below:")
        st.code(st.session_state['example_text'])

# ----------------------------
# Footer: instructions & saving model (optional)
# ----------------------------
st.markdown("---")
st.caption("Developed for educational/demo purposes. For production use: add URL/attachment scanning, human review, and other security checks. If you used the Kaggle dataset, cite the dataset in your readme when deploying.")
