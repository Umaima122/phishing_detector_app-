"""
AI Phishing Email Detector - Premium Professional UI
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
# Page config & Premium CSS
# ----------------------------
st.set_page_config(
    page_title="AI Phishing Shield — by Umaima Qureshi", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Hero Section */
    .hero-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(30%, -30%);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.15);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 24px rgba(102,126,234,0.3);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input Areas */
    .stTextArea textarea {
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }
    
    /* Alert Boxes */
    .alert-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 8px 24px rgba(239,68,68,0.3);
        margin: 1rem 0;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 8px 24px rgba(16,185,129,0.3);
        margin: 1rem 0;
    }
    
    .confidence-bar {
        height: 12px;
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 0.75rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    /* Hints Panel */
    .hints-panel {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
    }
    
    .hint-item {
        display: flex;
        align-items: start;
        gap: 0.75rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #475569;
    }
    
    .hint-icon {
        min-width: 24px;
        height: 24px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(102,126,234,0.1);
        border-radius: 12px;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Footer */
    .footer {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        color: #64748b;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .footer-name {
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 1.5rem;
        background: rgba(248,250,252,0.5);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: rgba(102,126,234,0.05);
    }
    
    /* Metric Cards */
    .metric-container {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .glass-card {
        animation: fadeIn 0.6s ease forwards;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
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
# Hero Header
# ----------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🛡️ AI Phishing Shield</div>
        <div class="hero-subtitle">Advanced machine learning protection against email threats</div>
        <div style="color: #64748b; font-size: 1rem; line-height: 1.6;">
            Powered by TF-IDF vectorization and Logistic Regression, trained on thousands of real-world phishing examples. 
            Get instant threat analysis with confidence scoring and explainable AI insights.
        </div>
        <div class="hero-badge">⚡ Developed by Umaima Qureshi</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load dataset
# ----------------------------
main_csv_path = "Phishing_Email.csv"
sample_csv_path = "Phishing_Email_Sample.csv"

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="section-icon">📂</div>Dataset Configuration</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your phishing dataset (optional)", type=["csv"], help="Upload Phishing_Email.csv for full training")

    if uploaded_file is not None:
        df = load_csv_from_bytes(uploaded_file.read())
    elif os.path.exists(main_csv_path):
        df = safe_read_csv(main_csv_path)
    elif os.path.exists(sample_csv_path):
        st.info("📊 Using sample dataset for demonstration")
        df = safe_read_csv(sample_csv_path)
    else:
        st.info("📊 Using built-in demo dataset")
        df = pd.DataFrame({
            "Email Text": [
                "Urgent! Your account has been suspended. Click http://fakebank.com to verify.",
                "Hi team, attached is the agenda for tomorrow's meeting. Regards.",
                "Dear user, update your password at http://phishingsite.com immediately to avoid suspension.",
                "Hello Omaima, congrats on your results. Let's celebrate this week!"
            ],
            "Email Type": ["Phishing Email", "Safe Email", "Phishing Email", "Safe Email"]
        })
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Clean & prepare dataset
# ----------------------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

text_col = "Email Text" if "Email Text" in df.columns else df.columns[0]
label_col = "Email Type" if "Email Type" in df.columns else df.columns[-1]

df[text_col] = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.strip() != ""].reset_index(drop=True)
df = df.drop(index=0, errors="ignore").reset_index(drop=True)

label_map = {"Phishing Email": 1, "Safe Email": 0}
if df[label_col].dtype == object:
    df['label'] = df[label_col].map(label_map)
    df['label'] = df['label'].fillna(0).astype(int)
else:
    df['label'] = df[label_col].astype(int)

df['processed_text'] = df[text_col].apply(preprocess_text)

# Dataset stats
phishing_count = (df['label'] == 1).sum()
safe_count = (df['label'] == 0).sum()
total_count = len(df)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><div class="section-icon">📊</div>Dataset Statistics</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{total_count}</div>
            <div class="stat-label">Total Emails</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{phishing_count}</div>
            <div class="stat-label">Phishing Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{safe_count}</div>
            <div class="stat-label">Safe Emails</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{(phishing_count/total_count*100):.1f}%</div>
            <div class="stat-label">Threat Rate</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("🔍 View Dataset Preview", expanded=False):
    st.dataframe(df[[text_col, label_col]].head(10), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

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
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><div class="section-icon">🎯</div>Model Performance</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-container">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Accuracy</div>
            <div style="font-size: 2rem; font-weight: 800; color: #667eea;">{accuracy:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    precision = model_info["report"].get("1", {}).get("precision", 0)
    st.markdown(
        f"""
        <div class="metric-container">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Precision</div>
            <div style="font-size: 2rem; font-weight: 800; color: #667eea;">{precision:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    recall = model_info["report"].get("1", {}).get("recall", 0)
    st.markdown(
        f"""
        <div class="metric-container">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Recall</div>
            <div style="font-size: 2rem; font-weight: 800; color: #667eea;">{recall:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("📈 Detailed Metrics & Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(model_info["confusion_matrix"], annot=True, fmt="d", ax=ax, cmap="RdPu", cbar=True, square=True)
    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax.set_ylabel("Actual", fontsize=12, fontweight='bold')
    ax.set_xticklabels(["Safe", "Phishing"], fontsize=11)
    ax.set_yticklabels(["Safe", "Phishing"], fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=20)
    st.pyplot(fig)
    
    st.write("**Classification Report:**")
    report_df = pd.DataFrame(model_info["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Inference UI
# ----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><div class="section-icon">✉️</div>Email Threat Scanner</div>', unsafe_allow_html=True)

col_input, col_hints = st.columns([2, 1])

with col_input:
    email_input = st.text_area(
        "Paste email content for analysis",
        height=250,
        placeholder="Example: Urgent! Your account has been compromised. Click here to verify your identity immediately...",
        help="Paste the full email content including subject and body"
    )
    
    uploaded_txt = st.file_uploader("Or upload a .txt file", type=["txt"], help="Upload a text file containing the email")
    
    if uploaded_txt is not None and not email_input:
        try:
            email_input = uploaded_txt.read().decode("utf-8", errors="ignore")
        except Exception:
            email_input = str(uploaded_txt.getvalue())

    if st.button("🔍 Analyze Email Threat"):
        if not email_input.strip():
            st.warning("⚠️ Please paste or upload email content to analyze")
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
                conf_pct = f"{proba:.1%}" if proba is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="alert-danger">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                            <div style="font-size: 2.5rem;">⚠️</div>
                            <div>
                                <div style="font-size: 1.4rem; font-weight: 800;">PHISHING DETECTED</div>
                                <div style="font-size: 1rem; opacity: 0.95;">Threat Confidence: {conf_pct}</div>
                            </div>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {proba*100 if proba else 0}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("**🔍 Threat Indicators Detected:**")
                indicators = []
                if "url" in processed_input:
                    indicators.append("🔗 Suspicious URL tokens detected")
                if re.search(r'\b(urgent|immediately|verify|password|suspended|click|act now)\b', processed_input):
                    indicators.append("⚡ Urgency manipulation tactics")
                if re.search(r'\b(bank|account|verify|login|password|security|credential)\b', processed_input):
                    indicators.append("🏦 Financial/security keywords present")
                if re.search(r'\b(winner|prize|congratulations|claim|free)\b', processed_input):
                    indicators.append("🎁 Reward/prize baiting language")
                
                for indicator in indicators:
                    st.markdown(f"- {indicator}")
                
                if not indicators:
                    st.markdown("- ⚠️ Content pattern matches known phishing templates")
                    
            else:
                conf_pct = f"{(1-proba):.1%}" if proba is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="alert-success">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                            <div style="font-size: 2.5rem;">✅</div>
                            <div>
                                <div style="font-size: 1.4rem; font-weight: 800;">EMAIL APPEARS SAFE</div>
                                <div style="font-size: 1rem; opacity: 0.95;">Safety Confidence: {conf_pct}</div>
                            </div>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {(1-proba)*100 if proba else 100}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("**✓ No obvious threat indicators found in content analysis**")
                st.info("💡 Remember: Always verify sender identity and be cautious with unexpected emails, even if they appear safe.")

with col_hints:
    st.markdown(
        """
        <div class="hints-panel">
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 1rem; color: #1e293b;">🧠 AI Detection Insights</div>
            
            <div class="hint-item">
                <div class="hint-icon">1</div>
                <div><strong>Urgency words</strong> like "urgent", "verify", "immediately" raise red flags</div>
            </div>
            
            <div class="hint-item">
                <div class="hint-icon">2</div>
                <div><strong>Suspicious links</strong> or email addresses are automatically flagged</div>
            </div>
            
            <div class="hint-item">
                <div class="hint-icon">3</div>
                <div><strong>Financial keywords</strong> combined with urgency indicate high risk</div>
            </div>
            
            <div class="hint-item">
                <div class="hint-icon">4</div>
                <div>Confidence <strong>>70%</strong> warrants immediate caution</div>
            </div>
            
            <div class="hint-item">
                <div class="hint-icon">⚠️</div>
                <div><strong>Limitations:</strong> This tool analyzes text content only. Always verify sender identity separately.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <div class="footer">
        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            Developed and Deployed by <span class="footer-name">Umaima Qureshi</span>
        </div>
        <div style="font-size: 0.9rem; color: #94a3b8;">
            🎓 Educational demonstration of ML-powered email security<br>
            For production use: Implement additional verification layers, link scanning, attachment analysis, and human oversight
        </div>
        <div style="margin-top: 1rem; font-size: 0.85rem; color: #cbd5e1;">
            Powered by TF-IDF • Logistic Regression • Scikit-learn • Streamlit
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

