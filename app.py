import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Railway AI | Complaint Intelligence",
    page_icon="🚆",
    layout="wide"
)

# -------------------- CUSTOM PROFESSIONAL THEME --------------------
def local_css():
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }

        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        [data-testid="stSidebar"] * { color: white !important; }

        .main-header {
            background-color: #003366; 
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #003366;
            color: white;
            font-weight: bold;
            border: none;
        }

        .stButton>button:hover {
            background-color: #00509d;
        }

        .result-card {
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            border-top: 5px solid #003366;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 20px;
        }

        .footer {
            text-align: center;
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 50px;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("Railway AI")
    st.markdown("---")
    st.subheader("Navigation")
    st.markdown("---")
    st.caption("v2.2.0 | Powered by Logistic Regression")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_models():
    try:
        with open("logistic_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_models()

# -------------------- TEXT PROCESSING --------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def rule_based_override(text):
    medical_keywords = ["heart attack", "cardiac arrest", "stroke", "severe injury", "heavy bleeding", "electric shock", "unconscious"]
    safety_keywords = ["weapon", "robbery", "assault", "harassment", "physical fight", "bomb threat"]

    for phrase in medical_keywords:
        if phrase in text:
            return "Medical Emergency", "High Priority 🚨"

    for phrase in safety_keywords:
        if phrase in text:
            return "Safety & Security", "High Priority 🚨"

    return None, None

def detect_urgency(text):
    urgent_keywords = ["delay", "late", "urgent", "broken", "water", "stink"]
    for word in urgent_keywords:
        if word in text:
            return "Medium Priority ⚠"
    return "Normal Priority"

# -------------------- HEADER --------------------
st.markdown("""
<div class="main-header">
    <h1>🚆 RailWay AI Prediction Portal</h1>
    <p>Official Intelligence System for Automated Complaint Categorization</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,3,1])

with col2:
    st.markdown("### 📝 Submit Your Complaint")
    user_input = st.text_area(
        "Provide full details of the incident (Train No, Coach, PNR):",
        height=150,
        placeholder="Describe the issue here..."
    )

    predict_btn = st.button("Analyze Complaint")

    if predict_btn:
        if user_input.strip() == "":
            st.warning("Input required.")
        elif model is None:
            st.error("Model files missing.")
        else:
            cleaned = clean_text(user_input)
            final = remove_stopwords(cleaned)
            vectorized = vectorizer.transform([final])

            # PREDICTION
            ml_prediction = model.predict(vectorized)[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vectorized)[0]
                confidence = np.max(probs) * 100
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3 = [(model.classes_[i], probs[i]*100) for i in top3_idx]
            else:
                confidence = 0
                top3 = []

            override_category, override_priority = rule_based_override(cleaned)

            if override_category:
                prediction = override_category
                priority = override_priority
                is_override = True
            else:
                prediction = ml_prediction
                priority = detect_urgency(user_input.lower())
                is_override = False

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📋 Classification Report")

            m1, m2 = st.columns(2)
            m1.metric("Predicted Category", str(prediction))
            m2.metric("ML Confidence Score", f"{confidence:.1f}%")

            # Confidence Level
            if confidence >= 80:
                st.success("High Model Certainty")
            elif confidence >= 60:
                st.warning("Moderate Model Certainty")
            else:
                st.error("Low Model Certainty - Manual Review Suggested")

            st.markdown("---")

            # Top 3 Probabilities
            st.markdown("### 🔎 Top 3 Probable Categories")
            for cat, prob in top3:
                st.write(f"{cat} — {prob:.2f}%")

            st.markdown("---")

            # Priority
            if "High" in priority:
                st.error(f"**Action Level:** {priority}")
            elif "Medium" in priority:
                st.warning(f"**Action Level:** {priority}")
            else:
                st.success(f"**Action Level:** {priority}")

            if is_override:
                st.info("💡 Categorized using Emergency Override System.")

            # Important Keywords (Logistic Regression Coefficients)
            try:
                feature_names = vectorizer.get_feature_names_out()
                class_index = list(model.classes_).index(prediction)
                coefs = model.coef_[class_index]
                top_features_idx = np.argsort(coefs)[-5:]
                important_words = [feature_names[i] for i in top_features_idx]

                st.markdown("### 🧠 Important Keywords Influencing Prediction")
                st.write(", ".join(important_words))
            except:
                pass

            st.markdown("---")

            # Model Performance Section
            st.markdown("### 📊 Model Performance (Test Dataset)")
            perf1, perf2, perf3, perf4 = st.columns(4)

            perf1.metric("Accuracy", "75%")
            perf2.metric("Precision", "0.74")
            perf3.metric("Recall", "0.72")
            perf4.metric("F1 Score", "0.73")

            st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
<hr>
© 2026 Ministry of Railways - AI Intelligence Division<br>
Digital India Initiative
</div>
""", unsafe_allow_html=True)
