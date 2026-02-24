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
        /* Main background */
        .main { background-color: #f8f9fa; }
        
        /* Sidebar styling to match Dashboard */
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        [data-testid="stSidebar"] * { color: white !important; }

        /* Header Styling */
        .main-header {
            background-color: #003366; 
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Button Styling */
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
            color: white;
        }

        /* Results Card */
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

# -------------------- SIDEBAR (New Professional Column) --------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/41/IR_Logo.svg/1200px-IR_Logo.svg.png", width=80)
    st.title("Railway AI")
    st.markdown("---")
    st.subheader("Navigation")
    # Using simple buttons for navigation as placeholders
    st.markdown("---")
    st.caption("v2.1.0 | Powered by ML")

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

# -------------------- LOGIC FUNCTIONS (Unchanged) --------------------
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
        if phrase in text: return "Medical Emergency", "High Priority 🚨"
    for phrase in safety_keywords:
        if phrase in text: return "Safety & Security", "High Priority 🚨"
    return None, None

def detect_urgency(text):
    urgent_keywords = ["delay", "late", "urgent", "broken", "water", "stink"]
    for word in urgent_keywords:
        if word in text: return "Medium Priority ⚠"
    return "Normal Priority"

# -------------------- MAIN UI LAYOUT --------------------
st.markdown("""
    <div class="main-header">
        <h1>🚆 RailWay AI Prediction Portal</h1>
        <p>Official Intelligence System for Automated Complaint Categorization</p>
    </div>
""", unsafe_allow_html=True)

# Centered Input Area
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("### 📝 Submit Your Complaint")
    user_input = st.text_area("Provide full details of the incident (Train No, Coach, PNR):", 
                              height=150, 
                              placeholder="Describe the issue here...")
    
    predict_btn = st.button("Analyze Complaint")

    if predict_btn:
        if user_input.strip() == "":
            st.warning("Input required.")
        elif model is None:
            st.error("Model files missing.")
        else:
            # PROCESS
            cleaned = clean_text(user_input)
            final = remove_stopwords(cleaned)
            vectorized = vectorizer.transform([final])

            # PREDICT
            ml_prediction = model.predict(vectorized)[0]
            confidence = np.max(model.predict_proba(vectorized)) * 100 if hasattr(model, "predict_proba") else 0
            
            override_category, override_priority = rule_based_override(cleaned)

            if override_category:
                prediction, priority, is_override = override_category, override_priority, True
            else:
                prediction, priority, is_override = ml_prediction, detect_urgency(user_input.lower()), False

            # RESULTS CARD
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📋 Classification Report")
            
            # Metric Layout
            m1, m2 = st.columns(2)
            m1.metric("Predicted Category", str(prediction))
            m2.metric("ML Confidence Score", f"{confidence:.1f}%")

            st.markdown("---")
            
            # Priority Indicator
            if "High" in priority:
                st.error(f"**Action Level:** {priority}")
            elif "Medium" in priority:
                st.warning(f"**Action Level:** {priority}")
            else:
                st.success(f"**Action Level:** {priority}")
            
            if is_override:
                st.info("💡 **Safety Note:** This categorization was generated by the Emergency Override System.")

            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <hr>
        © 2026 Ministry of Railways - AI Intelligence Division<br>
        Digital India Initiative
    </div>
""", unsafe_allow_html=True)
