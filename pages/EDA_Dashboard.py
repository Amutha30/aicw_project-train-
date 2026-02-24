import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix

# -------------------- UI/UX HEADER & CONFIG --------------------
st.set_page_config(page_title="RailMadad AI | EDA Dashboard", layout="wide")

# Custom CSS for the "Government Blue" Theme
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        [data-testid="stSidebar"] { background-color: #002244; }
        [data-testid="stSidebar"] * { color: white !important; }
        .main-header {
            background-color: #003366; 
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
        }
        .chart-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-top: 5px solid #003366;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR (Left Column) --------------------
with st.sidebar:
    st.title("🚆 RailMadad AI")
    st.subheader("Admin Dashboard")
    st.divider()
    st.page_link("app.py", label="Main Complaint Portal", icon="🏠")
    st.page_link("pages/EDA_Dashboard.py", label="EDA Insights", icon="📊")
    st.divider()
    st.info("System Status: Active")

# -------------------- HEADER --------------------
st.markdown("""
    <div class="main-header">
        <h1>📊 Railway Complaint EDA Dashboard</h1>
        <p>Comprehensive exploratory data analysis and model behavior insights.</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- YOUR ORIGINAL LOGIC --------------------

# Load Data
# Note: Ensure this path is correct for your local machine
df = pd.read_csv("train.csv", encoding="latin-1")

# Load Model & Vectorizer
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# --- DATASET OVERVIEW ---
st.subheader("📌 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Complaints", df.shape[0])
col2.metric("Total Categories", df['Recommended complaint category'].nunique())
col3.metric("Total Features", len(vectorizer.get_feature_names_out()))

st.divider()

# --- CATEGORY DISTRIBUTION ---
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.subheader("📊 Complaint Category Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 4))
df['Recommended complaint category'].value_counts().plot(kind='bar', ax=ax1, color='#003366')
plt.xticks(rotation=45)
st.pyplot(fig1)

st.markdown("""
**Insight:** Ticketing & Reservation and Cleanliness categories dominate the dataset, 
while Medical Emergency has fewer samples, leading to moderate class imbalance.
""")
st.markdown('</div>', unsafe_allow_html=True)

# --- TEXT LENGTH ANALYSIS ---
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.subheader("📝 Complaint Length Analysis")
df['word_count'] = df['final_text'].apply(lambda x: len(str(x).split()))
fig2, ax2 = plt.subplots(figsize=(10, 4))
df.groupby('Recommended complaint category')['word_count'].mean().sort_values().plot(kind='bar', ax=ax2, color='#00509d')
plt.xticks(rotation=45)
st.pyplot(fig2)

st.markdown("""
**Insight:** Medical Emergency complaints are shorter on average (~8 words), 
which reduces contextual richness and makes classification more challenging.
""")
st.markdown('</div>', unsafe_allow_html=True)

# --- CONFUSION MATRIX ---
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.subheader("🔎 Model Confusion Matrix")
X = vectorizer.transform(df['final_text'])
y_true = df['Recommended complaint category']
y_pred = model.predict(X)
cm = confusion_matrix(y_true, y_pred, labels=model.classes_)

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax3)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
st.pyplot(fig3)

st.markdown("""
**Key Observations:**
- Strong prediction performance for Ticketing and Cleanliness.
- Some confusion between Delay and Cleanliness due to overlapping keywords.
- Irrelevant class shows weaker semantic distinction.
""")
st.markdown('</div>', unsafe_allow_html=True)
