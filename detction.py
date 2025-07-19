import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import classification_report

# ---------------------- Header and Layout ----------------------
st.set_page_config(page_title="Internet Banking Fraud Detection", layout="wide")

st.markdown("<h1 style='text-align: center; color: darkblue;'>Bharat Institute of Engineering and Technology</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: green;'>Mini Project - Fraud Detection in Internet Banking Using Machine Learning</h2>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------- Sidebar Details ----------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/13/Bharat_Institute_of_Engineering_and_Technology_logo.png", use_column_width=True)
st.sidebar.title("Team Details")
st.sidebar.write("*Batch:* 12")
st.sidebar.write("*Guide:* Mr./Ms. NAzeen Fathima")
st.sidebar.write("*Team Members:*")
st.sidebar.write("- VVS Siva Kumara Sarma")
st.sidebar.write("- o. rajesh")
st.sidebar.write("- k. udayram")
st.sidebar.write("- y. d sai krishna")
st.sidebar.write("- m . shiva prasad")
st.sidebar.write("- p . nitish")
st.sidebar.markdown("---")

st.sidebar.markdown("---")
model_option = st.sidebar.radio("🔍 Select Model for Evaluation", ["Logistic Regression", "KNN", "Random Forest"])
test_size = st.sidebar.slider("Test Size", 0.2, 0.5, 0.33)

# ---------------------- Data Processing ----------------------
df = pd.read_csv('fraud_dataset_example_2.csv')
l = LabelEncoder()
df['type_code'] = l.fit_transform(df['type'])
X = df.drop(['isFlaggedFraud', 'isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = df['isFraud']

# ---------------------- Mutual Info Plot ----------------------
def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    return pd.Series(mi_scores, name="MI Scores", index=X.columns).sort_values(ascending=False)

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.figure(figsize=(8, 10))
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    st.pyplot(plt)

mi_scores = make_mi_scores(X, y)

# ---------------------- Model Training ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)

lr = LogisticRegression().fit(X_train, y_train)
knc = KNeighborsClassifier(n_neighbors=3).fit(X, y)
rf = RandomForestClassifier().fit(X_train, y_train)

lr_pred = lr.predict(X_test)
knc_pred = knc.predict(X_test)
rf_pred = rf.predict(X_test)

pickle.dump(knc, open("model.pkl", "wb"))

# ---------------------- Main Layout ----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Fraud Count:", df['isFraud'].value_counts())

with col2:
    st.subheader("📈 Feature Importance")
    plot_mi_scores(mi_scores)

st.markdown("---")

# ---------------------- Model Results ----------------------
st.subheader(f"🔎 {model_option} - Evaluation Report")

if model_option == "Logistic Regression":
    st.text(classification_report(y_test, lr_pred))
    st.write("Accuracy:", metrics.accuracy_score(y_test, lr_pred))
elif model_option == "KNN":
    st.text(classification_report(y_test, knc_pred))
    st.write("Accuracy:", metrics.accuracy_score(y_test, knc_pred))
elif model_option == "Random Forest":
    st.text(classification_report(y_test, rf_pred))
    st.write("Accuracy:", metrics.accuracy_score(y_test, rf_pred))
    st.write("Train Accuracy:", rf.score(X_train, y_train))

st.markdown("---")

# ---------------------- Predict Section ----------------------
st.subheader("🧪 Live Fraud Prediction (KNN Model)")

input_data = {}
for feature in X.columns:
    val = st.number_input(f"{feature}", value=float(df[feature].mean()))
    input_data[feature] = val

if st.button("Predict Fraud"):
    model = pickle.load(open("model.pkl", "rb"))
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)
    if pred[0] == 1:
        st.error("⚠ Fraud Detected!")
    else:
        st.success("✅ No Fraud Detected.")