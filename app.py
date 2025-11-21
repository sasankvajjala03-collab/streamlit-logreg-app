import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# LOAD TRAINED MODEL & ENCODERS
# -----------------------------
model = pickle.load(open("titanic_model.pkl", "rb"))
le_sex = pickle.load(open("le_sex.pkl", "rb"))
le_embarked = pickle.load(open("le_embarked.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict if they survived the Titanic disaster.")

# -----------------------------
# USER INPUT FIELDS
# -----------------------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# -----------------------------
# ENCODE USER INPUTS
# -----------------------------
sex_value = le_sex.transform([sex])[0]
embarked_value = le_embarked.transform([embarked])[0]

# Create DataFrame with exact feature order used during training
input_df = pd.DataFrame({
    "PassengerId": [0],     # placeholder (model ignores it)
    "Pclass": [pclass],
    "Sex": [sex_value],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_value]
})

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"🎉 The passenger **SURVIVED** (Probability: {prob:.2f})")
    else:
        st.error(f"❌ The passenger **DID NOT SURVIVE** (Probability: {prob:.2f})")
