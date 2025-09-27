import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load(r"e:\Study\Sprints Project\Heart_Disease_Project\Models\final_model_pipeline.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("Heart Disease Prediction App")
st.write("Fill in the form below to predict the risk of heart disease.")

# --- Input form ---
with st.form("heart_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age (years)")
        sex = st.selectbox("Sex", ["Select...", 0, 1], format_func=lambda x: "Male" if x == 1 else ("Female" if x == 0 else x))
        cp = st.selectbox("Chest pain type", ["Select...", 1, 2, 3, 4],
                          format_func=lambda x: {1: "Typical angina", 2: "Atypical angina",
                                                 3: "Non-anginal pain", 4: "Asymptomatic"}.get(x, x))
        trestbps = st.text_input("Resting blood pressure (mm Hg)")
        chol = st.text_input("Serum cholesterol (mg/dl)")
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["Select...", 0, 1],
                           format_func=lambda x: "True" if x == 1 else ("False" if x == 0 else x))
        restecg = st.selectbox("Resting ECG results", ["Select...", 0, 1, 2],
                               format_func=lambda x: {0: "Normal", 1: "ST-T abnormality",
                                                      2: "LVH by Estes’ criteria"}.get(x, x))

    with col2:
        thalach = st.text_input("Max heart rate achieved")
        exang = st.selectbox("Exercise induced angina", ["Select...", 0, 1],
                             format_func=lambda x: "Yes" if x == 1 else ("No" if x == 0 else x))
        oldpeak = st.text_input("ST depression (oldpeak)")
        slope = st.selectbox("Slope of peak exercise ST segment", ["Select...", 1, 2, 3],
                             format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get(x, x))
        ca = st.selectbox("Number of major vessels (0–3)", ["Select...", 0, 1, 2, 3])
        thal = st.selectbox("Thalassemia status", ["Select...", 3, 6, 7],
                            format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}.get(x, x))
        
        _, col_right = st.columns([3, 4])
        with col_right:
            submitted = st.form_submit_button("🔍 Predict")


    # _, col_right = st.columns([3, 1])
    # with col_right:
    #     submitted = st.form_submit_button("🔍 Predict")


# --- Prediction ---
if submitted:
    try:
        # Convert numeric fields from text to float/int
        input_data = pd.DataFrame([{
            "age": int(age),
            "sex": int(sex) if sex != "Select..." else None,
            "cp": int(cp) if cp != "Select..." else None,
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": int(fbs) if fbs != "Select..." else None,
            "restecg": int(restecg) if restecg != "Select..." else None,
            "thalach": float(thalach),
            "exang": int(exang) if exang != "Select..." else None,
            "oldpeak": float(oldpeak),
            "slope": int(slope) if slope != "Select..." else None,
            "ca": int(ca) if ca != "Select..." else None,
            "thal": int(thal) if thal != "Select..." else None
        }])

        # Run prediction
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High risk of Heart Disease (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low risk of Heart Disease (Probability: {prob:.2f})")

    except Exception as e:
        st.warning("⚠️ Please fill in all fields with valid numbers before predicting.")
