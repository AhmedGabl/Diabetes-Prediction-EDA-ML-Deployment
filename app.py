import streamlit as st
import pandas as pd
import pickle

# Load the model and scaler
@st.cache_resource
def load_model():
    with open("artifacts/model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("artifacts/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, scaler

rf_model, scaler = load_model()

# Define the input fields for the app
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict the likelihood of diabetes:")

# Create a form for user input
with st.form("diabetes_form"):
    st.header("Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120, help="Plasma glucose concentration")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, help="Body mass index (weight in kg/(height in m)^2)")
    
    with col2:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, help="Diabetes pedigree function")
        age = st.number_input("Age", min_value=0, max_value=120, value=25, help="Age in years")
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# Create a dataframe for the input data
input_data = pd.DataFrame({
    'Glucose': [glucose],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make a prediction
if submit_button:
    try:
        prediction = rf_model.predict(input_data_scaled)
        # Display the prediction
        if prediction[0] == 1:
            st.markdown(
                '<div style="background-color: blue; padding: 10px; border-radius: 5px;">'
                'The model predicts that the person is likely to have diabetes.'
                '</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background-color: green; padding: 10px; border-radius: 5px;">'
                'The model predicts that the person is not likely to have diabetes.'
                '</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add some additional information
st.sidebar.header("About")
st.sidebar.write("""
This app uses a machine learning model to predict the likelihood of diabetes based on user input.
The model was trained on the Pima Indians Diabetes Database.
""")
