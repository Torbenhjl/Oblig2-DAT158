import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('rf_model.joblib')

# Function to process input and make predictions


def predict(input_data):
    # Convert input data to DataFrame or perform any necessary preprocessing
    # input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_data)

    return prediction


# Title of the web app
st.title('Revenue Prediction Model')

# Get user input
budget = st.number_input('Enter the budget')
popularity = st.number_input('Enter the popularity')
runtime = st.number_input('Enter the runtime')

# When 'Predict' is clicked, make the prediction and display it
if st.button('Predict'):
    input_data = pd.DataFrame([[budget, popularity, runtime]], columns=[
                              'budget', 'popularity', 'runtime'])
    prediction = predict(input_data)
    st.success(f'The predicted revenue is ${prediction[0]:,.2f}')
