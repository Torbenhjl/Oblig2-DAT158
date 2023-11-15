import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('rf_model.joblib')

# Define a placeholder value for 'collection_id' if it's not part of user input
# This could be the median or mode of 'collection_id' from the training dataset, or another appropriate placeholder value
# Replace with your actual placeholder value for 'collection_id'
placeholder_collection_id = 0

# Function to process input and make predictions


def predict(input_data):
    # Add 'collection_id' to the DataFrame
    input_data['collection_id'] = placeholder_collection_id

    # Make prediction
    prediction = model.predict(input_data)

    return prediction


# Title of the web app
st.title('Revenue Prediction Model')

# Get user input for the features
budget = st.number_input('Enter the budget')
popularity = st.number_input('Enter the popularity')
runtime = st.number_input('Enter the runtime')

# When 'Predict' is clicked, make the prediction and display it
if st.button('Predict'):
    # Create a DataFrame with the same features as the model was trained on
    input_data = pd.DataFrame([[budget, popularity, runtime, placeholder_collection_id]],
                              columns=['budget', 'popularity', 'runtime', 'collection_id'])

    prediction = predict(input_data)
    st.success(f'The predicted revenue is ${prediction[0]:,.2f}')
