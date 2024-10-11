import streamlit as st
import pandas as pd
import joblib

# Load the trained Naive Bayes model and vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define a function to make predictions
def predict_language(text):
    # Transform the input text using the vectorizer
    input_data_vectorized = vectorizer.transform([text])

    # Make a prediction using the loaded model
    prediction = model.predict(input_data_vectorized)
    return prediction[0]

# Streamlit app layout
st.title("Language Prediction App")
st.write("Predict whether the input text is in Welsh or English.")

# Input field for user text
user_input = st.text_area("Enter text:", "")

# Predict button
if st.button("Predict Language"):
    if user_input:
        prediction = predict_language(user_input)
        # Display the actual predicted label directly
        st.success(f"The predicted language is: {'Welsh' if prediction == 'cy' else 'English'}")
    else:
        st.warning("Please enter some text to predict.")

