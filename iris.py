# app.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

def main():
    st.title("Iris Flower Classification")
    st.sidebar.header("Input Parameters")
    
    # Load the saved model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Input fields for sepal and petal measurements
    s_length = st.sidebar.slider('Sepal Length', 0.0, 10.0, 5.0, step=0.1)
    s_width = st.sidebar.slider('Sepal Width', 0.0, 10.0, 5.0, step=0.1)
    p_length = st.sidebar.slider('Petal Length', 0.0, 10.0, 5.0, step=0.1)
    p_width = st.sidebar.slider('Petal Width', 0.0, 10.0, 5.0, step=0.1)

    # Predict button
    if st.sidebar.button('Predict'):
        # Make predictions
        input_features = [[s_length, s_width, p_length, p_width]]
        prediction = model.predict(input_features)[0]
        
        # Display prediction
        st.subheader(f"Predicted Iris Species: {prediction}")

if __name__ == "__main__":
    main()
