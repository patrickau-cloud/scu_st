import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('flood_rainfall_model.h5')

# Function to normalize data
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Load and preprocess mock data
@st.cache
def load_data(file_path):
    mock_data = pd.read_csv(file_path)
    rainfall_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    mock_data[rainfall_columns] = mock_data[rainfall_columns].apply(normalize)
    mock_data['FLOODS'] = mock_data['FLOODS'].apply(lambda x: 1 if x == 'YES' else 0)
    return mock_data

# Make predictions
def predict_floods(data):
    rainfall_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    X = data[rainfall_columns].values.reshape((data.shape[0], 12, 1))
    predictions = model.predict(X)
    predictions = (predictions > 0.5).astype(int)
    return predictions

# Streamlit App
st.title('Flood Prediction using LSTM')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write(data)

    # Make predictions
    predictions = predict_floods(data)
    data['Predicted Floods'] = ['YES' if pred == 1 else 'NO' for pred in predictions]

    # Show actual vs predicted
    st.subheader('Actual vs Predicted Floods')
    st.write(data[['YEAR', 'FLOODS', 'Predicted Floods']])

    # Visualization
    st.subheader('Visualization')
    actual_vs_predicted = data[['YEAR', 'FLOODS', 'Predicted Floods']].melt(id_vars=['YEAR'], var_name='Type', value_name='Flood Status')
    st.line_chart(actual_vs_predicted, x='YEAR', y='Flood Status', color='Type')

st.write('Upload a CSV file to see the predictions.')
