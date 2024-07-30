import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import requests
import os

# Set ggplot style
plt.style.use('ggplot')

# Function to download the model
def download_model(url, model_name):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(model_name, 'wb') as file:
            file.write(response.content)
        st.write(f"Model {model_name} downloaded successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {model_name}: {e}")

# Function to load the model
def load_model(model_name):
    try:
        return tf.keras.models.load_model(model_name)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

# Function to preprocess the data from the uploaded file
def preprocess_data(file):
    try:
        df = pd.read_excel(file)
        st.write("File uploaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# URLs for the models
cnn_model_url = 'https://github.com/patrickau-cloud/scu_st/raw/91415962c62be79a5ad893eb5e2e6df7e050c71f/cnn_rainfall_watercourse_model.h5'
lstm_model_url = 'https://github.com/patrickau-cloud/scu_st/raw/91415962c62be79a5ad893eb5e2e6df7e050c71f/lstm_rainfall_watercourse_model.h5'

# Local model filenames
cnn_model_file = 'cnn_rainfall_watercourse_model.h5'
lstm_model_file = 'lstm_rainfall_watercourse_model.h5'

# Download models if they don't exist locally
if not os.path.exists(cnn_model_file):
    download_model(cnn_model_url, cnn_model_file)

if not os.path.exists(lstm_model_file):
    download_model(lstm_model_url, lstm_model_file)

# Streamlit app
st.title('Rainfall to Watercourse Level Prediction')

# Dropdown for model selection
model_name = st.selectbox('Choose a model', (cnn_model_file, lstm_model_file))

# File uploader for the Excel file
uploaded_file = st.file_uploader('Upload your rainfall and watercourse data (xlsx format)', type='xlsx')

if uploaded_file is not None:
    # Load the model
    model = load_model(model_name)
    
    if model:
        # Preprocess the data
        data = preprocess_data(uploaded_file)
        
        if data is not None:
            # Extract necessary columns
            try:
                dates = data['MonthDay']
                rainfall = data['SumRainfall']
                actual_levels = data['DeltaWatercourseLevel']
            except KeyError as e:
                st.error(f"Missing expected column in data: {e}")
                st.stop()
            
            # Prepare input data for the model
            input_data = rainfall.values.reshape(-1, 1)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Predict using the model and measure the running time
            start_time = time.time()
            try:
                predicted_levels = model.predict(input_data).flatten()
            except Exception as e:
                st.error(f"Error making predictions: {e}")
                st.stop()
            
            running_time = time.time() - start_time
            
            # Add the predicted levels to the data frame
            data['PredictedDelta'] = predicted_levels
            
            # Plot the results
            fig, ax1 = plt.subplots()
            
            # Plot the rainfall as bars
            ax1.bar(dates, rainfall, color='b', alpha=0.6, label='Rainfall (mm)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Rainfall (mm)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_ylim(0, 240)  # Set the y-axis limit for rainfall
            
            # Create a second y-axis for the predicted delta watercourse level
            ax2 = ax1.twinx()
            ax2.plot(dates, predicted_levels, color='r', label='Predicted Delta Watercourse Level')
            ax2.plot(dates, actual_levels, color='g', linestyle='--', label='Actual Delta Watercourse Level')
            ax2.set_ylabel('Delta Watercourse Level (m)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(-0.5, 1.9)  # Set the y-axis limit for watercourse level
            
            # Add legends
            fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
            
            # Rotate x-axis labels for better readability
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Title and show plot
            plt.title('Rainfall and Predicted Delta Watercourse Level Nerang River at Glenhurst')
            
            st.pyplot(fig)
            
            # Display running time
            st.write(f"Running time: {running_time:.4f} seconds")
            
            # Display data
            st.write("### Data")
            st.write(data)
