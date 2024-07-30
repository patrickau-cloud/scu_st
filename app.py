import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import requests
import os
import matplotlib.dates as mdates

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

# Button to run the chart generation
if st.button('Run Chart'):
    # Load the model
    model = load_model(model_name)
    
    if model:
        # Example dataset (you should replace this with your actual dataset)
        data = pd.DataFrame({
            'MonthDay': pd.date_range(start='2015-04-01', end='2015-05-31'),
            'SumRainfall': np.random.uniform(0, 200, size=61),
            'DeltaWatercourseLevel': np.random.uniform(-0.5, 1.5, size=61)
        })
        
        # Prepare the input data (example: SumRainfall level of 8)
        input_rainfall = np.array([[8]], dtype=np.float32)
        input_rainfall_reshaped = input_rainfall.reshape((input_rainfall.shape[0], 1, input_rainfall.shape[1]))

        # Make a prediction
        start_time = time.time()
        predicted = model.predict(input_rainfall_reshaped)
        running_time = time.time() - start_time

        st.write(f'Predicted Delta Watercourse Level: {predicted[0][0]}')

        # Filter data for a specific period
        start_date = '2015-04-25'
        end_date = '2015-05-07'
        mask = (data['MonthDay'] >= start_date) & (data['MonthDay'] <= end_date)
        filtered_data = data.loc[mask]

        # Prepare the input data for prediction
        X = filtered_data[['SumRainfall']].values
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))

        # Predict the watercourse delta
        predicted_deltas = model.predict(X_reshaped)

        # Add the predictions to the dataframe
        filtered_data['PredictedDelta'] = predicted_deltas

        # Plot the results
        fig, ax1 = plt.subplots()

        # Plot the rainfall as bars
        ax1.bar(filtered_data['MonthDay'], filtered_data['SumRainfall'], color='b', alpha=0.6, label='Rainfall (mm)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Rainfall (mm)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0, 240)  # Set the y-axis limit for rainfall
        ax1.grid(True)  # Enable grid for ax1

        # Set the x-axis major locator to show every 2 days
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Create a second y-axis for the predicted delta watercourse level
        ax2 = ax1.twinx()
        ax2.plot(filtered_data['MonthDay'], filtered_data['PredictedDelta'], color='r', label='Predicted Delta Watercourse Level')
        ax2.plot(filtered_data['MonthDay'], filtered_data['DeltaWatercourseLevel'], color='g', linestyle='--', label='Actual Delta Watercourse Level')
        ax2.set_ylabel('Delta Watercourse Level (m)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.grid(False)  # Disable grid for ax2

        # Add legends
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Title and show plot
        plt.title('Rainfall and Predicted Delta Watercourse Level (2015-04-25 to 2015-05-07)')
        plt.tight_layout()

        st.pyplot(fig)
        
        # Display running time
        st.write(f"Running time: {running_time:.4f} seconds")
