import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("siva.h5")  # Replace with the path to your model file

# Define the Streamlit app
st.title("Yield Prediction App")

# Upload a CSV file for prediction
uploaded_file = st.file_uploader("Upload a CSV file with the following columns: id, clonesize, honeybee, bumbles, andrena, osmia, MaxOfUpperTRange, MinOfUpperTRange, AverageOfUpperTRange, MaxOfLowerTRange, MinOfLowerTRange, AverageOfLowerTRange, RainingDays, AverageRainingDays, fruitset, fruitmass, seeds", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    input_data = pd.read_csv(uploaded_file)

    # Ensure that the input data has the correct columns (16 features)
    expected_columns = [
        "id", "clonesize", "honeybee", "bumbles", "andrena", "osmia",
        "MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
        "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange",
        "RainingDays", "AverageRainingDays", "fruitset", "fruitmass", "seeds"
    ]

    if set(expected_columns).issubset(input_data.columns):
        # Exclude the 'id' column before making predictions
        input_data = input_data.drop(columns=["id"])

        # Make predictions
        predictions = model.predict(input_data)

        # Display the predictions
        st.write("Predictions:")
        st.write(predictions)
    else:
        st.error("Input data is missing some of the required columns.")

# Display model architecture
st.subheader("Model Architecture")
# Describe your model architecture here

# Display training information (you can add more details)
st.subheader("Training Information")
# Describe your training details here

# Optionally, display model summary
show_summary = st.checkbox("Show Model Summary")
if show_summary:
    st.subheader("Model Summary")
    st.text(model.summary())
