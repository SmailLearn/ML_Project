import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
import requests

# Error Handling for Data Loading
try:
    with st.spinner("Loading data..."):
        df_p_h = pd.read_csv('https://raw.githubusercontent.com/SmailLearn/ML_Project/main/df_town.csv', header=None)
        df_p_h.columns = ['Town/City', 'Town/City_mean']
        df_p_h['Town/City_mean'] = pd.to_numeric(df_p_h['Town/City_mean'], errors='coerce')
    st.success("Data loaded successfully!")
except pd.errors.EmptyDataError:
    st.error("Error: Empty data. Please check the data source.")
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")

# making the df_p_h to a dictionary where the first column is the key and the second is the value
dictionary = dict(zip(df_p_h['Town/City'], df_p_h['Town/City_mean']))
city_names = list(dictionary.keys())
city_names_sorted = sorted(city_names)

# Error Handling for Model Loading
try:
    with st.spinner("Loading model..."):
        url = 'https://raw.githubusercontent.com/SmailLearn/ML_Project/main/xgboost_model_1.pkl'
        response = requests.get(url)
        response.raise_for_status()
        content = response.content
        model = pickle.loads(content)
    st.success("Model loaded successfully!")
except requests.exceptions.RequestException as req_err:
    st.error(f"Error in making a request: {req_err}")
except pickle.PickleError as pickle_err:
    st.error(f"Error loading the model from pickle: {pickle_err}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# Streamlit App
st.title("Price Housing Prediction App")

# user input
st.sidebar.title("User Inputs")
feature5 = st.sidebar.slider("FreeHold(Duration_F): 1-Oui, 0-Non", min_value=0, max_value=1, value=0)
feature6 = st.sidebar.slider("Detached(Prop_D): 1-Oui, 0-Non", min_value=0, max_value=1, value=0)
feature8 = st.sidebar.slider("maison Semi_Detache(Prop_S): 1-oui, 0- Non", min_value=0, max_value=1, value=0)
feature9 = st.sidebar.slider("Prop_T()", min_value=0, max_value=1, value=0)
feature10 = st.sidebar.slider("Sans paiement supplementaires(PPD Prop_A): 1-sans frais supplementaire, 0-Avec frais supplementaire", min_value=0, max_value=1, value=0)
feature11 = st.sidebar.slider("New_Home(Old/New_N): 0-Old, 1-New", min_value=0, max_value=1, value=0)
feature12 = st.sidebar.slider("Type de maison (Price_Category_Town/City): 1-Low, 2-Mid, 3-High, 4-Very High", min_value=1, max_value=4, value=1)
feature2 = st.sidebar.selectbox("Select Town/City", city_names_sorted, key="city_selector")

# Make prediction
try:
    input_data = pd.DataFrame({
        'Duration_F': [feature5],
        'Prop_D': [feature6],
        'Prop_S': [feature8],
        'Prop_T': [feature9],
        'PPD Prop_A': [feature10],
        'Old/New_N': [feature11],
        'Price_Category_Town/City': [feature12],
        'Town/City_mean': [dictionary.get(str(feature2), None)]
    })

    prediction = model.predict(input_data)
    prediction = np.exp(prediction)
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")

# Display the prediction
background_color = "#ADD8E6" 
st.title("Prediction Result")
st.subheader("The predicted price: ")
prediction_text = f"{prediction[0]:,.2f} £"  # Format the prediction as currency
border_style = "2px solid #008080"  # Border style with color code
st.markdown(f"<div style='text-align: center; padding: 20px; border: {border_style}; border-radius: 10px; background-color: #f0f8ff;'><h1 style='font-weight: bold;'>{prediction_text}</h1></div>", unsafe_allow_html=True)

#st.write(f"The predicted price is: {prediction[0]}")