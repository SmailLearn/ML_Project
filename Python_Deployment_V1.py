import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
import requests
import io

#df_p_h = pd.read_csv('https://raw.githubusercontent.com/SmailLearn/ML_Project/main/pp.csv', header = None)
df_p_h = pd.read_csv(r"C:\Users\1\Desktop\MSDE5\Machine_Learning_MODULE_6\Projet\df_town.csv", header = None)
df_p_h.columns = ['Town/City' ,'Town/City_mean']
df_p_h['Town/City_mean'] = pd.to_numeric(df_p_h['Town/City_mean'], errors='coerce')



dictionary = dict(zip(df_p_h['Town/City'], df_p_h['Town/City_mean']))
city_names = list(dictionary.keys())


# Load the trained model
file = r"C:\Users\1\Desktop\MSDE5\Machine_Learning_MODULE_6\Projet\dt_model_1.pkl"

#file = 'https://raw.githubusercontent.com/SmailLearn/ML_Project/main/xgboost_model_1.pkl'
# Load the trained model
file_path = r"C:\Users\1\Desktop\MSDE5\Machine_Learning_MODULE_6\Projet\xgboost_model_1.pkl"
with open(file_path, 'rb') as file:
    model = pickle.load(file)


# Streamlit App
st.title("Price Housing Prediction App")


feature5 = st.slider("FreeHold(Duration_F): 1-Oui, 0-Non ", min_value=0, max_value=1, value=0)
feature6 = st.slider("Detached(Prop_D): 1-Oui, 0-Non", min_value=0, max_value=1, value=0)
feature8 = st.slider("maison Semi_Detache(Prop_S): 1-oui, 0- Non", min_value=0, max_value=1, value=0)
feature9 = st.slider("Prop_T()", min_value=0, max_value=1, value=0)
feature10 = st.slider("Sans paiement supplementaires(PPD Prop_A): 1-sans frais supplementaire, 0-Avec frais supplementaire", min_value=0, max_value=1, value=0)
feature11 = st.slider("New_Home(Old/New_N): 0-Old, 1-New", min_value=0, max_value=1, value=0)
feature12 = st.slider("Type de maison (Price_Category_Town/City): 1-Low, 2-Mid, 3-High, 4-Very High", min_value=1, max_value=4, value=1)

feature2 = st.selectbox("Town/City", city_names, key="city_selector")


# Make prediction
input_data = pd.DataFrame({
    'Duration_F' : [feature5],
    'Prop_D' : [feature6],
   'Prop_S' : [feature8],
    'Prop_T' : [feature9],
    'PPD Prop_A' : [feature10],
    'Old/New_N' : [feature11],
    'Price_Category_Town/City' : [feature12],
    'Town/City_mean' : [dictionary.get(str(feature2), None)]
})


prediction = model.predict(input_data)
prediction = np.exp(prediction)

# Display the prediction
st.subheader("Prediction:")
st.write(f"The predicted price is: {prediction[0]}")