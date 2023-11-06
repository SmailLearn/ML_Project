import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
from sklearn.preprocessing import RobustScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

df_p_h = pd.read_csv(r"C:\Users\1\Desktop\MSDE5\Machine_Learning_module_6\Projet\pp-2021.csv", header = None)
df_p_h.columns = ['Transaction unique identifier','Price','Date of Transfer','Postcode', 'Property Type', 'Old/New', 'Duration', 'PAON', 'SAON', 'Street', 'Locality', 'Town/City', 'District','County', 'PPD Category Type', 'Record Status - monthly file only']

from scipy import stats

# Exemple avec une colonne 'Price'
z_scores = stats.zscore(df_p_h['Price'])
outliers = (z_scores > 3) | (z_scores < -3)
outlier_values = df_p_h['Price'][outliers]

df_no_out = df_p_h[~outliers]

df_encoded = df_no_out

#Grouping by Town/City
df=df_encoded.groupby("Town/City")['Price'].mean().reset_index()
Q1 = df['Price'].quantile(0.25)
Q2 = df['Price'].quantile(0.50)
Q3 = df['Price'].quantile(0.75)
df_encoded.loc[df_encoded['Price'] <= Q1, 'Price_Category_Town/City'] = 'Low'
df_encoded.loc[(df_encoded['Price'] <= Q2) & (df_encoded['Price'] > Q1), 'Price_Category_Town/City'] = 'Medium'
df_encoded.loc[(df_encoded['Price'] <= Q3) & (df_encoded['Price'] > Q2), 'Price_Category_Town/City'] = 'High'
df_encoded.loc[df_encoded['Price'] > Q3, 'Price_Category_Town/City'] = 'Very High'
labels = {
    'Low' : 1,
    'Medium' : 2,
    'High' : 3,
    'Very High' : 4
}
df_encoded["Price_Category_Town/City"] = df_encoded["Price_Category_Town/City"].map(labels)

df_encoded['Town/City_mean'] = round(df_encoded.groupby('Town/City')['Price'].transform('mean'), 2)

dictionary = dict(zip(df_encoded['Town/City'], df_encoded['Town/City_mean']))
city_names = list(dictionary.keys())


# Load the trained model
file = r"C:\Users\1\Desktop\MSDE5\Machine_Learning_MODULE_6\Projet\xgboost_model_1.pkl"
model = pickle.load(open(file, "rb"))

# Streamlit App
st.title("Price Housing Prediction App")

# Input for categorical variables
#category_options = ['Old/New', 'Town/City', 'PPD Category Type', 'Property Type', 'Duration']  # Replace with your actual categories
#selected_category = st.selectbox("Select Category:", category_options)

feature2 = st.selectbox("Town/City", city_names, 0, key="city_selector")
feature5 = st.slider("FreeHold(Duration_F): 1-Oui, 0-Non ", min_value=0, max_value=1, value=0)
feature6 = st.slider("Detached(Prop_D): 1-Oui, 0-Non", min_value=0, max_value=1, value=0)
feature7 = st.slider("Appartement(Prop_F): 1-Oui, 2-Non", min_value=0, max_value=1, value=0)
feature8 = st.slider("maison Semi_Detache(Prop_S): 1-oui, 0- Non", min_value=0, max_value=1, value=0)
feature9 = st.slider("Prop_T()", min_value=0, max_value=1, value=0)
feature10 = st.slider("Sans paiement supplementaires(PPD Prop_A): 1-sans frais supplementaire, 0-Avec frais supplementaire", min_value=0, max_value=1, value=0)
feature11 = st.slider("New_Home(Old/New_N): 0-Old, 1-New", min_value=0, max_value=1, value=0)
feature12 = st.slider("Type de maison (Price_Category_Town/City): 1-Low, 2-Mid, 3-High, 4-Very High", min_value=1, max_value=4, value=1)

feature2 = dictionary[feature2]

# Make prediction
input_data = pd.DataFrame({
    'Duration_F' : [feature5],
    'Prop_D' : [feature6],
    'Prop_F' : [feature7],
    'Prop_S' : [feature8],
    'Prop_T' : [feature9],
    'PPD Prop_A' : [feature10],
    'Old/New_N' : [feature11],
    'Price_Category_Town/City' : [feature12],
    'Town/City_mean' : [feature2]
})

prediction = model.predict(input_data)
prediction = np.exp(prediction)

#scaler = RobustScaler()
#prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

# Display the prediction
st.subheader("Prediction:")
st.write(f"The predicted price is: {prediction[0]}")