import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Sample Pokémon stats data (replace with your full dataset)
data = {
    'HP': [45, 60, 80, 39],
    'Attack': [49, 62, 82, 52],
    'Defense': [49, 63, 83, 43],
    'Sp. Atk': [65, 80, 100, 60],
    'Sp. Def': [65, 80, 100, 50],
    'Speed': [45, 60, 80, 65],
    'Type': ['Grass', 'Fire', 'Water', 'Bug']
}

df = pd.DataFrame(data)

# Prepare features and labels
X = df.drop('Type', axis=1)
y = df['Type']

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

st.title("Pokémon Type Classifier")

# Input fields for base stats
hp = st.number_input('HP', min_value=1, max_value=255, value=50)
attack = st.number_input('Attack', min_value=1, max_value=255, value=50)
defense = st.number_input('Defense', min_value=1, max_value=255, value=50)
sp_atk = st.number_input('Sp. Atk', min_value=1, max_value=255, value=50)
sp_def = st.number_input('Sp. Def', min_value=1, max_value=255, value=50)
speed = st.number_input('Speed', min_value=1, max_value=255, value=50)

# Predict button
if st.button('Predict Type'):
    input_stats = np.array([[hp, attack, defense, sp_atk, sp_def, speed]])
    prediction = model.predict(input_stats)
    st.success(f"Predicted Pokémon type: {prediction[0]}")
