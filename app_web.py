import streamlit as st
import requests
import json
import numpy as np
import os

# Titulo de la aplicación
st.title('API de Predicción de Modelo Iris')

# Descripción de la aplicación
st.write('Ingresa las características de la flor de iris para obtener una predicción de su especie.')

# Campos de entrada para las características del iris
sepal_length = st.slider('Longitud del sépalo (cm)', 0.0, 10.0, 5.0)
sepal_width = st.slider('Ancho del sépalo (cm)', 0.0, 10.0, 3.0)
petal_length = st.slider('Longitud del pétalo (cm)', 0.0, 10.0, 4.0)
petal_width = st.slider('Ancho del pétalo (cm)', 0.0, 10.0, 1.0)

# Botón para hacer la predicción
if st.button('Obtener Predicción'):
    # Crear los datos en formato JSON
    features = [sepal_length, sepal_width, petal_length, petal_width]
    payload = {'features': features}

    # URL de la API de predicción (cambia esta URL si la API no está en esa url)
    api_url = os.environ.get('API_URL', 'http://localhost:5000/predict')

    try:
        # Enviar la petición a la API
        response = requests.post(api_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

        # Verificar si la petición fue exitosa
        if response.status_code == 200:
            prediction_result = response.json().get('prediction')

            # Mapear el resultado numérico a una especie
            species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            predicted_species = species_map.get(prediction_result, 'Desconocida')
            
            st.success(f"La predicción es: **{predicted_species}**")

        else:
            st.error(f"Error en la petición: {response.status_code} -{response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API. Asegúrate de que está en ejecución. Error: {e}")