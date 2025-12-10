import joblib
from flask import Flask, request, jsonify
import numpy as np

# Cargar el modelo entrenado
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: 'model.pkl' no encontrado. Por favor, asegúrate de haber ejecutado el script de entrenamiento.")
    model = None

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado. Por favor, entrene el modelo primero.'}), 500
    try:
        # Obtener los datos de la petición en formato JSON
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        # Realizar la predicción
        prediction = model.predict(features)
        #Devolver la predicción en formato JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
