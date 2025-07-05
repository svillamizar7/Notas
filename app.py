import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('./naive_bayes_model.pkl')

# Define los mapeos inversos para decodificar los resultados
mapeo_resultado_inverso = {1: 'Felicitaciones', 0: 'No Aprueba'}
mapeo_horas_inverso = {1: 'Alta', 0: 'Baja'}
mapeo_asistencia_inverso = {1: 'Buena', 0: 'Mala'}

# Título de la aplicación
st.title("Predicción de Clase")

# Subtítulo con formato HTML para color rojo
st.markdown("<h2 style='text-align: center; color: red;'>Elaborado por SergioVilla</h2>", unsafe_allow_html=True)


st.write("Por favor, elija los valores para las variables de entrada:")

# Entradas para el usuario
horas_estudio_input = st.selectbox("Horas de Estudio:", list(mapeo_horas_inverso.values()))
asistencia_input = st.selectbox("Asistencia:", list(mapeo_asistencia_inverso.values()))

# Mapear las entradas del usuario a los valores numéricos utilizados en el entrenamiento
horas_estudio_encoded = [key for key, value in mapeo_horas_inverso.items() if value == horas_estudio_input][1]
asistencia_encoded = [key for key, value in mapeo_asistencia_inverso.items() if value == asistencia_input][1]


# Crear un DataFrame con la nueva observación (debe tener el mismo formato que X_train)
new_observation = pd.DataFrame([[horas_estudio_encoded, asistencia_encoded]], columns=['Horas de Estudio', 'Asistencia'])

# Realizar la predicción
prediction_encoded = model.predict(new_observation)

# Decodificar la predicción
prediction_result = mapeo_resultado_inverso[prediction_encoded[0]]

# Mostrar el resultado de la predicción
if prediction_result == 'Felicitaciones':
    st.success(f"Resultado: {prediction_result} 😊")
else:
    st.error(f"Resultado: {prediction_result} 😞")
