import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('/naive_bayes_model.pkl')

# Define los mapeos inversos para mostrar los resultados originales
mapeo_horas_inverso = {1: 'Alta', 0: 'Baja'}
mapeo_asistencia_inverso = {1: 'Buena', 0: 'Mala'}
mapeo_resultado_inverso = {1: 'Sí', 0: 'No'}

# Título y subtítulo de la aplicación
st.title("Predicción de Clase")
st.markdown("<h2 style='color: red;'>Elaborado por SergioVilla</h2>", unsafe_allow_html=True)

st.write("""
Este modelo predice si un estudiante aprobará o no basándose en las Horas de Estudio y la Asistencia.
""")

# Entradas del usuario
st.subheader("Seleccione los valores de entrada:")

horas_estudio_input = st.selectbox("Horas de Estudio", ['Alta', 'Baja'])
asistencia_input = st.selectbox("Asistencia", ['Buena', 'Mala'])

# Convertir las entradas del usuario a los valores numéricos usados por el modelo
horas_estudio_codificado = 1 if horas_estudio_input == 'Alta' else 0
asistencia_codificado = 1 if asistencia_input == 'Buena' else 0

# Crear un DataFrame con las entradas del usuario
input_data = pd.DataFrame({
    'Horas de Estudio': [horas_estudio_codificado],
    'Asistencia': [asistencia_codificado]
})

# Realizar la predicción
prediction_encoded = model.predict(input_data)
prediction_result = mapeo_resultado_inverso[prediction_encoded[0]]

# Mostrar el resultado de la predicción con iconos
st.subheader("Resultado de la Predicción:")

if prediction_result == 'Sí':
    st.write(f"El estudiante: **Felicitaciones** 😊")
else:
    st.write(f"El estudiante: **No Aprueba** 😞")

