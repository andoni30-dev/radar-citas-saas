import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="Radar de Citas", page_icon="🏥")
st.title("🏥 Radar de Citas: Predicción de Ausencias")
st.write("Sube el archivo Excel/CSV de tu agenda para saber qué pacientes tienen riesgo de faltar.")

# --- SELECTOR DE PLAN EN LA BARRA LATERAL ---
st.sidebar.header("Configuración de la cuenta")
plan_seleccionado = st.sidebar.selectbox(
    "Selecciona tu plan:",
    ("Plan Estándar", "Plan Premium")
)

# Cargar el modelo dependiendo del plan elegido
@st.cache_resource
def cargar_modelos(plan):
    scaler = joblib.load('escalador.pkl')
    if plan == "Plan Premium":
        modelo = joblib.load('modelo_premium.pkl') # XGBoost
    else:
        modelo = joblib.load('modelo_estandar.pkl') # Regresión Logística
    return modelo, scaler

modelo, scaler = cargar_modelos(plan_seleccionado)

st.info(f"Estás usando el **{plan_seleccionado}**.")

# Variables exactas que espera el modelo
features_modelo = ['Age', 'WaitingDays', 'SMS_received', 'Scholarship', 'Hipertension', 
                   'Diabetes', 'Alcoholism', 'Prior_no_shows', 'Total_prior_appointments', 
                   'No_show_rate', 'Days_since_last_noshow']

archivo_subido = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo_subido is not None:
    df_nuevo = pd.read_csv(archivo_subido)
    st.write("Vista previa de tus datos (primeras 5 filas):")
    st.dataframe(df_nuevo.head())
    
    if st.button("Analizar Riesgos"):
        try:
            # 1. Detectar columnas faltantes y sobrantes
            columnas_faltantes = [col for col in features_modelo if col not in df_nuevo.columns]
            columnas_sobrantes = [col for col in df_nuevo.columns if col not in features_modelo]
            
            df_procesado = df_nuevo.copy()

            # --- TRADUCTOR DE TEXTOS A NÚMEROS (Para la columna No-show) ---
            posibles_columnas_asistencia = ['No-show', 'Asistio', 'Asistió', 'Estado', 'Asistencia', 'Faltó', 'Falto']
            
            for col in posibles_columnas_asistencia:
                if col in df_procesado.columns:
                    df_procesado[col] = df_procesado[col].astype(str).str.lower()
                    df_procesado[col] = df_procesado[col].replace({
                        'no': 1, 'no asistio': 1, 'no asistió': 1, 'falto': 1, 'faltó': 1, 'ausente': 1, 'canceló': 1, 'cancelo': 1,
                        'si': 0, 'sí': 0, 'asistio': 0, 'asistió': 0, 'vino': 0, 'presente': 0
                    })
                    df_procesado[col] = pd.to_numeric(df_procesado[col], errors='coerce').fillna(0)
                    if col != 'No-show':
                        df_procesado = df_procesado.rename(columns={col: 'No-show'})
                    break 
            
            # 2. Manejar columnas faltantes (Rellenar con valores por defecto)
            if columnas_faltantes:
                st.warning(f"⚠️ Faltan columnas: {columnas_faltantes}. Se usarán valores por defecto (la precisión puede bajar).")
                
                valores_defecto = {
                    'Age': 40, 'WaitingDays': 7, 'SMS_received': 0, 'Scholarship': 0,
                    'Hipertension': 0, 'Diabetes': 0, 'Alcoholism': 0, 'Prior_no_shows': 0,
                    'Total_prior_appointments': 0, 'No_show_rate': 0.0, 'Days_since_last_noshow': -1
                }
                
                for col in columnas_faltantes:
                    df_procesado[col] = valores_defecto[col]
            
            if columnas_sobrantes:
                st.info(f"ℹ️ Se ignorarán {len(columnas_sobrantes)} columnas irrelevantes.")

            # 3. Si falta WaitingDays pero hay fechas, lo calculamos
            if 'WaitingDays' in columnas_faltantes:
                if 'ScheduledDay' in df_procesado.columns and 'AppointmentDay' in df_procesado.columns:
                    df_procesado['ScheduledDay'] = pd.to_datetime(df_procesado['ScheduledDay'])
                    df_procesado['AppointmentDay'] = pd.to_datetime(df_procesado['AppointmentDay'])
                    df_procesado['WaitingDays'] = (df_procesado['AppointmentDay'] - df_procesado['ScheduledDay']).dt.days
                    df_procesado['WaitingDays'] = df_procesado['WaitingDays'].clip(lower=0)
            
            # 4. Preparar datos para el modelo
            X_nuevo = df_procesado[features_modelo]
            X_nuevo_scaled = scaler.transform(X_nuevo)
            
            # 5. Predecir (con umbral ajustado según el plan)
            probabilidades = modelo.predict_proba(X_nuevo_scaled)[:, 1]
            
            if plan_seleccionado == "Plan Premium":
                umbral = 0.50
            else:
                umbral = 0.473
                
            df_procesado['Probabilidad_Falta'] = probabilidades
            df_procesado['Riesgo'] = np.where(probabilidades >= umbral, "🔴 ALTO (Llamar)", "🟢 BAJO")
            
            st.success("¡Análisis completado con éxito!")
            
            # Mostrar resultados
            df_alto_riesgo = df_procesado[df_procesado['Riesgo'] == "🔴 ALTO (Llamar)"].sort_values(by='Probabilidad_Falta', ascending=False)
            
            st.subheader(f"Pacientes en Riesgo Alto: {len(df_alto_riesgo)}")
            
            cols_mostrar = ['Probabilidad_Falta', 'Riesgo']
            for c in ['PatientId', 'Age', 'AppointmentDay']:
                if c in df_alto_riesgo.columns:
                    cols_mostrar.append(c)
            
            st.dataframe(df_alto_riesgo[cols_mostrar])
            
            # Botón de descarga
            csv = df_procesado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Reporte Completo (CSV)",
                data=csv,
                file_name='reporte_riesgos.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Error inesperado al procesar el archivo: {e}")