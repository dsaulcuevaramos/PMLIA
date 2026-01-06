import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Librerías de Machine Learning (Basadas en tus notebooks)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración de la página
st.set_page_config(page_title="Predicción Clínica Juan Pablo II", layout="wide", page_icon="🏥")

# --- CSS PARA ESTILO ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;}
    .main-header {font-size: 2.5rem; color: #1E88E5;}
</style>
""", unsafe_allow_html=True)

# --- 1. FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data
def cargar_datos(uploaded_file=None):
    """Carga los datos y maneja diferencias de separadores (; o ,)"""
    file_path = "dataset_pacientes_v2.csv" # Archivo por defecto
    
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file, sep=';')
        except:
            return pd.read_csv(uploaded_file, sep=',')
    
    try:
        # Intenta cargar el local si no se sube nada
        return pd.read_csv(file_path, sep=';')
    except:
        st.error("⚠️ No se encontró 'dataset_pacientes_v2.csv'. Por favor sube el archivo en el menú lateral.")
        return None

# --- 2. MOTOR DE ENTRENAMIENTO ---
def entrenar_y_evaluar(X, y, test_size, random_state, usar_scaler, modelos_seleccionados):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = None
    if usar_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Definición de algoritmos disponibles
    algoritmos = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "SVR (Support Vector)": SVR() 
    }
    
    resultados = []
    trained_models = {}

    progreso = st.progress(0)
    step = 0
    
    for nombre in modelos_seleccionados:
        base_model = algoritmos[nombre]
        
        # Usamos MultiOutputRegressor para predecir todas las áreas (Rayos X, Eco, etc.) a la vez
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Métricas Globales (Promedio de todas las salidas)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        resultados.append({
            "Algoritmo": nombre,
            "R2 Score (%)": round(r2 * 100, 2),
            "Error Promedio (MAE)": round(mae, 2),
            "Obj": model
        })
        
        step += 1
        progreso.progress(step / len(modelos_seleccionados))
        
    df_res = pd.DataFrame(resultados).sort_values(by="R2 Score (%)", ascending=False)
    return df_res, scaler, X_test, y_test

# --- INTERFAZ PRINCIPAL ---

st.markdown('<h1 class="main-header">🏥 Sistema Inteligente - Clínica Juan Pablo II</h1>', unsafe_allow_html=True)
st.write("Predicción de demanda de pacientes basada en modelos de Regresión Multivariable.")

# --- SIDEBAR: CONFIGURACIONES ---
st.sidebar.header("⚙️ Panel de Control")

# 1. Carga de Datos
archivo = st.sidebar.file_uploader("Subir dataset (Opcional)", type=["csv"])
df = cargar_datos(archivo)

if df is not None:
    # Identificar Columnas
    cols_objetivo = ['total_pacientes', 'pacientes_rayos_x', 'pacientes_mamografia', 
                     'pacientes_ecografia', 'pacientes_laboratorio', 'pacientes_densitometria']
    
    # Filtrar solo columnas numéricas para X
    cols_posibles_features = [c for c in df.columns if c not in cols_objetivo and np.issubdtype(df[c].dtype, np.number)]
    
    # 2. Manipulación Dinámica de Datos (Feature Selection)
    st.sidebar.subheader("1. Selección de Datos (K-Best)")
    k_value = st.sidebar.slider("Filtrar Mejores 'K' Características:", 2, len(cols_posibles_features), 10)
    
    # Lógica de Selección
    X_raw = df[cols_posibles_features].fillna(0)
    y_raw = df[cols_objetivo].fillna(0)
    
    selector = SelectKBest(score_func=f_regression, k=k_value)
    selector.fit(X_raw, y_raw['total_pacientes']) # Usamos total como referencia
    cols_seleccionadas = X_raw.columns[selector.get_support()].tolist()
    
    st.sidebar.write(f"**Variables seleccionadas ({k_value}):**")
    st.sidebar.code(", ".join(cols_seleccionadas))
    
    X_final = X_raw[cols_seleccionadas]

    # 3. Parámetros de Modelo
    st.sidebar.subheader("2. Configuración de Entrenamiento")
    test_size_param = st.sidebar.slider("Tamaño de Test (test_size)", 0.1, 0.5, 0.2)
    random_state_param = st.sidebar.number_input("Semilla Aleatoria (random_state)", value=42)
    estandarizar = st.sidebar.checkbox("Estandarizar Datos (Recomendado)", value=True)
    
    modelos_user = st.sidebar.multiselect("Algoritmos a Comparar:", 
                                          ["Linear Regression", "Decision Tree", "Random Forest", "SVR (Support Vector)"],
                                          default=["Random Forest", "Linear Regression"])

    # --- TABS (PESTAÑAS) ---
    tab1, tab2, tab3 = st.tabs(["📊 Análisis de Datos", "🏆 Entrenamiento & Ranking", "🔮 Predicción Futura"])

    # --- TAB 1: DATOS ---
    with tab1:
        st.subheader("Visualización Dinámica de Datos")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(10))
            st.caption(f"Mostrando primeros 10 registros. Total filas: {len(df)}")
        with col2:
            st.info("Variables Objetivo:")
            st.write(cols_objetivo)
            
        st.subheader("Correlación de las 'Mejores Variables'")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(X_final.corr(), annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --- TAB 2: ENTRENAMIENTO ---
    with tab2:
        st.subheader("Comparativa de Modelos de Regresión")
        
        if st.button("🚀 Ejecutar Entrenamiento y Comparar"):
            if not modelos_user:
                st.warning("Selecciona al menos un algoritmo en la barra lateral.")
            else:
                resultados_df, scaler_trained, X_test_sample, y_test_sample = entrenar_y_evaluar(
                    X_final, y_raw, test_size_param, random_state_param, estandarizar, modelos_user
                )
                
                # Mostrar Tabla Ranking
                st.write("### 🥇 Ranking de Algoritmos (Mejor a Peor)")
                st.dataframe(resultados_df.drop(columns="Obj").style.background_gradient(cmap="Greens", subset=["R2 Score (%)"]))
                
                # Guardar el mejor modelo en sesión
                mejor_modelo = resultados_df.iloc[0]
                st.session_state['modelo_activo'] = mejor_modelo['Obj']
                st.session_state['nombre_modelo'] = mejor_modelo['Algoritmo']
                st.session_state['scaler'] = scaler_trained
                st.session_state['features'] = cols_seleccionadas
                
                st.success(f"Modelo ganador: **{mejor_modelo['Algoritmo']}** cargado para predicciones.")
                
                # Gráfica de error
                st.write("#### Comparativa Visual de Precisión (R2)")
                st.bar_chart(resultados_df.set_index("Algoritmo")["R2 Score (%)"])

    # --- TAB 3: PREDICCIONES ---
    with tab3:
        st.subheader("Simulador de Demanda Futura")
        
        if 'modelo_activo' not in st.session_state:
            st.warning("⚠️ Primero debes entrenar los modelos en la pestaña anterior.")
        else:
            modelo = st.session_state['modelo_activo']
            features = st.session_state['features']
            scaler = st.session_state['scaler']
            
            tipo_pred = st.radio("Modo de Predicción:", ["Fecha Específica", "Proyección (7, 15, 30 días)"])
            
            if tipo_pred == "Fecha Específica":
                col_d, col_c = st.columns(2)
                with col_d:
                    fecha = st.date_input("Seleccionar Fecha", datetime.today())
                
                # Generar inputs dinámicos para las features seleccionadas
                input_data = {}
                with col_c:
                    st.write("Configuración de variables:")
                    for f in features:
                        # Lógica inteligente para pre-llenar datos si son de fecha
                        val = 0.0
                        if f == 'mes': val = float(fecha.month)
                        elif f == 'dia_semana': val = float(fecha.weekday())
                        elif f == 'anio': val = float(fecha.year)
                        elif f == 'trimestre': val = float((fecha.month-1)//3 + 1)
                        else:
                            # Usar promedio histórico como default
                            val = float(df[f].mean())
                        
                        input_data[f] = st.number_input(f"Valor para '{f}':", value=val)
                
                if st.button("Predecir Demanda"):
                    df_in = pd.DataFrame([input_data])
                    if scaler: df_in = scaler.transform(df_in)
                    pred = modelo.predict(df_in)[0]
                    
                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🏥 Total Pacientes", int(pred[0]))
                    c2.metric("🩻 Rayos X", int(pred[1]))
                    c3.metric("🧪 Laboratorio", int(pred[4]))
                    
                    # Gráfico de áreas
                    st.bar_chart(pd.Series(pred[1:], index=cols_objetivo[1:]))

            else: # PROYECCIÓN
                dias = st.slider("Días a proyectar:", 7, 30, 7)
                if st.button("Generar Proyección"):
                    fechas = []
                    preds = []
                    start = datetime.today()
                    
                    for i in range(dias):
                        curr = start + timedelta(days=i)
                        row = {}
                        for f in features:
                            # Lógica simple de proyección (promedios históricos para clima)
                            if f == 'mes': row[f] = curr.month
                            elif f == 'dia_semana': row[f] = curr.weekday()
                            elif f == 'es_fin_semana': row[f] = 1 if curr.weekday() >= 5 else 0
                            else: row[f] = df[f].mean() # Promedio para clima
                        
                        # Predecir
                        df_row = pd.DataFrame([row])
                        if scaler: df_row = scaler.transform(df_row)
                        res = modelo.predict(df_row)[0]
                        preds.append(res)
                        fechas.append(curr.strftime("%Y-%m-%d"))
                    
                    df_proj = pd.DataFrame(preds, columns=cols_objetivo)
                    df_proj['Fecha'] = fechas
                    df_proj.set_index('Fecha', inplace=True)
                    
                    st.write("### Tendencia Estimada")
                    st.line_chart(df_proj['total_pacientes'])
                    st.dataframe(df_proj.style.format("{:.0f}"))

else:
    st.info("Esperando carga de datos...")