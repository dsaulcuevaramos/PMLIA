import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Librerías de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE # <--- Agregado RFE
from sklearn.multioutput import MultiOutputRegressor

# Algoritmos
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# Configuración de la página
st.set_page_config(page_title="Predicción Clínica Juan Pablo II", layout="wide", page_icon="🏥")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;}
    .main-header {font-size: 2.5rem; color: #1E88E5;}
</style>
""", unsafe_allow_html=True)

# --- 1. FUNCIÓN DE CARGA ---
@st.cache_data
def cargar_datos(uploaded_file=None):
    file_path = "dataset_pacientes_v2.csv"
    if uploaded_file is not None:
        try: return pd.read_csv(uploaded_file, sep=';')
        except: return pd.read_csv(uploaded_file, sep=',')
    try: return pd.read_csv(file_path, sep=';')
    except: return None

# --- 2. MOTOR DE ENTRENAMIENTO ---
def entrenar_y_evaluar(X, y, test_size, random_state, tipo_escalado, modelos_seleccionados, poly_degree):
    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = None
    
    # 2. Escalamiento (Standard vs Normalization)
    if tipo_escalado == "Estandarización (StandardScaler)":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif tipo_escalado == "Normalización (MinMaxScaler)":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    resultados = []
    progreso = st.progress(0)
    step = 0
    
    for nombre in modelos_seleccionados:
        model = None
        
        # Lógica específica para Regresión Polinomial
        if nombre == "Regresión Polinomial":
            poly = PolynomialFeatures(degree=poly_degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            
            base_model = LinearRegression()
            model = base_model 
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
            
            model.poly_transformer = poly 
            
        else:
            # Diccionario de modelos estándar
            if nombre == "Linear Regression":
                base_model = LinearRegression()
            elif nombre == "Decision Tree": # <--- Algoritmo Solicitado
                base_model = DecisionTreeRegressor(random_state=random_state)
            elif nombre == "Random Forest":
                base_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif nombre == "SVR (Support Vector)":
                base_model = SVR()
            
            # Wrapper MultiOutput para manejar N variables objetivo a la vez
            model = MultiOutputRegressor(base_model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            model.poly_transformer = None

        # Métricas
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
    return df_res, scaler

# --- INTERFAZ PRINCIPAL ---

st.markdown('<h1 class="main-header">🏥 Sistema Inteligente - Clínica Juan Pablo II</h1>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuración Avanzada")

# Carga
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv"])
df = cargar_datos(archivo)

if df is not None:
    # --- 1. SELECCIÓN DE TARGETS (MODIFICADO) ---
    st.sidebar.subheader("1. Variables Objetivo (Target)")
    
    all_targets = ['total_pacientes', 'pacientes_rayos_x', 'pacientes_mamografia', 
                   'pacientes_ecografia', 'pacientes_laboratorio', 'pacientes_densitometria']
    
    # Usuario selecciona qué predecir
    targets_seleccionados = st.sidebar.multiselect(
        "Selecciona qué predecir:", 
        all_targets, 
        default=all_targets
    )
    
    if not targets_seleccionados:
        st.warning("Selecciona al menos una variable objetivo.")
        st.stop()

    cols_features = [c for c in df.columns if c not in all_targets and np.issubdtype(df[c].dtype, np.number)]
    
    # Filtramos Dataframes según selección
    X_raw = df[cols_features].fillna(0)
    y_raw = df[targets_seleccionados].fillna(0)
    
    # --- 2. SELECCIÓN DE FEATURES (MODIFICADO: KBEST vs RFE) ---
    st.sidebar.subheader("2. Selección de Características")
    
    k_value = st.sidebar.slider("Cantidad de Variables 'K':", 2, len(cols_features), 11)
    metodo_seleccion = st.sidebar.radio("Método de Selección:", ["SelectKBest (Estadístico)", "RFE (Recursivo)"])

    cols_sel_final = []
    
    # Lógica de Selección ACTIVA
    if metodo_seleccion == "SelectKBest (Estadístico)":
        selector = SelectKBest(score_func=f_regression, k=k_value)
        # Ajustamos con la primera columna target para referencia
        selector.fit(X_raw, y_raw.iloc[:, 0]) 
        cols_sel_final = X_raw.columns[selector.get_support()].tolist()
    else:
        # RFE (Recursive Feature Elimination)
        estimator = DecisionTreeRegressor(random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=k_value)
        selector.fit(X_raw, y_raw.iloc[:, 0])
        cols_sel_final = X_raw.columns[selector.get_support()].tolist()

    X_final = X_raw[cols_sel_final]
    
    st.sidebar.caption(f"Variables seleccionadas: {len(cols_sel_final)}")
    st.sidebar.code("\n".join(cols_sel_final))

    # --- 3. CONFIGURACIÓN MODELO ---
    st.sidebar.subheader("3. Preprocesamiento y Modelos")
    
    tipo_escalado = st.sidebar.radio(
        "Técnica de Escalado:",
        ["Sin Escalar", "Estandarización (StandardScaler)", "Normalización (MinMaxScaler)"],
        index=1
    )
    
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    random_state = st.sidebar.number_input("Random State", value=42)
    
    modelos_disponibles = [
        "Decision Tree", # <--- Agregado explícitamente al menú
        "Random Forest", 
        "Linear Regression", 
        "Regresión Polinomial", 
        "SVR (Support Vector)"
    ]
    modelos_user = st.sidebar.multiselect("Algoritmos:", modelos_disponibles, default=["Decision Tree", "Random Forest"])
    
    poly_degree = 2
    if "Regresión Polinomial" in modelos_user:
        poly_degree = st.sidebar.slider("Grado Polinomial (Degree)", 2, 5, 2, help="Cuidado: Grados altos >3 pueden ser muy lentos.")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Datos y Comparativa", "🏆 Entrenamiento", "🔮 Predicctor"])

    with tab1:
        st.subheader("Datos Filtrados")
        st.dataframe(df.head())
        
        st.divider()
        st.subheader("🆚 Comparativa: K-Best vs RFE")
        st.info(f"Comparación de las {k_value} mejores variables seleccionadas por cada método.")
        
        # Calculamos ambos solo para mostrar la tabla
        # 1. KBest
        sel_kb = SelectKBest(score_func=f_regression, k=k_value)
        sel_kb.fit(X_raw, y_raw.iloc[:, 0])
        cols_kb = X_raw.columns[sel_kb.get_support()].tolist()
        
        # 2. RFE
        sel_rfe = RFE(estimator=DecisionTreeRegressor(random_state=42), n_features_to_select=k_value)
        sel_rfe.fit(X_raw, y_raw.iloc[:, 0])
        cols_rfe = X_raw.columns[sel_rfe.get_support()].tolist()
        
        # Tabla lado a lado
        max_len = max(len(cols_kb), len(cols_rfe))
        # Rellenar con vacíos si hay diferencia (raro si k es fijo, pero por seguridad)
        cols_kb += [''] * (max_len - len(cols_kb))
        cols_rfe += [''] * (max_len - len(cols_rfe))
        
        df_compare = pd.DataFrame({
            'SelectKBest (Filtro)': cols_kb,
            'RFE (Wrapper)': cols_rfe
        })
        st.table(df_compare)
        st.write(f"**Método activo para entrenamiento:** {metodo_seleccion}")

    with tab2:
        st.subheader("Entrenamiento y Evaluación")
        st.write(f"**Objetivos a predecir:** {', '.join(targets_seleccionados)}")
        
        if st.button("🚀 Entrenar Modelos"):
            res_df, scaler_trained = entrenar_y_evaluar(
                X_final, y_raw, test_size, random_state, tipo_escalado, modelos_user, poly_degree
            )
            
            st.dataframe(res_df.drop(columns="Obj").style.background_gradient(cmap="Greens", subset=["R2 Score (%)"]))
            
            # Guardar mejor modelo y metadatos
            best = res_df.iloc[0]
            st.session_state['modelo'] = best['Obj']
            st.session_state['scaler'] = scaler_trained
            st.session_state['features'] = cols_sel_final
            st.session_state['target_names'] = targets_seleccionados # Guardamos los nombres
            st.success(f"Modelo cargado: {best['Algoritmo']}")

    with tab3:
        st.subheader("Simulador de Predicción")
        if 'modelo' in st.session_state:
            features = st.session_state['features']
            modelo_actual = st.session_state['modelo']
            target_names = st.session_state['target_names']
            
            # Inputs
            col1, col2 = st.columns(2)
            input_data = {}
            with col1:
                fecha = st.date_input("Fecha", datetime.today())
            with col2:
                for f in features:
                    val_def = float(df[f].mean())
                    if f == 'mes': val_def = float(fecha.month)
                    elif f == 'anio': val_def = float(fecha.year)
                    elif f == 'dia_semana': val_def = float(fecha.weekday())
                    input_data[f] = st.number_input(f"{f}", value=val_def)
            
            if st.button("Predecir"):
                X_new = pd.DataFrame([input_data])
                
                # 1. Escalar
                if st.session_state['scaler']:
                    X_new_sc = st.session_state['scaler'].transform(X_new)
                else:
                    X_new_sc = X_new

                # 2. Polinomial
                if hasattr(modelo_actual, 'poly_transformer') and modelo_actual.poly_transformer is not None:
                      X_new_final = modelo_actual.poly_transformer.transform(X_new_sc)
                else:
                      X_new_final = X_new_sc
                
                pred = modelo_actual.predict(X_new_final)[0]
                
                st.divider()
                st.subheader("Resultados Estimados")

                # Visualización Dinámica (Depende de si eligió 1 o varios targets)
                if len(target_names) == 1:
                    st.metric(label=target_names[0], value=f"{int(pred):,}")
                else:
                    # Si eligió varios, buscamos si 'total_pacientes' está entre ellos para destacarlo
                    if 'total_pacientes' in target_names:
                        idx = target_names.index('total_pacientes')
                        st.metric("Total Pacientes", int(pred[idx]))
                    
                    # Gráfico de barras con los targets seleccionados
                    chart_data = pd.DataFrame({'Area': target_names, 'Pacientes': pred})
                    st.bar_chart(chart_data.set_index('Area'))
        else:
            st.warning("Entrena primero los modelos.")
else:
    st.info("Sube el archivo CSV.")