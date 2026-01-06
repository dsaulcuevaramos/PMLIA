import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Librerías de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
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
    .stButton>button {width: 100%; background-color: #1E88E5; color: white;}
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
    
    # 2. Escalamiento
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
            # Algoritmos estándar
            if nombre == "Linear Regression":
                base_model = LinearRegression()
            elif nombre == "Decision Tree": 
                base_model = DecisionTreeRegressor(random_state=random_state)
            elif nombre == "Random Forest":
                base_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif nombre == "SVR (Support Vector)":
                base_model = SVR()
            
            # Wrapper MultiOutput
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

archivo = st.sidebar.file_uploader("Subir dataset", type=["csv"])
df = cargar_datos(archivo)

if df is not None:
    # --- 1. SELECCIÓN DE TARGETS ---
    st.sidebar.subheader("1. Variables Objetivo (Target)")
    
    all_targets = ['total_pacientes', 'pacientes_rayos_x', 'pacientes_mamografia', 
                   'pacientes_ecografia', 'pacientes_laboratorio', 'pacientes_densitometria']
    
    targets_seleccionados = st.sidebar.multiselect(
        "Selecciona qué predecir:", 
        all_targets, 
        default=all_targets
    )
    
    if not targets_seleccionados:
        st.warning("Selecciona al menos una variable objetivo.")
        st.stop()

    cols_features = [c for c in df.columns if c not in all_targets and np.issubdtype(df[c].dtype, np.number)]
    
    X_raw = df[cols_features].fillna(0)
    y_raw = df[targets_seleccionados].fillna(0)
    
    # --- 2. SELECCIÓN DE FEATURES ---
    st.sidebar.subheader("2. Selección de Características")
    
    k_value = st.sidebar.slider("Cantidad de Variables 'K':", 2, len(cols_features), 11)
    metodo_seleccion = st.sidebar.radio("Método de Selección:", ["SelectKBest (Estadístico)", "RFE (Recursivo)"])

    cols_sel_final = []
    
    if metodo_seleccion == "SelectKBest (Estadístico)":
        selector = SelectKBest(score_func=f_regression, k=k_value)
        selector.fit(X_raw, y_raw.iloc[:, 0]) 
        cols_sel_final = X_raw.columns[selector.get_support()].tolist()
    else:
        # RFE
        estimator = DecisionTreeRegressor(random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=k_value)
        selector.fit(X_raw, y_raw.iloc[:, 0])
        cols_sel_final = X_raw.columns[selector.get_support()].tolist()

    X_final = X_raw[cols_sel_final]
    st.sidebar.caption(f"Variables Activas: {len(cols_sel_final)}")

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
        "Decision Tree",
        "Random Forest", 
        "Linear Regression", 
        "Regresión Polinomial", 
        "SVR (Support Vector)"
    ]
    modelos_user = st.sidebar.multiselect("Algoritmos:", modelos_disponibles, default=["Decision Tree", "Random Forest"])
    
    poly_degree = 2
    if "Regresión Polinomial" in modelos_user:
        poly_degree = st.sidebar.slider("Grado Polinomial", 2, 5, 2)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Datos y Comparativa", "🏆 Entrenamiento", "🔮 Predicctor Pro"])

    with tab1:
        st.subheader("Datos Filtrados")
        st.dataframe(df.head())
        
        st.divider()
        st.subheader("🆚 Comparativa: K-Best vs RFE")
        st.info(f"Comparación de las {k_value} mejores variables seleccionadas por cada método.")
        
        sel_kb = SelectKBest(score_func=f_regression, k=k_value)
        sel_kb.fit(X_raw, y_raw.iloc[:, 0])
        cols_kb = X_raw.columns[sel_kb.get_support()].tolist()
        
        sel_rfe = RFE(estimator=DecisionTreeRegressor(random_state=42), n_features_to_select=k_value)
        sel_rfe.fit(X_raw, y_raw.iloc[:, 0])
        cols_rfe = X_raw.columns[sel_rfe.get_support()].tolist()
        
        max_len = max(len(cols_kb), len(cols_rfe))
        cols_kb += [''] * (max_len - len(cols_kb))
        cols_rfe += [''] * (max_len - len(cols_rfe))
        
        df_compare = pd.DataFrame({
            'SelectKBest (Filtro)': cols_kb,
            'RFE (Wrapper)': cols_rfe
        })
        st.table(df_compare)

    with tab2:
        st.subheader("Entrenamiento y Ranking")
        st.write(f"**Objetivos a predecir:** {', '.join(targets_seleccionados)}")
        
        if st.button("🚀 Entrenar Modelos"):
            res_df, scaler_trained = entrenar_y_evaluar(
                X_final, y_raw, test_size, random_state, tipo_escalado, modelos_user, poly_degree
            )
            
            # Tabla de Resultados
            st.dataframe(res_df.drop(columns="Obj").style.background_gradient(cmap="Greens", subset=["R2 Score (%)"]))
            
            # --- AGREGADO: GRÁFICO DE BARRAS COMPARATIVO ---
            st.divider()
            st.subheader("📊 Comparativa Visual de Rendimiento (R2 Score)")
            
            # Preparamos datos para el gráfico (Algoritmo en Eje Y, Score en Eje X)
            chart_data = res_df.set_index("Algoritmo")[["R2 Score (%)"]]
            st.bar_chart(chart_data)
            # -----------------------------------------------
            
            best = res_df.iloc[0]
            st.session_state['modelo'] = best['Obj']
            st.session_state['scaler'] = scaler_trained
            st.session_state['features'] = cols_sel_final
            st.session_state['target_names'] = targets_seleccionados
            st.success(f"Modelo cargado: {best['Algoritmo']}")

    with tab3:
        st.subheader("Simulador de Predicción")
        
        if 'modelo' in st.session_state:
            features = st.session_state['features']
            modelo_actual = st.session_state['modelo']
            target_names = st.session_state['target_names']
            
            # Contenedor de Inputs
            with st.container():
                c_fechas, c_vars = st.columns([1, 3])
                
                with c_fechas:
                    st.markdown("### 📅 Temporalidad")
                    fecha_inicio = st.date_input("Fecha Inicio", datetime.today())
                    dias_proyeccion = st.slider("Días a proyectar (Rango)", 1, 30, 1)
                
                # Inputs dinámicos para las features (excepto las de fecha que calculamos auto)
                input_base = {}
                with c_vars:
                    st.markdown("### 🌡️ Variables Externas")
                    cols = st.columns(3)
                    idx = 0
                    for f in features:
                        # Omitimos variables de fecha porque las calcularemos en el loop
                        if f in ['mes', 'anio', 'dia_semana']:
                            continue
                            
                        val_def = float(df[f].mean())
                        with cols[idx % 3]:
                            input_base[f] = st.number_input(f"{f}", value=val_def)
                        idx += 1
            
            st.divider()
            
            # --- BOTÓN ARRIBA ---
            if st.button("🔮 Calcular Predicción", type="primary"):
                
                # Generar rango de fechas
                fechas_rango = [fecha_inicio + timedelta(days=i) for i in range(dias_proyeccion)]
                
                # Construir DataFrame para todo el rango
                rows = []
                for fecha in fechas_rango:
                    row = input_base.copy()
                    
                    # Inyectar datos de fecha si el modelo los usa
                    if 'mes' in features: row['mes'] = fecha.month
                    if 'anio' in features: row['anio'] = fecha.year
                    if 'dia_semana' in features: row['dia_semana'] = fecha.weekday() # 0=Lunes
                    
                    # Asegurar orden de columnas
                    ordered_row = {k: row.get(k, 0) for k in features}
                    rows.append(ordered_row)
                
                X_batch = pd.DataFrame(rows)
                
                # 1. Escalar
                if st.session_state['scaler']:
                    X_batch_sc = st.session_state['scaler'].transform(X_batch)
                else:
                    X_batch_sc = X_batch

                # 2. Polinomial
                if hasattr(modelo_actual, 'poly_transformer') and modelo_actual.poly_transformer is not None:
                      X_batch_final = modelo_actual.poly_transformer.transform(X_batch_sc)
                else:
                      X_batch_final = X_batch_sc
                
                # Predicción del Batch (Matriz: filas=días, cols=targets)
                preds_batch = modelo_actual.predict(X_batch_final)
                
                # --- RESULTADOS ---
                st.subheader(f"📊 Resultados ({dias_proyeccion} días)")

                # Cálculo de Promedios
                promedios = preds_batch.mean(axis=0)
                
                # VISUALIZACIÓN 1: Tarjetas Métricas (Promedio del periodo)
                if len(target_names) == 1:
                    val = int(promedios[0])
                    st.metric(label=f"Promedio {target_names[0]}", value=f"{val:,}")
                else:
                    cols_res = st.columns(min(len(target_names), 4))
                    for i, t_name in enumerate(target_names):
                        with cols_res[i % 4]:
                            val = int(promedios[i])
                            st.metric(label=t_name, value=f"{val:,}")
                
                # VISUALIZACIÓN 2: Gráfico de Tendencia (Si hay > 1 día)
                if dias_proyeccion > 1:
                    st.markdown("### 📈 Tendencia Estimada")
                    df_trend = pd.DataFrame(preds_batch, columns=target_names)
                    df_trend['Fecha'] = fechas_rango
                    df_trend.set_index('Fecha', inplace=True)
                    
                    st.line_chart(df_trend)
                    
                    with st.expander("Ver tabla de datos detallada"):
                        st.dataframe(df_trend.style.format("{:.0f}"))

        else:
            st.warning("⚠️ Por favor, ve a la pestaña 'Entrenamiento' y entrena un modelo primero.")
else:
    st.info("Sube el archivo CSV para comenzar.")