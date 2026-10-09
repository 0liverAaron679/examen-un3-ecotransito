import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# 1. Configuración de la página
st.set_page_config(
    page_title="Sistema EcoTránsito CDMX",
    layout="wide"
)

st.title("Sistema EcoTránsito CDMX")

# 2. Carga y filtrado de datos con barra lateral
try:
    df = pd.read_csv("cdmx_air_traffic_dirty.csv")
except FileNotFoundError:
    df = pd.read_csv("cdmx_air_traffic_clean.csv")

# Mapeo o unificación de nombres de columnas
rename_dict = {col: col for col in df.columns}
for col in df.columns:
    if col.lower() == "concentracion_pm25":
        rename_dict[col] = "Concentracion_PM25"
    elif col.lower() == "nivel_no2":
        rename_dict[col] = "Nivel_NO2"
    elif col.lower() == "indice_trafico_vehicular":
        rename_dict[col] = "Indice_Trafico_Vehicular"
    elif col.lower() == "zona_estacion":
        rename_dict[col] = "Zona_Estacion"
    elif col.lower() == "timestamp":
        rename_dict[col] = "Timestamp"

df = df.rename(columns=rename_dict)

# Limpieza básica
df["Concentracion_PM25"] = df["Concentracion_PM25"].replace(-999.0, np.nan)
df["Nivel_NO2"] = df["Nivel_NO2"].replace([-999.0, -995.0], np.nan)
df["Indice_Trafico_Vehicular"] = df["Indice_Trafico_Vehicular"].replace(-999.0, np.nan)
df = df.drop_duplicates()

# Limpieza estructural
df_clean = df.copy()

# Homogeneizar Zona_Estacion
df_clean["Zona_Estacion"] = df_clean["Zona_Estacion"].astype(str).str.strip().str.replace(" ", "_").str.title()

# Timestamp y Hora_Dia
df_clean["Timestamp"] = pd.to_datetime(df_clean["Timestamp"])
df_clean["Hora_Dia"] = df_clean["Timestamp"].dt.hour

# Outliers con IQR en Concentracion_PM25
Q1 = df_clean["Concentracion_PM25"].quantile(0.25)
Q3 = df_clean["Concentracion_PM25"].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR
df_clean.loc[df_clean["Concentracion_PM25"] > limite_sup, "Concentracion_PM25"] = np.nan
df_clean.loc[df_clean["Concentracion_PM25"] < limite_inf, "Concentracion_PM25"] = np.nan

# Imputación k-NN
columnas_num = ["Concentracion_PM25", "Nivel_NO2", "Indice_Trafico_Vehicular"]
imputer = KNNImputer(n_neighbors=2)
df_clean[columnas_num] = imputer.fit_transform(df_clean[columnas_num])

# Filtro por zona 
st.sidebar.header("Filtros")
zonas = st.sidebar.multiselect(
    "Selecciona las zonas:",
    options=df_clean["Zona_Estacion"].unique(),
    default=df_clean["Zona_Estacion"].unique()
)

# Aplicar el filtro
df_filtrado = df_clean[df_clean["Zona_Estacion"].isin(zonas)]

# 3. Métricas principales (KPIs)
st.subheader("Indicadores principales")

col1, col2, col3 = st.columns(3)

with col1:
    total_lecturas = len(df_filtrado)
    st.metric("Total de lecturas activas", total_lecturas)

with col2:
    promedio_pm25 = round(df_filtrado["Concentracion_PM25"].mean(), 2)
    st.metric("Promedio PM₂.₅ (µg/m³)", promedio_pm25)

with col3:
    max_trafico = round(df_filtrado["Indice_Trafico_Vehicular"].max(), 2)
    st.metric("Índice máximo de tráfico", max_trafico)

# 4. Pestañas interactivas de visualización (Gráficos 2D y 3D incluidos)
tab1, tab2, tab3, tab4 = st.tabs([
    "Gráfico de dispersión", 
    "Mapa de calor de correlación", 
    "Gráfico 2D (Línea temporal)", 
    "Gráfico 3D (Multivariado)"
])

# Pestaña 1: Scatter plot
with tab1:
    st.subheader("Relación entre Tráfico y PM₂.₅")
    fig1 = px.scatter(
        df_filtrado,
        x="Indice_Trafico_Vehicular",
        y="Concentracion_PM25",
        color="Zona_Estacion",
        title="Índice de Tráfico vs Concentración de PM₂.₅",
        labels={
            "Indice_Trafico_Vehicular": "Índice de Tráfico Vehicular",
            "Concentracion_PM25": "Concentración PM₂.₅ (µg/m³)"
        }
    )
    st.plotly_chart(fig1, use_container_width=True)

# Pestaña 2: Heatmap de correlación
with tab2:
    st.subheader("Mapa de calor de correlaciones")
    columnas_corr = ["Concentracion_PM25", "Nivel_NO2", "Indice_Trafico_Vehicular", "Hora_Dia"]
    matriz_corr = df_filtrado[columnas_corr].corr()
    
    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlación")
    st.pyplot(fig2)

# Pestaña 3: Gráfico 2D
with tab3:
    st.subheader("Gráfico 2D: Tendencia Promedio de PM₂.₅ por Hora del Día")
    df_2d = df_filtrado.groupby(["Hora_Dia", "Zona_Estacion"])["Concentracion_PM25"].mean().reset_index()
    
    fig3 = px.line(
        df_2d,
        x="Hora_Dia",
        y="Concentracion_PM25",
        color="Zona_Estacion",
        markers=True,
        title="Evolución de PM₂.₅ a lo largo del día (2D)",
        labels={
            "Hora_Dia": "Hora del Día (0-23h)",
            "Concentracion_PM25": "Promedio PM₂.₅ (µg/m³)"
        }
    )
    st.plotly_chart(fig3, use_container_width=True)

# Pestaña 4: Gráfico 3D
with tab4:
    st.subheader("Gráfico 3D: Tráfico vs PM₂.₅ vs NO₂")
    fig4 = px.scatter_3d(
        df_filtrado,
        x="Indice_Trafico_Vehicular",
        y="Concentracion_PM25",
        z="Nivel_NO2",
        color="Zona_Estacion",
        size="Hora_Dia",
        title="Visualización 3D Multivariable",
        labels={
            "Indice_Trafico_Vehicular": "Tráfico",
            "Concentracion_PM25": "PM₂.₅",
            "Nivel_NO2": "NO₂"
        }
    )
    fig4.update_layout(height=700)
    st.plotly_chart(fig4, use_container_width=True)