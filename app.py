import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stats
import pickle

st.set_page_config(page_title="EDA Cancer",
                   page_icon="📊",
                   layout="wide")

# -----------------------------
# Datos
# -----------------------------

# En esta parte importamos los datos desde un archivo csv, este dataframe se llamara df.
datos = pd.read_csv("data.csv")

X = datos.drop(columns=["id", "diagnosis"])
feature_names = X.columns.tolist()
y = datos["diagnosis"]
df = pd.DataFrame(X, columns=feature_names)
df["Diagnóstico"] = y.map({"M": "Maligno", "B": "Benigno"})



# Título y descripción
st.title("Análisis Exploratorio de Datos - Breast Cancer (Wisconsin)")

st.markdown("""
        Este análisis permite explorar las características más relevantes del dataset Breast Cancer Wisconsin Diagnostic, 
        proporcionando visualizaciones y estadísticas descriptivas para facilitar la comprensión del comportamiento de cada variable 
        según el diagnóstico.
    """)

st.markdown("""
    <div style="text-align: justify;">
        Este análisis permite explorar las características más relevantes del dataset <strong>Breast Cancer Wisconsin Diagnostic</strong>, 
        proporcionando visualizaciones y estadísticas descriptivas para facilitar la comprensión del comportamiento de cada variable 
        según el diagnóstico.
    </div>
    """, unsafe_allow_html=True)

st.subheader("Primeras filas del dataset")
st.dataframe(df.head(6))
st.markdown("""
    <div style="text-align: justify;">
        Tras llevar a cabo un análisis exploratorio inicial del conjunto de datos, se verificó la ausencia de valores faltantes en las variables consideradas. 
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
---
""")

# -----------------------------
# Datos faltantes en el dataset
# -----------------------------


#st.subheader("Conteo de datos faltantes en el dataset")

#valores_nulos = df.isnull().sum()

#tabla_nulos = pd.DataFrame({
#    'Variable': valores_nulos.index,
#    'Cantidad de valores faltantes': valores_nulos.values
#})

#st.markdown("""
#    <div style="text-align: justify;">
#        Como se evidencia en la presente tabla, vemos que no hay valores faltantes en el dataset.
#    </div>
#    """, unsafe_allow_html=True)

#st.table(tabla_nulos)


# Sidebar

st.sidebar.header('Variables de estudio')
variables = df.columns.drop("Diagnóstico")
variable_seleccionada = st.sidebar.selectbox('Por favor, seleccione la variable de interés!', variables)

# Autor

with st.sidebar:
    st.markdown("""
    <hr>
    <div style="text-align: center; font-size: 0.9em; color: gray;">
        Desarrollado por Carlos D. López P.
    </div>
    """, unsafe_allow_html=True)




# Título variable seleccionada
#st.markdown(f"## Análisis de la variable: {variable_seleccionada}")
st.markdown(f"## Análisis de la variable: <span style='color:#2a9df4; font-weight:bold'>{variable_seleccionada}</span>", unsafe_allow_html=True)

# Subconjunto de datos
valores = df[variable_seleccionada]
diagnostico = df['Diagnóstico']

# Gráficos
st.subheader("Distribución de la variable")
st.markdown("""
    <div style="text-align: justify;">
        A continuacion se presenta un histograma y un diagrama de caja y bigotes interactivos de la variable seleccionada por tipo de diagnostico
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Figuras y ejemplos
# -----------------------------

#fig, ax = plt.subplots()
#ax.hist(df[variable_seleccionada], bins=30, color="steelblue", edgecolor="black")
#ax.set_title("Histograma 1", fontsize=14)
#ax.set_xlabel(variable_seleccionada, fontsize=12)
#ax.set_ylabel("Frecuencia", fontsize=12)
#st.pyplot(fig)


#figura = px.histogram(
#        df,
#        x=variable_seleccionada,
#        nbins=30,
#        marginal="rug",
#        title="Histograma 1",
#        color_discrete_sequence=["steelblue"]
#    )
#    
#figura.update_traces(marker=dict(line=dict(color="black", width=1)))  # Línea negra alrededor de las barras
#st.plotly_chart(figura, use_container_width=True)

col1, col2 = st.columns(2)

# Histograma
with col1:
    fig = px.histogram(
        df,
        x=variable_seleccionada,
        nbins=30,
        marginal="rug",
        title="Histograma",
        color_discrete_sequence=["steelblue"]
    )
    
    fig.update_traces(marker=dict(line=dict(color="black", width=1)))  # Línea negra alrededor de las barras
    
    st.plotly_chart(fig, use_container_width=True)

# Boxplot con colores personalizados
with col2:
    fig2 = px.box(
        df,
        x='Diagnóstico',
        y=variable_seleccionada,
        color='Diagnóstico',
        title="Boxplot por Diagnóstico",
        color_discrete_map={
            'Benigno': 'steelblue',  # Azul para 'Benigno'
            'Maligno': 'firebrick'    # Rojo para 'Maligno'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

# Filtrar los datos por diagnóstico
grupo_benigno = df[df['Diagnóstico'] == 'Benigno'][variable_seleccionada].dropna()
grupo_maligno = df[df['Diagnóstico'] == 'Maligno'][variable_seleccionada].dropna()

# Prueba t de Student para muestras independientes
t_stat, p_valor = stats.ttest_ind(grupo_benigno, grupo_maligno, equal_var=False)  # Welch’s t-test

# Resultados
st.subheader("Comparación de medias por diagnóstico")
st.markdown(f"""
<div style="text-align: justify;">
    Se realizó una prueba t de Student para comparar las medias de la variable <strong>{variable_seleccionada}</strong> entre los grupos <strong>Benigno</strong> y <strong>Maligno</strong>.
    El valor p obtenido fue <strong>{p_valor:.4f}</strong>. {"Esto indica una diferencia significativa entre los grupos." if p_valor < 0.05 else "No se encontraron diferencias significativas entre los grupos."}
</div>
""", unsafe_allow_html=True)



# Estadísticas descriptivas
st.subheader("Estadísticas Descriptivas")
st.dataframe(valores.describe().to_frame().T.round(2))


# Comparación por diagnóstico
st.markdown(f"### Resumen de <span style='color:#2a9df4; font-weight:bold'>{variable_seleccionada}</span> por tipo de diagnóstico", unsafe_allow_html=True)
st.write(df.groupby("Diagnóstico")[variable_seleccionada].describe())


st.markdown("""
---
""")

# Gráfico de dispersión con otras variables
st.subheader("Relación con otras variables")
otras_variables = [var for var in variables if var != variable_seleccionada]
otra_variable = st.selectbox("Seleccione otra variable para comparar", otras_variables)

fig3 = px.scatter(
    df,
    x=variable_seleccionada,
    y=otra_variable,
    color='Diagnóstico',
    title="Gráfico de dispersión interactivo",
    color_discrete_sequence=px.colors.qualitative.Set1,
    hover_data=df.columns
)

st.plotly_chart(fig3, use_container_width=True)


# Tabla de datos
st.subheader("Vista previa de los datos")
st.dataframe(df[[variable_seleccionada, otra_variable, 'Diagnóstico']].head(6))

# Correlacion


st.markdown("""
---
""")


st.subheader("Matriz de correlación entre las variables de estudio")
st.markdown("""
    <div style="text-align: justify;">
        A continuación, se presenta la matriz de correlación, la cual permite identificar la intensidad y dirección de las relaciones entre las variables numéricas del dataset. Este análisis resulta útil para detectar posibles asociaciones relevantes que podrían influir en el modelado posterior.
    </div>
    """, unsafe_allow_html=True)

# Copia del DataFrame y codificación
df_temp = df.copy()
df_temp['Diagnóstico'] = df_temp['Diagnóstico'].map({'Benigno': 0, 'Maligno': 1})

# Filtrado numérico y cálculo de correlación
df_numericas = df_temp.select_dtypes(include=[float, int])
matriz_correlacion = df_numericas.corr()

# Forzamos la diagonal a 1
np.fill_diagonal(matriz_correlacion.values, 1)

# Heatmap sin anotaciones
fig_corr = go.Figure(
    data=go.Heatmap(
        z=matriz_correlacion.values,
        x=matriz_correlacion.columns,
        y=matriz_correlacion.index,
        colorscale='RdYlBu_r',
        zmin=-1, zmax=1,
        showscale=True
    )
)

fig_corr.update_layout(
    width=1000,
    height=600,
    margin=dict(l=100, r=20, t=30, b=30),
    xaxis=dict(tickangle=45),
    yaxis=dict(autorange='reversed')  # opcional: mantiene orden original
)

st.plotly_chart(fig_corr, use_container_width=False)




st.markdown("""
## 🔍 Análisis de Correlaciones

El siguiente mapa de calor representa las correlaciones entre las variables del conjunto de datos, incluyendo la variable objetivo **`Diagnóstico`**.

### 📌 Conclusiones principales

---

### 🔹 1. Fuertes correlaciones entre ciertas variables
- Variables como `radius_mean`, `perimeter_mean` y `area_mean` muestran una **fuerte correlación positiva** entre sí.
- Este patrón también se repite en sus versiones `_worst`: `radius_worst`, `perimeter_worst`, `area_worst`.
- Es esperable, ya que están relacionadas geométricamente: un mayor radio implica mayor perímetro y mayor área.

---

### 🔹 2. Posible multicolinealidad
- Se observan correlaciones altas entre variables similares medidas en distintas etapas (por ejemplo, `radius_mean`, `radius_se`, `radius_worst`).
- Esta redundancia sugiere **potencial multicolinealidad**, que puede afectar negativamente a algunos modelos como la regresión logística.
- Técnicas como **PCA (Análisis de Componentes Principales)** o métodos de selección de variables pueden ayudar a mitigar este problema.

---

### 🔹 3. Correlación con el `Diagnóstico`
- Algunas variables tienen una **correlación positiva clara con el diagnóstico maligno**:
  - `concave points_mean`, `concavity_mean`, `radius_mean`, `perimeter_mean`, `area_mean`
  - También sus equivalentes `_worst` y algunas `_se`.
- Esto sugiere que a medida que aumentan estos valores, **es más probable que el diagnóstico sea maligno**.
- Estas variables son buenas candidatas para modelos de clasificación.

---

### 🔹 4. Variables menos correlacionadas
- Variables como `fractal_dimension_mean`, `fractal_dimension_se`, y `symmetry_mean` presentan **baja o nula correlación con el diagnóstico**.
- Aunque esto puede sugerir menor relevancia, **no se deben descartar sin una evaluación más profunda**, ya que algunas variables pueden tener relaciones no lineales con el diagnóstico.

---
""")
