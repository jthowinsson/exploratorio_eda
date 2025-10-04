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