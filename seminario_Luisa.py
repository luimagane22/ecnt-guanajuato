"""
seminario_Luisa.py

Requisitos:
    Ver requirements.txt. Las dependencias principales son:
    streamlit, pandas, numpy, scikit-learn, plotly, folium,
    statsmodels (opcional, para series de tiempo), pyyaml, reportlablib.

Uso:
    python3 -m streamlit run seminario_Luisa.py
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================

import io
import json
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Tuple

from normalizador_sis2024 import (
    normalizar_multiples_excels,
    reporte_normalizacion,
)
from score_riesgo import calcular_score_df, render_score_paciente, render_tabla_riesgo

from seguimiento_pacientes import render_panel_seguimiento

from reporte_pdf import render_seccion_pdf

from alertas import procesar_alertas

from conclusiones import render_conclusiones

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.express as px 
import yaml

import folium
from streamlit_folium import st_folium

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    _HAVE_STATSMODELS = True
except Exception:
    _HAVE_STATSMODELS = False

# =============================================================================
# CONFIGURACION INICIAL DE LA APLICACION STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Guanajuato", #Titulo que aparece en la pestana del navegador
    layout="wide", #Utiliza todo el ancho disponible de la ventana
    page_icon=" " #Icono de la pestana del navegador
)

# =============================================================================
# SECCION 1: UTILIDADES PARA CARGA Y LECTURA DE ARCHIVOS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_excel(path: str) -> pd.DataFrame:
    """
    Lee un archivo Excel desde una ruta local y lo regresa como DataFrame.

    El decorador @st.cache_data evita releer el archivo en cada interaccion
    del usuario, lo que mejora significativamente el rendimiento de la app.

    Parametros:
        path (str): Ruta relativa o absoluta al archivo .xlsx o .xls.

    Retorna:
        pd.DataFrame: Contenido del Excel como tabla de datos.
    """
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    """
    Carga un archivo GeoJSON desde disco y lo regresa como diccionario de Python.

    El GeoJSON contiene los limites geograficos de los municipios de Guanajuato
    y se utiliza para construir el mapa coropletico de distribucion de pacientes.

    Parametros:
        path (str): Ruta al archivo .geojson.

    Retorna:
        dict: Estructura GeoJSON con features (municipios) y sus geometrias.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_costs(path: str) -> dict:
    """
    Carga el archivo YAML de costos en salud y lo regresa como diccionario.

    El YAML define los costos unitarios por enfermedad (consultas, medicamentos,
    eventos adversos), las tasas economicas (inflacion, descuento) y los
    parametros de adherencia terapeutica utilizados en la proyeccion presupuestal.

    Parametros:
        path (str): Ruta al archivo costos.yaml.

    Retorna:
        dict: Parametros economicos y de costos en salud estructurados por enfermedad.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# SECCION 2: VALIDACION DE ESQUEMA DE DATOS
# =============================================================================

def ensure_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verifica que el DataFrame contenga todas las columnas clinicas requeridas
    para el funcionamiento correcto del panel de vigilancia.

    Las columnas se dividen en cuatro grupos:
      - Demograficas: id_paciente, edad, sexo, municipio, entidad, origen_indigena
      - Enfermedades principales: diabetes, hipertension, dislipidemia, obesidad
      - Antecedentes familiares (prefijo af_): cardiovascular, HTA, diabetes,
        dislipidemias, obesidad, cerebrovascular
      - Antecedentes personales (prefijo ap_): tabaquismo, alcoholismo, sedentarismo,
        diabetes gestacional, postmenopausia, sobrepeso, cardiovascular, VIH,
        cerebrovascular, tuberculosis
      - Parametros clinicos: glucosa, HbA1c, presion arterial, peso, IMC, cintura,
        perfil lipidico (colesterol total, LDL, HDL, trigliceridos)

    Parametros:
        df (pd.DataFrame): DataFrame a validar.

    Retorna:
        Tuple[bool, List[str]]:
          - True si todas las columnas estan presentes, False en caso contrario.
          - Lista con los nombres de las columnas faltantes (vacia si no hay ninguna).
    """
    required = [
        "id_paciente","edad","sexo","municipio","entidad","origen_indigena",
        "diabetes","hipertension","dislipidemia","obesidad",
        "af_enf_cardiovascular", "af_hta", "af_diabetes", "af_dislipidemias",
        "af_obesidad", "af_enf_cerebrovascular",
        "ap_fumador","ap_alcoholismo","ap_sedentarismo", "ap_diabetes_gestacional",
        "ap_postMenopausia", "ap_sobrepeso", "ap_enf_cardiovascular", "ap_vih",
        "ap_enf_cerebrovascular", "ap_tuberculosis",
        "glucosa_ayunas","hba1c_pct","presion_sistolica","presion_diastolica",
        "peso_kg","imc","cintura_cm","col_total","ldl","hdl","trigliceridos"
    ]
    # Identificamos cuales columnas requeridas no existen en el DataFrame recibido
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


# =============================================================================
# SECCION 3: GENERACION DE DATOS SINTETICOS
# =============================================================================

def synthetic_patients(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame con datos sinteticos de pacientes del estado de Guanajuato.

    Se utiliza como fuente de datos de respaldo cuando no existe el archivo
    data/pacientes.xlsx, lo que permite ejecutar y evaluar la aplicacion
    sin necesidad de contar con datos reales.

    Las variables numericas se generan con distribuciones normales cuyos
    parametros (media y desviacion estandar) intentan aproximarse a valores
    clinicamente plausibles. Las enfermedades se simulan como variables
    binarias cuya probabilidad depende de la edad y el IMC del paciente,
    reflejando de manera simplificada las correlaciones epidemiologicas conocidas.

    Nota: Los datos generados NO deben usarse para analisis epidemiologico real.

    Parametros:
        n    (int): Numero de pacientes a generar. Por defecto 1500.
        seed (int): Semilla del generador de numeros aleatorios para reproducibilidad.

    Retorna:
        pd.DataFrame: DataFrame con todas las columnas del esquema requerido.
    """
    rng = np.random.default_rng(seed)

    # Catalogo de los 46 municipios del estado de Guanajuato
    municipios = [
        "Abasolo","Acambaro","San Miguel de Allende","Apaseo el Alto",
        "Apaseo el Grande","Atarjea","Celaya","Manuel Doblado","Comonfort",
        "Coroneo","Cortazar","Cueramaro","Doctor Mora","Dolores Hidalgo",
        "Guanajuato", "Huanimaro", "Irapuato", "Jaral del Progreso", "Jerecuaro",
        "Leon", "Moroleon", "Ocampo", "Penjamo", "Pueblo Nuevo",
        "Purisima del Rincon", "Romita", "Salamanca", "Salvatierra",
        "San Diego de la Union", "San Felipe", "San Francisco del Rincon",
        "San Jose Iturbide", "San Luis de la Paz", "Santa Catarina",
        "Santa Cruz de Juventino Rosas", "Santiago Maravatio",
        "Silao de la Victoria", "Tarandacuo", "Tarimoro", "Tierra Blanca",
        "Uriangato", "Valle de Santiago", "Victoria", "Villagran", "Xichu", "Yuriria"
    ]
    entidades = ["Guanajuato"] * len(municipios)

    # Distribucion de sexo: 55% femenino / 45% masculino (aproximacion al censo de Guanajuato)
    sexo = rng.choice(["F","M"], size=n, p=[0.55, 0.45])

    # Edad uniforme entre 18 y 89 anos
    edad = rng.integers(18, 90, size=n)

    # Asignacion aleatoria de municipio de residencia
    municipio = rng.choice(municipios, size=n)
    entidad = ["Guanajuato"] * n

    # Origen indigena: 10% de probabilidad, refleja la proporcion aproximada estatal
    origen_indigena = rng.choice([0, 1], size=n, p=[0.9, 0.1])

    # -------------------------------------------------------------------------
    # Variables antropometricas y enfermedades principales
    # -------------------------------------------------------------------------

    # IMC distribucion normal con media 28 kg/m^2, recortada a limites fisiologicos
    imc = rng.normal(28, 5, size=n).clip(16, 60)

    # Obesidad se define como IMC >= 30 (criterio OMS)
    obesidad = (imc >= 30).astype(int)

    # Diabetes probabilidad creciente con la edad y el IMC (modelo simplificado)
    diabetes_prob = np.clip((edad - 35) / 50 + (imc - 25) / 20, 0, 0.85)
    diabetes = (rng.random(n) < diabetes_prob).astype(int)

    # Hipertension probabilidad ligada a la edad y el IMC
    hipertension_prob = np.clip((edad - 30) / 45 + (imc - 27) / 25, 0, 0.9)
    hipertension = (rng.random(n) < hipertension_prob).astype(int)

    # Dislipidemia probabilidad proporcional al exceso de peso
    dislipidemia_prob = np.clip(0.25 + (imc - 25) / 25, 0, 0.8)
    dislipidemia = (rng.random(n) < dislipidemia_prob).astype(int)

    # -------------------------------------------------------------------------
    # Antecedentes familiares
    # Probabilidades basadas en prevalencias reportadas en la literatura
    # -------------------------------------------------------------------------
    af_enf_cardiovascular = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    af_hta = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    af_diabetes = rng.choice([0, 1], size=n, p=[0.65, 0.35])
    af_dislipidemias = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    af_obesidad = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    af_enf_cerebrovascular = rng.choice([0, 1], size=n, p=[0.65, 0.35])

    # -------------------------------------------------------------------------
    # Antecedentes personales
    # -------------------------------------------------------------------------
    ap_fumador = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    ap_alcoholismo = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    ap_sedentarismo = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    ap_diabetes_gestacional = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    ap_postMenopausia = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    ap_sobrepeso = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    ap_enf_cardiovascular = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    ap_vih = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    ap_enf_cerebrovascular = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    ap_tuberculosis = rng.choice([0, 1], size=n, p=[0.6, 0.4])

    # -------------------------------------------------------------------------
    # Parametros biometricos y de laboratorio
    # Los pacientes con la enfermedad respectiva tienen valores desplazados
    # hacia rangos patologicos, simulando un control deficiente.
    # -------------------------------------------------------------------------

    # Glucosa en ayunas (mg/dL) media de 90 en no diabeticos, ~140 en diabeticos
    glucosa_ayunas = rng.normal(
        90 + 50 * diabetes, 15 + 10 * diabetes
    ).clip(60, 350).round(1)

    # HbA1c (%): indicador de control glucemico a largo plazo
    # Se eleva en diabeticos y cuando la glucosa supera 180 mg/dL
    hba1c_pct = (
        rng.normal(5.4, 0.4, n) + 2.2 * diabetes + 0.3 * (glucosa_ayunas > 180)
    ).clip(4.5, 14).round(2)

    # Presion arterial (mmHg): sistolica y diastolica con incremento en hipertensos
    presion_sistolica  = rng.normal(120 + 25 * hipertension, 12).clip(85, 220).round(0)
    presion_diastolica = rng.normal(78 + 15 * hipertension, 8).clip(50, 130).round(0)

    # Peso y medidas corporales
    peso_kg    = rng.normal(75 + 8 * (imc > 30), 15, n).clip(40, 180).round(1)
    cintura_cm = rng.normal(95 + 12 * (imc > 30), 10, n).clip(55, 180).round(1)

    # Perfil lipidico (mg/dL): LDL calculado como fraccion del colesterol total
    col_total    = rng.normal(180 + 40 * dislipidemia, 25).clip(90, 350).round(0)
    ldl          = (col_total * 0.6 + rng.normal(0, 10, n)).clip(40, 250).round(0)
    hdl          = rng.normal(50 - 8 * dislipidemia, 10).clip(20, 100).round(0)
    trigliceridos = rng.normal(140 + 80 * dislipidemia, 40).clip(50, 600).round(0)

    # Construimos el DataFrame final ensamblando todas las variables generadas
    df = pd.DataFrame({
        # Identificador unico en formato "P100000", "P100001", ...
        "id_paciente": [f"P{100000 + i}" for i in range(n)],
        "edad": edad,
        "sexo": sexo,
        "municipio": municipio,
        "entidad": entidad,
        "origen_indigena": origen_indigena.astype(int),

        # Enfermedades principales (variable binaria: 1 = presente, 0 = ausente)
        "diabetes":      diabetes,
        "hipertension":  hipertension,
        "dislipidemia":  dislipidemia,
        "obesidad":      obesidad,

        # Antecedentes familiares
        "af_enf_cardiovascular":  af_enf_cardiovascular,
        "af_hta":                 af_hta,
        "af_diabetes":            af_diabetes,
        "af_dislipidemias":       af_dislipidemias,
        "af_obesidad":            af_obesidad,
        "af_enf_cerebrovascular": af_enf_cerebrovascular,

        # Antecedentes personales
        "ap_fumador":               ap_fumador,
        "ap_alcoholismo":           ap_alcoholismo,
        "ap_sedentarismo":          ap_sedentarismo,
        "ap_diabetes_gestacional":  ap_diabetes_gestacional,
        "ap_postMenopausia":        ap_postMenopausia,
        "ap_sobrepeso":             ap_sobrepeso,
        "ap_enf_cardiovascular":    ap_enf_cardiovascular,
        "ap_vih":                   ap_vih,
        "ap_enf_cerebrovascular":   ap_enf_cerebrovascular,
        "ap_tuberculosis":          ap_tuberculosis,

        # Parametros clinicos y biometricos
        "glucosa_ayunas":       glucosa_ayunas,
        "hba1c_pct":            hba1c_pct,
        "presion_sistolica":    presion_sistolica,
        "presion_diastolica":   presion_diastolica,
        "peso_kg":              peso_kg,
        "imc":                  imc.round(1),
        "cintura_cm":           cintura_cm,
        "col_total":            col_total,
        "ldl":                  ldl,
        "hdl":                  hdl,
        "trigliceridos":        trigliceridos
    })
    return df


# =============================================================================
# SECCION 4: UTILIDADES DE EXPORTACION Y DESCARGA
# =============================================================================

def save_bytes_as_download(name: str, buffer: bytes) -> None:
    """
    Renderiza un enlace de descarga HTML en la interfaz de Streamlit
    para que el usuario pueda guardar un archivo en su equipo.

    Convierte los bytes a Base64 y los embebe en un hipervínculo con
    el atributo 'download', lo que funciona en cualquier navegador moderno.

    Parametros:
        name   (str):   Nombre del archivo que se sugiere al guardar (p. ej. "reporte.csv").
        buffer (bytes): Contenido binario del archivo (CSV, Excel, PDF, etc.).
    """
    b64  = base64.b64encode(buffer).decode()
    href = (
        f'<a href="data:application/octet-stream;base64,{b64}" '
        f'download="{name}">Descargar {name}</a>'
    )
    st.markdown(href, unsafe_allow_html=True)


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "datos") -> bytes:
    """
    Serializa un DataFrame como archivo Excel (.xlsx) en memoria y
    retorna los bytes resultantes, listos para ofrecerse como descarga.

    Utiliza un buffer en memoria (BytesIO) para no escribir archivos
    temporales en disco, lo que es mas eficiente en entornos de nube.

    Parametros:
        df         (pd.DataFrame): Tabla de datos a exportar.
        sheet_name (str):          Nombre de la hoja dentro del archivo Excel.

    Retorna:
        bytes: Contenido binario del archivo .xlsx generado.
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


# =============================================================================
# SECCION 5: ANONIMIZACION DE DATOS
# =============================================================================

def anonymize_df(
    df: pd.DataFrame,
    hash_ids: bool = True,
    generalize_age: bool = True,
    drop_quasi: bool = False
) -> pd.DataFrame:
    """
    Aplica tecnicas basicas de anonimizacion al DataFrame antes de mostrarlo
    en pantalla, con el objetivo de reducir el riesgo de reidentificacion
    de los pacientes.

    Las tres estrategias implementadas son:
      1. Hashing de identificadores: sustituye el ID real por los primeros
         12 caracteres de su hash SHA-256, eliminando el identificador directo.
      2. Generalizacion de edad: agrupa la edad exacta en intervalos de 10 anos
         (tecnica de k-anonimato), reduciendo la especificidad del dato.
      3. Supresion de cuasi-identificadores: elimina columnas que, combinadas,
         podrian permitir identificar a un individuo (municipio, peso, cintura).

    Nota: Esta funcion solo afecta la vista en pantalla (df_filt_display).
    Los datos base usados para calculos y analisis no se modifican.

    Parametros:
        df            (pd.DataFrame): DataFrame original a anonimizar.
        hash_ids      (bool): Si True, aplica hash SHA-256 al campo id_paciente.
        generalize_age(bool): Si True, convierte la edad a intervalos categoricos.
        drop_quasi    (bool): Si True, elimina municipio, peso y cintura.

    Retorna:
        pd.DataFrame: Copia anonimizada del DataFrame de entrada.
    """
    df2 = df.copy()

    # Paso 1: Reemplazar el ID real por su huella SHA-256 (primeros 12 caracteres)
    if hash_ids and "id_paciente" in df2.columns:
        df2["id_paciente"] = df2["id_paciente"].astype(str).apply(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:12]
        )

    # Paso 2: Convertir la edad exacta a rangos etarios (k-anonimato)
    if generalize_age and "edad" in df2.columns:
        bins   = [0, 17, 24, 34, 44, 54, 64, 74, 200]
        labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
        df2["edad"] = pd.cut(
            df2["edad"], bins=bins, labels=labels, right=True, include_lowest=True
        )

    # Paso 3: Eliminar cuasi-identificadores geograficos y fisicos
    if drop_quasi:
        for c in ["municipio", "entidad", "cintura_cm", "peso_kg"]:
            if c in df2.columns:
                df2 = df2.drop(columns=[c])

    return df2


# =============================================================================
# SECCION 6: FILTRADO DE DATOS
# =============================================================================

def apply_filters(df: pd.DataFrame, f: Dict) -> pd.DataFrame:
    """
    Aplica sobre el DataFrame los filtros seleccionados por el usuario
    en la barra lateral de la interfaz.

    Los filtros son acumulativos: cada condicion activa reduce aun mas
    el subconjunto de pacientes que se muestra en todas las vistas.

    Filtros disponibles:
      - Sexo biologico (uno o varios valores de la lista: F, M)
      - Origen indigena (Si / No / ambos)
      - Municipio(s) de residencia
      - Rango de edad (valores enteros minimo y maximo)
      - Enfermedades presentes (el paciente debe tener TODAS las seleccionadas)

    Parametros:
        df (pd.DataFrame): DataFrame completo de pacientes.
        f  (Dict):         Diccionario de filtros activos con las claves:
                           'sexo', 'origen', 'municipios', 'rango_edad', 'enfermedades'.

    Retorna:
        pd.DataFrame: Subconjunto del DataFrame que cumple todos los filtros.
    """
    out = df.copy()

    # Filtro por sexo: se aplica solo si hay al menos una opcion seleccionada
    if f.get("sexo"):
        out = out[out["sexo"].isin(f["sexo"])]

    # Filtro por origen indigena: se aplica solo si se eligio exactamente una opcion
    if f.get("origen") is not None and len(f["origen"]) == 1:
        out = out[out["origen_indigena"] == (1 if f["origen"][0] == "Si" else 0)]

    # Filtro por municipio(s) seleccionados
    if f.get("municipios"):
        out = out[out["municipio"].isin(f["municipios"])]

    # Filtro por rango de edad
    if f.get("rango_edad"):
        lo, hi = f["rango_edad"]
        out = out[(out["edad"] >= lo) & (out["edad"] <= hi)]

    # Filtro por enfermedades: operacion AND (el paciente debe tener todas las elegidas)
    if f.get("enfermedades"):
        for enf in f["enfermedades"]:
            out = out[out[enf] == 1]

    return out


# =============================================================================
# SECCION 7: AGRUPAMIENTO DE PACIENTES CON K-MEANS
# =============================================================================

def kmeans_pipeline(df: pd.DataFrame, features: List[str], k: int, random_state: int = 7):
    """
    Aplica un pipeline de Machine Learning para agrupar pacientes con
    perfil clinico similar mediante el algoritmo K-Means.

    El pipeline encadena tres pasos:
      1. Imputacion: rellena los valores faltantes con la mediana de cada columna,
         evitando que los NaN excluyan pacientes del analisis.
      2. Estandarizacion: transforma cada variable a media 0 y desviacion estandar 1
         para que todas contribuyan por igual independientemente de su escala.
      3. K-Means: divide a los pacientes en k grupos minimizando la distancia
         euclidiana intra-cluster.

    Parametros:
        df          (pd.DataFrame): DataFrame de pacientes.
        features    (List[str]):    Nombres de las columnas numericas a utilizar.
        k           (int):          Numero de clusters deseado.
        random_state(int):          Semilla para reproducibilidad del resultado.

    Retorna:
        Tuple:
          - labels    (ndarray): Etiqueta de cluster asignada a cada paciente.
          - centroids (ndarray): Coordenadas de los centroides en el espacio escalado.
          - pipe      (Pipeline): Pipeline completo ya entrenado (reutilizable para predecir).
    """
    X = df[features].copy()

    # Definicion del pipeline con los tres pasos encadenados
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")), # Paso 1: imputacion
        ("sc",  StandardScaler(with_mean=True, with_std=True)), # Paso 2: estandarizacion
        ("km",  KMeans(n_clusters=k, n_init="auto", random_state=random_state))  # Paso 3: K-Means
    ])

    # Entrenamos el pipeline y obtenemos la etiqueta de cluster para cada fila
    labels = pipe.fit_predict(X)
    centroids = pipe.named_steps["km"].cluster_centers_
    return labels, centroids, pipe


def elbow_and_silhouette(
    df: pd.DataFrame,
    features: List[str],
    kmin: int = 2,
    kmax: int = 10,
    random_state: int = 7
):
    """
    Calcula las metricas de validacion interna para distintos valores de k,
    con el fin de apoyar la seleccion del numero optimo de clusters.

    Se calculan dos criterios complementarios:
      - Inercia (metodo del codo): suma de las distancias cuadradas de cada
        punto a su centroide. A medida que k crece la inercia disminuye; el
        punto donde la reduccion se aplana sugiere el k optimo.
      - Coeficiente de Silhouette: mide que tan similar es un punto a su propio
        cluster comparado con los demas. Valores cercanos a 1 indican clusters
        bien definidos; valores cercanos a 0 indican solapamiento.

    Parametros:
        df          (pd.DataFrame): DataFrame con los datos de los pacientes.
        features    (List[str]):    Columnas numericas a utilizar en el analisis.
        kmin        (int):          Valor minimo de k a evaluar (default 2).
        kmax        (int):          Valor maximo de k a evaluar (default 10).
        random_state(int):          Semilla para reproducibilidad.

    Retorna:
        Tuple[List[int], List[float], List[float]]:
          - ks:       Lista de valores de k evaluados.
          - inertias: Inercia correspondiente a cada k.
          - sils:     Silhouette correspondiente a cada k (NaN si no se pudo calcular).
    """
    X = df[features].copy()

    # Preprocesamiento previo: imputacion y estandarizacion
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xp  = sc.fit_transform(imp.fit_transform(X))

    inertias = []
    sils     = []
    ks       = list(range(kmin, kmax + 1))

    for k in ks:
        km  = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        lab = km.fit_predict(Xp)

        # Inercia: suma de distancias cuadradas al centroide asignado
        inertias.append(km.inertia_)

        # Silhouette: requiere al menos 2 clusters con mas de un punto
        try:
            sils.append(silhouette_score(Xp, lab))
        except Exception:
            # Si K-Means genera un cluster con un solo punto, silhouette_score
            # lanza un error; en ese caso registramos NaN para no interrumpir
            sils.append(np.nan)

    return ks, inertias, sils


# =============================================================================
# SECCION 8: SERIES DE TIEMPO Y PROYECCIONES EPIDEMIOLOGICAS
# =============================================================================

def infer_time_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detecta automaticamente las columnas temporales disponibles en el DataFrame.

    Busca en orden de preferencia:
      1. Columna 'fecha' (tipo datetime o convertible).
      2. Columnas 'anio' y 'mes' como enteros.
      3. Solo columna 'anio'.

    Si no se detecta ninguna columna temporal, retorna cadenas vacias.

    Parametros:
        df (pd.DataFrame): DataFrame a inspeccionar.

    Retorna:
        Tuple[str, str]: Nombre de la columna de fecha/ano y de mes, o ('','').
    """
    if "fecha" in df.columns:
        return "fecha", ""
    if "anio" in df.columns and "mes" in df.columns:
        return "anio", "mes"
    if "anio" in df.columns:
        return "anio", ""
    return "", ""


def series_by_year(
    df: pd.DataFrame,
    disease: str,
    date_col: str,
    month_col: str = ""
) -> pd.Series:
    """
    Construye una serie de tiempo anual con el conteo de pacientes unicos
    que presentan una enfermedad determinada.

    Soporta tres formatos de columna temporal:
      - Columna 'fecha' tipo datetime: se extrae el ano.
      - Columnas 'anio' y 'mes' como enteros.
      - Solo columna 'anio'.

    Parametros:
        df        (pd.DataFrame): DataFrame con los registros de pacientes.
        disease   (str):          Nombre de la enfermedad (columna binaria, p.ej. 'diabetes').
        date_col  (str):          Nombre de la columna de fecha o ano.
        month_col (str):          Nombre de la columna de mes (si existe).

    Retorna:
        pd.Series: Serie indexada por ano con el numero de pacientes unicos por ano.
                   Si no hay historico temporal, retorna una Serie vacia.
    """
    # Filtramos unicamente los pacientes con la enfermedad activa
    dff = df[df[disease] == 1].copy()

    # Caso 1: columna de fecha tipo datetime -> extraemos el ano
    if date_col == "fecha":
        dff["anio"] = pd.to_datetime(dff["fecha"]).dt.year
        s = dff.groupby("anio")["id_paciente"].nunique().sort_index()
    # Casos 2 y 3: columna de ano numerica
    elif date_col in ("anio",):
        s = dff.groupby("anio")["id_paciente"].nunique().sort_index()
    # Sin datos historicos: se retorna una serie vacia
    else:
        s = pd.Series(dtype=int)

    return s


def forecast_counts(s: pd.Series, horizon_years: List[int]) -> Dict[int, float]:
    """
    Pronostica el numero de pacientes para anos futuros a partir de una
    serie de tiempo historica.

    Estrategia de pronostico (en orden de prioridad):
      1. Suavizamiento Exponencial (ETS): modelo de tendencia aditiva sin
         componente estacional. Se usa cuando hay al menos 5 observaciones
         y statsmodels esta disponible.
      2. ARIMA(1,1,0): modelo autorregresivo de primer orden con una
         diferenciacion. Se intenta si ETS falla.
      3. Si ambos modelos fallan o hay menos de 5 datos, retorna un
         diccionario vacio y la funcion llamadora aplica una proyeccion
         de crecimiento simple (growth_project).

    Parametros:
        s             (pd.Series):  Serie historica indexada por ano (int -> conteo).
        horizon_years (List[int]):  Lista de anos futuros a pronosticar.

    Retorna:
        Dict[int, float]: Diccionario {ano: conteo_pronosticado}.
                          Puede estar vacio si no fue posible aplicar ningun modelo.
    """
    out = {}

    # Solo aplicamos modelos de series de tiempo si hay suficientes datos
    if len(s) >= 5 and _HAVE_STATSMODELS:
        try:
            # Intento 1: Suavizamiento Exponencial con tendencia aditiva
            model = ExponentialSmoothing(s.values, trend="add", seasonal=None)
            fit   = model.fit(optimized=True)
            last_year = int(s.index.max())

            for h in horizon_years:
                steps = h - last_year
                if steps <= 0:
                    # El ano ya existe en la serie: usamos el valor real
                    out[h] = float(s.get(h, np.nan))
                else:
                    # Proyectamos los pasos necesarios hacia adelante
                    out[h] = float(fit.forecast(steps)[-1])
            return out
        except Exception:
            pass  # Si ETS falla, intentamos ARIMA

        try:
            # Intento 2: ARIMA(1,1,0) — autorregresivo con una diferenciacion
            model = ARIMA(s.values, order=(1, 1, 0))
            fit   = model.fit()
            last_year = int(s.index.max())

            for h in horizon_years:
                steps = h - last_year
                if steps <= 0:
                    out[h] = float(s.get(h, np.nan))
                else:
                    out[h] = float(fit.forecast(steps)[-1])
            return out
        except Exception:
            pass  # Si ARIMA tambien falla, retornamos out vacio

    # Retorno vacio: la logica de crecimiento simple se aplica en el modulo llamador
    return out


def growth_project(current_count: int, years: int, rate: float, mode: str = "compuesto") -> float:
    """
    Proyeccion simple del numero de casos esperados en un horizonte temporal,
    dada una tasa de crecimiento anual.

    Modos disponibles:
      - 'lineal':    N(t) = N + (rate * N * t)
      - 'compuesto': N(t) = N * (1 + rate)^t  (recomendado para mayor realismo)

    Parametros:
        current_count (int):   Numero actual de pacientes con la enfermedad.
        years         (int):   Horizonte de proyeccion en anos.
        rate          (float): Tasa de crecimiento anual (p. ej. 0.03 = 3%).
        mode          (str):   'compuesto' (default) o 'lineal'.

    Retorna:
        float: Estimacion del numero de casos al cabo de 'years' anos.
               Nunca retorna un valor negativo (minimo 0.0).
    """
    if mode == "lineal":
        return max(0.0, current_count + rate * current_count * years)

    # Modo compuesto por defecto (interes compuesto aplicado a poblacion)
    return max(0.0, current_count * ((1 + rate) ** years))


# =============================================================================
# SECCION 9: CLASIFICACION DE SEVERIDAD CLINICA
# =============================================================================

def severity_rules(row: pd.Series) -> Dict[str, str]:
    """
    Clasifica la severidad de cada enfermedad cronica para un paciente
    a partir de sus parametros de laboratorio y clinicos.

    Criterios utilizados (simplificados para fines del sistema):
      - Diabetes:     HbA1c < 7% = leve | 7-8.9% = moderada | >= 9% = severa
      - Hipertension: PA < 140/90 = leve | PA < 160/100 = moderada | resto = severa
      - Dislipidemia: LDL < 130 = leve | 130-159 = moderada | >= 160 = severa
      - Obesidad:     IMC 30-34.9 = leve | 35-39.9 = moderada | >= 40 = severa

    ADVERTENCIA: Estos criterios son una aproximacion para fines de demostracion.
    No deben emplearse para decision clinica real.

    Parametros:
        row (pd.Series): Fila del DataFrame correspondiente a un paciente.

    Retorna:
        Dict[str, str]: Diccionario con la severidad por enfermedad presente
                        (p. ej. {'diabetes': 'moderada', 'hipertension': 'leve'}).
    """
    sev = {}

    # Clasificacion de diabetes basada en HbA1c (%)
    if row.get("diabetes", 0) == 1:
        h = row.get("hba1c_pct", np.nan)
        if pd.isna(h):
            sev["diabetes"] = "desconocida"
        elif h < 7:
            sev["diabetes"] = "leve"
        elif h < 9:
            sev["diabetes"] = "moderada"
        else:
            sev["diabetes"] = "severa"

    # Clasificacion de hipertension basada en presion arterial (mmHg)
    if row.get("hipertension", 0) == 1:
        s = row.get("presion_sistolica", np.nan)
        d = row.get("presion_diastolica", np.nan)
        if pd.isna(s) or pd.isna(d):
            sev["hipertension"] = "desconocida"
        elif s < 140 and d < 90:
            sev["hipertension"] = "leve"
        elif s < 160 or d < 100:
            sev["hipertension"] = "moderada"
        else:
            sev["hipertension"] = "severa"

    # Clasificacion de dislipidemia basada en LDL (mg/dL)
    if row.get("dislipidemia", 0) == 1:
        l = row.get("ldl", np.nan)
        if pd.isna(l):
            sev["dislipidemia"] = "desconocida"
        elif l < 130:
            sev["dislipidemia"] = "leve"
        elif l < 160:
            sev["dislipidemia"] = "moderada"
        else:
            sev["dislipidemia"] = "severa"

    # Clasificacion de obesidad basada en IMC (kg/m^2)
    if row.get("obesidad", 0) == 1:
        bmi = row.get("imc", np.nan)
        if pd.isna(bmi):
            sev["obesidad"] = "desconocida"
        elif bmi < 35:
            sev["obesidad"] = "leve"
        elif bmi < 40:
            sev["obesidad"] = "moderada"
        else:
            sev["obesidad"] = "severa"

    return sev


# =============================================================================
# SECCION 10: CALCULO DE COSTOS EN SALUD
# =============================================================================

def costo_paciente(costos: dict, enfermedad: str, severidad: str) -> float:
    """
    Estima el costo anual de atencion por paciente para una enfermedad
    cronica en un nivel de severidad determinado.

    Lee los parametros del archivo costos.yaml, que debe tener la siguiente
    estructura por enfermedad:

        <enfermedad>:
          severidad:
            <nivel>:             # leve | moderada | severa
              q:                 # cantidad de recursos consumidos por tipo
                consultas_mes: N
                medicamento_mes: N
              eventos:           # complicaciones con probabilidad y costo
                <nombre>:
                  p: 0.05        # probabilidad anual del evento
                  k: 15000       # costo del evento en pesos

        recursos:                # costos unitarios
          consultas_mes:
            u: 250
          ...

        adherencia:
          alpha: 0.9             # factor de ajuste de frecuencia de recursos
          beta: 1.1              # factor de ajuste de costo de eventos

        economia:
          inflacion_g: 0.05
          descuento_r: 0.03

    El costo total se calcula como:
      Costo = sum(recursos * alpha) + sum(p_evento * k_evento * beta)

    Parametros:
        costos     (dict): Diccionario cargado desde costos.yaml.
        enfermedad (str):  Nombre de la enfermedad ('diabetes', 'hipertension', etc.).
        severidad  (str):  Nivel de severidad ('leve', 'moderada', 'severa').

    Retorna:
        float: Costo anual estimado en pesos. Retorna 0.0 si la enfermedad
               no existe en el YAML o no tiene datos para esa severidad.
    """
    # Verificamos que la enfermedad este definida en el YAML
    if enfermedad not in costos:
        return 0.0
    enf_data = costos[enfermedad]

    # Parametros globales de recursos y adherencia
    recursos   = costos.get("recursos", {})
    adherencia = costos.get("adherencia", {"alpha": 1.0, "beta": 1.0})
    alpha      = adherencia.get("alpha", 1.0)
    beta       = adherencia.get("beta", 1.0)

    # Estructura de costos para el nivel de severidad solicitado
    sev     = enf_data.get("severidad", {}).get(severidad, {})
    q       = sev.get("q", {})        # Cantidades de recursos
    eventos = sev.get("eventos", {})  # Eventos adversos con probabilidad y costo

    total = 0.0

    # Componente 1: costo de recursos de atencion regular (consultas, medicamentos)
    for nombre, cantidad in q.items():
        if nombre in recursos:
            costo_unit = recursos[nombre]["u"]
            # Ajuste de adherencia: los recursos de uso mensual se reducen por el factor alpha
            if "mes" in nombre:
                cantidad *= alpha
            total += costo_unit * cantidad

    # Componente 2: valor esperado de eventos adversos (probabilidad x costo x beta)
    for ev, info in eventos.items():
        p = info.get("p", 0)   # Probabilidad anual del evento
        k = info.get("k", 0)   # Costo del evento en pesos
        total += p * k * beta

    return total


# =============================================================================
# SECCION 11: INTERPRETACION DE TRAYECTORIAS CLINICAS
# =============================================================================

def describe_trajectory(series: pd.Series, cfg: Dict) -> str:
    """
    Genera una descripcion textual de la evolucion de un indicador clinico
    a lo largo del tiempo para un paciente individual.

    La funcion evalua:
      1. La tendencia general: si el valor sube, baja o se mantiene estable
         respecto a la primera medicion registrada.
      2. Si el ultimo valor se encuentra dentro, por debajo o por encima del
         rango clinico objetivo definido en METRIC_CONFIG.

    Parametros:
        series (pd.Series): Valores del indicador ordenados cronologicamente.
        cfg    (Dict):      Configuracion del indicador con las claves:
                              - 'tolerance' (float): diferencia maxima para considerar estable.
                              - 'unit'      (str):   unidad de medicion para el texto.
                              - 'good_range'(tuple): (min, max) del rango saludable (opcional).

    Retorna:
        str: Descripcion en lenguaje natural de la trayectoria del indicador.
             Retorna un mensaje indicativo si no hay mediciones disponibles.
    """
    # Eliminamos valores faltantes para evitar errores de calculo
    s = series.dropna()
    if s.empty:
        return "No hay mediciones registradas."

    s      = s.astype(float)
    first  = s.iloc[0]   # Primera medicion del historial
    last   = s.iloc[-1]  # Medicion mas reciente
    diff   = last - first

    tol    = cfg.get("tolerance", 0.0)  # Umbral para considerar estabilidad
    unidad = cfg.get("unit", "")

    # Determinacion de tendencia segun la variacion total
    if abs(diff) <= tol:
        tendencia = "se ha mantenido relativamente estable"
    elif diff < 0:
        tendencia = "muestra una tendencia a la baja (mejoria)"
    else:
        tendencia = "muestra una tendencia al alza (empeoramiento)"

    # Evaluacion del ultimo valor contra el rango clinico objetivo, si esta definido
    good = cfg.get("good_range", None)
    if good:
        lo, hi = good
        if last < lo:
            estado = "por debajo del objetivo"
        elif last <= hi:
            estado = "dentro del objetivo (controlada)"
        else:
            estado = "por encima del objetivo"
        extra = f" El valor mas reciente esta {estado}."
    else:
        extra = ""

    return (
        f"Paso de {first:.1f} a {last:.1f} {unidad}; "
        f"a lo largo del tiempo {tendencia}.{extra}"
    )


# =============================================================================
# SECCION 12: CONFIGURACION DE INDICADORES CLINICOS
# =============================================================================
# Se usa en la pagina de paciente individual para interpretar el historial.
METRIC_CONFIG = {
    "glucosa_ayunas": {
        "label":      "Glucosa en ayunas",
        "unit":       "mg/dL",
        "tolerance":  10.0,
        "good_range": (70, 130) # Criterio: glucosa en ayunas normal segun ADA
    },
    "hba1c_pct": {
        "label":      "HbA1c",
        "unit":       "%",
        "tolerance":  0.3,
        "good_range": (0, 7.0) # Objetivo de control glucemico para diabeticos (ADA)
    },
    "peso_kg": {
        "label":     "Peso",
        "unit":      "kg",
        "tolerance": 2.0,
        # Sin rango objetivo: el peso saludable depende de la talla de cada paciente
    },
    "imc": {
        "label":      "IMC",
        "unit":       "kg/m2",
        "tolerance":  1.0,
        "good_range": (18.5, 24.9) # Clasificacion OMS: normopeso
    },
    "presion_sistolica": {
        "label":      "Presion sistolica",
        "unit":       "mmHg",
        "tolerance":  5.0,
        "good_range": (90, 129) # Rango normal segun JNC-8
    },
    "presion_diastolica": {
        "label":      "Presion diastolica",
        "unit":       "mmHg",
        "tolerance":  5.0,
        "good_range": (60, 79)
    },
    "col_total": {
        "label":      "Colesterol total",
        "unit":       "mg/dL",
        "tolerance":  10.0,
        "good_range": (0, 200) # Objetivo: < 200 mg/dL para adultos
    },
    "ldl": {
        "label":      "LDL",
        "unit":       "mg/dL",
        "tolerance":  10.0,
        "good_range": (0, 100) # Objetivo estricto para pacientes de alto riesgo
    },
    "hdl": {
        "label":      "HDL",
        "unit":       "mg/dL",
        "tolerance":  5.0,
        # Sin good_range: para HDL, mayor es mejor (no hay maximo patologico universal)
    },
    "trigliceridos": {
        "label":      "Trigliceridos",
        "unit":       "mg/dL",
        "tolerance":  15.0,
        "good_range": (0, 150)  # Objetivo: < 150 mg/dL (OMS)
    },
    "cintura_cm": {
        "label":     "Circunferencia de cintura",
        "unit":      "cm",
        "tolerance": 2.0,
        # Sin good_range: el umbral de riesgo varia por sexo (>90 cm H / >80 cm M)
    },
}


# =============================================================================
# SECCION 13: PROYECCION PRESUPUESTAL
# =============================================================================

def compute_budget(df: pd.DataFrame, costos: dict, anos: List[int] = [5, 10, 15, 20]) -> pd.DataFrame:
    """
    Calcula el presupuesto proyectado para la atencion de pacientes cronicos
    en distintos horizontes temporales, desagregado por enfermedad y severidad.

    Metodologia:
      Para cada combinacion (enfermedad, severidad, horizonte):
        costo_proyectado = costo_unitario * factor_economico * N_pacientes

      Donde:
        factor_economico = (1 + inflacion)^t / (1 + descuento)^t

      La inflacion y la tasa de descuento se leen del YAML en la clave 'economia'.
      El numero de pacientes (N) corresponde a los del DataFrame actual con esa
      enfermedad activa (columna == 1).

    Parametros:
        df     (pd.DataFrame): DataFrame de pacientes con enfermedades como columnas binarias.
        costos (dict):         Diccionario de costos cargado desde costos.yaml.
        anos   (List[int]):    Horizontes de proyeccion en anos. Por defecto [5, 10, 15, 20].

    Retorna:
        pd.DataFrame: Tabla con columnas:
          enfermedad | severidad | anos | pacientes | costo_unitario | costo_total
    """
    # Parametros macroeconomicos del YAML
    inflacion = costos.get("economia", {}).get("inflacion_g", 0.0)
    descuento = costos.get("economia", {}).get("descuento_r", 0.0)

    enfermedades = ["diabetes", "hipertension", "dislipidemia", "obesidad"]
    severidades  = ["leve", "moderada", "severa"]
    resultados   = []

    for enf in enfermedades:
        for sev in severidades:
            # Contamos los pacientes con esa enfermedad activa
            mask = df[enf] == 1
            N    = mask.sum()
            if N == 0:
                continue   # No hay pacientes con esta combinacion, se omite

            # Costo base anual por paciente para esa enfermedad y severidad
            costo_base = costo_paciente(costos, enf, sev)
            if costo_base == 0:
                continue   # El YAML no tiene datos para esta combinacion

            # Proyeccion a cada horizonte temporal con ajuste economico
            for t in anos:
                # El factor combina el efecto de la inflacion (encarece el servicio)
                # con el descuento financiero (reduce el valor presente)
                factor     = ((1 + inflacion) ** t) / ((1 + descuento) ** t)
                costo_proj = costo_base * factor * N

                resultados.append({
                    "enfermedad":    enf,
                    "severidad":     sev,
                    "anos":          t,
                    "pacientes":     int(N),
                    "costo_unitario": round(costo_base, 2),
                    "costo_total":   round(costo_proj, 2)
                })

    return pd.DataFrame(resultados)


# =============================================================================
# SECCION 14: DATOS DE REFERENCIA — POBLACION MUNICIPAL 2018
# =============================================================================

# Poblacion aproximada de cada municipio de Guanajuato segun el censo 2018.
# Se usa como denominador para calcular tasas de prevalencia por 1,000 habitantes
# en el mapa coropletico (pacientes registrados / poblacion * 1000).
poblacion_2018 = {
    "Abasolo": 94097, "Acambaro": 117476, "San Miguel de Allende": 179331,
    "Apaseo el Alto": 71115, "Apaseo el Grande": 96729, "Atarjea": 5521,
    "Celaya": 521300, "Comonfort": 85836, "Coroneo": 12790, "Cortazar": 99730,
    "Cueramaro": 29515, "Doctor Mora": 25271, "Dolores Hidalgo": 158969,
    "Guanajuato": 194067, "Huanimaro": 22281, "Irapuato": 598542,
    "Jaral del Progreso": 40329, "Jerecuaro": 51217, "Leon": 1645986,
    "Moroleon": 53575, "Manuel Doblado": 40131, "Ocampo": 24420,
    "Penjamo": 158343, "Pueblo Nuevo": 12359, "Purisima del Rincon": 83245,
    "Romita": 61781, "Salamanca": 288173, "Salvatierra": 105505,
    "San Diego de la Union": 40955, "San Felipe": 116647,
    "San Francisco del Rincon": 124314, "San Jose Iturbide": 81597,
    "San Luis de la Paz": 125741, "Santa Catarina": 5499,
    "Santa Cruz de Juventino Rosas": 87119, "Santiago Maravatio": 7190,
    "Silao de la Victoria": 196430, "Tarimoro": 36335, "Tierra Blanca": 19858,
    "Uriangato": 65975, "Valle de Santiago": 150820, "Victoria": 21240,
    "Villagran": 61800, "Xichu": 12132, "Yuriria": 73189
}


# =============================================================================
# SECCION 15: RUTAS DE ARCHIVOS POR DEFECTO
# =============================================================================

# Rutas relativas a los archivos de datos del proyecto.
# Se espera que exista una carpeta 'data/' en el mismo directorio que este script.
DATA_XLSX   = "data/pacientes.xlsx"  # Datos de pacientes en formato propio o SIS-2024
GTO_GEOJSON = "data/guanajuato_municipios.geojson"  # Limites municipales para el mapa
COSTS_YAML  = "data/costos.yaml" # Parametros economicos y de costos en salud


# =============================================================================
# SECCION 16: BARRA LATERAL — NAVEGACION, CARGA DE DATOS Y FILTROS
# =============================================================================

# -----------------------------------------------------------------------------
# 16.1 NAVEGACION PRINCIPAL
# Siempre visible al inicio de la barra lateral para acceso rapido a secciones.
# -----------------------------------------------------------------------------

st.sidebar.header("Navegacion")
pagina_sidebar = st.sidebar.radio(
    "Ir a seccion:",
    options=[
        "Inicio y conclusiones",
        "Alertas clinicas",
        "Paciente individual",
        "Seguimiento y riesgo",
        "Mapa por municipio",
        "Registro de pacientes",
        "Grupos de pacientes",
        "Costos proyectados",
        "Reporte PDF",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# 16.2 CARGA DE ARCHIVOS EXCEL
# Permite subir uno o varios archivos con datos de pacientes.
# El normalizador los convierte automaticamente al esquema unificado del sistema.
# -----------------------------------------------------------------------------

st.sidebar.header("Cargar datos de pacientes")

uploaded_files = st.sidebar.file_uploader(
    "Sube uno o varios archivos Excel con datos de pacientes. "
    "El sistema acepta tanto el formato propio como el formato oficial SIS-2024 de la SSA.",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Intentamos leer cada archivo y registramos los que fallen
    archivos_y_dfs  = []
    errores_lectura = []

    for f in uploaded_files:
        try:
            df_crudo = pd.read_excel(f)
            archivos_y_dfs.append((getattr(f, "name", "archivo"), df_crudo))
        except Exception as e:
            errores_lectura.append(f"{getattr(f, 'name', 'archivo')}: {e}")

    # Mostramos los errores de lectura sin interrumpir la ejecucion
    for err in errores_lectura:
        st.error(f"No se pudo leer: {err}")

    if not archivos_y_dfs:
        st.error("Ningun archivo pudo leerse correctamente.")
        st.stop()  # Detiene la ejecucion de la app si no hay datos validos

    # Normalizamos todos los archivos al esquema unificado del sistema
    df_raw, reportes = normalizar_multiples_excels(archivos_y_dfs)

    # Mostramos el reporte de normalizacion en un expander de la barra lateral
    with st.sidebar.expander("Ver resultado de la carga", expanded=False):
        for rep in reportes:
            reporte_normalizacion(rep)

else:
    # Si el usuario no subio archivos, intentamos cargar el Excel local de respaldo
    try:
        df_crudo = load_excel(DATA_XLSX)
        st.info("Usando `data/pacientes.xlsx` porque no se subieron archivos.")
        df_raw, _ = normalizar_multiples_excels([("pacientes.xlsx", df_crudo)])
    except Exception:
        # Si tampoco existe el archivo local, generamos datos sinteticos
        st.warning("No se encontro `data/pacientes.xlsx`. Generando datos sinteticos...")
        df_raw = synthetic_patients(1500)

        # Asignamos fechas aleatorias dentro de los ultimos dos anos
        rng2 = np.random.default_rng(99)
        df_raw["__fecha__"] = pd.to_datetime("today") - pd.to_timedelta(
            rng2.integers(0, 730, size=len(df_raw)), unit="D"
        )

        # Intentamos guardar los datos sinteticos como Excel para reutilizarlos
        try:
            import os
            os.makedirs("data", exist_ok=True)
            df_raw.to_excel(DATA_XLSX, index=False)
            st.info("Se guardo un Excel sintetico en `data/pacientes.xlsx`.")
        except Exception:
            pass  # Si no se puede escribir en disco, continuamos sin guardar

# Estandarizacion del identificador de paciente: mayusculas, sin espacios extra
if "id_paciente" in df_raw.columns:
    df_raw["id_paciente"] = (
        df_raw["id_paciente"]
        .astype(str).str.strip().str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )

# Validacion del esquema: se notifica si faltan columnas pero no se interrumpe la app
ok, missing = ensure_columns(df_raw)
if not ok:
    st.warning(
        f"Algunas columnas no se encontraron en los datos y se rellenaron con 0/NaN: "
        f"{missing}. El analisis continua con los datos disponibles."
    )

# Carga del GeoJSON para el mapa; si no existe, el mapa queda deshabilitado
try:
    geojson = load_geojson(GTO_GEOJSON)
except Exception:
    geojson = None

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# 16.3 FILTROS BASICOS DE POBLACION
# Todos los filtros se aplican globalmente a todas las secciones del panel.
# -----------------------------------------------------------------------------

st.sidebar.header("Filtrar poblacion")

min_edad = int(df_raw["edad"].min())
max_edad = int(df_raw["edad"].max())

# Rango de edad: slider de doble extremo entre el minimo y maximo del dataset
rango_edad = st.sidebar.slider(
    "Rango de edad",
    min_value=min_edad, max_value=max_edad,
    value=(min_edad, max_edad), step=1
)

# Seleccion multiple de sexo biologico
sexo_opts = st.sidebar.multiselect(
    "Sexo biologico",
    options=sorted(df_raw["sexo"].dropna().unique().tolist())
)

# Seleccion multiple de municipio de residencia
municipios_opts = st.sidebar.multiselect(
    "Municipio",
    options=sorted(df_raw["municipio"].dropna().unique().tolist())
)

# Filtro por origen indigena (Si / No)
origen_opts = st.sidebar.multiselect(
    "Origen indigena",
    options=["Si", "No"]
)

# Filtro por enfermedades: se muestran solo pacientes con TODAS las seleccionadas (AND logico)
enfs = st.sidebar.multiselect(
    "Filtrar por enfermedad (muestra pacientes con TODAS las seleccionadas)",
    options=["diabetes", "hipertension", "dislipidemia", "obesidad"]
)

# Diccionario que consolida todos los filtros activos para pasarlos a apply_filters()
filters = {
    "rango_edad":   rango_edad,
    "sexo":         sexo_opts,
    "municipios":   municipios_opts,
    "origen":       origen_opts,
    "enfermedades": enfs,
}

# -----------------------------------------------------------------------------
# 16.4 OPCIONES AVANZADAS (expanders colapsables en la barra lateral)
# -----------------------------------------------------------------------------

# Opciones de anonimizacion para proteccion de datos personales
with st.sidebar.expander("Opciones de privacidad", expanded=False):
    hash_ids   = st.checkbox(
        "Proteger identificadores de pacientes", True,
        help="Reemplaza los IDs reales con un codigo cifrado (SHA-256)."
    )
    gen_age    = st.checkbox(
        "Mostrar edad en rangos (no exacta)", False,
        help="Agrupa la edad en intervalos de 10 anos para mayor privacidad."
    )
    drop_quasi = st.checkbox(
        "Ocultar municipio y datos fisicos", False,
        help="Elimina municipio, peso y cintura de la vista en pantalla."
    )

# Guardado y recuperacion de configuraciones de filtros en la sesion activa
with st.sidebar.expander("Guardar filtros", expanded=False):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Guardar", help="Guarda los filtros actuales en la sesion"):
            st.session_state["saved_filters"] = json.dumps(filters)
            st.success("Filtros guardados.")
    with col_s2:
        if st.button("Recuperar", help="Muestra los filtros guardados previamente"):
            if "saved_filters" in st.session_state:
                try:
                    st.info("Filtros guardados:")
                    st.code(st.session_state["saved_filters"], language="json")
                except Exception:
                    st.error("No se pudieron cargar los filtros.")

# Tasas de crecimiento anual configurables por enfermedad, usadas en proyecciones
with st.sidebar.expander("Tasas de crecimiento (proyecciones)", expanded=False):
    rate_diab = st.number_input(
        "Diabetes (ej. 0.03 = 3%)", value=0.03, step=0.005, format="%.3f"
    )
    rate_htn  = st.number_input(
        "Hipertension", value=0.025, step=0.005, format="%.3f"
    )
    rate_dlp  = st.number_input(
        "Dislipidemia", value=0.02, step=0.005, format="%.3f"
    )
    rate_ob   = st.number_input(
        "Obesidad", value=0.03, step=0.005, format="%.3f"
    )
    growth_mode = st.selectbox(
        "Tipo de proyeccion", ["compuesto", "lineal"],
        help="Compuesto: mas realista (interes compuesto). Lineal: crecimiento constante."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Sistema de Vigilancia ECNT · Secretaria de Salud GTO")


# =============================================================================
# SECCION 17: PREPARACION DE DATOS PARA LA VISTA
# =============================================================================

# Aplicamos los filtros sobre df_raw (datos completos y numericos) para que
# el slider de edad funcione correctamente incluso cuando la generalizacion
# de edad esta activada en la vista de pantalla.
df_filt = apply_filters(df_raw, filters)

# Calculamos el score de riesgo de cada paciente y lo agregamos como columna
df_filt = calcular_score_df(df_filt)

# df_filt_display es una copia anonimizada solo para mostrar en pantalla.
# Los calculos internos (clustering, score, seguimiento) siempre usan df_filt.
df_filt_display = anonymize_df(
    df_filt,
    hash_ids=hash_ids,
    generalize_age=gen_age,
    drop_quasi=drop_quasi,
)


# =============================================================================
# SECCION 18: TITULO PRINCIPAL Y KPIs GLOBALES
# =============================================================================

st.title("Sistema de Vigilancia de Enfermedades Cronicas — Guanajuato")
st.caption(
    "Selecciona una seccion en el menu de la izquierda. "
    "Los filtros aplican a todas las vistas."
)

# Fila de metricas principales: conteos de pacientes por categoria
colk1, colk2, colk3, colk4, colk5 = st.columns(5)
with colk1:
    st.metric("Pacientes en el registro", len(df_filt))
with colk2:
    st.metric("Diabetes",      int((df_filt["diabetes"]     == 1).sum()))
with colk3:
    st.metric("Hipertension",  int((df_filt["hipertension"] == 1).sum()))
with colk4:
    st.metric("Dislipidemia",  int((df_filt["dislipidemia"] == 1).sum()))
with colk5:
    st.metric("Obesidad",      int((df_filt["obesidad"]     == 1).sum()))

st.markdown("---")


# =============================================================================
# SECCION 19: ENRUTAMIENTO POR PAGINAS
# Cada bloque if/elif corresponde a una seccion del panel de navegacion.
# =============================================================================

# -----------------------------------------------------------------------------
# PAGINA 1: INICIO Y CONCLUSIONES
# Muestra una descripcion del sistema y las conclusiones epidemiologicas
# generadas automaticamente a partir de los datos filtrados.
# -----------------------------------------------------------------------------
if pagina_sidebar == "Inicio y conclusiones":
    st.markdown("""
    > **Que hace este sistema?** Carga los archivos Excel del registro SIS-2024,
    > analiza automaticamente los indicadores clinicos de cada paciente y genera
    > alertas, reportes y conclusiones en lenguaje natural para apoyar la toma
    > de decisiones del personal de salud en Guanajuato.
    >
    > **Como empezar?** Sube tus archivos Excel en el panel izquierdo y
    > navega por las secciones usando el menu de navegacion.
    """)
    render_conclusiones(df_filt, df_raw)

# -----------------------------------------------------------------------------
# PAGINA 2: ALERTAS CLINICAS
# Lista los pacientes que requieren atencion urgente o seguimiento proximo,
# clasificados automaticamente por el modulo de alertas.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Alertas clinicas":
    st.header("Pacientes que requieren atencion")
    st.caption(
        "El sistema reviso automaticamente los indicadores de todos los pacientes. "
        "Atencion urgente: el medico debe intervenir esta semana. "
        "Seguimiento: programar consulta en el proximo mes."
    )
    procesar_alertas(df_filt)

# -----------------------------------------------------------------------------
# PAGINA 3: PACIENTE INDIVIDUAL
# Permite seleccionar un paciente por ID y visualizar su historial clinico
# completo, la evolucion grafica de sus indicadores y su nivel de riesgo.
# Requiere que los datos tengan una columna de fecha de visita.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Paciente individual":
    st.header("Seguimiento individual de paciente")

    if "id_paciente" not in df_raw.columns:
        st.info(
            "No se puede mostrar el historial porque los datos "
            "no tienen columna de identificador de paciente."
        )
    elif "__fecha__" not in df_raw.columns:
        st.info(
            "Para ver el historial de un paciente, los datos deben tener una columna "
            "de fecha de visita. Verifica que tus archivos Excel incluyan una columna "
            "con la palabra 'fecha', 'anio' o 'mes'."
        )
    else:
        df_hist = df_raw.copy()
        df_hist = df_hist.dropna(subset=["__fecha__"])

        if df_hist.empty:
            st.info("No se encontraron fechas validas en los datos.")
        else:
            # Normalizamos los IDs para que la busqueda no sea sensible a espacios
            df_hist["id_paciente"] = (
                df_hist["id_paciente"]
                .astype(str).str.strip().str.upper()
                .str.replace(r"\s+", " ", regex=True)
            )
            pacientes_ids = sorted(df_hist["id_paciente"].unique().tolist())

            # Selector de paciente: lista ordenada de IDs
            sel_id = st.selectbox(
                "Buscar paciente por ID", pacientes_ids,
                help="Selecciona el identificador del paciente para ver su historial."
            )

            # Filtramos las filas del paciente seleccionado y las ordenamos por fecha
            df_p = df_hist[df_hist["id_paciente"] == sel_id].copy()
            df_p = df_p.sort_values("__fecha__")

            if df_p.empty:
                st.info("No hay registros para ese paciente.")
            else:
                st.subheader(f"Historial clinico — Paciente {sel_id}")

                # Columnas a mostrar en la tabla de historial
                cols_hist = [
                    "__fecha__", "edad", "glucosa_ayunas", "hba1c_pct",
                    "presion_sistolica", "presion_diastolica",
                    "peso_kg", "imc", "col_total", "ldl", "hdl",
                    "trigliceridos", "cintura_cm"
                ]
                # Solo mostramos las columnas que existen en los datos
                cols_hist = [c for c in cols_hist if c in df_p.columns]
                st.dataframe(df_p[cols_hist], use_container_width=True)

                st.markdown("### Evolucion de indicadores clinicos")

                # Graficas de linea para los principales indicadores a lo largo del tiempo
                plot_cols = [
                    c for c in [
                        "glucosa_ayunas", "hba1c_pct", "peso_kg",
                        "presion_sistolica", "presion_diastolica", "imc"
                    ]
                    if c in df_p.columns
                ]
                if plot_cols:
                    st.line_chart(df_p.set_index("__fecha__")[plot_cols])
                else:
                    st.info("No hay suficientes datos numericos para mostrar las graficas.")

                # Score de riesgo calculado con la ultima visita registrada
                st.markdown("### Nivel de riesgo cardiovascular (ultima visita)")
                fila_reciente = df_p.sort_values("__fecha__").iloc[-1]
                render_score_paciente(fila_reciente)

                # Interpretacion textual de la trayectoria de cada indicador
                st.markdown("### Interpretacion del historial clinico")
                for col, cfg in METRIC_CONFIG.items():
                    if col in df_p.columns:
                        texto = describe_trajectory(df_p.sort_values("__fecha__")[col], cfg)
                        st.markdown(f"**{cfg['label']}**: {texto}")

# -----------------------------------------------------------------------------
# PAGINA 4: SEGUIMIENTO Y RIESGO
# Muestra la tabla de priorizacion de pacientes por score de riesgo y el
# panel de pacientes sin consulta reciente o fuera de control.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Seguimiento y riesgo":
    st.header("Priorizacion de pacientes por nivel de riesgo")
    st.caption(
        "Cada paciente recibe una puntuacion de 0 a 100 segun que tan controladas "
        "estan sus enfermedades. A mayor puntuacion, mayor urgencia de atencion. "
        "Bajo riesgo (<30) | Riesgo moderado (30-54) | Riesgo alto (55-74) | Riesgo muy alto (>=75)"
    )
    render_tabla_riesgo(df_filt)
    st.markdown("---")
    st.header("Pacientes sin consulta reciente o sin control")
    render_panel_seguimiento(df_filt)

# -----------------------------------------------------------------------------
# PAGINA 5: MAPA POR MUNICIPIO
# Genera un mapa coropletico con la tasa de pacientes por 1,000 habitantes
# para cada municipio del estado, usando el GeoJSON y los datos de poblacion.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Mapa por municipio":
    st.header("Distribucion geografica por municipio")

    if geojson is None:
        st.info(
            "Para habilitar el mapa, coloca el archivo "
            "`guanajuato_municipios.geojson` dentro de la carpeta `data/`."
        )
    else:
        # Calculamos el numero de pacientes unicos por municipio
        muni_counts = df_filt.groupby("municipio")["id_paciente"].nunique().reset_index()
        muni_counts.columns = ["municipio", "pacientes"]

        # Agregamos la poblacion de referencia y calculamos la tasa por 1,000 hab.
        muni_counts["poblacion_2018"] = muni_counts["municipio"].map(poblacion_2018)
        muni_counts["tasa_por_1000"]  = (
            muni_counts["pacientes"] / muni_counts["poblacion_2018"]
        ) * 1000

        # Detectamos el nombre de la propiedad geografica en el GeoJSON
        prop_keys = []
        try:
            prop_keys = list(geojson["features"][0]["properties"].keys())
        except Exception:
            pass
        prefer   = ["municipio", "NOMGEO", "name", "NOM_MUN", "NOM_MPIO", "mpio", "NOMBRE", "mun_name"]
        key_name = next(
            (k for k in prefer if k in prop_keys),
            prop_keys[0] if prop_keys else "municipio"
        )

        # Construimos el mapa centrado en Guanajuato
        m = folium.Map(location=[21.0, -101.25], zoom_start=7)

        # Capa coropletica: color proporcional a la tasa por 1,000 habitantes
        chor = folium.Choropleth(
            geo_data=geojson,
            name="Tasa por 1000 habitantes",
            data=muni_counts,
            columns=["municipio", "tasa_por_1000"],
            key_on=f"feature.properties.{key_name}",
            fill_color="YlOrRd",      # Amarillo-Naranja-Rojo: mayor tasa = mas rojo
            fill_opacity=0.7,
            line_opacity=0.2,
            nan_fill_color="lightgray",  # Municipios sin datos en gris
            legend_name="Pacientes por 1000 habitantes (poblacion 2018)"
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # Tooltip: muestra el nombre del municipio al pasar el cursor
        try:
            chor.geojson.add_child(
                folium.features.GeoJsonTooltip(
                    fields=[key_name], aliases=["Municipio:"], localize=True
                )
            )
        except Exception as e:
            st.warning(f"No se pudo agregar tooltip al mapa: {e}")

        st_folium(m, use_container_width=True, returned_objects=[])

# -----------------------------------------------------------------------------
# PAGINA 6: REGISTRO DE PACIENTES
# Muestra la tabla completa de pacientes con los filtros activos, con opciones
# de descarga en formato CSV y Excel.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Registro de pacientes":
    st.header("Registro de pacientes")
    st.caption("Vista completa de los datos cargados con los filtros activos.")
    st.dataframe(df_filt_display, use_container_width=True)

    # Botones de descarga: CSV y Excel
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_bytes = df_filt_display.to_csv(index=False).encode("utf-8")
        save_bytes_as_download("pacientes_filtrados.csv", csv_bytes)
    with exp_col2:
        xlsx_bytes = to_excel_bytes(df_filt_display, "pacientes_filtrados")
        save_bytes_as_download("pacientes_filtrados.xlsx", xlsx_bytes)

# -----------------------------------------------------------------------------
# PAGINA 7: GRUPOS DE PACIENTES (CLUSTERING)
# Aplica K-Means sobre indicadores clinicos numericos para identificar
# subgrupos de pacientes con perfiles similares.
# Incluye analisis del codo y Silhouette para seleccion del k optimo.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Grupos de pacientes":
    st.header("Grupos de pacientes con perfil similar")
    st.caption(
        "El sistema identifica grupos de pacientes que comparten caracteristicas "
        "clinicas similares, util para disenar intervenciones dirigidas a cada perfil."
    )

    # Columnas numericas disponibles para el analisis de clustering
    clu_cols_default = [
        "edad", "imc", "glucosa_ayunas", "hba1c_pct",
        "presion_sistolica", "presion_diastolica",
        "col_total", "ldl", "hdl", "trigliceridos", "cintura_cm"
    ]
    clu_cols = [
        c for c in clu_cols_default
        if c in df_filt.columns and pd.api.types.is_numeric_dtype(df_filt[c])
    ]

    if len(clu_cols) < 2:
        st.info("No hay suficientes datos numericos para identificar grupos de pacientes.")
    else:
        st.caption(f"Se analizan {len(clu_cols)} indicadores clinicos para agrupar pacientes.")

        # El usuario define el numero de grupos con el slider
        k = st.slider(
            "En cuantos grupos quieres dividir a los pacientes?",
            min_value=2, max_value=10, value=4, step=1,
            help="Se recomienda entre 3 y 5 grupos."
        )

        # Expander con el analisis tecnico para apoyar la eleccion del k optimo
        with st.expander("Ver analisis tecnico para elegir el numero de grupos"):
            ks, inertias, sils = elbow_and_silhouette(df_filt, clu_cols, kmin=2, kmax=10)
            c1, c2 = st.columns(2)
            with c1:
                # Grafica de inercia: metodo del codo
                st.line_chart(pd.DataFrame({"inertia": inertias}, index=ks))
            with c2:
                # Grafica de Silhouette: calidad de separacion entre clusters
                st.line_chart(pd.DataFrame({"silhouette": sils}, index=ks))

            # Recomendacion automatica: k con mayor coeficiente de Silhouette
            valid = [(k_val, sil) for k_val, sil in zip(ks, sils) if not np.isnan(sil)]
            if valid:
                best_k, best_sil = max(valid, key=lambda t: t[1])
                st.markdown(
                    f"Recomendacion: El numero optimo de grupos es **{best_k}** "
                    f"(indice de calidad: {best_sil:.3f})."
                )
            else:
                st.markdown("No fue posible calcular una recomendacion automatica.")

        # Ejecutamos el pipeline de K-Means con el k seleccionado por el usuario
        labels, centroids, pipe = kmeans_pipeline(df_filt, clu_cols, k=k)
        df_clu = df_filt.copy()
        df_clu["cluster"] = labels   # Agregamos la etiqueta de cluster al DataFrame

        # Vista previa de los primeros 20 pacientes con su cluster asignado
        st.dataframe(df_clu[["id_paciente", "cluster"] + clu_cols].head(20), use_container_width=True)

        # Perfil promedio de cada cluster para facilitar su interpretacion clinica
        prof = df_clu.groupby("cluster")[clu_cols].mean().round(2)
        st.subheader("Caracteristicas promedio de cada grupo")
        st.dataframe(prof, use_container_width=True)

        # Descarga de la asignacion de cluster por paciente
        st.markdown("**Descargar la asignacion de grupo de cada paciente**")
        save_bytes_as_download(
            "asignaciones_cluster.csv",
            df_clu[["id_paciente", "cluster"]].to_csv(index=False).encode("utf-8")
        )

# -----------------------------------------------------------------------------
# PAGINA 8: COSTOS PROYECTADOS
# Muestra la estimacion del costo economico de atencion a la poblacion
# registrada en horizontes de 5, 10, 15 y 20 anos, por enfermedad.
# Requiere el archivo costos.yaml en la carpeta data/.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Costos proyectados":
    st.header("Proyeccion de costos de atencion a futuro")
    st.caption(
        "Estimacion del costo economico de atender a la poblacion registrada "
        "en horizontes de 5, 10, 15 y 20 anos."
    )

    # Intentamos cargar el YAML de costos; si no existe, se informa al usuario
    try:
        with open(COSTS_YAML, "r", encoding="utf-8") as f:
            costos = yaml.safe_load(f)
    except Exception:
        costos = None
        st.info(
            "Para ver las proyecciones de costos, agrega el archivo `costos.yaml` "
            "en la carpeta `data/`. Contacta al administrador del sistema si no tienes este archivo."
        )

    if costos:
        # Calculamos la proyeccion presupuestal y la mostramos en tabla y grafica
        df_costos = compute_budget(df_raw, costos)
        st.subheader("Tabla de costos proyectados por enfermedad")
        st.dataframe(df_costos)

        # Grafica de barras agrupadas: costo total por enfermedad y horizonte temporal
        fig = px.bar(
            df_costos,
            x="anos",
            y="costo_total",
            color="enfermedad",
            barmode="group",
            title="Costo proyectado por enfermedad y horizonte temporal"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Las proyecciones no estan disponibles. "
            "Agrega el archivo `costos.yaml` en la carpeta `data/`."
        )

# -----------------------------------------------------------------------------
# PAGINA 9: REPORTE PDF
# Genera un reporte oficial descargable con los indicadores de salud,
# graficas, lista de pacientes de alto riesgo y proyeccion de costos.
# -----------------------------------------------------------------------------
elif pagina_sidebar == "Reporte PDF":
    st.header("Generar reporte oficial por municipio")
    st.caption(
        "Descarga un reporte en PDF listo para presentar en reuniones o enviar a autoridades. "
        "Incluye indicadores de salud, graficas, lista de pacientes de alto riesgo "
        "y proyeccion de costos."
    )

    # Intentamos cargar los costos para incluirlos en el reporte (opcional)
    try:
        with open(COSTS_YAML, "r", encoding="utf-8") as f:
            costos = yaml.safe_load(f)
    except Exception:
        costos = None

    # El modulo de reporte maneja internamente la generacion y el enlace de descarga
    render_seccion_pdf(df_filt, costos=costos)

st.divider()