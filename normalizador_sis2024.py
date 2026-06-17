"""
normalizador_sis2024.py
=======================
Modulo de deteccion automatica de formato y normalizacion de columnas
para archivos Excel del sistema de salud de Guanajuato.

Descripcion general:
    Este modulo resuelve el problema de interoperabilidad entre los
    distintos formatos de Excel que puede recibir el panel ECNT. En
    la practica, los datos pueden provenir de dos fuentes con estructuras
    completamente distintas:

      1. Formato propio: columnas ya estandarizadas al esquema de
         seminario_Luisa.py. Se identifica porque contiene columnas
         como 'id_paciente', 'glucosa_ayunas', 'hba1c_pct', etc.

      2. Formato SIS-2024: encabezados crudos de la Tarjeta de Registro
         y Control de Enfermedades Cronicas (SINBA/SSA, clave SIS-2024).
         Los encabezados pueden aparecer con mayusculas, acentos, abreviaturas
         o variantes regionales (p.ej. "Glucemia en ayuno mg/dL" o "Glicemia
         ayuno" para referirse al mismo campo).

    El modulo detecta automaticamente el formato de cada archivo y aplica
    el pipeline de normalizacion correspondiente, produciendo siempre un
    DataFrame con el esquema propio independientemente del origen.

Pipeline de normalizacion (aplicado en orden):
    1. Deteccion de formato (propio / sis2024 / desconocido).
    2. Mapeo de columnas SIS-2024 al esquema propio (si aplica).
    3. Normalizacion de sexo (MUJER/HOMBRE/1/2 -> F/M).
    4. Normalizacion de origen indigena (Si/No/X -> 0/1).
    5. Normalizacion de variables binarias (Si/No/X/True -> 0/1).
    6. Normalizacion numerica: extraccion de cifras, conversion a float,
       reemplazo de valores fuera de rango clinico por NaN.
    7. Calculo de IMC si no viene en el Excel pero si hay peso y talla.
    8. Inferencia de obesidad a partir del IMC calculado.
    9. Construccion de id_paciente a partir de CURP si no hay expediente.
   10. Deteccion y estandarizacion de columna de fecha (__fecha__).
   11. Relleno de columnas faltantes con 0 (binarios) o NaN (numericos).
   12. Reordenamiento de columnas: primero las del esquema propio, luego extras.

Uso desde seminario_Luisa.py:
    from normalizador_sis2024 import normalizar_multiples_excels, reporte_normalizacion
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher   # Calculo de similitud entre cadenas de texto
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# SECCION 1: ESQUEMA PROPIO — COLUMNAS ESTANDAR DEL SISTEMA
# =============================================================================

COLUMNAS_PROPIAS: List[str] = [
    # Datos demograficos del paciente
    "id_paciente", "edad", "sexo", "municipio", "entidad", "origen_indigena",

    # Enfermedades cronicas principales (variable binaria: 1 = presente)
    "diabetes", "hipertension", "dislipidemia", "obesidad",

    # Antecedentes familiares (prefijo af_)
    "af_enf_cardiovascular", "af_hta", "af_diabetes",
    "af_dislipidemias", "af_obesidad", "af_enf_cerebrovascular",

    # Antecedentes personales (prefijo ap_)
    "ap_fumador", "ap_alcoholismo", "ap_sedentarismo",
    "ap_diabetes_gestacional", "ap_postMenopausia", "ap_sobrepeso",
    "ap_enf_cardiovascular", "ap_vih", "ap_enf_cerebrovascular",
    "ap_tuberculosis",

    # Parametros clinicos y biometricos
    "glucosa_ayunas", "hba1c_pct", "presion_sistolica", "presion_diastolica",
    "peso_kg", "imc", "cintura_cm",
    "col_total", "ldl", "hdl", "trigliceridos",
]

# Subconjunto de columnas usadas como "firma" del formato propio.

FIRMA_FORMATO_PROPIO: List[str] = [
    "id_paciente", "glucosa_ayunas", "hba1c_pct",
    "presion_sistolica", "imc",
]


# =============================================================================
# SECCION 2: DICCIONARIO DE MAPEO SIS-2024 → ESQUEMA PROPIO
# =============================================================================
# Estructura del diccionario:
#   clave  : cadena con uno o varios alias separados por "|"
#             (variantes del encabezado tal como aparecen en el Excel fisico)
#   valor  : nombre de la columna en el esquema propio
#
# Antes de comparar, tanto la clave como el encabezado del Excel se limpian
# con _limpiar_texto(): se convierten a minusculas, se eliminan acentos y
# se reemplazan caracteres especiales por espacios. Esto garantiza que
# "HbA1c %" y "hemoglobina glicosilada" se resuelvan al mismo campo.
# =============================================================================

MAPA_SIS2024: Dict[str, str] = {

    # --- Identificacion del establecimiento y del paciente ---
    "clave|clues|id unidad":                              "clues",
    "nombre de la unidad|unidad medica":                  "nombre_unidad",
    "localidad":                                          "localidad",
    "municipio":                                          "municipio",
    "jurisdiccion":                                       "jurisdiccion",
    "expediente|num expediente|numero expediente":        "id_paciente",
    # Nombre del paciente: se captura pero se elimina en la anonimizacion
    "nombre paciente|nombre del paciente|apellido paterno|apellido materno|"
    "nombre|primer nombre":                               "nombre_paciente",
    "curp":                                               "curp",
    "fecha de nacimiento|fecha nacimiento|f nacimiento":  "fecha_nacimiento",
    "edad":                                               "edad",
    # sexo_raw se normaliza a F/M en _normalizar_sexo()
    "sexo|genero|mujer|hombre":                           "sexo_raw",
    "entidad de nacimiento|entidad nacimiento":           "entidad",
    # origen_indigena_raw se normaliza a 0/1 en _normalizar_origen_indigena()
    "pueblo indigena|indigena|origen indigena":           "origen_indigena_raw",

    # --- Fechas de ingreso y seguimiento ---
    "fecha ingreso|fecha de ingreso|ingreso":             "fecha_ingreso",
    "reingreso":                                          "es_reingreso",

    # --- Diagnosticos (presencia de la enfermedad como columna binaria) ---
    "diabetes mellitus|diabetes":                         "diabetes",
    "hipertension arterial|hipertension|hta":             "hipertension",
    "obesidad":                                           "obesidad",
    "dislipidemia|dislipidemias":                         "dislipidemia",
    "sindrome metabolico":                                "sindrome_metabolico",

    # --- Parametros basales: Diabetes ---
    "glucemia en ayuno|glucemia ayuno|glucosa ayunas|glucosa en ayunas|"
    "glucosa ayuno|glicemia ayuno":                       "glucosa_ayunas",
    "hba1c|hemoglobina glucosilada|hemoglobina glicosilada|hba1c %|"
    "hemoglobina a1c":                                    "hba1c_pct",
    "fondo de ojo|fondo ojo":                             "fondo_ojo",
    "revision de pies|revision pies|pie diabetico":       "revision_pies",

    # --- Parametros basales: Hipertension ---
    "presion sistolica|presion arterial sistolica|pa sistolica|"
    "sistolica|sistole":                                  "presion_sistolica",
    "presion diastolica|presion arterial diastolica|pa diastolica|"
    "diastolica|diastole":                                "presion_diastolica",

    # --- Parametros basales: Obesidad y antropometria ---
    "peso|peso kg|peso (kg)":                             "peso_kg",
    "imc|indice de masa corporal|indice masa corporal":   "imc",
    "circunferencia de cintura|cintura|perimetro cintura|"
    "circ cintura|circ. cintura":                         "cintura_cm",
    # talla_m se usa internamente para calcular IMC si no viene en el Excel
    "talla|estatura|talla (m)|talla m":                   "talla_m",

    # --- Parametros basales: Perfil lipidico ---
    "colesterol total|col total|colesterol":              "col_total",
    "ldl|c-ldl|ldl colesterol|colesterol ldl":            "ldl",
    "hdl|c-hdl|hdl colesterol|colesterol hdl":            "hdl",
    "trigliceridos|trigliceridos mg|trigliceridos (mg/dl)": "trigliceridos",

    # --- Antecedentes familiares (AF) ---
    "af enf cardiovascular|af cardiovascular|antecedente familiar cardiovascular|"
    "af cardio":                                          "af_enf_cardiovascular",
    "af hta|af hipertension|antecedente familiar hipertension|"
    "antecedentes familiares hta":                        "af_hta",
    "af diabetes|antecedente familiar diabetes|"
    "antecedentes familiares diabetes":                   "af_diabetes",
    "af dislipidemias|af dislipidemia|antecedente familiar dislipidemia": "af_dislipidemias",
    "af obesidad|antecedente familiar obesidad":          "af_obesidad",
    "af enf cerebrovascular|af cerebrovascular|"
    "antecedente familiar cerebrovascular":               "af_enf_cerebrovascular",

    # --- Antecedentes personales (AP) ---
    "sedentarismo|ap sedentarismo":                       "ap_sedentarismo",
    "sobrepeso|ap sobrepeso":                             "ap_sobrepeso",
    "tabaquismo|fumador|ap fumador|ap tabaquismo":        "ap_fumador",
    "alcoholismo|ap alcoholismo":                         "ap_alcoholismo",
    "vih|ap vih|vih sida":                                "ap_vih",
    "tuberculosis|ap tuberculosis|tb":                    "ap_tuberculosis",
    "post menopausia|postmenopausia|ap postmenopausia|"
    "menopausia":                                         "ap_postMenopausia",
    "diabetes gestacional|ap diabetes gestacional":       "ap_diabetes_gestacional",
    "enf cardiovascular personal|ap enf cardiovascular|"
    "ap cardiovascular|enf cardio personal":              "ap_enf_cardiovascular",
    "enf cerebrovascular personal|ap enf cerebrovascular|"
    "ap cerebrovascular":                                 "ap_enf_cerebrovascular",
    "terapia de reemplazo hormonal|terapia hormonal|trh": "ap_terapia_hormonal",
    "producto macrosomico|macrosomico":                   "ap_macrosomico",

    # --- Visitas domiciliarias y seguimiento ---
    "fecha visita|fecha de visita|visita":                "fecha_visita",
    "resultado visita|resultado":                         "resultado_visita",

    # --- Tipo de deteccion y tratamiento previo ---
    "tipo deteccion|deteccion realizada por|deteccion por|"
    "pesquisa|sintomatologia":                            "tipo_deteccion",
    "tratamiento previo farmacologico|tratamiento previo|"
    "trat previo farmacologico":                          "tratamiento_previo_farmacologico",

    # --- Columnas temporales tipicas en reportes de la SSA ---
    "anio|año|year":                                      "anio",
    "mes|month":                                          "mes",
}


# =============================================================================
# SECCION 3: RANGOS CLINICOS VALIDOS
# =============================================================================
# Para cada variable numerica, se definen los limites fisiologicamente
# posibles. Cualquier valor fuera de estos rangos se considera un error
# de captura y se reemplaza por NaN durante la normalizacion numerica.
# Los limites son deliberadamente amplios para no descartar valores
# extremos pero reales (p.ej. glucosa 600 en una crisis hiperglicemica).

RANGOS_CLINICOS: Dict[str, Tuple[float, float]] = {
    "edad":               (0,    120),    # Anos cumplidos
    "glucosa_ayunas":     (40,   600),    # mg/dL
    "hba1c_pct":          (3.0,  20.0),  # Porcentaje
    "presion_sistolica":  (60,   280),    # mmHg
    "presion_diastolica": (30,   160),    # mmHg
    "peso_kg":            (20,   300),    # Kilogramos
    "imc":                (10,   80),     # kg/m^2
    "cintura_cm":         (40,   200),    # Centimetros
    "col_total":          (50,   500),    # mg/dL
    "ldl":                (20,   400),    # mg/dL
    "hdl":                (10,   150),    # mg/dL
    "trigliceridos":      (30,   2000),   # mg/dL (limite alto por pancreatitis hipertrigliceridemica)
    "talla_m":            (0.5,  2.5),    # Metros (usada internamente para calcular IMC)
}


# =============================================================================
# SECCION 4: UTILIDADES INTERNAS DE COMPARACION DE TEXTO
# =============================================================================

def _limpiar_texto(texto: str) -> str:
    """
    Normaliza una cadena de texto para hacer comparaciones robustas
    entre encabezados de Excel y los patrones del diccionario MAPA_SIS2024.

    Transformaciones aplicadas en orden:
      1. Transliteracion de caracteres acentuados y especiales al ASCII
         equivalente (p.ej. 'a' -> 'a', 'n' -> 'n', 'c' -> 'c').
      2. Conversion a minusculas.
      3. Reemplazo de caracteres no alfanumericos por espacios
         (elimina parentesis, barras, guiones, simbolos de unidad, etc.).
      4. Colapso de espacios multiples en uno solo y eliminacion de
         espacios al inicio y al final.

    Ejemplo:
        "HbA1c (%)" -> "hba1c"
        "Glucemia en ayuno (mg/dL)" -> "glucemia en ayuno mg dl"

    Parametros:
        texto (str): Cadena original (encabezado de Excel o patron del mapa).

    Retorna:
        str: Cadena normalizada lista para comparacion.
    """
    # Tabla de transliteracion: caracteres acentuados -> ASCII equivalente
    reemplazos = str.maketrans(
        "áéíóúÁÉÍÓÚàèìòùäëïöüñç",
        "aeiouAEIOUaeiouaeiouac"
    )
    texto = texto.translate(reemplazos)
    texto = texto.lower()
    # Eliminamos todo lo que no sea letra, numero o espacio
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    # Normalizamos espacios multiples
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _similitud(a: str, b: str) -> float:
    """
    Calcula el ratio de similitud entre dos cadenas de texto
    usando el algoritmo de Ratcliff/Obershelp (SequenceMatcher).

    El valor retornado va de 0.0 (completamente distintas) a 1.0
    (identicas). Se usa como criterio de desempate cuando no hay
    coincidencia exacta ni de subcadena entre un encabezado del Excel
    y los alias de MAPA_SIS2024.

    Parametros:
        a (str): Primera cadena (ya normalizada con _limpiar_texto).
        b (str): Segunda cadena (ya normalizada con _limpiar_texto).

    Retorna:
        float: Ratio de similitud entre 0.0 y 1.0.
    """
    return SequenceMatcher(None, a, b).ratio()


def _mejor_match(col_excel: str, mapa: Dict[str, str],
                 umbral: float = 0.72) -> str | None:
    """
    Busca el patron en MAPA_SIS2024 que mejor coincide con el nombre
    de una columna del Excel, usando una estrategia de dos pasos:

    Paso 1 — Coincidencia exacta o de subcadena:
      Se compara la columna limpia contra cada alias del patron limpio.
      Si la columna es igual al alias, o uno contiene al otro
      (p.ej. "glucemia en ayuno mg dl" contiene "glucemia en ayuno"),
      se asigna un score de 0.95 y se continua buscando por si hay
      una coincidencia exacta aun mejor.

    Paso 2 — Similitud difusa:
      Si ningun alias supero el umbral en el paso 1, se calcula el
      ratio de SequenceMatcher entre la columna y cada alias. Si el
      mejor score supera el umbral (0.72 por defecto), se acepta.

    El umbral de 0.72 se calibro empiricamente para evitar falsos
    positivos entre campos distintos (p.ej. "ldl" vs "hdl") mientras
    se aceptan variantes razonables.

    Parametros:
        col_excel (str):        Nombre de la columna tal como viene en el Excel.
        mapa      (Dict):       Diccionario de patrones (MAPA_SIS2024).
        umbral    (float):      Score minimo para aceptar una coincidencia difusa.
                                Por defecto 0.72.

    Retorna:
        str | None: Nombre de la columna estandar si se encontro coincidencia,
                    None si ninguna coincidencia supero el umbral.
    """
    col_limpia    = _limpiar_texto(col_excel)
    mejor_col_std = None
    mejor_score   = 0.0

    for patrones_raw, col_std in mapa.items():
        # Cada entrada del mapa puede tener multiples alias separados por "|"
        aliases = [_limpiar_texto(p) for p in patrones_raw.split("|")]

        for alias in aliases:
            # Coincidencia exacta: retornamos inmediatamente, es el caso ideal
            if col_limpia == alias:
                return col_std

            # Coincidencia de subcadena: uno contiene al otro
            if alias in col_limpia or col_limpia in alias:
                score = 0.95
            else:
                # Similitud difusa como ultimo recurso
                score = _similitud(col_limpia, alias)

            if score > mejor_score:
                mejor_score   = score
                mejor_col_std = col_std

    # Solo aceptamos la coincidencia si supera el umbral de confianza
    if mejor_score >= umbral:
        return mejor_col_std
    return None


# =============================================================================
# SECCION 5: DETECCION DE FORMATO
# =============================================================================

def detectar_formato(df: pd.DataFrame) -> str:
    """
    Analiza los encabezados del DataFrame y determina si el archivo
    corresponde al formato propio, al SIS-2024 o si es desconocido.

    Criterio de decision:
      1. Se buscan las columnas de FIRMA_FORMATO_PROPIO en el DataFrame
         (comparacion en minusculas para ignorar mayusculas).
         Si al menos el 60% estan presentes -> "propio".
      2. Si no pasa el criterio anterior, se intenta mapear cada columna
         contra MAPA_SIS2024. Si al menos 3 columnas mapean exitosamente
         -> "sis2024".
      3. Si ninguno de los criterios se cumple -> "desconocido".
         El procesamiento continua de todas formas, pero se emite una
         advertencia en el reporte.

    El umbral del 60% para el formato propio permite que archivos con
    algunas columnas opcionales ausentes sigan siendo reconocidos
    correctamente.

    Parametros:
        df (pd.DataFrame): DataFrame crudo leido con pd.read_excel().

    Retorna:
        str: "propio", "sis2024" o "desconocido".
    """
    cols_df     = {c.lower().strip() for c in df.columns}
    firma_lower = {c.lower() for c in FIRMA_FORMATO_PROPIO}
    hits_propios = len(firma_lower & cols_df)
    porcentaje   = hits_propios / len(firma_lower)

    if porcentaje >= 0.6:
        return "propio"

    # Intentamos contar cuantas columnas del Excel mapean al diccionario SIS-2024
    n_mapeadas = sum(
        1 for col in df.columns
        if _mejor_match(col, MAPA_SIS2024) is not None
    )
    if n_mapeadas >= 3:
        return "sis2024"

    return "desconocido"


# =============================================================================
# SECCION 6: MAPEO DE COLUMNAS SIS-2024 AL ESQUEMA PROPIO
# =============================================================================

def _mapear_columnas_sis2024(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Renombra las columnas de un Excel SIS-2024 a sus equivalentes en el
    esquema propio, usando el diccionario MAPA_SIS2024 y la funcion
    _mejor_match() para la busqueda con tolerancia a variantes.

    Manejo de colisiones: si dos columnas distintas del Excel mapean al
    mismo nombre estandar (p.ej. dos columnas con variantes de "glucosa"),
    solo se renombra la primera y la segunda se agrega a no_mapeadas para
    conservarla sin perder datos.

    Parametros:
        df (pd.DataFrame): DataFrame con columnas crudas del SIS-2024.

    Retorna:
        Tuple[pd.DataFrame, Dict]:
          - DataFrame con columnas renombradas al esquema propio.
          - Diccionario de auditoria con las claves:
              'mapeadas'    : {col_original: col_estandar} para cada mapeo exitoso.
              'no_mapeadas' : [col_original, ...] para columnas sin correspondencia
                              o que causarian colision.
    """
    mapeo_aplicado: Dict[str, str] = {}
    no_mapeadas:    List[str]      = []
    rename_dict:    Dict[str, str] = {}

    for col in df.columns:
        col_std = _mejor_match(col, MAPA_SIS2024)
        if col_std is not None:
            # Verificamos que no exista ya otra columna mapeada al mismo nombre estandar
            if col_std not in rename_dict.values():
                rename_dict[col]      = col_std
                mapeo_aplicado[col]   = col_std
            else:
                # Colision: conservamos la columna con su nombre original
                no_mapeadas.append(col)
        else:
            no_mapeadas.append(col)

    df_out    = df.rename(columns=rename_dict)
    auditoria = {"mapeadas": mapeo_aplicado, "no_mapeadas": no_mapeadas}
    return df_out, auditoria


# =============================================================================
# SECCION 7: TRANSFORMACIONES POST-MAPEO
# Normalizan los valores de las columnas ya renombradas al formato
# esperado por seminario_Luisa.py.
# =============================================================================

def _normalizar_sexo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza la columna de sexo biologico al formato 'F' / 'M'.

    Los Excels del SIS-2024 pueden registrar el sexo de multiples formas:
    'MUJER' / 'HOMBRE', 'M' / 'H', '1' / '2' (codificacion numerica SSA),
    'FEMENINO' / 'MASCULINO'. Esta funcion unifica todas las variantes.

    Logica de decision:
      - Si ya existe la columna 'sexo': se aplican solo las conversiones
        necesarias (p.ej. FEMENINO -> F) y se retorna.
      - Si existe 'sexo_raw' (columna mapeada desde el SIS-2024): se aplica
        el mapeo completo y se elimina 'sexo_raw'.
      - Si ninguna de las dos existe: se retorna el DataFrame sin cambios.

    Parametros:
        df (pd.DataFrame): DataFrame con columna 'sexo' o 'sexo_raw'.

    Retorna:
        pd.DataFrame: DataFrame con la columna 'sexo' estandarizada a 'F'/'M'.
    """
    if "sexo" in df.columns:
        # La columna ya existe: aplicamos solo las conversiones necesarias
        df["sexo"] = df["sexo"].astype(str).str.strip().str.upper()
        df["sexo"] = df["sexo"].replace({
            "FEMENINO": "F", "MASCULINO": "M",
            "MUJER":    "F", "HOMBRE":    "M",
            "H":        "M", "1":         "F", "2": "M"
        })
        return df

    if "sexo_raw" not in df.columns:
        return df   # No hay columna de sexo: no podemos hacer nada

    # Mapeo completo desde sexo_raw
    mapa_sexo = {
        "mujer": "F", "femenino": "F", "f": "F", "1": "F",
        "hombre": "M", "masculino": "M", "m": "M", "h": "M", "2": "M",
    }
    df["sexo"] = (
        df["sexo_raw"]
        .astype(str).str.strip().str.lower()
        .map(mapa_sexo)
    )
    df = df.drop(columns=["sexo_raw"], errors="ignore")
    return df


def _normalizar_origen_indigena(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna de origen indigena a variable binaria entera (0 o 1).

    Los Excels del SIS-2024 pueden registrar este campo como:
    'SI' / 'NO', 'Si' / 'No', 'X' (marca en casilla), 1 / 0,
    'True' / 'False', o incluso vacio (interpretado como 0 = No).

    Logica de decision:
      - Se busca primero 'origen_indigena_raw' (columna mapeada desde SIS-2024).
      - Si no existe, se busca 'origen_indigena' directamente.
      - Si ninguna existe, se crea la columna con valor 0 para todos los registros.
      - Despues del parseo, se elimina 'origen_indigena_raw' si era diferente
        de 'origen_indigena'.

    Parametros:
        df (pd.DataFrame): DataFrame con columna de origen indigena en cualquier formato.

    Retorna:
        pd.DataFrame: DataFrame con la columna 'origen_indigena' como entero 0 o 1.
    """
    col_raw   = "origen_indigena_raw"
    col_final = "origen_indigena"

    # Determinamos la fuente de datos
    src = (
        col_raw   if col_raw   in df.columns else
        col_final if col_final in df.columns else
        None
    )
    if src is None:
        # Si no hay ninguna columna de origen indigena, asumimos 0 para todos
        df["origen_indigena"] = 0
        return df

    def _parsear(val):
        """Convierte un valor individual al binario 0 o 1."""
        v = str(val).strip().lower()
        if v in ("si", "sí", "yes", "1", "x", "true"):
            return 1
        if v in ("no", "0", "false", "nan", "none", ""):
            return 0
        return 0   # Valor desconocido: se trata como No

    df[col_final] = df[src].apply(_parsear)

    # Eliminamos la columna temporal _raw si es distinta de la final
    if col_raw in df.columns and col_raw != col_final:
        df = df.drop(columns=[col_raw], errors="ignore")
    return df


def _normalizar_binarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a entero 0 o 1 todas las columnas de diagnostico y antecedentes.

    Las columnas binarias del SIS-2024 pueden registrarse como:
      - Texto: 'SI' / 'NO', 'Si' / 'No', 'Sí'
      - Marca de casilla: 'X', 'x', simbolos de palomita ('v', 'checkmark')
      - Booleano: True / False
      - Numerico: 1 / 0, o cualquier numero positivo para verdadero
      - Vacio o 'sin dato': se interpreta como 0 (ausencia del antecedente)

    La funcion itera sobre la lista de columnas binarias conocidas y aplica
    la conversion solo a las que existen en el DataFrame, ignorando las demas.

    Parametros:
        df (pd.DataFrame): DataFrame con columnas de diagnostico y antecedentes.

    Retorna:
        pd.DataFrame: DataFrame con todas las columnas binarias como entero 0 o 1.
    """
    cols_binarias = [
        # Enfermedades principales
        "diabetes", "hipertension", "dislipidemia", "obesidad",
        "sindrome_metabolico",
        # Antecedentes familiares
        "af_enf_cardiovascular", "af_hta", "af_diabetes",
        "af_dislipidemias", "af_obesidad", "af_enf_cerebrovascular",
        # Antecedentes personales
        "ap_fumador", "ap_alcoholismo", "ap_sedentarismo",
        "ap_diabetes_gestacional", "ap_postMenopausia", "ap_sobrepeso",
        "ap_enf_cardiovascular", "ap_vih", "ap_enf_cerebrovascular",
        "ap_tuberculosis", "ap_terapia_hormonal", "ap_macrosomico",
        # Otros campos binarios del SIS-2024
        "es_reingreso", "tratamiento_previo_farmacologico",
    ]

    def _a_binario(val) -> int:
        """Convierte un valor individual al entero 0 o 1."""
        v = str(val).strip().lower()
        if v in ("si", "sí", "yes", "1", "x", "true", "marcado", "v", "checkmark"):
            return 1
        if v in ("no", "0", "false", "nan", "none", "", "sin dato"):
            return 0
        # Ultimo recurso: intento de conversion numerica
        try:
            return 1 if float(v) > 0 else 0
        except ValueError:
            return 0  # Valor no interpretable: se trata como ausente

    for col in cols_binarias:
        if col in df.columns:
            df[col] = df[col].apply(_a_binario)

    return df


def _normalizar_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a tipo float las columnas numericas clinicas y reemplaza
    por NaN los valores que caen fuera del rango clinico definido en
    RANGOS_CLINICOS.

    El proceso para cada columna numerica es:
      1. Extraccion de la cifra numerica con una expresion regular que elimina
         unidades pegadas al valor (p.ej. "120 mmHg" -> "120", "1.75m" -> "1.75").
      2. Conversion a float con pd.to_numeric(..., errors='coerce'),
         que convierte cualquier cadena no numerica a NaN.
      3. Comparacion contra los limites fisiologicos del RANGOS_CLINICOS.
         Los valores fuera de rango se reemplazan por NaN.

    El conteo de valores invalidos se registra en el reporte de normalizacion
    para dar visibilidad al usuario, pero no se lanza ninguna excepcion.

    Parametros:
        df (pd.DataFrame): DataFrame con columnas numericas en formato mixto.

    Retorna:
        pd.DataFrame: DataFrame con columnas numericas como float, sin valores
                      fuera de rango fisiologico.
    """
    cols_num = list(RANGOS_CLINICOS.keys())

    for col in cols_num:
        if col not in df.columns:
            continue

        # Extraemos solo la parte numerica, descartando unidades de medicion
        # Ejemplo: "7.5 %" -> "7.5", "140/90 mmHg" -> "140" (primer numero)
        df[col] = (
            df[col].astype(str)
            .str.extract(r"([-+]?\d*\.?\d+)")[0]
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Reemplazamos los valores fuera del rango fisiologico por NaN
        lo, hi = RANGOS_CLINICOS[col]
        fuera_rango = (df[col] < lo) | (df[col] > hi)
        if fuera_rango.sum() > 0:
            df.loc[fuera_rango, col] = np.nan
            # El conteo de invalidos se registra en normalizar_excel() antes
            # de llamar a esta funcion, por eso no lo registramos aqui de nuevo

    return df


def _calcular_imc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Indice de Masa Corporal (IMC = peso / talla^2) cuando no
    esta disponible en el Excel pero si estan el peso y la talla.

    La funcion detecta automaticamente si la talla viene en metros o en
    centimetros: si la mediana de los valores de talla es mayor que 3,
    se interpreta que estan en centimetros y se divide entre 100 antes
    de aplicar la formula.

    Si la columna 'imc' ya existe y tiene al menos un valor no nulo,
    la funcion no realiza ningun calculo para no sobreescribir datos reales.

    El IMC calculado se recorta dentro del rango fisiologico de RANGOS_CLINICOS
    para evitar valores imposibles por errores de captura en peso o talla.

    Parametros:
        df (pd.DataFrame): DataFrame con columnas 'peso_kg' y 'talla_m'
                           (esta ultima puede venir en metros o centimetros).

    Retorna:
        pd.DataFrame: DataFrame con la columna 'imc' calculada o sin cambios
                      si ya existia o si falta peso o talla.
    """
    # No sobreescribimos el IMC si ya tiene datos validos
    if "imc" in df.columns and df["imc"].notna().any():
        return df

    # No podemos calcular si falta peso o talla
    if "peso_kg" not in df.columns or "talla_m" not in df.columns:
        return df

    talla = df["talla_m"].copy()

    # Deteccion automatica de unidad: mediana > 3 implica que viene en centimetros
    if talla.dropna().median() > 3:
        talla = talla / 100   # Conversion de cm a metros

    df["imc"] = (df["peso_kg"] / (talla ** 2)).round(1)

    # Recortamos el IMC dentro del rango clinico valido
    df["imc"] = df["imc"].clip(
        RANGOS_CLINICOS["imc"][0],
        RANGOS_CLINICOS["imc"][1]
    )
    return df


def _inferir_obesidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infiere la presencia de obesidad a partir del IMC cuando la columna
    'obesidad' no esta disponible en los datos.

    Criterio: IMC >= 30 kg/m^2 (clasificacion OMS: obesidad grado I o superior).

    Solo actua si la columna 'obesidad' no existe pero si hay un IMC disponible.
    No sobreescribe datos de obesidad que ya vengan en el Excel original.

    Parametros:
        df (pd.DataFrame): DataFrame con la columna 'imc' ya normalizada.

    Retorna:
        pd.DataFrame: DataFrame con la columna 'obesidad' como entero 0 o 1,
                      o sin cambios si 'obesidad' ya existia.
    """
    if "obesidad" not in df.columns and "imc" in df.columns:
        df["obesidad"] = (df["imc"] >= 30).astype(int)
    return df


def _construir_id_paciente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que el DataFrame tenga una columna 'id_paciente' valida.

    Estrategia de construccion (en orden de preferencia):
      1. Si ya existe 'id_paciente': no se hace nada.
      2. Si existe 'curp': se usa la CURP como identificador unico del paciente.
      3. En caso contrario (fallback): se genera un ID provisional con el
         formato "P_SIS_N" donde N es el indice de la fila. Este ID es
         funcional para el sistema pero no identifica al paciente de forma
         persistente entre distintas cargas del mismo archivo.

    Parametros:
        df (pd.DataFrame): DataFrame sin columna 'id_paciente' o con ella.

    Retorna:
        pd.DataFrame: DataFrame con la columna 'id_paciente' garantizada.
    """
    if "id_paciente" in df.columns:
        return df

    if "curp" in df.columns:
        # La CURP es un identificador unico oficial: la usamos directamente
        df["id_paciente"] = df["curp"].astype(str).str.strip().str.upper()
        return df

    # Fallback: ID provisional basado en el indice de fila
    # Nota: estos IDs cambiaran si el archivo se carga de nuevo con filas reordenadas
    df["id_paciente"] = ["P_SIS_" + str(i) for i in range(len(df))]
    return df


def _detectar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta y estandariza la columna de fecha de visita como '__fecha__',
    que es la convencion interna utilizada por seminario_Luisa.py para
    el historial de visitas de cada paciente.

    La funcion busca en orden de prioridad:
      1. '__fecha__' ya existe -> no hace nada.
      2. 'fecha_ingreso' del SIS-2024 -> conversion directa a datetime.
      3. 'fecha_visita' del SIS-2024 -> conversion directa a datetime.
      4. Cualquier columna cuyo nombre contenga la cadena "fecha"
         (busqueda insensible a mayusculas) -> usa la primera encontrada.
      5. Columnas 'anio' + 'mes' como enteros -> construye una fecha
         del primer dia del mes (dia=1) para representar el periodo.
      6. Solo columna 'anio' -> construye una fecha del 1 de enero
         del ano correspondiente.
      7. Si ninguna opcion es exitosa -> asigna pd.NaT a toda la columna.
         El sistema seguira funcionando sin historial temporal.

    Parametros:
        df (pd.DataFrame): DataFrame ya normalizado en columnas.

    Retorna:
        pd.DataFrame: DataFrame con la columna '__fecha__' como tipo datetime.
    """
    # Si ya existe la columna, no la sobreescribimos
    if "__fecha__" in df.columns:
        return df

    # Prioridad 1 y 2: columnas de fecha especificas del SIS-2024
    for candidata in ["fecha_ingreso", "fecha_visita"]:
        if candidata in df.columns:
            df["__fecha__"] = pd.to_datetime(df[candidata], errors="coerce")
            if df["__fecha__"].notna().any():
                return df

    # Prioridad 3: cualquier columna con "fecha" en el nombre
    cands = [c for c in df.columns if "fecha" in c.lower()]
    if cands:
        df["__fecha__"] = pd.to_datetime(df[cands[0]], errors="coerce")
        return df

    # Prioridad 4: reconstruccion a partir de anio y mes numericos
    # Se usa un diccionario de mapeo minusculas->original para ser insensible a mayusculas
    cols_lower = {c.lower(): c for c in df.columns}
    if "anio" in cols_lower and "mes" in cols_lower:
        df["__fecha__"] = pd.to_datetime(
            dict(
                year  = df[cols_lower["anio"]],
                month = df[cols_lower["mes"]],
                day   = 1          # Primer dia del mes como representacion del periodo
            ),
            errors="coerce"
        )
        return df

    # Prioridad 5: solo el ano disponible -> 1 de enero de ese ano
    if "anio" in cols_lower:
        df["__fecha__"] = pd.to_datetime(
            df[cols_lower["anio"]].astype(str) + "-01-01",
            errors="coerce"
        )
        return df

    # Sin informacion temporal: la columna existe pero con todos los valores NaT
    df["__fecha__"] = pd.NaT
    return df


# =============================================================================
# SECCION 8: FUNCION PRINCIPAL DE NORMALIZACION
# =============================================================================

def normalizar_excel(
    df: pd.DataFrame,
    nombre_archivo: str = "archivo",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Punto de entrada principal del modulo. Aplica el pipeline completo de
    normalizacion a un DataFrame crudo leido con pd.read_excel().

    Pipeline aplicado en orden:
      8.1 Deteccion del formato del archivo.
      8.2 Mapeo de columnas SIS-2024 al esquema propio (si aplica).
      8.3 Normalizacion de valores: sexo, origen indigena, binarios, numericos.
      8.4 Calculo de IMC y deduccion de obesidad.
      8.5 Construccion de id_paciente y deteccion de fecha.
      8.6 Estandarizacion del id_paciente (mayusculas, sin espacios extra).
      8.7 Relleno de columnas faltantes con 0 (binarios) o NaN (numericos).
      8.8 Reordenamiento: primero columnas del esquema propio, luego extras.

    Parametros:
        df             (pd.DataFrame): DataFrame crudo leido con pd.read_excel().
        nombre_archivo (str):          Nombre del archivo fuente (para el reporte).
        verbose        (bool):         Si True, imprime advertencias a stdout
                                       (util para depuracion en desarrollo).

    Retorna:
        Tuple[pd.DataFrame, Dict]:
          - DataFrame normalizado con el esquema propio de columnas.
          - Reporte de auditoria con las claves:
              'archivo'             : nombre del archivo procesado
              'formato'             : 'propio' | 'sis2024' | 'desconocido'
              'filas_entrada'       : numero de filas del DataFrame crudo
              'filas_salida'        : numero de filas del DataFrame normalizado
              'columnas_mapeadas'   : {col_original: col_estandar}
              'columnas_no_mapeadas': [col_original, ...] no mapeadas
              'columnas_faltantes'  : [col_estandar, ...] rellenadas con 0/NaN
              'valores_invalidos'   : {col: n_valores_reemplazados_por_NaN}
              'advertencias'        : [str, ...] mensajes al operador
    """
    reporte: Dict = {
        "archivo":              nombre_archivo,
        "formato":              "desconocido",
        "filas_entrada":        len(df),
        "filas_salida":         0,
        "columnas_mapeadas":    {},
        "columnas_no_mapeadas": [],
        "columnas_faltantes":   [],
        "valores_invalidos":    {},
        "advertencias":         [],
    }

    # --- Paso 8.1: Detectar el formato del archivo ---
    fmt = detectar_formato(df)
    reporte["formato"] = fmt

    if verbose:
        print(f"[normalizador] {nombre_archivo}: formato detectado -> '{fmt}'")

    # --- Paso 8.2: Mapeo de columnas (solo necesario para SIS-2024) ---
    if fmt == "sis2024":
        df, auditoria = _mapear_columnas_sis2024(df)
        reporte["columnas_mapeadas"]    = auditoria["mapeadas"]
        reporte["columnas_no_mapeadas"] = auditoria["no_mapeadas"]
    elif fmt == "propio":
        # En formato propio no hay renombrado; solo documentamos que columnas coinciden
        reporte["columnas_mapeadas"] = {c: c for c in df.columns if c in COLUMNAS_PROPIAS}
    else:
        reporte["advertencias"].append(
            "Formato no reconocido. Se intentara procesar igual, "
            "pero pueden faltar columnas criticas."
        )

    # --- Paso 8.3: Normalizacion de valores ---
    df = _normalizar_sexo(df)
    df = _normalizar_origen_indigena(df)
    df = _normalizar_binarios(df)

    # Contamos valores fuera de rango ANTES de reemplazarlos por NaN
    # para que el reporte refleje cuantos valores crudos eran invalidos
    for col, (lo, hi) in RANGOS_CLINICOS.items():
        if col in df.columns:
            serie_num   = pd.to_numeric(df[col], errors="coerce")
            n_invalidos = int(((serie_num < lo) | (serie_num > hi)).sum())
            if n_invalidos:
                reporte["valores_invalidos"][col] = n_invalidos

    df = _normalizar_numericos(df)

    # --- Paso 8.4: IMC y obesidad derivados ---
    df = _calcular_imc(df)
    df = _inferir_obesidad(df)

    # --- Paso 8.5: Identificador y fecha ---
    df = _construir_id_paciente(df)
    df = _detectar_fecha(df)

    # --- Paso 8.6: Estandarizacion del id_paciente ---
    # Mayusculas, sin espacios al inicio/final, sin espacios multiples internos
    if "id_paciente" in df.columns:
        df["id_paciente"] = (
            df["id_paciente"]
            .astype(str).str.strip().str.upper()
            .str.replace(r"\s+", " ", regex=True)
        )

    # --- Paso 8.7: Relleno de columnas faltantes ---
    # Las columnas binarias ausentes se rellenan con 0 (ausencia del antecedente).
    # Las columnas numericas ausentes se rellenan con NaN (dato desconocido).
    for col in COLUMNAS_PROPIAS:
        if col not in df.columns:
            reporte["columnas_faltantes"].append(col)
            es_binaria = col.startswith(("af_", "ap_")) or col in (
                "diabetes", "hipertension", "dislipidemia", "obesidad", "origen_indigena"
            )
            df[col] = 0 if es_binaria else np.nan

    # --- Paso 8.8: Reordenamiento de columnas ---
    # Las columnas del esquema propio van primero (en el orden de COLUMNAS_PROPIAS),
    # seguidas de las columnas extra que no pertenecen al esquema (datos del SIS-2024
    # que no tienen equivalente en el esquema propio).
    cols_extra  = [c for c in df.columns if c not in COLUMNAS_PROPIAS]
    cols_orden  = [c for c in COLUMNAS_PROPIAS if c in df.columns] + cols_extra
    df          = df[cols_orden]

    reporte["filas_salida"] = len(df)

    # Mensajes de depuracion a stdout si verbose=True
    if verbose and reporte["columnas_no_mapeadas"]:
        print(f"[normalizador] Columnas no mapeadas en '{nombre_archivo}':",
              reporte["columnas_no_mapeadas"])
    if verbose and reporte["columnas_faltantes"]:
        print(f"[normalizador] Columnas faltantes (rellenadas con 0/NaN):",
              reporte["columnas_faltantes"])

    return df, reporte


# =============================================================================
# SECCION 9: VISUALIZACION DEL REPORTE EN STREAMLIT
# =============================================================================

def reporte_normalizacion(reporte: Dict) -> None:
    """
    Renderiza el reporte de normalizacion en la interfaz de Streamlit,
    dando visibilidad al operador de salud sobre lo que ocurrio con
    su archivo Excel durante el proceso de carga.

    El reporte se muestra dentro de un expander en la barra lateral
    (llamado desde seminario_Luisa.py) y contiene:
      - Linea de resumen: nombre del archivo, formato detectado y
        conteo de filas antes y despues de la normalizacion.
      - Expander de columnas mapeadas (solo para SIS-2024): tabla
        con la correspondencia columna_original -> columna_estandar.
      - Expander de columnas no mapeadas: columnas del Excel que no
        coincidieron con ningun patron del MAPA_SIS2024. Se conservan
        en el DataFrame con su nombre original.
      - Expander de columnas faltantes: columnas del esquema propio
        que no estaban en el Excel y se rellenaron con 0 o NaN.
      - Expander de valores invalidos: tabla con el numero de valores
        que estaban fuera del rango clinico y se reemplazaron por NaN.
      - Advertencias en amarillo (st.warning) para el formato desconocido
        u otras situaciones que requieran atencion del operador.

    Parametros:
        reporte (Dict): Diccionario retornado por normalizar_excel().

    Retorna:
        None. Escribe directamente en la interfaz de Streamlit.
    """
    import streamlit as st

    fmt     = reporte["formato"]
    archivo = reporte["archivo"]

    # Colores e interpretaciones segun el formato detectado
    color    = {"propio": "green", "sis2024": "blue", "desconocido": "orange"}
    etiqueta = {
        "propio":      "Formato propio (ya estandarizado)",
        "sis2024":     "Formato SIS-2024 (tarjeta fisica)",
        "desconocido": "Formato no reconocido",
    }

    # Linea de resumen en la parte superior del reporte
    st.markdown(
        f"**{archivo}** - "
        f":{color.get(fmt, 'gray')}[{etiqueta.get(fmt, fmt)}] - "
        f"{reporte['filas_entrada']:,} filas de entrada, "
        f"{reporte['filas_salida']:,} tras normalizacion"
    )

    # Tabla de mapeo de columnas (solo relevante para archivos SIS-2024)
    if reporte["columnas_mapeadas"] and fmt == "sis2024":
        with st.expander(
            f"Columnas mapeadas SIS-2024 -> estandar "
            f"({len(reporte['columnas_mapeadas'])})"
        ):
            st.dataframe(
                pd.DataFrame(
                    list(reporte["columnas_mapeadas"].items()),
                    columns=["Columna original (SIS-2024)", "Columna estandar"]
                ),
                use_container_width=True,
                hide_index=True,
            )

    # Columnas que no pudieron mapearse: se conservan con su nombre original
    if reporte["columnas_no_mapeadas"]:
        with st.expander(
            f"Columnas NO mapeadas ({len(reporte['columnas_no_mapeadas'])}) "
            "- se conservan tal cual"
        ):
            st.write(reporte["columnas_no_mapeadas"])

    # Columnas del esquema propio que no estaban en el Excel
    if reporte["columnas_faltantes"]:
        with st.expander(
            f"Columnas faltantes ({len(reporte['columnas_faltantes'])}) "
            "- rellenadas con 0 o NaN"
        ):
            st.write(reporte["columnas_faltantes"])

    # Valores que se reemplazaron por NaN por estar fuera de rango clinico
    if reporte["valores_invalidos"]:
        with st.expander("Valores fuera de rango clinico -> reemplazados por NaN"):
            st.dataframe(
                pd.DataFrame(
                    [(col, n) for col, n in reporte["valores_invalidos"].items()],
                    columns=["Columna", "Valores invalidos"]
                ),
                use_container_width=True,
                hide_index=True,
            )

    # Advertencias: formato desconocido u otras situaciones de atencion
    for adv in reporte["advertencias"]:
        st.warning(adv)


# =============================================================================
# SECCION 10: NORMALIZACION DE MULTIPLES ARCHIVOS
# =============================================================================

def normalizar_multiples_excels(
    archivos_y_dfs: List[Tuple[str, pd.DataFrame]],
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Normaliza y concatena multiples DataFrames provenientes de archivos
    Excel con distintos formatos en un unico DataFrame maestro.

    Esta funcion es el punto de entrada que llama seminario_Luisa.py
    cuando el usuario sube uno o varios archivos. Cada archivo se procesa
    de forma independiente con normalizar_excel() y luego se concatenan
    todos los resultados.

    La columna 'archivo_origen' se agrega a cada DataFrame antes de
    concatenar, lo que permite rastrear de que archivo proviene cada
    registro en el DataFrame maestro.

    Si algun archivo individual falla durante la normalizacion, la
    excepcion se propaga (no se captura aqui); el manejo de errores
    de lectura se realiza en seminario_Luisa.py antes de llamar a esta funcion.

    Parametros:
        archivos_y_dfs (List[Tuple[str, pd.DataFrame]]):
            Lista de tuplas (nombre_archivo, dataframe_crudo) donde:
              - nombre_archivo (str): nombre del archivo para el reporte.
              - dataframe_crudo (pd.DataFrame): contenido del Excel leido
                                               con pd.read_excel().

    Retorna:
        Tuple[pd.DataFrame, List[Dict]]:
          - df_maestro: DataFrame unico con todos los pacientes de todos
                        los archivos, normalizado al esquema propio.
                        Tiene una columna adicional 'archivo_origen' con
                        el nombre del archivo de procedencia de cada fila.
          - reportes:   Lista de reportes individuales, uno por archivo,
                        en el mismo orden que archivos_y_dfs.
          Si la lista de entrada esta vacia, retorna (DataFrame vacio, []).
    """
    dfs_norm: List[pd.DataFrame] = []
    reportes: List[Dict]         = []

    for nombre, df_crudo in archivos_y_dfs:
        df_norm, rep = normalizar_excel(df_crudo, nombre_archivo=nombre)
        # Agregamos la columna de trazabilidad antes de concatenar
        df_norm["archivo_origen"] = nombre
        dfs_norm.append(df_norm)
        reportes.append(rep)

    if not dfs_norm:
        return pd.DataFrame(), reportes

    # Concatenamos todos los DataFrames ignorando los indices originales
    # para obtener un indice consecutivo en el DataFrame maestro
    df_maestro = pd.concat(dfs_norm, ignore_index=True)
    return df_maestro, reportes