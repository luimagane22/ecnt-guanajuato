"""
score_riesgo.py
===============
Modulo de calculo del score de riesgo cardiovascular y metabolico por paciente.

Descripcion general:
    Este modulo implementa un modelo de puntuacion (scoring) para cuantificar
    el riesgo cardiovascular y metabolico individual de cada paciente registrado
    en el sistema ECNT Guanajuato. El score resultante es un numero entre 0 y 100,
    donde valores mas altos indican mayor urgencia de atencion clinica.

    El modelo esta disenado para dos audiencias con necesidades distintas:
      - Medico en consulta: visualiza un semaforo con la explicacion clinica
        detallada de cada factor que contribuye al score del paciente.
      - Administracion / gobierno: accede a una tabla priorizable con scores
        numericos de toda la poblacion, descargable en CSV.

Modelo de puntuacion:
    El score se calcula como la suma de los puntos aportados por seis factores
    de riesgo independientes. Cada factor tiene un peso maximo (en puntos) y
    una funcion de evaluacion que devuelve una fraccion de ese peso segun la
    severidad del hallazgo clinico del paciente:

      Factor              Peso maximo   Indicador principal
      -------             -----------   -------------------
      Diabetes            25 pts        HbA1c (o glucosa en ayunas como proxy)
      Hipertension        25 pts        Presion arterial sistolica y diastolica
      Dislipidemia        15 pts        LDL (+ bonus por trigliceridos y HDL bajo)
      Obesidad            15 pts        IMC (+ bonus por obesidad abdominal)
      Edad                10 pts        Factor de riesgo independiente
      Multimorbilidad     10 pts        Numero de enfermedades cronicas simultaneas
      -------             -----------
      TOTAL              100 pts

    Los pesos fueron asignados siguiendo las guias internacionales de riesgo
    cardiovascular (AHA/ACC, ESC) que identifican la diabetes y la hipertension
    como los factores modificables de mayor impacto en la morbimortalidad cardiovascular.

    ADVERTENCIA: Este score es un instrumento de priorizacion clinica y de
    gestion de salud publica, NO un modelo validado de riesgo cardiovascular
    a 10 anos (como Framingham o SCORE2). No debe usarse como unico criterio
    para decisiones clinicas individuales.

Clasificacion del score en niveles:
      0  – 29 : Bajo      (verde)
      30 – 54 : Moderado  (amarillo)
      55 – 74 : Alto      (naranja)
      75 – 100: Muy alto  (rojo)

Uso desde seminario_Luisa.py:
    from score_riesgo import calcular_score_df, render_score_paciente, render_tabla_riesgo
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# SECCION 1: FUNCIONES DE EVALUACION POR FACTOR DE RIESGO
#
# Cada funcion recibe una fila del DataFrame (pd.Series) y el peso maximo
# asignado a ese factor, y retorna una tupla (puntos, explicacion) donde:
#   - puntos      : float en el rango [0, peso]. Representa que tan alto es
#                   el riesgo del paciente en ese factor especifico.
#   - explicacion : str con el valor detectado y su interpretacion clinica,
#                   listo para mostrar al medico en la interfaz.
#
# Si la enfermedad no aplica al paciente (columna == 0), la funcion retorna
# (0.0, "") sin calcular nada. Si los datos de laboratorio no estan disponibles,
# se asigna el 50% del peso como estimacion conservadora de riesgo desconocido.
# =============================================================================

def _eval_diabetes(row: pd.Series, peso: float) -> Tuple[float, str]:
    """
    Evalua el nivel de control glucemico del paciente diabetico.

    Indicador principal: HbA1c (hemoglobina glucosilada), que refleja
    el promedio de glucosa en sangre de los ultimos 2-3 meses y es
    el estandar de oro para monitorear el control de la diabetes (ADA).

    Escala de puntuacion basada en los umbrales de la ADA:
      HbA1c < 7.0%  : 30% del peso  (meta terapeutica alcanzada)
      HbA1c 7.0-7.9%: 55% del peso  (control moderado, fuera de meta)
      HbA1c 8.0-8.9%: 75% del peso  (mal control, riesgo aumentado)
      HbA1c >= 9.0% : 100% del peso (control muy deficiente, accion requerida)

    Si no hay HbA1c disponible, se usa la glucosa en ayunas como indicador
    proxy con umbrales propios. Si tampoco hay glucosa, se asigna el 50%
    del peso como penalizacion por datos faltantes.

    Parametros:
        row   (pd.Series): Fila del DataFrame con datos del paciente.
        peso  (float):     Peso maximo del factor diabetes (25 puntos).

    Retorna:
        Tuple[float, str]: (puntos_asignados, texto_explicativo_para_medico).
                           (0.0, "") si el paciente no tiene diabetes.
    """
    if row.get("diabetes", 0) != 1:
        return 0.0, ""

    hba1c        = row.get("hba1c_pct",       np.nan)
    glucosa      = row.get("glucosa_ayunas",   np.nan)
    explicaciones = []
    puntos        = 0.0

    if pd.notna(hba1c):
        # Clasificacion segun umbrales ADA de control glucemico
        if hba1c < 7.0:
            puntos += peso * 0.30
            explicaciones.append(f"HbA1c {hba1c:.1f}% (controlada)")
        elif hba1c < 8.0:
            puntos += peso * 0.55
            explicaciones.append(f"HbA1c {hba1c:.1f}% (control moderado)")
        elif hba1c < 9.0:
            puntos += peso * 0.75
            explicaciones.append(f"HbA1c {hba1c:.1f}% (mal control)")
        else:
            puntos += peso * 1.0
            explicaciones.append(f"HbA1c {hba1c:.1f}% (control muy deficiente)")
    else:
        # Sin HbA1c: usamos glucosa en ayunas como indicador de control aproximado
        if pd.notna(glucosa):
            if glucosa < 130:
                puntos += peso * 0.30
                explicaciones.append(f"Glucosa {glucosa:.0f} mg/dL (aceptable, sin HbA1c)")
            elif glucosa < 180:
                puntos += peso * 0.60
                explicaciones.append(f"Glucosa {glucosa:.0f} mg/dL (elevada, sin HbA1c)")
            else:
                puntos += peso * 0.90
                explicaciones.append(f"Glucosa {glucosa:.0f} mg/dL (muy elevada, sin HbA1c)")
        else:
            # Sin ningun dato metabolico: penalizacion del 50% por informacion faltante
            puntos += peso * 0.50
            explicaciones.append("Diabetes sin datos de laboratorio disponibles")

    return puntos, "; ".join(explicaciones)


def _eval_hipertension(row: pd.Series, peso: float) -> Tuple[float, str]:
    """
    Evalua el nivel de control de la presion arterial en el paciente hipertenso.

    Se aplica una clasificacion simplificada basada en el JNC-8
    (Joint National Committee on Prevention, Detection, Evaluation,
    and Treatment of High Blood Pressure):
      PA < 130/80 mmHg  : 25% del peso  (meta terapeutica, bien controlada)
      PA < 140/90 mmHg  : 50% del peso  (normal-alta, fuera de meta optima)
      PA < 160 o < 100  : 75% del peso  (HTA grado 1, control insuficiente)
      PA >= 160/100 mmHg: 100% del peso (HTA grado 2+, riesgo cardiovascular alto)

    Se requieren ambas mediciones (sistolica y diastolica) para la clasificacion.
    Si no hay datos de PA, se asigna el 50% como penalizacion por datos faltantes.

    Parametros:
        row   (pd.Series): Fila del DataFrame con datos del paciente.
        peso  (float):     Peso maximo del factor hipertension (25 puntos).

    Retorna:
        Tuple[float, str]: (puntos_asignados, texto_explicativo_para_medico).
                           (0.0, "") si el paciente no tiene hipertension.
    """
    if row.get("hipertension", 0) != 1:
        return 0.0, ""

    sis           = row.get("presion_sistolica",  np.nan)
    dia           = row.get("presion_diastolica", np.nan)
    explicaciones = []
    puntos        = 0.0

    if pd.notna(sis) and pd.notna(dia):
        # Ambas mediciones disponibles: clasificacion JNC-8 simplificada
        if sis < 130 and dia < 80:
            puntos += peso * 0.25
            explicaciones.append(f"PA {sis:.0f}/{dia:.0f} mmHg (controlada)")
        elif sis < 140 and dia < 90:
            puntos += peso * 0.50
            explicaciones.append(f"PA {sis:.0f}/{dia:.0f} mmHg (normal-alta)")
        elif sis < 160 or dia < 100:
            puntos += peso * 0.75
            explicaciones.append(f"PA {sis:.0f}/{dia:.0f} mmHg (HTA grado 1)")
        else:
            puntos += peso * 1.0
            explicaciones.append(f"PA {sis:.0f}/{dia:.0f} mmHg (HTA grado 2+)")
    else:
        # Sin medicion disponible: penalizacion del 50%
        puntos += peso * 0.50
        explicaciones.append("Hipertension sin medicion de PA disponible")

    return puntos, "; ".join(explicaciones)


def _eval_dislipidemia(row: pd.Series, peso: float) -> Tuple[float, str]:
    """
    Evalua el perfil lipidico del paciente con dislipidemia.

    Indicador principal: LDL (lipoproteinas de baja densidad), que es
    el objetivo primario del tratamiento hipolipemiante segun las guias
    ATP III y ESC 2019. Se complementa con dos bonificaciones de riesgo:
      - Trigliceridos >= 500 mg/dL: +20% del peso (riesgo de pancreatitis).
      - Trigliceridos >= 200 mg/dL: se documenta sin puntos adicionales.
      - HDL < 40 mg/dL: +15% del peso (HDL bajo es factor de riesgo independiente).

    Los bonificaciones estan limitadas por el peso maximo del factor
    (no pueden hacer que el total supere el peso asignado a dislipidemia).

    Escala LDL (ATP III):
      LDL < 100 mg/dL  : 20% del peso (optimo para alto riesgo)
      LDL 100-129 mg/dL: 45% del peso (cercano al optimo)
      LDL 130-159 mg/dL: 70% del peso (limítrofe alto)
      LDL >= 160 mg/dL : 100% del peso (alto)

    Si no hay perfil lipidico, se asigna el 50% como penalizacion.

    Parametros:
        row   (pd.Series): Fila del DataFrame con datos del paciente.
        peso  (float):     Peso maximo del factor dislipidemia (15 puntos).

    Retorna:
        Tuple[float, str]: (puntos_asignados, texto_explicativo_para_medico).
                           (0.0, "") si el paciente no tiene dislipidemia.
    """
    if row.get("dislipidemia", 0) != 1:
        return 0.0, ""

    ldl           = row.get("ldl",           np.nan)
    trig          = row.get("trigliceridos",  np.nan)
    hdl           = row.get("hdl",            np.nan)
    explicaciones = []
    puntos        = 0.0

    # --- Componente principal: LDL (ATP III) ---
    if pd.notna(ldl):
        if ldl < 100:
            puntos += peso * 0.20
            explicaciones.append(f"LDL {ldl:.0f} mg/dL (optimo)")
        elif ldl < 130:
            puntos += peso * 0.45
            explicaciones.append(f"LDL {ldl:.0f} mg/dL (cercano al optimo)")
        elif ldl < 160:
            puntos += peso * 0.70
            explicaciones.append(f"LDL {ldl:.0f} mg/dL (limítrofe alto)")
        else:
            puntos += peso * 1.0
            explicaciones.append(f"LDL {ldl:.0f} mg/dL (alto)")

    # --- Bonus de riesgo: trigliceridos muy elevados ---
    # >= 500 mg/dL: riesgo de pancreatitis aguda (AHA/ACC), suma puntos adicionales
    if pd.notna(trig) and trig >= 500:
        puntos = min(puntos + peso * 0.20, peso)
        explicaciones.append(f"Trigliceridos {trig:.0f} mg/dL (muy elevados, riesgo pancreatitis)")
    elif pd.notna(trig) and trig >= 200:
        # Elevados pero sin riesgo de pancreatitis: documentamos sin puntaje extra
        explicaciones.append(f"Trigliceridos {trig:.0f} mg/dL (elevados)")

    # --- Bonus de riesgo: HDL bajo ---
    # HDL < 40 mg/dL es un factor de riesgo cardiovascular independiente (ATP III)
    if pd.notna(hdl) and hdl < 40:
        puntos = min(puntos + peso * 0.15, peso)
        explicaciones.append(f"HDL {hdl:.0f} mg/dL (bajo, factor de riesgo adicional)")

    # Si no se agrego ninguna explicacion, no hay datos del perfil lipidico
    if not explicaciones:
        puntos = peso * 0.50
        explicaciones.append("Dislipidemia sin perfil lipidico disponible")

    return puntos, "; ".join(explicaciones)


def _eval_obesidad(row: pd.Series, peso: float) -> Tuple[float, str]:
    """
    Evalua el grado de obesidad del paciente usando el IMC como indicador
    principal, con un bonus por obesidad abdominal basado en la circunferencia
    de cintura.

    Clasificacion de IMC (OMS):
      IMC 25-29.9 kg/m^2: 20% del peso (sobrepeso, puede haber flag de obesidad)
      IMC 30-34.9 kg/m^2: 50% del peso (obesidad grado I)
      IMC 35-39.9 kg/m^2: 75% del peso (obesidad grado II)
      IMC >= 40.0 kg/m^2 : 100% del peso (obesidad morbida, grado III)

    Bonus por obesidad abdominal (IDF/ATP III):
      Cintura >= 88 cm en mujeres o >= 102 cm en hombres: +20% del peso.
      La obesidad abdominal es un predictor independiente de riesgo metabolico
      y cardiovascular, especialmente para sindrome metabolico.

    Parametros:
        row   (pd.Series): Fila del DataFrame con datos del paciente.
                           Se usa 'sexo' para determinar el umbral de cintura.
        peso  (float):     Peso maximo del factor obesidad (15 puntos).

    Retorna:
        Tuple[float, str]: (puntos_asignados, texto_explicativo_para_medico).
                           (0.0, "") si el paciente no tiene obesidad.
    """
    if row.get("obesidad", 0) != 1:
        return 0.0, ""

    imc           = row.get("imc",        np.nan)
    cintura       = row.get("cintura_cm", np.nan)
    sexo          = str(row.get("sexo", "")).upper()
    explicaciones = []
    puntos        = 0.0

    # --- Componente principal: IMC (clasificacion OMS) ---
    if pd.notna(imc):
        if imc < 30:
            puntos += peso * 0.20
            explicaciones.append(f"IMC {imc:.1f} (sobrepeso leve)")
        elif imc < 35:
            puntos += peso * 0.50
            explicaciones.append(f"IMC {imc:.1f} (obesidad grado I)")
        elif imc < 40:
            puntos += peso * 0.75
            explicaciones.append(f"IMC {imc:.1f} (obesidad grado II)")
        else:
            puntos += peso * 1.0
            explicaciones.append(f"IMC {imc:.1f} (obesidad morbida)")
    else:
        puntos += peso * 0.50
        explicaciones.append("Obesidad sin IMC disponible")

    # --- Bonus por obesidad abdominal (IDF/ATP III) ---
    # Los umbrales de cintura difieren por sexo biologico
    if pd.notna(cintura):
        umbral = 88 if sexo == "F" else 102   # cm: mujeres=88, hombres=102
        if cintura >= umbral:
            puntos = min(puntos + peso * 0.20, peso)
            explicaciones.append(
                f"Cintura {cintura:.0f} cm (obesidad abdominal, umbral {umbral} cm)"
            )

    return puntos, "; ".join(explicaciones)


def _eval_edad(row: pd.Series, peso: float) -> Tuple[float, str]:
    """
    Evalua la edad como factor de riesgo cardiovascular independiente.

    La edad es uno de los factores de riesgo no modificables mas importantes
    en los modelos de riesgo cardiovascular (Framingham, SCORE2). A mayor
    edad, mayor es la probabilidad acumulada de eventos cardiovasculares,
    independientemente de la presencia de otras enfermedades.

    Escala de puntuacion por grupo etario:
      < 40 anos : 10% del peso (riesgo bajo por edad)
      40-49 anos: 30% del peso
      50-59 anos: 55% del peso (riesgo moderado)
      60-69 anos: 75% del peso (riesgo elevado)
      >= 70 anos: 100% del peso (riesgo alto)

    Si no hay dato de edad, se retorna 0 puntos sin penalizacion.

    Parametros:
        row   (pd.Series): Fila del DataFrame con datos del paciente.
        peso  (float):     Peso maximo del factor edad (10 puntos).

    Retorna:
        Tuple[float, str]: (puntos_asignados, texto_explicativo_para_medico).
                           (0.0, "") si no hay dato de edad disponible.
    """
    edad = row.get("edad", np.nan)
    if pd.isna(edad):
        return 0.0, ""

    if edad < 40:
        return peso * 0.10, f"Edad {int(edad)} anos (riesgo bajo por edad)"
    elif edad < 50:
        return peso * 0.30, f"Edad {int(edad)} anos"
    elif edad < 60:
        return peso * 0.55, f"Edad {int(edad)} anos (riesgo moderado por edad)"
    elif edad < 70:
        return peso * 0.75, f"Edad {int(edad)} anos (riesgo elevado por edad)"
    else:
        return peso * 1.0,  f"Edad {int(edad)} anos (riesgo alto por edad)"


def _eval_multimorbilidad(row: pd.Series, peso: float) -> Tuple[float, str]:
    """
    Evalua el efecto multiplicador del riesgo cardiovascular derivado de
    la coexistencia de multiples enfermedades cronicas en un mismo paciente.

    La multimorbilidad no es simplemente la suma de riesgos individuales:
    la combinacion de diabetes, hipertension, dislipidemia y obesidad
    constituye el sindrome metabolico, que conlleva un riesgo cardiovascular
    significativamente superior al que predice cada enfermedad por separado.

    Escala de puntuacion:
      0 enfermedades: 0%   del peso  (no aplica)
      1 enfermedad  : 20%  del peso  (riesgo individual)
      2 enfermedades: 50%  del peso  (riesgo combinado, interaccion entre factores)
      3 enfermedades: 80%  del peso  (sindrome metabolico probable)
      4 enfermedades: 100% del peso  (sindrome metabolico completo)

    Parametros:
        row   (pd.Series): Fila del DataFrame con columnas binarias de enfermedades.
        peso  (float):     Peso maximo del factor multimorbilidad (10 puntos).

    Retorna:
        Tuple[float, str]: (puntos_asignados, texto_descriptivo).
                           (0.0, "") si el paciente no tiene ninguna enfermedad cronica.
    """
    enfs = ["diabetes", "hipertension", "dislipidemia", "obesidad"]
    n    = sum(int(row.get(e, 0) == 1) for e in enfs)

    if n == 0:
        return 0.0,        ""
    elif n == 1:
        return peso * 0.20, "1 enfermedad cronica"
    elif n == 2:
        return peso * 0.50, "2 enfermedades cronicas (riesgo combinado)"
    elif n == 3:
        return peso * 0.80, "3 enfermedades cronicas (sindrome metabolico probable)"
    else:
        return peso * 1.0,  "4 enfermedades cronicas (sindrome metabolico completo)"


# =============================================================================
# SECCION 2: DEFINICION DE FACTORES CON SUS PESOS
# =============================================================================
# Lista de los seis factores de riesgo que componen el modelo de scoring.
# La estructura de cada elemento es:
#   id       : identificador unico del factor (clave en el dict de resultados)
#   etiqueta : nombre legible para mostrar en la interfaz al medico
#   peso     : contribucion maxima al score total; la suma de todos debe ser 100
#   evaluar  : referencia a la funcion de evaluacion correspondiente
#
# INVARIANTE: sum(f["peso"] for f in FACTORES) == 100
# Esta restriccion se verifica con un assert al definir la lista para
# detectar errores de configuracion en tiempo de importacion del modulo.
# =============================================================================

FACTORES: List[Dict] = [
    {
        "id":       "diabetes",
        "etiqueta": "Diabetes",
        "peso":     25,          # Mayor peso: la diabetes es el principal factor modificable
        "evaluar":  _eval_diabetes,
    },
    {
        "id":       "hipertension",
        "etiqueta": "Hipertension",
        "peso":     25,          # Mismo peso que diabetes: ambas son causas principales de ECV
        "evaluar":  _eval_hipertension,
    },
    {
        "id":       "dislipidemia",
        "etiqueta": "Dislipidemia",
        "peso":     15,
        "evaluar":  _eval_dislipidemia,
    },
    {
        "id":       "obesidad",
        "etiqueta": "Obesidad",
        "peso":     15,
        "evaluar":  _eval_obesidad,
    },
    {
        "id":       "edad",
        "etiqueta": "Edad",
        "peso":     10,          # Factor no modificable; menor peso que los clinicos
        "evaluar":  _eval_edad,
    },
    {
        "id":       "multimorbilidad",
        "etiqueta": "Multimorbilidad",
        "peso":     10,          # Captura el efecto sinergico entre enfermedades
        "evaluar":  _eval_multimorbilidad,
    },
]

# Verificacion en tiempo de importacion: la suma de pesos debe ser exactamente 100.
# Si se modifica la lista FACTORES y la suma cambia, este assert lo detecta
# inmediatamente en lugar de producir scores silenciosamente incorrectos.
assert sum(f["peso"] for f in FACTORES) == 100, \
    "Los pesos de FACTORES deben sumar exactamente 100."


# =============================================================================
# SECCION 3: CLASIFICACION POR NIVEL DE RIESGO
# =============================================================================

def clasificar_riesgo(score: float) -> Tuple[str, str, str]:
    """
    Convierte el score numerico (0-100) en una clasificacion cualitativa
    de nivel de riesgo con su color e indicador visual asociado.

    Los umbrales de clasificacion se fijaron para que la distribucion
    esperada de la poblacion resulte en aproximadamente:
      - Bajo:      pacientes bien controlados (mayoria ideal)
      - Moderado:  pacientes con control parcial o comorbilidades leves
      - Alto:      pacientes fuera de metas terapeuticas
      - Muy alto:  pacientes con descontrol severo o multimorbilidad avanzada

    Umbrales:
      score  0 – 29 : Bajo     (verde)
      score 30 – 54 : Moderado (amarillo)
      score 55 – 74 : Alto     (naranja)
      score 75 – 100: Muy alto (rojo)

    Parametros:
        score (float): Puntuacion total calculada por calcular_score_fila().

    Retorna:
        Tuple[str, str, str]:
          - nivel    : "Bajo", "Moderado", "Alto" o "Muy alto"
          - color    : "green", "orange" o "red" (para widgets de Streamlit)
          - semaforo : emoji de color correspondiente al nivel
    """
    if score < 30:
        return "Bajo",     "green",  "verde"
    elif score < 55:
        return "Moderado", "orange", "amarillo"
    elif score < 75:
        return "Alto",     "orange", "naranja"
    else:
        return "Muy alto", "red",    "rojo"


# =============================================================================
# SECCION 4: CALCULO DEL SCORE PARA UNA FILA
# =============================================================================

def calcular_score_fila(row: pd.Series) -> Dict:
    """
    Calcula el score de riesgo completo para un paciente individual
    representado como una fila del DataFrame.

    Itera sobre todos los factores definidos en FACTORES, invoca la funcion
    de evaluacion de cada uno y acumula los puntos. Adicionalmente construye
    un diccionario de detalle por factor y una lista con los tres factores
    que mas contribuyeron al score total, para facilitar la explicabilidad
    del resultado al medico.

    El puntaje de cada factor se recorta a su peso maximo para evitar que
    los bonificadores (trigliceridos, cintura, HDL bajo) lo excedan.

    Parametros:
        row (pd.Series): Fila del DataFrame con todos los indicadores del paciente.

    Retorna:
        Dict con las claves:
          - score          (float):      Puntuacion total redondeada a 1 decimal (0-100).
          - nivel          (str):        Clasificacion: "Bajo", "Moderado", "Alto", "Muy alto".
          - color          (str):        Color para widgets de Streamlit: "green", "orange", "red".
          - semaforo       (str):        Indicador visual de color del nivel de riesgo.
          - detalle        (Dict):       Diccionario por factor con las sub-claves:
                                           etiqueta   : nombre legible del factor
                                           puntos     : puntos obtenidos (float, 2 decimales)
                                           peso_max   : peso maximo del factor
                                           pct        : porcentaje del maximo alcanzado (0-100)
                                           explicacion: texto clinico del hallazgo
          - explicacion_top (List[str]): Explicaciones de los 3 factores con mayor contribucion
                                         absoluta, ordenadas de mayor a menor.
    """
    detalle = {}
    total   = 0.0

    for factor in FACTORES:
        puntos, explicacion = factor["evaluar"](row, factor["peso"])

        # Recortamos al peso maximo del factor para evitar desbordamiento por bonificadores
        puntos = min(puntos, factor["peso"])

        detalle[factor["id"]] = {
            "etiqueta":    factor["etiqueta"],
            "puntos":      round(puntos, 2),
            "peso_max":    factor["peso"],
            # Porcentaje del maximo alcanzado: util para la barra de progreso por factor
            "pct":         round(puntos / factor["peso"] * 100) if factor["peso"] else 0,
            "explicacion": explicacion,
        }
        total += puntos

    nivel, color, semaforo = clasificar_riesgo(total)

    # Identificamos los 3 factores con mayor contribucion absoluta al score
    # (excluyendo los que no aplican al paciente, cuya explicacion es vacia)
    top3 = sorted(
        [(fid, d) for fid, d in detalle.items() if d["explicacion"]],
        key=lambda x: x[1]["puntos"],
        reverse=True,
    )[:3]
    explicacion_top = [d["explicacion"] for _, d in top3 if d["explicacion"]]

    return {
        "score":            round(total, 1),
        "nivel":            nivel,
        "color":            color,
        "semaforo":         semaforo,
        "detalle":          detalle,
        "explicacion_top":  explicacion_top,
    }


# =============================================================================
# SECCION 5: CALCULO VECTORIZADO PARA TODO EL DATAFRAME
# =============================================================================

def calcular_score_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica calcular_score_fila() a todo el DataFrame y agrega tres columnas
    con los resultados del scoring.

    Trabaja sobre una copia del DataFrame para no modificar el original,
    lo que permite llamar a esta funcion multiples veces sin efectos secundarios.

    Columnas agregadas:
      - riesgo_score    (float): Puntuacion numerica de 0 a 100.
      - riesgo_nivel    (str):   Clasificacion cualitativa del riesgo.
      - riesgo_semaforo (str):   Indicador visual del nivel de riesgo.

    El uso de df.apply(..., axis=1) aplica la funcion fila por fila.
    Para DataFrames muy grandes (>100,000 filas) esto puede ser lento;
    en ese escenario se recomienda implementar una vectorizacion con numpy.

    Parametros:
        df (pd.DataFrame): DataFrame de pacientes con columnas clinicas.

    Retorna:
        pd.DataFrame: Copia del DataFrame con las tres columnas de score agregadas.
    """
    resultados = df.apply(calcular_score_fila, axis=1)

    df = df.copy()
    df["riesgo_score"]    = resultados.apply(lambda r: r["score"])
    df["riesgo_nivel"]    = resultados.apply(lambda r: r["nivel"])
    df["riesgo_semaforo"] = resultados.apply(lambda r: r["semaforo"])

    return df


# =============================================================================
# SECCION 6: VISTA INDIVIDUAL EN STREAMLIT (HISTORIAL DE PACIENTE)
# =============================================================================

def render_score_paciente(row: pd.Series) -> None:
    """
    Renderiza en Streamlit la ficha completa de riesgo de un paciente
    individual, orientada al medico en consulta.

    Componentes de la vista:
      1. Metrica principal: puntuacion numerica y nivel de riesgo con color.
      2. Barra de progreso visual: refleja visualmente el score de 0 a 100
         con el color correspondiente al nivel.
      3. Detalle por factor: cada factor que aplica al paciente se muestra
         en un expander con su contribucion en puntos, una mini-barra de
         progreso y el texto explicativo del hallazgo clinico.
         Los expanders con mayor contribucion (>60% del peso del factor)
         se abren por defecto para destacar los hallazgos mas urgentes.
      4. Recomendacion clinica: texto de accion sugerida segun el nivel
         de riesgo, adaptado al contexto del primer nivel de atencion.

    Parametros:
        row (pd.Series): Ultima fila del historial del paciente (visita mas reciente).
                         Debe contener las columnas clinicas del esquema propio.

    Retorna:
        None. Escribe directamente en la interfaz de Streamlit.
    """
    import streamlit as st

    resultado = calcular_score_fila(row)
    score   = resultado["score"]
    nivel   = resultado["nivel"]
    color   = resultado["color"]
    sem     = resultado["semaforo"]
    detalle = resultado["detalle"]

    st.markdown("---")
    st.subheader("Score de riesgo cardiovascular y metabolico")

    # --- Metrica principal: puntuacion y nivel de riesgo ---
    col_score, col_nivel, col_vacio = st.columns([1, 2, 3])
    with col_score:
        st.metric("Puntuacion", f"{score:.0f} / 100")
    with col_nivel:
        # Color del texto segun el nivel: verde, naranja o rojo
        color_texto = (
            "#1a7a1a" if color == "green" else
            "#b35c00" if color == "orange" else
            "#c0392b"
        )
        st.markdown(
            f"<div style='padding-top:8px;font-size:1.1rem;'>"
            f"{sem} <strong style='color:{color_texto}'>"
            f"Riesgo {nivel}</strong></div>",
            unsafe_allow_html=True,
        )

    # --- Barra de progreso coloreada segun el nivel de riesgo ---
    barra_color = {"green": "#2ecc71", "orange": "#e67e22", "red": "#e74c3c"}
    st.markdown(
        f"""
        <div style="background:#e0e0e0;border-radius:8px;height:14px;margin:8px 0 16px;">
          <div style="width:{score}%;background:{barra_color.get(color,'#888')};
               height:14px;border-radius:8px;transition:width 0.5s;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Detalle por factor de riesgo ---
    st.markdown("**Contribucion por factor de riesgo**")
    for fid, d in detalle.items():
        # Omitimos factores que no aplican a este paciente (sin puntos ni explicacion)
        if d["puntos"] == 0 and not d["explicacion"]:
            continue

        pct_factor = d["pct"]  # Porcentaje del maximo del factor que se alcanzo

        # Icono de semaforo por factor: verde <40%, amarillo 40-70%, rojo >70%
        icono = "verde" if pct_factor < 40 else ("amarillo" if pct_factor < 70 else "rojo")

        # Los expanders con contribucion alta (>60% del peso) se abren por defecto
        with st.expander(
            f"{icono} {d['etiqueta']}  -  {d['puntos']:.1f} / {d['peso_max']} pts",
            expanded=(d["puntos"] > d["peso_max"] * 0.6),
        ):
            if d["explicacion"]:
                st.caption(d["explicacion"])
            # Mini-barra de progreso: que tan cerca esta el factor de su maximo
            st.progress(pct_factor / 100)

    # --- Recomendacion clinica segun el nivel de riesgo ---
    st.markdown("**Recomendacion clinica**")
    recomendaciones = {
        "Bajo": (
            "El paciente tiene riesgo bajo en este momento. "
            "Mantener seguimiento anual y reforzar habitos preventivos."
        ),
        "Moderado": (
            "Riesgo moderado. Revisar adherencia a tratamiento farmacologico, "
            "reforzar dieta y actividad fisica. Cita de seguimiento en 3-6 meses."
        ),
        "Alto": (
            "Riesgo alto. Se recomienda consulta medica en el proximo mes, "
            "ajuste de tratamiento y evaluacion de comorbilidades. "
            "Considerar referencia a segundo nivel si no hay mejoria."
        ),
        "Muy alto": (
            "Riesgo muy alto. Atencion prioritaria. Evaluar hospitalizacion o "
            "referencia urgente a segundo/tercer nivel. "
            "Revisar todos los factores modificables de inmediato."
        ),
    }
    st.info(recomendaciones.get(nivel, ""))


# =============================================================================
# SECCION 7: TABLA PRIORIZABLE PARA ADMINISTRACION Y GOBIERNO
# =============================================================================

def render_tabla_riesgo(df: pd.DataFrame) -> None:
    """
    Renderiza en Streamlit una tabla interactiva con todos los pacientes
    de la poblacion filtrada, ordenados por score de riesgo descendente.

    Esta vista esta orientada a gestores y personal administrativo que
    necesitan priorizar la atencion sin revisar el detalle clinico de
    cada paciente individualmente.

    Componentes de la vista:
      1. KPIs de distribucion: cuatro metricas con el conteo y porcentaje
         de pacientes en cada nivel de riesgo (Bajo / Moderado / Alto / Muy alto).
      2. Filtro por nivel: multiselect que por defecto muestra solo los
         pacientes de riesgo Alto y Muy alto (los mas urgentes).
      3. Tabla interactiva: columnas con ID, semaforo, score (ProgressColumn),
         nivel, datos demograficos y presencia de cada enfermedad cronica.
         Ordenada por score descendente.
      4. Descarga en CSV: enlace para guardar la lista completa del filtro activo.

    Si la columna 'riesgo_score' no existe en el DataFrame, se calcula
    automaticamente antes de construir la vista.

    Parametros:
        df (pd.DataFrame): DataFrame de pacientes con columnas clinicas.
                           Idealmente ya procesado por calcular_score_df().

    Retorna:
        None. Escribe directamente en la interfaz de Streamlit.
    """
    import streamlit as st
    import base64

    # Calculamos los scores si no se hizo previamente
    if "riesgo_score" not in df.columns:
        df = calcular_score_df(df)

    st.markdown("### Distribucion de riesgo en la poblacion filtrada")

    # --- KPIs: conteo y porcentaje por nivel de riesgo ---
    niveles = ["Bajo", "Moderado", "Alto", "Muy alto"]
    etiquetas_nivel = {
        "Bajo": "Bajo", "Moderado": "Moderado",
        "Alto": "Alto", "Muy alto": "Muy alto"
    }
    cols = st.columns(4)
    for i, niv in enumerate(niveles):
        n   = int((df["riesgo_nivel"] == niv).sum())
        pct = n / len(df) * 100 if len(df) else 0
        cols[i].metric(
            f"{etiquetas_nivel[niv]}",
            f"{n:,}",
            f"{pct:.1f}% del total",
        )

    st.markdown("---")

    # --- Filtro por nivel de riesgo ---
    # Por defecto muestra Alto y Muy alto para enfocar la atencion en urgentes
    nivel_sel = st.multiselect(
        "Filtrar por nivel de riesgo",
        options=niveles,
        default=["Alto", "Muy alto"],
    )
    df_mostrar = df[df["riesgo_nivel"].isin(nivel_sel)] if nivel_sel else df

    # --- Construccion de la tabla compacta ---
    # Solo incluimos las columnas que existen en el DataFrame (tolerancia a datos faltantes)
    cols_tabla = [
        "id_paciente", "riesgo_semaforo", "riesgo_score",
        "riesgo_nivel", "edad", "sexo", "municipio",
        "diabetes", "hipertension", "dislipidemia", "obesidad"
    ]
    cols_tabla = [c for c in cols_tabla if c in df_mostrar.columns]

    # Ordenamos por score descendente para que los pacientes mas urgentes esten arriba
    df_tabla = (
        df_mostrar[cols_tabla]
        .sort_values("riesgo_score", ascending=False)
        .reset_index(drop=True)
    )

    # Renombramos columnas para una presentacion mas compacta en la tabla
    rename_cols = {
        "id_paciente":    "ID Paciente",
        "riesgo_semaforo": "",           # El semaforo no necesita encabezado
        "riesgo_score":   "Score",
        "riesgo_nivel":   "Nivel",
        "edad":           "Edad",
        "sexo":           "Sexo",
        "municipio":      "Municipio",
        "diabetes":       "DM",          # Abreviatura clinica estandar
        "hipertension":   "HTA",
        "dislipidemia":   "DLP",
        "obesidad":       "OB",
    }
    df_tabla = df_tabla.rename(
        columns={k: v for k, v in rename_cols.items() if k in df_tabla.columns}
    )

    st.dataframe(
        df_tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            # ProgressColumn: visualiza el score como barra de progreso en la tabla
            "Score": st.column_config.ProgressColumn(
                "Score de riesgo",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
        },
    )

    st.caption(
        f"Mostrando {len(df_tabla):,} pacientes de {len(df):,} en el filtro actual. "
        "Ordenados por score descendente."
    )

    # --- Descarga en CSV ---
    # Usamos base64 para generar un enlace de descarga compatible con todos los navegadores
    csv = df_mostrar.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    st.markdown(
        f'<a href="data:application/octet-stream;base64,{b64}" '
        f'download="pacientes_por_riesgo.csv">Descargar lista de pacientes</a>',
        unsafe_allow_html=True,
    )