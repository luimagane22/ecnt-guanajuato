"""
conclusiones.py
===============
Modulo de interpretacion automatica de datos y generacion de conclusiones
en lenguaje natural para el panel ECNT Guanajuato.
Descripcion general:
    Este modulo analiza el DataFrame de pacientes ya filtrado y produce
    textos interpretativos estructurados que se muestran en la pagina
    de inicio del panel.

    El modulo se organiza en dos capas:
      1. Funciones de analisis (prefijo '_'): cada una examina un aspecto
         especifico de la poblacion (prevalencias, control metabolico,
         riesgo, seguimiento, demografia, municipios) y retorna un par
         (hallazgos_cortos, parrafo_detallado) en formato Markdown.
      2. Funcion publica render_conclusiones(): orquesta todas las funciones
         de analisis, consolida los hallazgos mas relevantes y los presenta
         en la interfaz de Streamlit con pestanas, expanders y recomendaciones.

Salidas que genera:
    - Resumen ejecutivo con los 5 hallazgos clinicos mas relevantes.
    - Analisis detallado expandible por tema (riesgo, control metabolico,
      prevalencia, seguimiento, demografia, comparativa municipal).
    - Analisis especifico por municipio con selector interactivo.
    - Recomendaciones de accion priorizadas segun los hallazgos detectados.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# SECCION 1: FUNCIONES AUXILIARES DE FORMATO
# =============================================================================

def _pct(n: int, total: int) -> str:
    """
    Formatea una proporcion como porcentaje con un decimal.

    Maneja el caso de division por cero retornando "0.0%" cuando
    el total es cero, evitando excepciones en poblaciones vacias.

    Parametros:
        n     (int): Numerador (cantidad de casos).
        total (int): Denominador (total de la poblacion).

    Retorna:
        str: Cadena con el porcentaje formateado, p.ej. "34.7%".
    """
    if total == 0:
        return "0.0%"
    return f"{n / total * 100:.1f}%"


def _nivel_prevalencia(pct: float) -> str:
    """
    Clasifica una prevalencia numerica en una categoria textual.

    Los umbrales se basan en criterios epidemiologicos generales
    para enfermedades cronicas no transmisibles:
      - >= 60%: muy alta
      - 40–59%: alta
      - 20–39%: moderada
      - < 20%:  baja

    Parametros:
        pct (float): Porcentaje de prevalencia (0–100).

    Retorna:
        str: Categoria de prevalencia en texto ("muy alta", "alta",
             "moderada" o "baja").
    """
    if pct >= 60:
        return "muy alta"
    elif pct >= 40:
        return "alta"
    elif pct >= 20:
        return "moderada"
    else:
        return "baja"


# =============================================================================
# SECCION 2: FUNCIONES DE ANALISIS POR TEMA
# Cada funcion examina un aspecto especifico de la poblacion y retorna:
#   - hallazgos (List[str]): frases cortas listas para el resumen ejecutivo
#   - detalle   (str):       parrafo largo en Markdown para el expander
# Todas tienen prefijo '_' porque son de uso interno del modulo.
# =============================================================================

def _analizar_prevalencias(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Analiza las prevalencias de las cuatro enfermedades cronicas principales:
    diabetes, hipertension arterial, dislipidemia y obesidad.

    Para cada enfermedad calcula el conteo y el porcentaje de pacientes
    afectados. Adicionalmente evalua:
      - Multimorbilidad: pacientes con 2 o mas enfermedades simultaneas.
      - Sindrome metabolico completo: pacientes con las 4 enfermedades activas.

    La multimorbilidad es un indicador importante porque implica mayor
    complejidad clinica, mayor consumo de recursos y mayor riesgo cardiovascular.

    Parametros:
        df (pd.DataFrame): DataFrame de pacientes con columnas binarias
                           'diabetes', 'hipertension', 'dislipidemia', 'obesidad'.

    Retorna:
        Tuple[List[str], str]:
          - Lista de hallazgos cortos (maximo 3) para el resumen ejecutivo.
          - Parrafo detallado en Markdown con la tabla de prevalencias y
            la discusion de multimorbilidad.
          Retorna ([], "") si el DataFrame no contiene ninguna columna de enfermedad.
    """
    n    = len(df)
    enfs = {
        "diabetes":     "diabetes",
        "hipertension": "hipertension arterial",
        "dislipidemia": "dislipidemia",
        "obesidad":     "obesidad",
    }

    # Calculamos conteo y porcentaje para cada enfermedad presente en el DataFrame
    conteos = {}
    for col, nombre in enfs.items():
        if col in df.columns:
            conteos[col] = {
                "nombre": nombre,
                "n":   int((df[col] == 1).sum()),
                "pct": (df[col] == 1).mean() * 100,
            }

    if not conteos:
        return [], ""

    # Identificamos la enfermedad mas y menos prevalente para los hallazgos
    mas_prev   = max(conteos.values(), key=lambda x: x["pct"])
    menos_prev = min(conteos.values(), key=lambda x: x["pct"])

    # Calculamos multimorbilidad: sumamos cuantas enfermedades tiene cada paciente
    cols_enf = [c for c in enfs if c in df.columns]
    n_enfs   = df[cols_enf].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    n_multimorbilidad = int((n_enfs >= 2).sum())   # 2 o mas enfermedades
    n_sindrome        = int((n_enfs == 4).sum())   # Las 4 enfermedades: sindrome metabolico completo

    hallazgos = []

    # Hallazgo 1: enfermedad con mayor prevalencia y su clasificacion epidemiologica
    nivel = _nivel_prevalencia(mas_prev["pct"])
    hallazgos.append(
        f"La **{mas_prev['nombre']}** es la enfermedad mas frecuente, "
        f"presente en **{mas_prev['n']:,} pacientes** ({_pct(mas_prev['n'], n)}), "
        f"una prevalencia {nivel}."
    )

    # Hallazgo 2: multimorbilidad (solo si hay pacientes afectados)
    if n_multimorbilidad > 0:
        hallazgos.append(
            f"**{n_multimorbilidad:,} pacientes** ({_pct(n_multimorbilidad, n)}) "
            f"padecen 2 o mas enfermedades cronicas simultaneamente, "
            f"lo que aumenta significativamente su riesgo cardiovascular."
        )

    # Hallazgo 3: sindrome metabolico completo (las 4 enfermedades presentes)
    if n_sindrome > 0:
        hallazgos.append(
            f"**{n_sindrome:,} pacientes** ({_pct(n_sindrome, n)}) presentan "
            f"las 4 enfermedades cronicas principales (sindrome metabolico completo) "
            f"y representan la poblacion de mayor prioridad clinica."
        )

    # Construccion del parrafo detallado con una linea por enfermedad
    lineas = ["**Prevalencia por enfermedad:**"]
    for col, datos in conteos.items():
        nivel = _nivel_prevalencia(datos["pct"])
        lineas.append(
            f"- **{datos['nombre'].capitalize()}**: {datos['n']:,} pacientes "
            f"({_pct(datos['n'], n)}) - prevalencia {nivel}."
        )
    lineas.append("")
    lineas.append(
        f"La **{menos_prev['nombre']}** es la menos frecuente con "
        f"{_pct(menos_prev['n'], n)} de los pacientes registrados."
    )
    if n_multimorbilidad > 0:
        lineas.append(
            f"\nDe los {n:,} pacientes, **{n_multimorbilidad:,} ({_pct(n_multimorbilidad, n)})** "
            f"tienen 2 o mas enfermedades cronicas, lo que implica mayor complejidad "
            f"en su manejo clinico y mayores costos de atencion."
        )

    return hallazgos, "\n".join(lineas)


def _analizar_control_metabolico(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Evalua que tan controlados se encuentran los pacientes segun sus
    parametros de laboratorio mas relevantes.

    Indicadores evaluados:
      - HbA1c en pacientes diabeticos: meta < 7.0% (ADA). Se clasifica
        adicionalmente el descontrol severo (HbA1c >= 9%).
      - Presion arterial sistolica en hipertensos: meta < 130 mmHg (JNC-8).
        Se identifica hipertension grado 2 (PA >= 160 mmHg).
      - LDL en pacientes con dislipidemia: meta < 100 mg/dL (ATP III).

    Cada indicador genera un hallazgo con la interpretacion del nivel de
    control (desde "situacion preocupante" hasta "buen nivel de control").

    Parametros:
        df (pd.DataFrame): DataFrame de pacientes con columnas clinicas.

    Retorna:
        Tuple[List[str], str]:
          - Lista de hallazgos cortos para el resumen ejecutivo.
          - Parrafo detallado en Markdown con los porcentajes de control
            para cada indicador evaluado.
    """
    n        = len(df)
    hallazgos = []
    lineas    = ["**Control metabolico de la poblacion:**"]

    # --- HbA1c en pacientes diabeticos ---
    if "hba1c_pct" in df.columns and "diabetes" in df.columns:
        diabeticos = df[df["diabetes"] == 1]
        n_diab     = len(diabeticos)
        if n_diab > 0:
            hba1c = pd.to_numeric(diabeticos["hba1c_pct"], errors="coerce").dropna()
            if len(hba1c) > 0:
                n_controlados = int((hba1c < 7.0).sum())   # En meta terapeutica
                n_criticos    = int((hba1c >= 9.0).sum())  # Descontrol severo
                pct_ctrl      = n_controlados / n_diab * 100
                pct_crit      = n_criticos / n_diab * 100

                # Clasificacion cualitativa del nivel de control glucemico
                if pct_ctrl < 30:
                    interpretacion = "una situacion preocupante que requiere intervencion urgente"
                elif pct_ctrl < 50:
                    interpretacion = "un nivel de control insuficiente"
                elif pct_ctrl < 70:
                    interpretacion = "un nivel de control moderado con margen de mejora"
                else:
                    interpretacion = "un buen nivel de control glucemico"

                hallazgos.append(
                    f"Solo el **{pct_ctrl:.1f}%** de los pacientes diabeticos "
                    f"tiene HbA1c en meta (< 7%), lo que representa {interpretacion}. "
                    f"**{n_criticos:,} pacientes** ({pct_crit:.1f}%) tienen HbA1c >= 9% "
                    f"y requieren atencion prioritaria."
                )
                lineas.append(
                    f"- **Diabetes (HbA1c):** {n_controlados:,} de {n_diab:,} diabeticos "
                    f"({pct_ctrl:.1f}%) tienen glucemia controlada (HbA1c < 7%). "
                    f"{n_criticos:,} pacientes ({pct_crit:.1f}%) presentan HbA1c >= 9% "
                    f"(descontrol severo)."
                )

    # --- Presion arterial sistolica en hipertensos ---
    if "presion_sistolica" in df.columns and "hipertension" in df.columns:
        hipertensos = df[df["hipertension"] == 1]
        n_htn       = len(hipertensos)
        if n_htn > 0:
            pa = pd.to_numeric(hipertensos["presion_sistolica"], errors="coerce").dropna()
            if len(pa) > 0:
                n_ctrl_pa  = int((pa < 130).sum())   # Presion en meta
                n_crit_pa  = int((pa >= 160).sum())  # Hipertension grado 2
                pct_ctrl_pa = n_ctrl_pa / n_htn * 100
                hallazgos.append(
                    f"El **{pct_ctrl_pa:.1f}%** de los hipertensos tiene presion arterial "
                    f"sistolica en meta (< 130 mmHg). "
                    f"**{n_crit_pa:,} pacientes** presentan hipertension grado 2 "
                    f"(PA >= 160 mmHg) con riesgo cardiovascular elevado."
                )
                lineas.append(
                    f"- **Hipertension (PA sistolica):** {n_ctrl_pa:,} de {n_htn:,} hipertensos "
                    f"({pct_ctrl_pa:.1f}%) tienen presion arterial controlada. "
                    f"{n_crit_pa:,} pacientes ({_pct(n_crit_pa, n_htn)}) "
                    f"presentan HTA grado 2 o mayor."
                )

    # --- LDL en pacientes con dislipidemia ---
    if "ldl" in df.columns and "dislipidemia" in df.columns:
        dislip = df[df["dislipidemia"] == 1]
        n_dlp  = len(dislip)
        if n_dlp > 0:
            ldl = pd.to_numeric(dislip["ldl"], errors="coerce").dropna()
            if len(ldl) > 0:
                n_ctrl_ldl  = int((ldl < 100).sum())   # LDL en meta para alto riesgo
                pct_ctrl_ldl = n_ctrl_ldl / n_dlp * 100
                lineas.append(
                    f"- **Dislipidemia (LDL):** {n_ctrl_ldl:,} de {n_dlp:,} pacientes "
                    f"({pct_ctrl_ldl:.1f}%) tienen LDL en meta (< 100 mg/dL)."
                )

    return hallazgos, "\n".join(lineas)


def _analizar_riesgo(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Analiza la distribucion de los niveles de riesgo cardiovascular
    de la poblacion a partir del score calculado por calcular_score_df().

    El score de riesgo (0–100) se clasifica en cuatro niveles:
      - Bajo      (score < 30)
      - Moderado  (score 30–54)
      - Alto      (score 55–74)
      - Muy alto  (score >= 75)

    Se considera de especial importancia el grupo de riesgo alto + muy alto,
    ya que estos pacientes requieren priorizacion en la agenda de consulta
    y posible referencia a segundo nivel de atencion.

    Parametros:
        df (pd.DataFrame): DataFrame con la columna 'riesgo_nivel' (generada
                           por calcular_score_df()) y opcionalmente 'riesgo_score'.

    Retorna:
        Tuple[List[str], str]:
          - Lista con el hallazgo sobre pacientes de riesgo alto/muy alto.
          - Parrafo con la distribucion completa por nivel y el score promedio.
          Retorna ([], "") si la columna 'riesgo_nivel' no existe.
    """
    if "riesgo_nivel" not in df.columns:
        return [], ""

    n      = len(df)
    niveles = ["Bajo", "Moderado", "Alto", "Muy alto"]

    # Contamos cuantos pacientes hay en cada nivel de riesgo
    conteos_riesgo = {niv: int((df["riesgo_nivel"] == niv).sum()) for niv in niveles}
    n_alto     = conteos_riesgo["Alto"] + conteos_riesgo["Muy alto"]
    n_muy_alto = conteos_riesgo["Muy alto"]

    hallazgos = []
    if n_alto > 0:
        hallazgos.append(
            f"**{n_alto:,} pacientes** ({_pct(n_alto, n)}) tienen riesgo cardiovascular "
            f"alto o muy alto y deben ser priorizados para atencion. "
            f"De estos, **{n_muy_alto:,}** ({_pct(n_muy_alto, n)}) requieren "
            f"intervencion urgente."
        )

    # Tabla de distribucion por nivel con indicadores visuales de color
    lineas  = ["**Distribucion por nivel de riesgo cardiovascular:**"]
    etiquetas = {"Bajo": "Bajo", "Moderado": "Moderado", "Alto": "Alto", "Muy alto": "Muy alto"}
    for niv in niveles:
        cnt = conteos_riesgo[niv]
        lineas.append(
            f"- **{etiquetas[niv]}**: {cnt:,} pacientes ({_pct(cnt, n)})"
        )

    # Score promedio de la poblacion (si la columna numerica esta disponible)
    if "riesgo_score" in df.columns:
        scores = pd.to_numeric(df["riesgo_score"], errors="coerce").dropna()
        if len(scores) > 0:
            lineas.append(
                f"\nEl score promedio de la poblacion es **{scores.mean():.1f}/100**, "
                f"con un maximo de {scores.max():.0f} puntos."
            )

    return hallazgos, "\n".join(lineas)


def _analizar_seguimiento(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Analiza el estado de seguimiento y adherencia de los pacientes
    a partir de las alertas generadas por el modulo de seguimiento.

    Categorias de alerta evaluadas (buscadas como subcadenas en la
    columna 'seguimiento_tipo_alerta'):
      - "Perdido": pacientes sin consulta en los ultimos 6 meses.
        Son prioritarios para busqueda activa mediante visita domiciliaria.
      - "sin control": pacientes que asisten pero no logran control metabolico.
      - "deterioro": pacientes cuyos indicadores empeoran entre visitas,
        lo que sugiere que el esquema de tratamiento no es efectivo.

    Parametros:
        df (pd.DataFrame): DataFrame con la columna 'seguimiento_tipo_alerta'
                           y opcionalmente 'seguimiento_tiene_alerta'.

    Retorna:
        Tuple[List[str], str]:
          - Lista de hallazgos sobre pacientes perdidos y en deterioro.
          - Parrafo con el resumen de cada categoria de alerta de seguimiento.
          Retorna ([], "") si la columna 'seguimiento_tipo_alerta' no existe.
    """
    if "seguimiento_tipo_alerta" not in df.columns:
        return [], ""

    n = len(df)

    # Contamos cada tipo de alerta buscando subcadenas en el campo de texto
    n_perdidos    = int(df["seguimiento_tipo_alerta"].str.contains("Perdido",     na=False).sum())
    n_sin_control = int(df["seguimiento_tipo_alerta"].str.contains("sin control", na=False).sum())
    n_deterioro   = int(df["seguimiento_tipo_alerta"].str.contains("deterioro",   na=False).sum())

    # Total de pacientes con cualquier tipo de alerta de seguimiento activa
    n_con_alerta = (
        int(df["seguimiento_tiene_alerta"].sum())
        if "seguimiento_tiene_alerta" in df.columns
        else 0
    )

    hallazgos = []

    # Hallazgo 1: pacientes perdidos para seguimiento (mas de 6 meses sin consulta)
    if n_perdidos > 0:
        hallazgos.append(
            f"**{n_perdidos:,} pacientes** ({_pct(n_perdidos, n)}) llevan mas de "
            f"6 meses sin consulta y se consideran perdidos para el seguimiento. "
            f"Se recomienda busqueda activa mediante visita domiciliaria."
        )

    # Hallazgo 2: pacientes con deterioro progresivo en sus indicadores
    if n_deterioro > 0:
        hallazgos.append(
            f"**{n_deterioro:,} pacientes** ({_pct(n_deterioro, n)}) muestran "
            f"deterioro progresivo en sus indicadores clinicos entre visitas consecutivas, "
            f"lo que sugiere que el tratamiento actual no esta siendo efectivo."
        )

    # Construccion del parrafo detallado
    lineas = ["**Estado de seguimiento de la poblacion:**"]
    if n_perdidos > 0:
        lineas.append(
            f"- **Pacientes perdidos** (sin consulta > 6 meses): "
            f"{n_perdidos:,} ({_pct(n_perdidos, n)})"
        )
    if n_sin_control > 0:
        lineas.append(
            f"- **Sin control metabolico**: {n_sin_control:,} ({_pct(n_sin_control, n)})"
        )
    if n_deterioro > 0:
        lineas.append(
            f"- **En deterioro progresivo**: {n_deterioro:,} ({_pct(n_deterioro, n)})"
        )
    if n_con_alerta > 0:
        lineas.append(
            f"- **Total con alguna alerta de seguimiento**: "
            f"{n_con_alerta:,} ({_pct(n_con_alerta, n)})"
        )

    return hallazgos, "\n".join(lineas)


def _analizar_demografia(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Analiza el perfil demografico de la poblacion registrada.

    Variables evaluadas:
      - Edad: promedio, rango y proporcion de pacientes de 60 anos o mas.
        Este grupo etario representa mayor vulnerabilidad clinica y requiere
        seguimiento mas frecuente y atencion especializada en geriatria.
      - Sexo: distribucion porcentual por categoria.
      - Origen indigena: si supera el 5%, se menciona la necesidad de
        consideraciones culturales en los protocolos de atencion.

    Parametros:
        df (pd.DataFrame): DataFrame con las columnas 'edad', 'sexo' y
                           opcionalmente 'origen_indigena'.

    Retorna:
        Tuple[List[str], str]:
          - Lista con hallazgo sobre envejecimiento (si edad promedio >= 55).
          - Parrafo con la distribucion demografica completa.
    """
    n         = len(df)
    hallazgos = []
    lineas    = ["**Perfil demografico:**"]

    # --- Distribucion de edad ---
    if "edad" in df.columns:
        edad = pd.to_numeric(df["edad"], errors="coerce").dropna()
        if len(edad) > 0:
            n_mayores = int((edad >= 60).sum())  # Adultos mayores: grupo de mayor riesgo
            edad_prom = edad.mean()

            lineas.append(
                f"- Edad promedio: **{edad_prom:.1f} anos** "
                f"(rango: {edad.min():.0f}-{edad.max():.0f} anos)"
            )
            if n_mayores > 0:
                lineas.append(
                    f"- Pacientes de 60 anos o mas: **{n_mayores:,}** "
                    f"({_pct(n_mayores, n)}) - grupo de mayor vulnerabilidad."
                )
            # Solo generamos un hallazgo si la poblacion tiene un perfil
            # de envejecimiento significativo (edad promedio >= 55)
            if edad_prom >= 55:
                hallazgos.append(
                    f"La poblacion registrada tiene una edad promedio de "
                    f"**{edad_prom:.1f} anos**, con **{n_mayores:,} pacientes mayores de 60** "
                    f"({_pct(n_mayores, n)}), un grupo que requiere atencion especializada "
                    f"y seguimiento mas frecuente."
                )

    # --- Distribucion por sexo biologico ---
    if "sexo" in df.columns:
        dist   = df["sexo"].value_counts(normalize=True) * 100
        partes = [f"{k}: {v:.1f}%" for k, v in dist.items()]
        lineas.append(f"- Distribucion por sexo: {', '.join(partes)}")

    # --- Poblacion indigena: consideraciones culturales en salud ---
    if "origen_indigena" in df.columns:
        pct_ind = df["origen_indigena"].mean() * 100
        if pct_ind > 5:
            lineas.append(
                f"- Poblacion de origen indigena: **{pct_ind:.1f}%** - "
                f"requiere consideraciones culturales en la atencion."
            )

    return hallazgos, "\n".join(lineas)


def _analizar_municipios(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Identifica los municipios con mayor carga de enfermedad y riesgo,
    generando una comparativa para orientar la asignacion de recursos.

    Analisis realizados:
      - Municipio con mayor numero de pacientes registrados (volumen).
      - Municipio con mayor prevalencia de diabetes e hipertension (porcentual).
      - Municipio con mayor numero de pacientes perdidos para seguimiento.
      - Municipio con mayor concentracion de pacientes de riesgo alto/muy alto.

    Esta informacion es util para priorizar visitas de brigadas de salud,
    apertura de nuevos servicios o refuerzo de los existentes.

    Parametros:
        df (pd.DataFrame): DataFrame con las columnas 'municipio', columnas
                           binarias de enfermedades, y opcionalmente
                           'seguimiento_tipo_alerta' y 'riesgo_nivel'.

    Retorna:
        Tuple[List[str], str]:
          - Lista de hallazgos sobre municipios con mayor carga o riesgo.
          - Parrafo con el comparativo municipal detallado.
          Retorna ([], "") si la columna municipio no existe o hay menos de 2.
    """
    if "municipio" not in df.columns or df["municipio"].nunique() < 2:
        return [], ""

    hallazgos = []
    lineas    = ["**Analisis por municipio:**"]

    # Municipio con el mayor numero absoluto de pacientes registrados
    top_mun    = df["municipio"].value_counts()
    mun_mayor  = top_mun.index[0]
    n_mayor    = top_mun.iloc[0]

    # Prevalencia de diabetes e hipertension por municipio (porcentaje)
    enfs_analizar = ["diabetes", "hipertension"]
    for enf in enfs_analizar:
        if enf not in df.columns:
            continue
        prev_mun = (
            df.groupby("municipio")[enf]
            .apply(lambda x: (x == 1).mean() * 100)
            .sort_values(ascending=False)
        )
        if len(prev_mun) > 0:
            mun_top    = prev_mun.index[0]
            pct_top    = prev_mun.iloc[0]
            nombre_enf = "diabetes" if enf == "diabetes" else "hipertension"
            lineas.append(
                f"- Mayor prevalencia de {nombre_enf}: **{mun_top}** ({pct_top:.1f}%)"
            )

    # Municipio con mayor numero absoluto de pacientes perdidos para seguimiento
    if "seguimiento_tipo_alerta" in df.columns:
        perdidos_mun = (
            df[df["seguimiento_tipo_alerta"].str.contains("Perdido", na=False)]
            .groupby("municipio")
            .size()
            .sort_values(ascending=False)
        )
        if len(perdidos_mun) > 0:
            mun_perdidos = perdidos_mun.index[0]
            n_perd       = perdidos_mun.iloc[0]
            hallazgos.append(
                f"El municipio de **{mun_perdidos}** concentra la mayor cantidad "
                f"de pacientes perdidos para seguimiento ({n_perd:,}), "
                f"lo que sugiere priorizar la busqueda activa en esa area."
            )
            lineas.append(
                f"- Municipio con mas pacientes perdidos: **{mun_perdidos}** ({n_perd:,})"
            )

    # Municipio con mayor numero de pacientes de riesgo alto o muy alto
    if "riesgo_nivel" in df.columns:
        alto_mun = (
            df[df["riesgo_nivel"].isin(["Alto", "Muy alto"])]
            .groupby("municipio")
            .size()
            .sort_values(ascending=False)
        )
        if len(alto_mun) > 0:
            mun_riesgo = alto_mun.index[0]
            n_riesgo   = alto_mun.iloc[0]
            hallazgos.append(
                f"**{mun_riesgo}** es el municipio con mayor numero de pacientes "
                f"de riesgo alto o muy alto ({n_riesgo:,}), "
                f"lo que indica una necesidad urgente de reforzar los servicios de salud en esa area."
            )
            lineas.append(
                f"- Municipio con mas pacientes de alto riesgo: "
                f"**{mun_riesgo}** ({n_riesgo:,})"
            )

    lineas.append(
        f"\nEl municipio con mayor numero de pacientes registrados es "
        f"**{mun_mayor}** con {n_mayor:,} registros."
    )

    return hallazgos, "\n".join(lineas)


# =============================================================================
# SECCION 3: GENERADOR DE RECOMENDACIONES DE ACCION
# =============================================================================

def _generar_recomendaciones(df: pd.DataFrame) -> str:
    """
    Genera recomendaciones de accion priorizadas a partir de los hallazgos
    detectados en el DataFrame de pacientes.

    Las recomendaciones se activan cuando se superan umbrales especificos:
      - HbA1c >= 9% en mas del 10% de los diabeticos:
        -> Intervencion glucemica urgente (revision de esquemas farmacologicos).
      - Mas del 15% de pacientes perdidos para seguimiento:
        -> Activar brigadas de busqueda activa domiciliaria.
      - Mas del 20% de pacientes con riesgo alto o muy alto:
        -> Refuerzo de la capacidad de referencia a segundo nivel.
      - Mas del 10% de pacientes con 3 o mas enfermedades simultaneas:
        -> Implementar modelo de atencion integrado para multimorbilidad.
      - Si ninguno de los criterios anteriores se cumple:
        -> Mensaje de seguimiento regular y prevencion.

    Parametros:
        df (pd.DataFrame): DataFrame de pacientes con indicadores clinicos.

    Retorna:
        str: Cadena Markdown con las recomendaciones separadas por saltos
             de linea dobles, listas para renderizarse en Streamlit.
    """
    n               = len(df)
    recomendaciones = []

    # Recomendacion 1: intervencion urgente en control glucemico
    if "hba1c_pct" in df.columns and "diabetes" in df.columns:
        diabeticos = df[df["diabetes"] == 1]
        if len(diabeticos) > 0:
            hba1c = pd.to_numeric(diabeticos["hba1c_pct"], errors="coerce").dropna()
            if len(hba1c) > 0 and (hba1c >= 9.0).mean() > 0.1:
                recomendaciones.append(
                    "**Intervencion glucemica urgente:** Mas del 10% de los pacientes "
                    "diabeticos tienen HbA1c >= 9%. Se recomienda revisar y ajustar "
                    "los esquemas de tratamiento farmacologico de forma prioritaria."
                )

    # Recomendacion 2: busqueda activa de pacientes perdidos para seguimiento
    if "seguimiento_tipo_alerta" in df.columns:
        n_perdidos = df["seguimiento_tipo_alerta"].str.contains("Perdido", na=False).sum()
        if n_perdidos / n > 0.15:
            recomendaciones.append(
                "**Busqueda activa de pacientes:** Mas del 15% de los pacientes "
                "llevan mas de 6 meses sin consulta. Se recomienda activar brigadas "
                "de visita domiciliaria focalizadas en los municipios con mayor abandono."
            )

    # Recomendacion 3: refuerzo de segundo nivel de atencion para alto riesgo
    if "riesgo_nivel" in df.columns:
        n_alto = (df["riesgo_nivel"].isin(["Alto", "Muy alto"])).sum()
        if n_alto / n > 0.20:
            recomendaciones.append(
                "**Refuerzo de segundo nivel:** Mas del 20% de los pacientes "
                "tienen riesgo cardiovascular alto o muy alto. Se recomienda evaluar "
                "la capacidad de referencia a segundo nivel y priorizar estos casos "
                "en la agenda de consulta."
            )

    # Recomendacion 4: atencion integrada para pacientes con multimorbilidad
    cols_enf = [c for c in ["diabetes", "hipertension", "dislipidemia", "obesidad"]
                if c in df.columns]
    if cols_enf:
        n_enfs = df[cols_enf].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        if (n_enfs >= 3).mean() > 0.10:
            recomendaciones.append(
                "**Manejo integral de multimorbilidad:** Mas del 10% de los pacientes "
                "tienen 3 o mas enfermedades cronicas simultaneas. Se recomienda "
                "implementar un modelo de atencion integrado que aborde todas las "
                "condiciones en una misma consulta."
            )

    # Si no se activo ninguna recomendacion critica, emitimos un mensaje positivo
    if not recomendaciones:
        recomendaciones.append(
            "Los indicadores generales de la poblacion se encuentran dentro "
            "de rangos aceptables. Se recomienda mantener el seguimiento regular "
            "y continuar con las estrategias de prevencion actuales."
        )

    return "\n\n".join(recomendaciones)


# =============================================================================
# SECCION 4: CONCLUSIONES POR MUNICIPIO
# =============================================================================

def _conclusiones_municipio(df: pd.DataFrame, municipio: str) -> Dict:
    """
    Genera el analisis completo de indicadores para un municipio especifico.

    Filtra el DataFrame al municipio seleccionado y ejecuta las cuatro
    funciones de analisis principales. De cada una toma el hallazgo mas
    relevante (primero de la lista) para construir el resumen ejecutivo
    del municipio, y concatena todos los parrafos detallados.

    Parametros:
        df        (pd.DataFrame): DataFrame completo de pacientes.
        municipio (str):          Nombre del municipio a analizar.

    Retorna:
        Dict con las claves:
          - n                (int):  Numero de pacientes en el municipio.
          - hallazgos        (List[str]): Hasta 4 hallazgos cortos del municipio.
          - detalle          (str):  Parrafos detallados concatenados en Markdown.
          - recomendaciones  (str):  Recomendaciones especificas para el municipio.
        Si no hay datos para el municipio, retorna un dict con listas vacias.
    """
    df_mun = df[df["municipio"] == municipio].copy()
    n      = len(df_mun)

    if n == 0:
        return {"hallazgos": [], "detalle": "No hay datos para este municipio."}

    # Ejecutamos todos los analisis sobre el subconjunto del municipio
    h_prev,   d_prev   = _analizar_prevalencias(df_mun)
    h_ctrl,   d_ctrl   = _analizar_control_metabolico(df_mun)
    h_riesgo, d_riesgo = _analizar_riesgo(df_mun)
    h_seg,    d_seg    = _analizar_seguimiento(df_mun)

    # Tomamos el hallazgo mas relevante de cada area para el resumen del municipio
    hallazgos = h_prev[:1] + h_ctrl[:1] + h_riesgo[:1] + h_seg[:1]

    # Concatenamos todos los parrafos no vacios
    detalles  = [d for d in [d_prev, d_ctrl, d_riesgo, d_seg] if d]

    return {
        "n":               n,
        "hallazgos":       hallazgos,
        "detalle":         "\n\n".join(detalles),
        "recomendaciones": _generar_recomendaciones(df_mun),
    }


# =============================================================================
# SECCION 5: INTERFAZ DE USUARIO EN STREAMLIT (PUNTO DE ENTRADA PRINCIPAL)
# =============================================================================

def render_conclusiones(df_filt: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """
    Punto de entrada principal del modulo. Se llama desde seminario_Luisa.py
    en la pagina de inicio, inmediatamente despues de los KPIs globales.

    Estructura de la interfaz generada:
      - Pestana 1 "Resumen general":
          * Resumen ejecutivo con los 5 hallazgos mas relevantes de la poblacion,
            seleccionados de las areas: riesgo, control metabolico, prevalencias,
            seguimiento y comparativa municipal.
          * Seis expanders con el analisis detallado de cada area.
          * Seccion de recomendaciones priorizadas.
      - Pestana 2 "Por municipio":
          * Selector desplegable de municipios disponibles en los datos.
          * Hallazgos y analisis detallado especifico del municipio elegido.
          * Recomendaciones adaptadas a ese municipio.

    El orden de los hallazgos en el resumen ejecutivo esta intencionalmente
    orientado a la toma de decisiones clinicas: primero riesgo (lo mas urgente),
    luego control (efectividad del tratamiento), prevalencias, seguimiento y
    geografia.

    Parametros:
        df_filt (pd.DataFrame): DataFrame filtrado con los filtros activos del sidebar.
                                 Es el que se usa para todos los analisis.
        df_raw  (pd.DataFrame): DataFrame completo sin filtros. Actualmente no se usa
                                 directamente, pero se recibe para posibles extensiones
                                 que requieran comparar la poblacion filtrada con la total.

    Retorna:
        None. Escribe directamente en la interfaz de Streamlit.
    """
    import streamlit as st

    st.markdown("---")
    st.header("Interpretacion automatica de los datos")
    st.caption(
        "El sistema analiza automaticamente los indicadores clinicos y genera "
        "conclusiones en lenguaje natural basadas en los datos cargados."
    )

    # Creamos dos pestanas: analisis general de la poblacion y analisis por municipio
    tab_general, tab_municipio = st.tabs([
        "Resumen general",
        "Por municipio",
    ])

    # -------------------------------------------------------------------------
    # PESTANA 1: RESUMEN GENERAL DE LA POBLACION
    # -------------------------------------------------------------------------
    with tab_general:

        # Ejecutamos todos los modulos de analisis sobre la poblacion filtrada
        h_prev,   d_prev   = _analizar_prevalencias(df_filt)
        h_ctrl,   d_ctrl   = _analizar_control_metabolico(df_filt)
        h_riesgo, d_riesgo = _analizar_riesgo(df_filt)
        h_seg,    d_seg    = _analizar_seguimiento(df_filt)
        h_demo,   d_demo   = _analizar_demografia(df_filt)
        h_mun,    d_mun    = _analizar_municipios(df_filt)

        # Consolidamos los 5 hallazgos mas importantes ordenados por relevancia clinica:
        # riesgo (urgencia) > control metabolico (efectividad) > prevalencias >
        # seguimiento > municipios
        todos_hallazgos = (
            h_riesgo[:1] + h_ctrl[:1] + h_prev[:1] + h_seg[:1] + h_mun[:1]
        )
        # Eliminamos hallazgos vacios y limitamos a 5
        todos_hallazgos = [h for h in todos_hallazgos if h][:5]

        # --- Resumen ejecutivo: los hallazgos numerados ---
        st.subheader("Hallazgos principales")
        if todos_hallazgos:
            for i, hallazgo in enumerate(todos_hallazgos, 1):
                st.markdown(f"**{i}.** {hallazgo}")
        else:
            st.info("Carga datos con indicadores clinicos para generar conclusiones.")

        st.markdown("")

        # --- Analisis detallado por area (cada uno en un expander colapsable) ---
        # El orden prioriza lo mas accionable primero

        if d_riesgo:
            with st.expander("Analisis de riesgo cardiovascular"):
                st.markdown(d_riesgo)

        if d_ctrl:
            with st.expander("Control metabolico de la poblacion"):
                st.markdown(d_ctrl)

        if d_prev:
            with st.expander("Prevalencia de enfermedades cronicas"):
                st.markdown(d_prev)

        if d_seg:
            with st.expander("Estado de seguimiento y adherencia"):
                st.markdown(d_seg)

        if d_demo:
            with st.expander("Perfil demografico"):
                st.markdown(d_demo)

        if d_mun:
            with st.expander("Analisis comparativo por municipio"):
                st.markdown(d_mun)

        # --- Recomendaciones de accion priorizadas ---
        st.subheader("Recomendaciones de accion")
        recomendaciones = _generar_recomendaciones(df_filt)
        st.markdown(recomendaciones)

    # -------------------------------------------------------------------------
    # PESTANA 2: ANALISIS POR MUNICIPIO
    # -------------------------------------------------------------------------
    with tab_municipio:

        # Obtenemos la lista de municipios disponibles en los datos filtrados
        municipios_disp = (
            sorted(df_filt["municipio"].dropna().unique().tolist())
            if "municipio" in df_filt.columns
            else []
        )

        if not municipios_disp:
            st.info("No hay datos de municipio disponibles.")
        else:
            # Selector desplegable de municipio
            mun_sel = st.selectbox(
                "Selecciona un municipio para ver su analisis",
                options=municipios_disp,
                key="conclusiones_mun_sel",
            )

            # Ejecutamos el analisis especifico para el municipio elegido
            resultado = _conclusiones_municipio(df_filt, mun_sel)
            n_mun     = resultado.get("n", 0)

            # Encabezado con el nombre del municipio y el numero de pacientes
            st.markdown(
                f"**Municipio:** {mun_sel} &nbsp;|&nbsp; "
                f"**Pacientes registrados:** {n_mun:,}"
            )
            st.markdown("")

            # Hallazgos del municipio (hasta 4, uno por area de analisis)
            st.subheader(f"Hallazgos - {mun_sel}")
            hallazgos_mun = resultado.get("hallazgos", [])
            if hallazgos_mun:
                for i, h in enumerate(hallazgos_mun, 1):
                    st.markdown(f"**{i}.** {h}")
            else:
                st.info("No hay suficientes datos clinicos para generar conclusiones.")

            # Analisis detallado del municipio en un expander colapsable
            detalle = resultado.get("detalle", "")
            if detalle:
                with st.expander("Ver analisis detallado"):
                    st.markdown(detalle)

            # Recomendaciones especificas para el municipio seleccionado
            recomendaciones_mun = resultado.get("recomendaciones", "")
            if recomendaciones_mun:
                st.subheader("Recomendaciones")
                st.markdown(recomendaciones_mun)