"""
seguimiento_pacientes.py
========================
Modulo de deteccion de pacientes perdidos para seguimiento y sin control metabolico.

Descripcion general:
    Este modulo identifica, dentro de la poblacion registrada, a los pacientes
    que requieren atencion activa por alguna de las siguientes razones:

      1. Perdido para seguimiento: no se ha registrado ninguna visita en los
         ultimos N meses (configurable, por defecto 6). Estos pacientes pueden
         haber abandonado el tratamiento o haberse atendido en otra unidad.

      2. Sin control metabolico: la ultima visita registrada muestra uno o mas
         indicadores clinicos por encima de las metas terapeuticas definidas en
         las normas mexicanas NOM-015-SSA2 (diabetes) y NOM-030-SSA2 (hipertension).

      3. En deterioro progresivo: al comparar la penultima y ultima visita, uno
         o mas indicadores empeoraron (subieron estando ya sobre la meta), lo que
         sugiere que el esquema de tratamiento actual no esta siendo efectivo.

    Los pacientes se clasifican en cinco categorias de alerta con orden de
    prioridad de atencion:
      1. Perdido + sin control  (prioridad maxima, rojo)
      2. Perdido sin consulta   (naranja)
      3. En deterioro           (amarillo)
      3. Sin control metabolico (amarillo, misma prioridad que deterioro)
      4. Sin fecha registrada   (blanco, dato faltante)
      5. Controlado             (verde, sin accion requerida)

    El modulo requiere que el DataFrame contenga la columna '__fecha__'
    (generada por normalizador_sis2024.py) para calcular el tiempo transcurrido
    desde la ultima visita. Sin ella, todos los pacientes se clasifican en
    la categoria "Sin fecha registrada".

Uso desde seminario_Luisa.py:
    from seguimiento_pacientes import render_panel_seguimiento
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# SECCION 1: METAS CLINICAS DE CONTROL
# =============================================================================
# Diccionario con los umbrales de control metabolico para cada indicador.
# Un paciente se considera "sin control" si el valor del indicador supera
# el umbral meta_max en su ultima visita registrada.
#
# Las metas estan basadas en:
#   - NOM-015-SSA2-2010: Para la prevencion, tratamiento y control de la
#     diabetes mellitus (HbA1c, glucosa en ayunas).
#   - NOM-030-SSA2-2009: Para la prevencion, deteccion, diagnostico,
#     tratamiento y control de la hipertension arterial sistemica (PA).
#   - Consenso de dislipidemia de la Sociedad Mexicana de Cardiologia
#     (LDL, trigliceridos).
#   - OMS: Clasificacion de obesidad por IMC.
#
# Estructura de cada entrada:
#   etiqueta      : nombre legible del indicador para mostrar al medico
#   unidad        : unidad de medicion (p.ej. "mg/dL", "%")
#   meta_max      : umbral maximo; valores >= meta_max se consideran fuera de meta
#   aplica_si     : columna binaria de enfermedad que debe estar activa (== 1)
#                   para que se evalúe este indicador en el paciente
#   mensaje_fuera : texto cuando el indicador esta fuera de meta
#   mensaje_ok    : texto cuando el indicador esta dentro de la meta
# =============================================================================

METAS_CLINICAS: Dict[str, Dict] = {
    # --- Diabetes: control glucemico ---
    "hba1c_pct": {
        "etiqueta":     "HbA1c",
        "unidad":       "%",
        "meta_max":     7.0,        # NOM-015-SSA2: meta < 7% para adultos con DM2
        "aplica_si":    "diabetes",
        "mensaje_fuera": "HbA1c >= 7% (diabetes sin control)",
        "mensaje_ok":    "HbA1c < 7% (diabetes controlada)",
    },
    "glucosa_ayunas": {
        "etiqueta":     "Glucosa en ayuno",
        "unidad":       "mg/dL",
        "meta_max":     130,        # NOM-015-SSA2: glucosa en ayunas < 130 mg/dL
        "aplica_si":    "diabetes",
        "mensaje_fuera": "Glucosa >= 130 mg/dL",
        "mensaje_ok":    "Glucosa < 130 mg/dL",
    },

    # --- Hipertension: presion arterial ---
    "presion_sistolica": {
        "etiqueta":     "Presion sistolica",
        "unidad":       "mmHg",
        "meta_max":     130,        # NOM-030-SSA2 / JNC-8: PA sistolica < 130 mmHg
        "aplica_si":    "hipertension",
        "mensaje_fuera": "PA sistolica >= 130 mmHg",
        "mensaje_ok":    "PA sistolica < 130 mmHg",
    },
    "presion_diastolica": {
        "etiqueta":     "Presion diastolica",
        "unidad":       "mmHg",
        "meta_max":     80,         # NOM-030-SSA2 / JNC-8: PA diastolica < 80 mmHg
        "aplica_si":    "hipertension",
        "mensaje_fuera": "PA diastolica >= 80 mmHg",
        "mensaje_ok":    "PA diastolica < 80 mmHg",
    },

    # --- Dislipidemia: perfil lipidico ---
    "ldl": {
        "etiqueta":     "LDL",
        "unidad":       "mg/dL",
        "meta_max":     100,        # SMC / ATP III: LDL < 100 mg/dL para alto riesgo
        "aplica_si":    "dislipidemia",
        "mensaje_fuera": "LDL >= 100 mg/dL",
        "mensaje_ok":    "LDL < 100 mg/dL",
    },
    "trigliceridos": {
        "etiqueta":     "Trigliceridos",
        "unidad":       "mg/dL",
        "meta_max":     150,        # Consenso SMC: trigliceridos < 150 mg/dL
        "aplica_si":    "dislipidemia",
        "mensaje_fuera": "Trigliceridos >= 150 mg/dL",
        "mensaje_ok":    "Trigliceridos < 150 mg/dL",
    },

    # --- Obesidad ---
    "imc": {
        "etiqueta":     "IMC",
        "unidad":       "",
        "meta_max":     30,         # OMS: IMC >= 30 = obesidad activa
        "aplica_si":    "obesidad",
        "mensaje_fuera": "IMC >= 30 (obesidad activa)",
        "mensaje_ok":    "IMC < 30",
    },
}

# Umbral por defecto de meses sin visita para clasificar a un paciente como "perdido".
# Es configurable por el usuario desde la interfaz (slider en render_panel_seguimiento).
MESES_PERDIDO_DEFAULT = 6


# =============================================================================
# SECCION 2: FUNCIONES DE ANALISIS POR PACIENTE
# Todas tienen prefijo '_' porque son de uso interno del modulo.
# =============================================================================

def _ultima_visita(df_paciente: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Obtiene la fecha de la visita mas reciente registrada para un paciente.

    Parametros:
        df_paciente (pd.DataFrame): Subconjunto del DataFrame con todos los
                                    registros de un mismo paciente, ordenados
                                    de cualquier forma.

    Retorna:
        pd.Timestamp: Fecha maxima de la columna '__fecha__', o
        None si la columna no tiene ningun valor valido (todas son NaT).
    """
    fechas = df_paciente["__fecha__"].dropna()
    return fechas.max() if not fechas.empty else None


def _meses_sin_visita(
    ultima: Optional[pd.Timestamp],
    hoy: pd.Timestamp
) -> Optional[float]:
    """
    Calcula el numero de meses transcurridos entre la ultima visita
    y la fecha de referencia (hoy).

    Usa 30.44 dias como promedio mensual (365.25 / 12) para mayor
    precision que un entero fijo de 30 dias.

    Parametros:
        ultima (pd.Timestamp | None): Fecha de la ultima visita del paciente.
                                      None si no hay fechas registradas.
        hoy    (pd.Timestamp):        Fecha de referencia para el calculo
                                      (normalmente la fecha actual del sistema).

    Retorna:
        float: Meses transcurridos, con un decimal de precision.
        None:  Si la fecha de ultima visita no esta disponible.
    """
    if ultima is None or pd.isna(ultima):
        return None
    delta = hoy - ultima
    return delta.days / 30.44


def _indicadores_fuera_de_meta(fila: pd.Series) -> List[str]:
    """
    Evalua los indicadores clinicos de la ultima visita del paciente y
    detecta cuales superan sus metas terapeuticas.

    Solo evalua los indicadores de las enfermedades que el paciente tiene
    activas (columna binaria == 1), para evitar clasificar como "sin control"
    a pacientes por indicadores que no les aplican clinicamente.

    Ejemplo: un paciente sin diabetes no sera evaluado por HbA1c ni glucosa,
    aunque esas columnas existan en su registro.

    Parametros:
        fila (pd.Series): Fila con los datos de la ultima visita del paciente.
                          Debe contener las columnas de METAS_CLINICAS y las
                          columnas binarias de enfermedades.

    Retorna:
        List[str]: Lista de mensajes descriptivos de cada indicador fuera de meta.
                   Cada mensaje incluye el nombre del indicador, el valor actual
                   y el umbral de referencia. Lista vacia si todos estan en meta.
    """
    fuera = []
    for col, meta in METAS_CLINICAS.items():
        # Omitimos el indicador si la enfermedad correspondiente no aplica al paciente
        enfermedad = meta.get("aplica_si")
        if enfermedad and fila.get(enfermedad, 0) != 1:
            continue

        val = fila.get(col, np.nan)
        if pd.isna(val):
            continue   # Sin dato: no penalizamos

        if val >= meta["meta_max"]:
            fuera.append(
                f"{meta['etiqueta']}: {val:.1f} {meta['unidad']} "
                f"(meta < {meta['meta_max']} {meta['unidad']})"
            )
    return fuera


def _indicadores_en_deterioro(df_paciente: pd.DataFrame) -> List[str]:
    """
    Detecta indicadores clinicos que empeoraron entre la penultima y la
    ultima visita registrada del paciente.

    Criterio de deterioro: un indicador se considera en deterioro si:
      1. Esta por encima de su meta terapeutica (val_actual >= meta_max), Y
      2. Su valor aumento respecto a la visita anterior (val_actual > val_prev).

    La condicion AND es importante: un indicador que sube pero sigue dentro
    de la meta NO se reporta como deterioro. Solo se reporta cuando el
    paciente ya estaba descontrolado y empeoro aun mas.

    Requiere al menos dos visitas con fecha para poder comparar. Si el
    paciente tiene solo un registro, retorna una lista vacia.

    Parametros:
        df_paciente (pd.DataFrame): Todos los registros del paciente con
                                    la columna '__fecha__' disponible.

    Retorna:
        List[str]: Lista de mensajes con el indicador, el valor anterior,
                   el valor actual y el incremento. Lista vacia si no hay
                   deterioro o si no hay suficiente historial.
    """
    # Ordenamos cronologicamente y descartamos registros sin fecha
    df_ord = df_paciente.sort_values("__fecha__").dropna(subset=["__fecha__"])

    if len(df_ord) < 2:
        return []   # Sin historial suficiente para comparar

    ultima    = df_ord.iloc[-1]    # Visita mas reciente
    penultima = df_ord.iloc[-2]    # Visita inmediatamente anterior
    deterioros = []

    for col, meta in METAS_CLINICAS.items():
        # Solo evaluamos indicadores de las enfermedades activas del paciente
        enfermedad = meta.get("aplica_si")
        if enfermedad and ultima.get(enfermedad, 0) != 1:
            continue

        val_actual = ultima.get(col, np.nan)
        val_prev   = penultima.get(col, np.nan)

        # Necesitamos ambos valores para poder comparar
        if pd.isna(val_actual) or pd.isna(val_prev):
            continue

        # Deterioro: el indicador sigue fuera de meta Y aumento respecto a la visita anterior
        if val_actual >= meta["meta_max"] and val_actual > val_prev:
            cambio = val_actual - val_prev
            deterioros.append(
                f"{meta['etiqueta']}: {val_prev:.1f} -> {val_actual:.1f} "
                f"(+{cambio:.1f} {meta['unidad']})"
            )

    return deterioros


def _clasificar_paciente(
    df_paciente: pd.DataFrame,
    hoy: pd.Timestamp,
    meses_umbral: int,
) -> Dict:
    """
    Analiza el historial completo de un paciente y lo clasifica en una
    categoria de alerta de seguimiento.

    Logica de clasificacion (en orden de prioridad decreciente):
      1. Perdido + sin control (prioridad 1, rojo):
         El paciente no ha asistido en mas de meses_umbral meses Y ademas
         tiene indicadores fuera de meta o en deterioro. Es el caso mas
         urgente porque combina abandono del tratamiento con descontrol activo.
      2. Perdido sin consulta (prioridad 2, naranja):
         El paciente no ha asistido en mas de meses_umbral meses, pero sus
         ultimos indicadores estaban en meta. Requiere busqueda activa.
      3. En deterioro (prioridad 3, amarillo):
         El paciente asiste, pero sus indicadores empeoran entre visitas.
         Sugiere que el tratamiento actual necesita ajuste.
      3. Sin control metabolico (prioridad 3, amarillo):
         El paciente asiste, pero sus indicadores estan fuera de meta
         sin mostrar una tendencia de empeoramiento clara.
      4. Sin fecha registrada (prioridad 4, blanco):
         No hay informacion de fechas para este paciente. Se conserva en
         el listado para que el operador pueda verificarlo manualmente.
      5. Controlado (prioridad 5, verde):
         Todos los indicadores evaluados estan dentro de sus metas y
         el paciente asiste dentro del periodo de seguimiento esperado.

    La flag tiene_alerta es True para las prioridades 1-3 (requieren accion).

    Parametros:
        df_paciente  (pd.DataFrame): Todos los registros del paciente.
        hoy          (pd.Timestamp): Fecha de referencia para calcular ausencia.
        meses_umbral (int):          Meses maximos sin visita antes de considerarse perdido.

    Retorna:
        Dict: Diccionario con todas las variables de clasificacion y datos
              demograficos del paciente para construir la tabla de alertas.
              Claves principales:
                id_paciente, ultima_visita, meses_sin_visita, es_perdido,
                sin_datos_fecha, indicadores_fuera, indicadores_deterioro,
                tiene_alerta, tipo_alerta, prioridad,
                municipio, edad, sexo, diabetes, hipertension,
                dislipidemia, obesidad, riesgo_score, riesgo_nivel.
    """
    # Tomamos los datos demograficos de la visita mas reciente
    fila_reciente = df_paciente.sort_values("__fecha__").iloc[-1]
    id_pac        = str(fila_reciente.get("id_paciente", "Desconocido"))

    # Calculamos ausencia
    ultima          = _ultima_visita(df_paciente)
    meses_ausente   = _meses_sin_visita(ultima, hoy)
    sin_datos_fecha = ultima is None or pd.isna(ultima)

    # Paciente perdido: tiene fecha registrada pero lleva mas del umbral sin asistir
    es_perdido = (
        not sin_datos_fecha
        and meses_ausente is not None
        and meses_ausente > meses_umbral
    )

    # Evaluamos estado clinico en la ultima visita
    fuera     = _indicadores_fuera_de_meta(fila_reciente)
    deterioro = _indicadores_en_deterioro(df_paciente)

    # Clasificacion jerarquica: el primer criterio que se cumple define la alerta
    if es_perdido and (fuera or deterioro):
        tipo_alerta = "Perdido + sin control"
        prioridad   = 1
    elif es_perdido:
        tipo_alerta = "Perdido (sin consulta)"
        prioridad   = 2
    elif deterioro:
        tipo_alerta = "En deterioro"
        prioridad   = 3
    elif fuera:
        tipo_alerta = "Sin control metabolico"
        prioridad   = 3
    elif sin_datos_fecha:
        tipo_alerta = "Sin fecha registrada"
        prioridad   = 4
    else:
        tipo_alerta = "Controlado"
        prioridad   = 5

    # tiene_alerta = True para prioridades 1, 2 y 3 (requieren intervencion activa)
    tiene_alerta = prioridad <= 3

    return {
        # Variables de clasificacion de seguimiento
        "id_paciente":            id_pac,
        "ultima_visita":          ultima,
        "meses_sin_visita":       round(meses_ausente, 1) if meses_ausente else None,
        "es_perdido":             es_perdido,
        "sin_datos_fecha":        sin_datos_fecha,
        "indicadores_fuera":      fuera,
        "indicadores_deterioro":  deterioro,
        "tiene_alerta":           tiene_alerta,
        "tipo_alerta":            tipo_alerta,
        "prioridad":              prioridad,
        # Datos demograficos y clinicos de la ultima visita (para la tabla)
        "municipio":    fila_reciente.get("municipio",    ""),
        "edad":         fila_reciente.get("edad",         np.nan),
        "sexo":         fila_reciente.get("sexo",         ""),
        "diabetes":     int(fila_reciente.get("diabetes",     0)),
        "hipertension": int(fila_reciente.get("hipertension", 0)),
        "dislipidemia": int(fila_reciente.get("dislipidemia", 0)),
        "obesidad":     int(fila_reciente.get("obesidad",     0)),
        "riesgo_score": fila_reciente.get("riesgo_score", np.nan),
        "riesgo_nivel": fila_reciente.get("riesgo_nivel", ""),
    }


# =============================================================================
# SECCION 3: CLASIFICACION DE TODO EL DATAFRAME
# =============================================================================

def clasificar_seguimiento_df(
    df: pd.DataFrame,
    meses_umbral: int = MESES_PERDIDO_DEFAULT,
    fecha_referencia: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clasifica el seguimiento de todos los pacientes del DataFrame y agrega
    columnas de alerta al DataFrame original.

    El proceso es:
      1. Agrupa el DataFrame por 'id_paciente'.
      2. Llama a _clasificar_paciente() para cada grupo.
      3. Construye df_alertas: una fila por paciente con todos los resultados.
      4. Hace un merge del df original con las columnas de clasificacion,
         usando prefijo 'seguimiento_' para no colisionar con otras columnas.

    El parametro fecha_referencia permite fijar una fecha de corte distinta
    a la actual, util para pruebas o para analisis retrospectivos.

    Columnas agregadas al DataFrame original (con prefijo 'seguimiento_'):
      seguimiento_tipo_alerta    (str):   Etiqueta de la categoria de alerta.
      seguimiento_prioridad      (int):   Valor numerico de prioridad (1=urgente, 5=ok).
      seguimiento_meses_ausente  (float): Meses desde la ultima visita.
      seguimiento_tiene_alerta   (bool):  True si el paciente requiere accion.

    Parametros:
        df               (pd.DataFrame):     DataFrame de pacientes con '__fecha__'.
        meses_umbral     (int):              Meses sin visita para ser "perdido".
        fecha_referencia (datetime | None):  Fecha de corte. None = fecha actual.

    Retorna:
        Tuple[pd.DataFrame, pd.DataFrame]:
          - df con las cuatro columnas de seguimiento agregadas.
          - df_alertas: una fila por paciente con todos los campos de
            clasificacion (incluye indicadores_fuera e indicadores_deterioro
            como listas, usados en el detalle expandible de la UI).
    """
    # Si no hay columna de fecha, la creamos con NaT para que el modulo no falle
    if "__fecha__" not in df.columns:
        df["__fecha__"] = pd.NaT

    hoy = pd.Timestamp(fecha_referencia or datetime.today())

    # Clasificamos cada paciente de forma independiente agrupando por ID
    resultados = []
    for pid, grupo in df.groupby("id_paciente", sort=False):
        r = _clasificar_paciente(grupo, hoy, meses_umbral)
        resultados.append(r)

    df_alertas = pd.DataFrame(resultados)

    # Hacemos merge solo de las columnas de clasificacion al DataFrame original.
    # Usamos prefijo 'seguimiento_' para evitar colisiones con columnas existentes.
    cols_merge = [
        "id_paciente", "tipo_alerta", "prioridad",
        "meses_sin_visita", "tiene_alerta",
    ]
    df = df.merge(
        df_alertas[cols_merge].rename(columns={
            "tipo_alerta":      "seguimiento_tipo_alerta",
            "prioridad":        "seguimiento_prioridad",
            "meses_sin_visita": "seguimiento_meses_ausente",
            "tiene_alerta":     "seguimiento_tiene_alerta",
        }),
        on="id_paciente",
        how="left",   # Left join: todos los registros del df original se conservan
    )

    return df, df_alertas


# =============================================================================
# SECCION 4: GENERACION DEL EXCEL DESCARGABLE
# =============================================================================

def _generar_excel(df_alertas: pd.DataFrame, meses_umbral: int) -> bytes:
    """
    Genera un archivo Excel con tres hojas que resume el estado de seguimiento
    de los pacientes con alguna alerta activa.

    Estructura del archivo:
      Hoja 1 - "Resumen por alerta":
        Tabla con el numero de pacientes en cada categoria de alerta,
        ordenada de mayor a menor frecuencia.
      Hoja 2 - "Resumen por municipio":
        Tabla con el numero de pacientes con alerta activa agrupados por
        municipio, ordenada de mayor a menor. Util para priorizar visitas
        de brigadas de salud por zona geografica.
      Hoja 3 - "Listado de pacientes":
        Tabla completa con una fila por paciente, incluyendo datos demograficos,
        tipo de alerta, indicadores fuera de meta e indicadores en deterioro.
        Las columnas de listas (indicadores_fuera, indicadores_deterioro) se
        convierten a texto con separador " | " para ser legibles en Excel.
        Las columnas se autoajustan al contenido (maximo 50 caracteres de ancho).

    Esta funcion tiene prefijo '_' porque es de uso interno; el punto de
    entrada publico es render_panel_seguimiento() que la llama al presionar
    el boton de descarga.

    Parametros:
        df_alertas   (pd.DataFrame): DataFrame de alertas generado por
                                     clasificar_seguimiento_df() (ya filtrado
                                     a solo los pacientes con alerta activa).
        meses_umbral (int):          Umbral de meses usado en la clasificacion
                                     (se incluye como contexto en el archivo).

    Retorna:
        bytes: Contenido binario del archivo .xlsx generado en memoria,
               listo para ofrecerse como descarga con st.download_button().
    """
    output = io.BytesIO()

    # Seleccionamos y ordenamos las columnas para la hoja de listado
    cols_export = [
        "id_paciente", "tipo_alerta", "meses_sin_visita", "ultima_visita",
        "municipio", "edad", "sexo",
        "diabetes", "hipertension", "dislipidemia", "obesidad",
        "riesgo_score", "riesgo_nivel",
        "indicadores_fuera", "indicadores_deterioro",
    ]
    cols_export = [c for c in cols_export if c in df_alertas.columns]
    df_export   = df_alertas[cols_export].copy()

    # Convertimos las listas de indicadores a cadenas de texto para Excel
    # (Excel no puede almacenar listas Python nativas)
    for col in ["indicadores_fuera", "indicadores_deterioro"]:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(
                lambda x: " | ".join(x) if isinstance(x, list) else str(x)
            )

    # Ordenamos por prioridad (si existe) o por tipo de alerta alfabeticamente
    sort_col = (
        "prioridad"   if "prioridad"   in df_export.columns else
        "tipo_alerta" if "tipo_alerta" in df_export.columns else
        None
    )
    if sort_col:
        df_export = df_export.sort_values(sort_col)

    # Hoja 2: resumen de pacientes con alerta por categoria
    resumen_alerta = (
        df_alertas
        .groupby("tipo_alerta")
        .size()
        .reset_index(name="n_pacientes")
        .sort_values("n_pacientes", ascending=False)
    )

    # Hoja 3: resumen de pacientes con alerta por municipio
    df_con_alerta = df_alertas[df_alertas["tiene_alerta"]]
    resumen_mun   = (
        df_con_alerta
        .groupby("municipio")
        .size()
        .reset_index(name="pacientes_con_alerta")
        .sort_values("pacientes_con_alerta", ascending=False)
    )

    # Escribimos las tres hojas en el mismo archivo Excel
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        resumen_alerta.to_excel(writer, sheet_name="Resumen por alerta",    index=False)
        resumen_mun.to_excel(   writer, sheet_name="Resumen por municipio", index=False)
        df_export.to_excel(     writer, sheet_name="Listado de pacientes",  index=False)

        # Autoajuste del ancho de columnas en la hoja de pacientes
        ws = writer.sheets["Listado de pacientes"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col) + 4
            ws.column_dimensions[col[0].column_letter].width = min(max_len, 50)

    return output.getvalue()


# =============================================================================
# SECCION 5: PANEL INTERACTIVO EN STREAMLIT (PUNTO DE ENTRADA PRINCIPAL)
# =============================================================================

def render_panel_seguimiento(
    df: pd.DataFrame,
    meses_umbral: int = MESES_PERDIDO_DEFAULT,
) -> None:
    """
    Punto de entrada principal del modulo. Renderiza el panel completo de
    seguimiento de pacientes en la interfaz de Streamlit.

    El panel esta orientado al coordinador de CEAPS o al responsable de
    seguimiento que necesita identificar rapidamente que pacientes requieren
    atencion activa (llamadas, visitas domiciliarias, nuevas citas).

    Estructura de la interfaz:
      1. Slider de umbral: permite al usuario ajustar cuantos meses sin visita
         definen a un paciente como "perdido" (rango 1-24, default 6).
      2. KPIs por categoria: cinco metricas con conteo y porcentaje de
         pacientes en cada nivel de alerta, mas una metrica global.
      3. Filtros de tabla: multiselect por tipo de alerta y por municipio.
         Por defecto muestra solo las categorias de maxima urgencia (rojo y naranja).
      4. Tabla interactiva: ordenada por prioridad descendente y luego por
         meses sin visita, con columnas especiales (DateColumn, ProgressColumn).
      5. Detalle por paciente: selector desplegable que muestra los indicadores
         fuera de meta y los que empeoraron respecto a la visita anterior.
      6. Descarga Excel: boton que genera el archivo .xlsx con tres hojas
         (resumen por alerta, resumen por municipio, listado completo).

    La clasificacion se recalcula cada vez que el usuario cambia el slider
    de umbral (no se usa cache para este calculo porque el umbral es variable).

    Parametros:
        df           (pd.DataFrame): DataFrame de pacientes filtrado, con la
                                     columna '__fecha__' y las columnas clinicas.
        meses_umbral (int):          Valor inicial del slider de umbral.
                                     Por defecto MESES_PERDIDO_DEFAULT (6).

    Retorna:
        None. Escribe directamente en la interfaz de Streamlit.
    """
    import streamlit as st

    st.markdown(
        "Identifica pacientes que han dejado de asistir o cuyos indicadores "
        "clinicos no han mejorado, para priorizar la busqueda activa."
    )

    # --- Slider: umbral de meses sin visita ---
    # El valor del slider sobreescribe el parametro meses_umbral recibido
    meses_umbral = st.slider(
        "Meses sin visita para considerar paciente perdido",
        min_value=1, max_value=24, value=meses_umbral, step=1,
        help="Pacientes cuya ultima consulta fue hace mas de este numero de meses.",
    )

    # Ejecutamos la clasificacion con el umbral actual del slider
    with st.spinner("Analizando historial de pacientes..."):
        _, df_alertas = clasificar_seguimiento_df(df, meses_umbral=meses_umbral)

    total = len(df_alertas)
    if total == 0:
        st.warning("No hay pacientes en el filtro actual.")
        return

    # --- KPIs: distribucion de pacientes por categoria de alerta ---
    st.markdown("### Resumen de alertas")

    # Definimos las cinco categorias con su color asociado para el display
    grupos_kpi = [
        ("Perdido + sin control",  "red"),
        ("Perdido (sin consulta)", "orange"),
        ("En deterioro",           "orange"),
        ("Sin control metabolico", "orange"),
        ("Controlado",             "green"),
    ]
    cols_kpi = st.columns(len(grupos_kpi))
    for i, (etiqueta, _) in enumerate(grupos_kpi):
        n   = int((df_alertas["tipo_alerta"] == etiqueta).sum())
        pct = n / total * 100
        cols_kpi[i].metric(etiqueta, f"{n:,}", f"{pct:.1f}%")

    # Metrica global: total de pacientes que requieren alguna accion
    n_alerta = int(df_alertas["tiene_alerta"].sum())
    st.info(
        f"**{n_alerta:,} de {total:,} pacientes** ({n_alerta / total * 100:.1f}%) "
        f"requieren atencion o seguimiento activo."
    )

    st.markdown("---")

    # --- Filtros de la tabla ---
    st.markdown("### Listado de pacientes")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        tipos_disponibles = sorted(df_alertas["tipo_alerta"].unique())
        # Por defecto mostramos solo las categorias de mayor urgencia (prioridad 1 y 2)
        tipos_sel = st.multiselect(
            "Tipo de alerta",
            options=tipos_disponibles,
            default=[t for t in tipos_disponibles
                     if "Perdido + sin control" in t or "Perdido (sin consulta)" in t],
        )

    with col_f2:
        municipios_disp = sorted(df_alertas["municipio"].dropna().unique())
        mun_sel = st.multiselect("Municipio", options=municipios_disp, key="seg_municipio_filtro")

    # Aplicamos los filtros seleccionados
    df_vista = df_alertas.copy()
    if tipos_sel:
        df_vista = df_vista[df_vista["tipo_alerta"].isin(tipos_sel)]
    if mun_sel:
        df_vista = df_vista[df_vista["municipio"].isin(mun_sel)]

    # Ordenamos: primero por prioridad (1=mas urgente), luego por meses sin visita (descendente)
    df_vista = df_vista.sort_values(
        ["prioridad", "meses_sin_visita"],
        ascending=[True, False]
    )

    # --- Tabla resumen de pacientes ---
    cols_tabla = [
        "id_paciente", "tipo_alerta", "meses_sin_visita", "ultima_visita",
        "municipio", "edad", "sexo", "riesgo_score", "riesgo_nivel",
    ]
    cols_tabla = [c for c in cols_tabla if c in df_vista.columns]

    st.dataframe(
        df_vista[cols_tabla].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "tipo_alerta":      st.column_config.TextColumn("Alerta"),
            "meses_sin_visita": st.column_config.NumberColumn(
                "Meses sin visita", format="%.1f"
            ),
            "ultima_visita":    st.column_config.DateColumn(
                "Ultima visita", format="DD/MM/YYYY"
            ),
            # ProgressColumn: muestra el score como barra de progreso de 0 a 100
            "riesgo_score":     st.column_config.ProgressColumn(
                "Score riesgo", min_value=0, max_value=100, format="%.0f"
            ),
        },
    )
    st.caption(f"Mostrando {len(df_vista):,} de {total:,} pacientes.")

    # --- Detalle expandible por paciente ---
    st.markdown("### Detalle por paciente")
    st.caption("Selecciona un paciente para ver sus indicadores fuera de meta.")

    ids_disponibles = df_vista["id_paciente"].tolist()
    if not ids_disponibles:
        st.info("No hay pacientes con los filtros seleccionados.")
        return

    id_sel = st.selectbox("Paciente", options=ids_disponibles)

    if id_sel:
        fila_det = df_alertas[df_alertas["id_paciente"] == id_sel].iloc[0]

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**Alerta:** {fila_det['tipo_alerta']}")
            ultima = fila_det.get("ultima_visita")
            if pd.notna(ultima):
                st.markdown(
                    f"**Ultima visita:** {pd.Timestamp(ultima).strftime('%d/%m/%Y')} "
                    f"({fila_det.get('meses_sin_visita', '?'):.1f} meses)"
                )
            else:
                st.markdown("**Ultima visita:** Sin registro")

        with col_d2:
            # Construimos la lista de enfermedades activas del paciente
            enfs = []
            for e, lbl in [
                ("diabetes",    "DM"),
                ("hipertension","HTA"),
                ("dislipidemia","DLP"),
                ("obesidad",    "OB")
            ]:
                if fila_det.get(e, 0) == 1:
                    enfs.append(lbl)
            st.markdown(f"**Enfermedades:** {', '.join(enfs) if enfs else 'Sin registro'}")

            if pd.notna(fila_det.get("riesgo_score")):
                st.markdown(
                    f"**Score de riesgo:** {fila_det['riesgo_score']:.0f}/100 "
                    f"- {fila_det.get('riesgo_nivel', '')}"
                )

        # Indicadores fuera de meta en la ultima visita
        fuera     = fila_det.get("indicadores_fuera",     [])
        deterioro = fila_det.get("indicadores_deterioro", [])

        if fuera:
            st.error("**Indicadores fuera de meta (ultima visita):**")
            for item in fuera:
                st.markdown(f"- {item}")

        if deterioro:
            st.warning("**Indicadores que empeoraron vs. visita anterior:**")
            for item in deterioro:
                st.markdown(f"- {item}")

        if not fuera and not deterioro:
            st.success(
                "Este paciente no tiene indicadores fuera de meta en su ultima visita."
            )

    # --- Descarga del listado en Excel ---
    st.markdown("---")
    st.markdown("### Descargar listado")

    # Generamos el Excel solo con los pacientes que tienen alerta activa
    excel_bytes = _generar_excel(
        df_alertas[df_alertas["tiene_alerta"]].copy(),
        meses_umbral,
    )
    st.download_button(
        label="Descargar pacientes con alerta (.xlsx)",
        data=excel_bytes,
        file_name=f"pacientes_seguimiento_{datetime.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption(
        "El Excel incluye tres hojas: resumen por tipo de alerta, "
        "resumen por municipio y listado completo de pacientes."
    )