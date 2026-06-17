"""
alertas.py
==========
Modulo de alertas clinicas automaticas para el panel ECNT Guanajuato.
Uso desde seminario_Luisa.py:
    from alertas import procesar_alertas
    procesar_alertas(df_raw)
"""

from __future__ import annotations

# Modulos de la biblioteca estandar de Python
import smtplib                         # Envio de correos electronicos via SMTP
import ssl                             # Contexto de seguridad para la conexion TLS
from datetime import datetime          # Generacion de la marca de tiempo en correos
from email.mime.multipart import MIMEMultipart  # Estructura del mensaje de correo
from email.mime.text import MIMEText   # Parte de texto/HTML del correo
from typing import Dict, List, Optional, Tuple  # Anotaciones de tipo

# Bibliotecas de terceros para analisis de datos
import numpy as np
import pandas as pd


# =============================================================================
# SECCION 1: UMBRALES CLINICOS DE ALERTA
# =============================================================================
# Cada elemento de la lista define un criterio clinico que, al cumplirse,
# genera una alerta para el paciente. La estructura de cada umbral es:
#
#   id        : identificador unico del umbral (cadena sin espacios)
#   etiqueta  : nombre legible del criterio para mostrarlo en pantalla
#   columna   : nombre de la columna del DataFrame que se evalua
#   condicion : funcion lambda que recibe el valor numerico y retorna True
#               si el umbral se supera (si debe generarse la alerta)
#   mensaje   : funcion lambda que recibe el valor y genera el texto descriptivo
#               de la alerta (incluye el valor detectado y el umbral de referencia)
#   severidad : "CRITICA" (requiere atencion inmediata) o
#               "ADVERTENCIA" (requiere seguimiento en el proximo mes)
#   aplica_si : nombre de la columna binaria de enfermedad que debe estar activa
#               para que el umbral aplique (p.ej. "diabetes"). None = aplica siempre.
# =============================================================================

UMBRALES: List[Dict] = [
    # --- Diabetes: control glucemico ---
    {
        "id":        "hba1c_critica",
        "etiqueta":  "HbA1c muy elevada",
        "columna":   "hba1c_pct",
        # HbA1c >= 9% indica control glucemico muy deficiente (ADA: nivel de accion)
        "condicion": lambda v: v >= 9.0,
        "mensaje":   lambda v: f"HbA1c = {v:.1f}% (meta < 7%, critico >= 9%)",
        "severidad": "CRITICA",
        "aplica_si": "diabetes",
    },
    {
        "id":        "hba1c_fuera_meta",
        "etiqueta":  "HbA1c fuera de meta",
        "columna":   "hba1c_pct",
        # HbA1c entre 7% y 8.9%: fuera del objetivo pero no en rango critico
        "condicion": lambda v: 7.0 <= v < 9.0,
        "mensaje":   lambda v: f"HbA1c = {v:.1f}% (meta < 7%)",
        "severidad": "ADVERTENCIA",
        "aplica_si": "diabetes",
    },
    {
        "id":        "glucosa_critica",
        "etiqueta":  "Glucosa en ayuno muy elevada",
        "columna":   "glucosa_ayunas",
        # Glucosa >= 300 mg/dL: riesgo de cetoacidosis diabetica
        "condicion": lambda v: v >= 300,
        "mensaje":   lambda v: f"Glucosa = {v:.0f} mg/dL (critico >= 300 mg/dL)",
        "severidad": "CRITICA",
        "aplica_si": "diabetes",
    },

    # --- Hipertension: presion arterial sistolica ---
    {
        "id":        "pa_grado2",
        "etiqueta":  "Hipertension grado 2",
        "columna":   "presion_sistolica",
        # PA sistolica >= 160 mmHg: hipertension grado 2 (JNC-8), riesgo cardiovascular elevado
        "condicion": lambda v: v >= 160,
        "mensaje":   lambda v: f"PA sistolica = {v:.0f} mmHg (critico >= 160 mmHg)",
        "severidad": "CRITICA",
        "aplica_si": "hipertension",
    },
    {
        "id":        "pa_grado1",
        "etiqueta":  "Hipertension grado 1",
        "columna":   "presion_sistolica",
        # PA sistolica entre 140 y 159 mmHg: hipertension grado 1, fuera de meta terapeutica
        "condicion": lambda v: 140 <= v < 160,
        "mensaje":   lambda v: f"PA sistolica = {v:.0f} mmHg (meta < 130 mmHg)",
        "severidad": "ADVERTENCIA",
        "aplica_si": "hipertension",
    },

    # --- Dislipidemia: perfil lipidico ---
    {
        "id":        "ldl_alto",
        "etiqueta":  "LDL elevado",
        "columna":   "ldl",
        # LDL >= 160 mg/dL: riesgo cardiovascular aumentado (ATP III / ESC 2019)
        "condicion": lambda v: v >= 160,
        "mensaje":   lambda v: f"LDL = {v:.0f} mg/dL (meta < 100 mg/dL)",
        "severidad": "ADVERTENCIA",
        "aplica_si": "dislipidemia",
    },
    {
        "id":        "trigliceridos_muy_altos",
        "etiqueta":  "Trigliceridos muy elevados",
        "columna":   "trigliceridos",
        # Trigliceridos >= 500 mg/dL: riesgo de pancreatitis aguda (AHA/ACC)
        "condicion": lambda v: v >= 500,
        "mensaje":   lambda v: f"Trigliceridos = {v:.0f} mg/dL (riesgo pancreatitis >= 500)",
        "severidad": "CRITICA",
        "aplica_si": "dislipidemia",
    },

    # --- Obesidad: indice de masa corporal ---
    {
        "id":        "imc_obesidad_morbida",
        "etiqueta":  "Obesidad morbida",
        "columna":   "imc",
        # IMC >= 40 kg/m2: obesidad grado III (OMS), asociada a complicaciones multiples
        "condicion": lambda v: v >= 40,
        "mensaje":   lambda v: f"IMC = {v:.1f} kg/m2 (obesidad morbida >= 40)",
        "severidad": "ADVERTENCIA",
        "aplica_si": "obesidad",
    },

    # --- Score de riesgo global ---
    {
        "id":        "riesgo_muy_alto",
        "etiqueta":  "Score de riesgo muy alto",
        "columna":   "riesgo_score",
        # Score >= 75/100 indica riesgo cardiovascular muy alto independientemente
        # de la enfermedad especifica. aplica_si=None: se evalua en todos los pacientes.
        "condicion": lambda v: v >= 75,
        "mensaje":   lambda v: f"Score de riesgo = {v:.0f}/100 (muy alto >= 75)",
        "severidad": "CRITICA",
        "aplica_si": None,
    },
]


# =============================================================================
# SECCION 2: DETECCION DE ALERTAS POR PACIENTE
# =============================================================================

def _evaluar_fila(row: pd.Series) -> List[Dict]:
    """
    Evalua todos los umbrales definidos en UMBRALES para un paciente y
    retorna la lista de alertas que se activaron.

    El proceso para cada umbral es:
      1. Verificar si el umbral aplica al paciente (segun la enfermedad registrada).
      2. Obtener el valor del indicador correspondiente en la fila.
      3. Ignorar el umbral si el valor es NaN o no es convertible a float.
      4. Evaluar la condicion: si se cumple, registrar la alerta.

    Esta funcion tiene prefijo '_' porque es de uso interno del modulo;
    no se importa directamente desde seminario_Luisa.py.

    Parametros:
        row (pd.Series): Fila del DataFrame con los datos de un paciente.

    Retorna:
        List[Dict]: Lista de alertas activas. Cada elemento contiene:
          - umbral_id  : identificador del umbral activado
          - etiqueta   : nombre legible del criterio
          - mensaje    : descripcion con el valor detectado y el umbral
          - severidad  : "CRITICA" o "ADVERTENCIA"
    """
    alertas_fila = []
    for u in UMBRALES:
        # Si el umbral requiere una enfermedad especifica, verificamos que el
        # paciente la tenga activa (columna == 1). Si no, omitimos el umbral.
        if u["aplica_si"] and row.get(u["aplica_si"], 0) != 1:
            continue

        # Leemos el valor del indicador; si es NaN, no podemos evaluarlo
        val = row.get(u["columna"], np.nan)
        if pd.isna(val):
            continue

        # Intentamos convertir a float para garantizar que la lambda funcione
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            continue

        # Si la condicion del umbral se cumple, registramos la alerta
        if u["condicion"](val_float):
            alertas_fila.append({
                "umbral_id": u["id"],
                "etiqueta":  u["etiqueta"],
                "mensaje":   u["mensaje"](val_float),
                "severidad": u["severidad"],
            })

    return alertas_fila


def detectar_alertas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recorre el DataFrame completo de pacientes y detecta todos aquellos
    que superan al menos un umbral clinico definido en UMBRALES.

    Si el DataFrame contiene una columna de fecha ('__fecha__'), se trabaja
    solo con la ultima visita registrada por paciente, evitando duplicar
    alertas por visitas anteriores. Si no hay columna de fecha, se usa el
    ultimo registro por id_paciente.

    Los resultados se ordenan por severidad (criticas primero) y por numero
    de alertas activas (de mayor a menor), para que los pacientes mas urgentes
    aparezcan al inicio de la tabla.

    Parametros:
        df (pd.DataFrame): DataFrame con todos los registros de pacientes.
                           Debe contener las columnas evaluadas en UMBRALES.

    Retorna:
        pd.DataFrame: Tabla de pacientes con alertas activas. Columnas:
          - id_paciente    : identificador del paciente
          - municipio      : municipio de residencia
          - edad           : edad en anos
          - sexo           : sexo biologico
          - correo_medico  : correo del medico responsable (si existe en los datos)
          - severidad_max  : "CRITICA" o "ADVERTENCIA" (la mas alta entre sus alertas)
          - n_alertas      : numero total de alertas activas
          - alertas        : lista de dicts con el detalle de cada alerta
          - resumen_alertas: texto con todos los mensajes concatenados ("|" como separador)
          - riesgo_score   : puntuacion de riesgo global (0-100)
          - riesgo_nivel   : etiqueta textual del nivel de riesgo
        Retorna un DataFrame vacio si no hay pacientes con alertas.
    """
    if df.empty:
        return pd.DataFrame()

    # Nos quedamos con la ultima visita por paciente para evitar alertas duplicadas
    if "__fecha__" in df.columns:
        df_ultimo = (
            df.sort_values("__fecha__")
            .groupby("id_paciente", sort=False)
            .last()
            .reset_index()
        )
    else:
        # Sin columna de fecha: usamos el ultimo registro encontrado por ID
        df_ultimo = df.drop_duplicates(subset=["id_paciente"], keep="last").copy()

    resultados = []
    for _, row in df_ultimo.iterrows():
        alertas_pac = _evaluar_fila(row)

        # Si el paciente no tiene ninguna alerta activa, lo omitimos
        if not alertas_pac:
            continue

        # Determinamos la severidad maxima: si hay al menos una alerta critica,
        # la clasificacion del paciente es CRITICA en su totalidad
        sev_max = (
            "CRITICA"
            if any(a["severidad"] == "CRITICA" for a in alertas_pac)
            else "ADVERTENCIA"
        )

        # Texto resumen: todos los mensajes de alerta separados por " | "
        resumen = " | ".join(a["mensaje"] for a in alertas_pac)

        resultados.append({
            "id_paciente":     str(row.get("id_paciente", "Desconocido")),
            "municipio":       str(row.get("municipio", "")),
            "edad":            row.get("edad", np.nan),
            "sexo":            str(row.get("sexo", "")),
            "correo_medico":   str(row.get("correo_medico", "")),
            "severidad_max":   sev_max,
            "n_alertas":       len(alertas_pac),
            "alertas":         alertas_pac,
            "resumen_alertas": resumen,
            "riesgo_score":    row.get("riesgo_score", np.nan),
            "riesgo_nivel":    str(row.get("riesgo_nivel", "")),
        })

    if not resultados:
        return pd.DataFrame()

    df_alertas = pd.DataFrame(resultados)

    # Ordenamos: primero por severidad (CRITICA = 0, ADVERTENCIA = 1),
    # luego por numero de alertas de mayor a menor para priorizar los casos mas graves
    orden_sev = {"CRITICA": 0, "ADVERTENCIA": 1}
    df_alertas["_orden"] = df_alertas["severidad_max"].map(orden_sev)
    df_alertas = (
        df_alertas
        .sort_values(["_orden", "n_alertas"], ascending=[True, False])
        .drop(columns=["_orden"])
        .reset_index(drop=True)
    )
    return df_alertas


# =============================================================================
# SECCION 3: CONSTRUCCION DEL CORREO ELECTRONICO
# =============================================================================

def _html_correo(paciente: Dict, correo_remitente: str) -> str:
    """
    Genera el cuerpo HTML del correo de alerta para un paciente especifico.

    El correo incluye:
      - Encabezado con el nivel de severidad (color rojo para critica, naranja para advertencia).
      - Ficha del paciente: ID, municipio, edad, sexo y score de riesgo.
      - Tabla con los indicadores fuera de rango, el valor detectado y la severidad de cada uno.
      - Recuadro de accion recomendada: diferenciado segun si la alerta es critica o advertencia.
      - Pie de pagina con la fecha de generacion y el remitente.

    Esta funcion tiene prefijo '_' porque es de uso interno del modulo.

    Parametros:
        paciente         (Dict): Diccionario con los datos del paciente y sus alertas.
                                 Debe contener las claves: id_paciente, municipio, edad,
                                 sexo, severidad_max, alertas, riesgo_score, riesgo_nivel.
        correo_remitente (str):  Direccion de correo desde la que se envia el mensaje
                                 (se muestra en el pie de pagina del correo).

    Retorna:
        str: Cadena con el HTML completo del correo, lista para adjuntarse
             como parte MIMEText en el mensaje SMTP.
    """
    sev       = paciente["severidad_max"]
    # Rojo para alertas criticas, naranja para advertencias
    color_sev = "#c0392b" if sev == "CRITICA" else "#e67e22"
    emoji_sev = "[CRITICA]" if sev == "CRITICA" else "[ADVERTENCIA]"
    fecha     = datetime.today().strftime("%d/%m/%Y %H:%M")

    # Construimos las filas de la tabla de indicadores, una por alerta activa
    filas_alertas = ""
    for a in paciente["alertas"]:
        color_fila = "#fdf2f2" if a["severidad"] == "CRITICA" else "#fef9ec"
        icono      = "[C]" if a["severidad"] == "CRITICA" else "[A]"
        filas_alertas += f"""
        <tr style="background:{color_fila}">
          <td style="padding:8px 12px">{icono} {a['etiqueta']}</td>
          <td style="padding:8px 12px">{a['mensaje']}</td>
          <td style="padding:8px 12px;font-weight:bold;color:{color_sev}">{a['severidad']}</td>
        </tr>"""

    # Bloque opcional del score de riesgo (solo si el valor no es NaN)
    score_html = ""
    if not pd.isna(paciente.get("riesgo_score", np.nan)):
        score_html = f"""
        <p style="margin:6px 0">
          <strong>Score de riesgo:</strong>
          {paciente['riesgo_score']:.0f}/100 - {paciente['riesgo_nivel']}
        </p>"""

    # Plantilla HTML completa del correo
    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:640px;margin:0 auto;color:#333">

  <div style="background:{color_sev};padding:16px 24px;border-radius:8px 8px 0 0">
    <h2 style="color:white;margin:0">
      {emoji_sev} Alerta ECNT - Paciente requiere atencion
    </h2>
    <p style="color:rgba(255,255,255,0.9);margin:4px 0 0">
      Secretaria de Salud de Guanajuato - Panel ECNT
    </p>
  </div>

  <div style="background:#f8f9fa;padding:16px 24px;border:1px solid #dee2e6">
    <h3 style="margin:0 0 12px;color:{color_sev}">
      Severidad: {sev}
    </h3>
    <p style="margin:6px 0">
      <strong>ID Paciente:</strong> {paciente['id_paciente']}
    </p>
    <p style="margin:6px 0">
      <strong>Municipio:</strong> {paciente['municipio']}
    </p>
    <p style="margin:6px 0">
      <strong>Edad / Sexo:</strong>
      {int(paciente['edad']) if not pd.isna(paciente.get('edad', np.nan)) else 'N/D'}
      anos / {paciente['sexo']}
    </p>
    {score_html}
  </div>

  <div style="padding:16px 24px">
    <h3 style="margin:0 0 12px">Indicadores fuera de rango</h3>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
      <thead>
        <tr style="background:#1a4e7a;color:white">
          <th style="padding:8px 12px;text-align:left">Indicador</th>
          <th style="padding:8px 12px;text-align:left">Valor detectado</th>
          <th style="padding:8px 12px;text-align:left">Severidad</th>
        </tr>
      </thead>
      <tbody>{filas_alertas}</tbody>
    </table>
  </div>

  <div style="background:#e8f4fd;padding:16px 24px;border-left:4px solid #1a4e7a">
    <strong>Accion recomendada:</strong>
    {"Atencion prioritaria. Evalúe intervencion inmediata o referencia a segundo nivel."
     if sev == "CRITICA"
     else "Revise el plan de tratamiento y programe seguimiento en las proximas 4 semanas."}
  </div>

  <div style="padding:16px 24px;border-top:1px solid #dee2e6;
              font-size:12px;color:#888;text-align:center">
    Generado automaticamente por el Panel ECNT - Guanajuato<br>
    {fecha} - Este mensaje es confidencial y de uso exclusivo del personal de salud.<br>
    Enviado desde: {correo_remitente}
  </div>

</body>
</html>"""


def _enviar_correo(
    paciente: Dict,
    destinatario: str,
    cfg: Dict,
) -> Tuple[bool, str]:
    """
    Envia el correo de alerta de un paciente especifico al destinatario indicado.

    Establece una conexion SMTP con autenticacion TLS (protocolo STARTTLS),
    construye el mensaje en formato HTML y lo envia. Maneja los errores mas
    comunes de forma diferenciada para dar mensajes utiles al usuario.

    Esta funcion tiene prefijo '_' porque es de uso interno del modulo.
    El punto de entrada publico para envio en lote es enviar_alertas_lote().

    Parametros:
        paciente     (Dict): Diccionario con los datos del paciente con alertas.
                             Debe incluir: id_paciente, municipio, severidad_max, alertas.
        destinatario (str):  Direccion de correo electronico del receptor.
        cfg          (Dict): Configuracion SMTP con las claves:
                               smtp_host, smtp_port, smtp_user, smtp_password.

    Retorna:
        Tuple[bool, str]:
          - True y mensaje de exito si el correo se envio correctamente.
          - False y descripcion del error en caso de fallo.
    """
    try:
        # Creamos el mensaje MIME con estructura multipart para soporte HTML
        msg = MIMEMultipart("alternative")
        sev = paciente["severidad_max"]

        # Asunto diferenciado segun severidad para facilitar la triaje en el correo
        msg["Subject"] = (
            f"[{'ALERTA CRITICA' if sev == 'CRITICA' else 'Advertencia'} ECNT] "
            f"Paciente {paciente['id_paciente']} - {paciente['municipio']}"
        )
        msg["From"] = cfg["smtp_user"]
        msg["To"]   = destinatario

        # Adjuntamos el cuerpo HTML generado por _html_correo()
        html = _html_correo(paciente, cfg["smtp_user"])
        msg.attach(MIMEText(html, "html", "utf-8"))

        # Conexion SMTP con TLS: primero se negocia la conexion sin cifrar,
        # luego se eleva a TLS con STARTTLS antes de enviar credenciales.
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.ehlo()                              # Presentacion al servidor SMTP
            server.starttls(context=context)           # Activacion de cifrado TLS
            server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.sendmail(cfg["smtp_user"], destinatario, msg.as_string())

        return True, f"Correo enviado a {destinatario}"

    except smtplib.SMTPAuthenticationError:
        # Error mas frecuente: contrasena incorrecta o cuenta sin App Password
        return False, (
            "Error de autenticacion SMTP. Verifica que estes usando "
            "un App Password de Google (no tu contrasena normal)."
        )
    except smtplib.SMTPException as e:
        # Otros errores del protocolo SMTP: servidor no disponible, TLS rechazado, etc.
        return False, f"Error SMTP: {e}"
    except Exception as e:
        # Cualquier otro error inesperado (red, configuracion, etc.)
        return False, f"Error inesperado: {e}"


# =============================================================================
# SECCION 4: ENVIO EN LOTE DE ALERTAS POR CORREO
# =============================================================================

def enviar_alertas_lote(
    df_alertas: pd.DataFrame,
    cfg: Dict,
    correo_default: str,
    solo_criticas: bool = False,
) -> List[Dict]:
    """
    Envia correos de alerta para todos los pacientes del DataFrame de alertas.

    Para cada paciente, determina el destinatario del correo con la siguiente
    logica de prioridad:
      1. Si el paciente tiene un correo_medico valido (que contenga "@"),
         se usa ese como destinatario.
      2. En caso contrario, se usa el correo_default proporcionado.

    Parametros:
        df_alertas     (pd.DataFrame): DataFrame generado por detectar_alertas(),
                                       con una fila por paciente con alertas activas.
        cfg            (Dict):         Configuracion SMTP (smtp_host, smtp_port,
                                       smtp_user, smtp_password).
        correo_default (str):          Correo de respaldo cuando el paciente no tiene
                                       medico asignado (ej. coordinador del CEAPS).
        solo_criticas  (bool):         Si True, solo se envian correos para los pacientes
                                       con severidad_max == "CRITICA". Default: False.

    Retorna:
        List[Dict]: Lista de resultados, uno por correo intentado, con las claves:
          - id_paciente  : identificador del paciente
          - destinatario : correo al que se intento enviar
          - exito        : True si el envio fue exitoso, False en caso de error
          - mensaje      : descripcion del resultado o del error
    """
    resultados = []

    # Filtramos segun la opcion de solo_criticas
    df_enviar = df_alertas
    if solo_criticas:
        df_enviar = df_alertas[df_alertas["severidad_max"] == "CRITICA"]

    for _, row in df_enviar.iterrows():
        # Determinamos el destinatario: medico del paciente o correo por defecto
        destinatario = (
            row["correo_medico"]
            if row.get("correo_medico") and "@" in str(row["correo_medico"])
            else correo_default
        )
        paciente = row.to_dict()
        exito, mensaje = _enviar_correo(paciente, destinatario, cfg)
        resultados.append({
            "id_paciente":  row["id_paciente"],
            "destinatario": destinatario,
            "exito":        exito,
            "mensaje":      mensaje,
        })

    return resultados


# =============================================================================
# SECCION 5: INTERFAZ DE USUARIO EN STREAMLIT (PUNTO DE ENTRADA PRINCIPAL)
# =============================================================================

def procesar_alertas(df: pd.DataFrame) -> None:
    """
    Punto de entrada principal del modulo. Se llama desde seminario_Luisa.py
    inmediatamente despues de cargar y normalizar los datos.

    Orquesta el flujo completo de alertas en dos pasos:
      1. Detecta las alertas y las renderiza en la interfaz de Streamlit:
           - Banner de resumen con contadores por nivel de severidad.
           - Tabla interactiva con filtro por severidad.
           - Vista de detalle expandible por paciente.
           - Boton de descarga de la lista en formato CSV.
      2. Si la configuracion SMTP esta disponible en secrets.toml, ofrece
         al usuario enviar los correos de alerta directamente desde la interfaz.
         Si no esta configurada, muestra instrucciones para habilitarla.

    Las alertas se almacenan en st.session_state para evitar recalcularlas
    en cada interaccion del usuario con la interfaz (reruns de Streamlit).
    La llave de cache incluye la longitud del DataFrame y su id de objeto para
    invalidarla automaticamente cuando se cargan datos nuevos.

    Parametros:
        df (pd.DataFrame): DataFrame completo de pacientes (df_raw o df_filt)
                           con todas las columnas clinicas requeridas.

    Retorna:
        None. Esta funcion escribe directamente en la interfaz de Streamlit.
    """
    import streamlit as st

    # Usamos session_state para cachear las alertas entre interacciones.
    # La llave combina la longitud y el id del DataFrame para detectar cambios.
    llave = f"alertas_{len(df)}_{id(df)}"
    if st.session_state.get("alertas_llave") != llave:
        # El DataFrame cambio: recalculamos las alertas
        with st.spinner("Analizando indicadores clinicos..."):
            df_alertas = detectar_alertas(df)
        st.session_state["alertas_df"]    = df_alertas
        st.session_state["alertas_llave"] = llave
    else:
        # El DataFrame no cambio: usamos el resultado cacheado
        df_alertas = st.session_state.get("alertas_df", pd.DataFrame())

    # Contadores para el banner de resumen
    n_criticas    = int((df_alertas["severidad_max"] == "CRITICA").sum())    if not df_alertas.empty else 0
    n_advertencia = int((df_alertas["severidad_max"] == "ADVERTENCIA").sum()) if not df_alertas.empty else 0
    n_total       = len(df_alertas)

    # -------------------------------------------------------------------------
    # Banner de resumen
    # -------------------------------------------------------------------------
    if n_total == 0:
        st.success("Sin alertas clinicas activas en los datos cargados.")
        return

    # Tres metricas en columnas: criticas, advertencias y total
    col_crit, col_adv, col_tot = st.columns(3)
    col_crit.metric("Alertas criticas",  n_criticas)
    col_adv.metric( "Advertencias",      n_advertencia)
    col_tot.metric( "Total con alertas", n_total,
                    f"{n_total / len(df) * 100:.1f}% del total")

    # Mensaje destacado segun el nivel de severidad predominante
    if n_criticas > 0:
        st.error(
            f"**{n_criticas} paciente(s) requieren atencion prioritaria.** "
            "Revisa la tabla de alertas criticas a continuacion."
        )
    elif n_advertencia > 0:
        st.warning(
            f"{n_advertencia} paciente(s) con indicadores fuera de meta."
        )

    # -------------------------------------------------------------------------
    # Tabla interactiva de alertas
    # -------------------------------------------------------------------------
    st.markdown("### Pacientes con alertas")

    # Radio de filtro rapido: permite mostrar todas, solo criticas o solo advertencias
    sev_filtro = st.radio(
        "Mostrar",
        options=["Todas", "Solo criticas", "Solo advertencias"],
        horizontal=True,
        key="alertas_filtro_sev",
    )
    df_vista = df_alertas.copy()
    if sev_filtro == "Solo criticas":
        df_vista = df_vista[df_vista["severidad_max"] == "CRITICA"]
    elif sev_filtro == "Solo advertencias":
        df_vista = df_vista[df_vista["severidad_max"] == "ADVERTENCIA"]

    # Columna de severidad con etiqueta legible para la tabla
    df_vista = df_vista.copy()
    df_vista["sev_display"] = df_vista["severidad_max"].map(
        {"CRITICA": "Critica", "ADVERTENCIA": "Advertencia"}
    )

    # Seleccionamos y renombramos las columnas para la vista en pantalla
    cols_tabla = [
        "id_paciente", "sev_display", "n_alertas",
        "municipio", "edad", "sexo",
        "riesgo_score", "resumen_alertas",
    ]
    cols_tabla = [c for c in cols_tabla if c in df_vista.columns]

    st.dataframe(
        df_vista[cols_tabla].rename(columns={
            "id_paciente":     "ID Paciente",
            "sev_display":     "Severidad",
            "n_alertas":       "# Alertas",
            "municipio":       "Municipio",
            "edad":            "Edad",
            "sexo":            "Sexo",
            "riesgo_score":    "Score riesgo",
            "resumen_alertas": "Detalle",
        }).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            # ProgressColumn: muestra el score como barra de progreso de 0 a 100
            "Score riesgo": st.column_config.ProgressColumn(
                "Score riesgo", min_value=0, max_value=100, format="%.0f"
            ),
        },
    )

    # Boton de descarga de la lista completa de alertas en CSV
    # Se excluye la columna 'alertas' (lista de dicts) porque no es serializable a CSV
    csv_alertas = (
        df_alertas
        .drop(columns=["alertas"], errors="ignore")
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        "Descargar lista de alertas (.csv)",
        data=csv_alertas,
        file_name=f"alertas_ecnt_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="dl_alertas_csv",
    )

    # -------------------------------------------------------------------------
    # Detalle expandible por paciente
    # -------------------------------------------------------------------------
    st.markdown("### Detalle por paciente")
    ids_alerta = df_vista["id_paciente"].tolist()

    if ids_alerta:
        # Selector desplegable: el usuario elige el ID del paciente a inspeccionar
        id_sel   = st.selectbox(
            "Ver detalle de paciente",
            options=ids_alerta,
            key="alertas_id_sel",
        )
        fila_sel = df_alertas[df_alertas["id_paciente"] == id_sel].iloc[0]
        sev_sel  = fila_sel["severidad_max"]

        # Bloque de severidad con color diferenciado
        if sev_sel == "CRITICA":
            st.error(f"Severidad: **{sev_sel}** - Atencion prioritaria")
        else:
            st.warning(f"Severidad: **{sev_sel}**")

        # Lista de alertas individuales con su etiqueta y mensaje
        for a in fila_sel["alertas"]:
            icono = "[C]" if a["severidad"] == "CRITICA" else "[A]"
            st.markdown(f"- {icono} **{a['etiqueta']}**: {a['mensaje']}")

    # -------------------------------------------------------------------------
    # Configuracion y envio de correos electronicos
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Enviar alertas por correo")

    # Intentamos leer la configuracion SMTP desde .streamlit/secrets.toml
    try:
        cfg_smtp = dict(st.secrets.get("alertas", {}))
        # Verificamos que esten todas las claves obligatorias para poder conectar al servidor
        smtp_ok = all(
            k in cfg_smtp
            for k in ["smtp_host", "smtp_port", "smtp_user", "smtp_password"]
        )
    except Exception:
        cfg_smtp = {}
        smtp_ok  = False

    if not smtp_ok:
        # SMTP no configurado: mostramos instrucciones para el administrador del sistema
        with st.expander("Configurar correo SMTP", expanded=True):
            st.info(
                "Agrega la configuracion en `.streamlit/secrets.toml` para "
                "habilitar el envio de correos:"
            )
            st.code("""[alertas]
smtp_host      = "smtp.gmail.com"
smtp_port      = 587
smtp_user      = "tu_correo@gmail.com"
smtp_password  = "tu_app_password"
correo_default = "coordinador@salud.gob.mx"
""", language="toml")
            st.markdown(
                "El campo `smtp_password` debe ser un **App Password** generado desde "
                "la configuracion de seguridad de la cuenta de Google "
                "(Cuenta -> Seguridad -> Verificacion en 2 pasos -> Contrasenas de aplicacion). "
                "**No uses tu contrasena normal de Gmail.**"
            )
        return

    # -------------------------------------------------------------------------
    # SMTP configurado: mostramos opciones de envio
    # -------------------------------------------------------------------------
    correo_default = cfg_smtp.get("correo_default", cfg_smtp["smtp_user"])

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        # Checkbox para restringir el envio solo a alertas criticas
        solo_criticas = st.checkbox(
            "Enviar solo alertas criticas",
            value=True,
            key="alertas_solo_criticas",
        )
    with col_e2:
        # Campo de texto para sobrescribir el correo destino configurado
        correo_override = st.text_input(
            "Correo destino (deja vacio para usar el configurado)",
            value="",
            placeholder=correo_default,
            key="alertas_correo_override",
        )

    # Si el usuario ingreso un correo, se usa ese; de lo contrario, el configurado
    destino_final = correo_override.strip() if correo_override.strip() else correo_default

    # Calculamos cuantos correos se enviaran segun la opcion de filtrado
    n_enviar = (
        len(df_alertas[df_alertas["severidad_max"] == "CRITICA"])
        if solo_criticas
        else len(df_alertas)
    )

    st.caption(
        f"Se enviaran {n_enviar} correo(s) a **{destino_final}** "
        f"(o al correo_medico de cada paciente si esta registrado)."
    )

    # Boton de envio: al presionarlo se dispara el envio en lote
    if st.button(
        f"Enviar {n_enviar} alerta(s) por correo",
        type="primary",
        key="btn_enviar_alertas",
    ):
        cfg_envio = dict(cfg_smtp)
        with st.spinner(f"Enviando {n_enviar} correo(s)..."):
            resultados_envio = enviar_alertas_lote(
                df_alertas,
                cfg=cfg_envio,
                correo_default=destino_final,
                solo_criticas=solo_criticas,
            )

        # Contamos exitos y errores para el mensaje de confirmacion
        exitos  = sum(1 for r in resultados_envio if r["exito"])
        errores = len(resultados_envio) - exitos

        if exitos:
            st.success(f"{exitos} correo(s) enviados correctamente.")
        if errores:
            st.error(f"{errores} correo(s) fallaron.")
            # Mostramos el detalle de cada fallo para facilitar el diagnostico
            for r in resultados_envio:
                if not r["exito"]:
                    st.caption(f"- {r['id_paciente']} -> {r['mensaje']}")