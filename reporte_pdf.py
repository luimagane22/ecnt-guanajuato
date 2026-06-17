"""
reporte_pdf.py
==============
Módulo de reporte PDF ejecutivo por municipio para el panel ECNT Guanajuato.

Genera reportes con:
  - Encabezado institucional
  - KPIs de prevalencia por enfermedad
  - Gráficas de distribución (barras y pastel)
  - Tabla de pacientes de alto riesgo
  - Proyección de costos a 5/10/15/20 años

Soporta:
  - PDF individual por municipio (elegido en la app)
  - PDF consolidado con todos los municipios (uno por página)

Uso desde seminario_Luisa.py:
    from reporte_pdf import render_seccion_pdf

    render_seccion_pdf(df_filt, df_raw, costos)
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# 1. PALETA Y ESTILOS
# ---------------------------------------------------------------------------

AZUL_INST   = colors.HexColor("#1a4e7a")   # azul institucional
AZUL_CLARO  = colors.HexColor("#d6e4f0")
VERDE       = colors.HexColor("#1a7a1a")
NARANJA     = colors.HexColor("#e67e22")
ROJO        = colors.HexColor("#c0392b")
GRIS_CLARO  = colors.HexColor("#f4f6f9")
GRIS_BORDE  = colors.HexColor("#cccccc")

COLORES_ENF = {
    "diabetes":     "#e74c3c",
    "hipertension": "#3498db",
    "dislipidemia": "#f39c12",
    "obesidad":     "#2ecc71",
}

ETIQUETAS_ENF = {
    "diabetes":     "Diabetes",
    "hipertension": "Hipertensión",
    "dislipidemia": "Dislipidemia",
    "obesidad":     "Obesidad",
}

def _estilos():
    base = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "TituloMunicipio",
        parent=base["Title"],
        fontSize=18,
        textColor=AZUL_INST,
        spaceAfter=4,
        spaceBefore=0,
        alignment=TA_LEFT,
    )
    subtitulo = ParagraphStyle(
        "Subtitulo",
        parent=base["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=8,
        alignment=TA_LEFT,
    )
    seccion = ParagraphStyle(
        "Seccion",
        parent=base["Heading2"],
        fontSize=12,
        textColor=AZUL_INST,
        spaceBefore=14,
        spaceAfter=4,
        borderPad=0,
    )
    normal = ParagraphStyle(
        "Normal2",
        parent=base["Normal"],
        fontSize=9,
        leading=13,
    )
    pie = ParagraphStyle(
        "Pie",
        parent=base["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
    )
    kpi_val = ParagraphStyle(
        "KpiVal",
        parent=base["Normal"],
        fontSize=22,
        textColor=AZUL_INST,
        alignment=TA_CENTER,
        leading=26,
    )
    kpi_lbl = ParagraphStyle(
        "KpiLbl",
        parent=base["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        leading=10,
    )
    return {
        "titulo": titulo, "subtitulo": subtitulo, "seccion": seccion,
        "normal": normal, "pie": pie, "kpi_val": kpi_val, "kpi_lbl": kpi_lbl,
    }


# ---------------------------------------------------------------------------
# 2. FIGURAS MATPLOTLIB → bytes para ReportLab
# ---------------------------------------------------------------------------

def _fig_to_image(fig, width_cm: float = 16, height_cm: float = 7) -> Image:
    """Convierte una figura matplotlib en un objeto Image de ReportLab."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def _grafica_barras_enfermedades(df_mun: pd.DataFrame, municipio: str) -> Image:
    """Barras horizontales de prevalencia por enfermedad."""
    enfs = ["diabetes", "hipertension", "dislipidemia", "obesidad"]
    etiquetas = [ETIQUETAS_ENF[e] for e in enfs]
    total = len(df_mun)
    conteos = [int((df_mun[e] == 1).sum()) for e in enfs]
    pcts    = [c / total * 100 if total else 0 for c in conteos]
    colores = [COLORES_ENF[e] for e in enfs]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(etiquetas, pcts, color=colores, height=0.5, edgecolor="white")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Prevalencia (%)", fontsize=8)
    ax.set_title(f"Prevalencia por enfermedad — {municipio}", fontsize=9,
                 color="#1a4e7a", pad=6)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, pct, n in zip(bars, pcts, conteos):
        ax.text(pct + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%  (n={n})", va="center", fontsize=7.5, color="#333")

    fig.tight_layout()
    return _fig_to_image(fig, width_cm=13, height_cm=5.5)


def _grafica_pastel_riesgo(df_mun: pd.DataFrame) -> Image:
    """Pastel de distribución por nivel de riesgo."""
    if "riesgo_nivel" not in df_mun.columns:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Sin datos de riesgo", ha="center", va="center")
        ax.axis("off")
        return _fig_to_image(fig, width_cm=7, height_cm=5.5)

    orden  = ["Bajo", "Moderado", "Alto", "Muy alto"]
    cols_r = {"Bajo": "#2ecc71", "Moderado": "#f1c40f",
              "Alto": "#e67e22", "Muy alto": "#e74c3c"}
    conteos = df_mun["riesgo_nivel"].value_counts()
    etiq = [n for n in orden if n in conteos.index]
    vals = [conteos[n] for n in etiq]
    cmap = [cols_r[n] for n in etiq]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    wedges, texts, autotexts = ax.pie(
        vals, labels=None, colors=cmap,
        autopct="%1.0f%%", startangle=90,
        pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 1.2}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, etiq, loc="lower center", fontsize=7,
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    ax.set_title("Distribución por\nnivel de riesgo", fontsize=9,
                 color="#1a4e7a", pad=4)
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=7, height_cm=5.5)


def _grafica_sexo_edad(df_mun: pd.DataFrame) -> Image:
    """Barras agrupadas de distribución por grupo de edad y sexo."""
    bins   = [0, 29, 39, 49, 59, 69, 200]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    df2 = df_mun.copy()
    df2["grupo_edad"] = pd.cut(df2["edad"], bins=bins, labels=labels,
                                right=True, include_lowest=True)
    pivot = (df2.groupby(["grupo_edad", "sexo"], observed=True)
               .size().unstack(fill_value=0))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.2))

    if "F" in pivot.columns:
        ax.bar(x - w/2, [pivot.loc[l, "F"] if l in pivot.index else 0 for l in labels],
               w, label="Femenino", color="#e91e8c", alpha=0.85)
    if "M" in pivot.columns:
        ax.bar(x + w/2, [pivot.loc[l, "M"] if l in pivot.index else 0 for l in labels],
               w, label="Masculino", color="#1565c0", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pacientes", fontsize=8)
    ax.set_title("Distribución por grupo de edad y sexo", fontsize=9,
                 color="#1a4e7a", pad=6)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=13, height_cm=5)


def _grafica_costos(df_costos_mun: Optional[pd.DataFrame]) -> Optional[Image]:
    """Barras agrupadas de proyección de costo por enfermedad y horizonte."""
    if df_costos_mun is None or df_costos_mun.empty:
        return None

    enfs   = df_costos_mun["enfermedad"].unique()
    anos   = sorted(df_costos_mun["anos"].unique())
    x      = np.arange(len(anos))
    n_enfs = len(enfs)
    w      = 0.8 / n_enfs

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, enf in enumerate(enfs):
        sub = df_costos_mun[df_costos_mun["enfermedad"] == enf]
        totales = [sub[sub["anos"] == a]["costo_total"].sum() / 1_000_000 for a in anos]
        offset  = (i - n_enfs / 2 + 0.5) * w
        bars    = ax.bar(x + offset, totales, w,
                         label=ETIQUETAS_ENF.get(enf, enf),
                         color=COLORES_ENF.get(enf, "#999"),
                         alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a} años" for a in anos], fontsize=8)
    ax.set_ylabel("Millones de pesos (MXN)", fontsize=8)
    ax.set_title("Proyección de costos por enfermedad", fontsize=9,
                 color="#1a4e7a", pad=6)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=14, height_cm=5.5)


# ---------------------------------------------------------------------------
# 3. COMPONENTES DE CONTENIDO
# ---------------------------------------------------------------------------

def _encabezado(municipio: str, n_total: int, estilos: dict) -> list:
    """Encabezado del reporte con título, fecha y línea divisora."""
    fecha = datetime.today().strftime("%d de %B de %Y").replace(
        "January","enero").replace("February","febrero").replace(
        "March","marzo").replace("April","abril").replace(
        "May","mayo").replace("June","junio").replace(
        "July","julio").replace("August","agosto").replace(
        "September","septiembre").replace("October","octubre").replace(
        "November","noviembre").replace("December","diciembre")

    items = [
        Paragraph(
            "Secretaría de Salud de Guanajuato",
            ParagraphStyle("inst", fontSize=8, textColor=colors.HexColor("#888"),
                           alignment=TA_LEFT)
        ),
        Paragraph(
            f"Reporte Epidemiológico ECNT — {municipio}",
            estilos["titulo"]
        ),
        Paragraph(
            f"Enfermedades Crónicas No Transmisibles &nbsp;|&nbsp; "
            f"Generado: {fecha} &nbsp;|&nbsp; "
            f"Total de pacientes registrados: <b>{n_total:,}</b>",
            estilos["subtitulo"]
        ),
        HRFlowable(width="100%", thickness=2, color=AZUL_INST,
                   spaceAfter=10),
    ]
    return items


def _tabla_kpis(df_mun: pd.DataFrame, estilos: dict) -> list:
    """Fila de 6 KPIs: total, 4 enfermedades y nivel de riesgo alto/muy alto."""
    n = len(df_mun)
    enfs = ["diabetes", "hipertension", "dislipidemia", "obesidad"]

    def kpi_cell(valor, etiqueta):
        return [
            Paragraph(str(valor), estilos["kpi_val"]),
            Paragraph(etiqueta,   estilos["kpi_lbl"]),
        ]

    n_alto = 0
    if "riesgo_nivel" in df_mun.columns:
        n_alto = int(df_mun["riesgo_nivel"].isin(["Alto", "Muy alto"]).sum())

    celdas = [
        kpi_cell(f"{n:,}", "Pacientes\ntotales"),
    ]
    for enf in enfs:
        cnt = int((df_mun[enf] == 1).sum())
        pct = cnt / n * 100 if n else 0
        celdas.append(kpi_cell(f"{cnt:,}", f"{ETIQUETAS_ENF[enf]}\n({pct:.1f}%)"))
    celdas.append(kpi_cell(f"{n_alto:,}", "Riesgo\nAlto / Muy alto"))

    data = [celdas]
    t = Table(data, colWidths=[2.8 * cm] * 6)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), GRIS_CLARO),
        ("BOX",         (0, 0), (-1, -1), 0.5, GRIS_BORDE),
        ("INNERGRID",   (0, 0), (-1, -1), 0.5, GRIS_BORDE),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("BACKGROUND",  (0, 0), (0, 0), AZUL_CLARO),
    ]))
    return [Paragraph("Indicadores clave", estilos["seccion"]), t]


def _tabla_alto_riesgo(df_mun: pd.DataFrame, estilos: dict, n_max: int = 15) -> list:
    """Tabla de los N pacientes con mayor score de riesgo."""
    if "riesgo_score" not in df_mun.columns:
        return [Paragraph("Sin datos de score de riesgo disponibles.",
                           estilos["normal"])]

    cols_show = ["id_paciente", "riesgo_semaforo", "riesgo_score",
                 "riesgo_nivel", "edad", "sexo", "municipio",
                 "diabetes", "hipertension", "dislipidemia", "obesidad"]
    cols_show = [c for c in cols_show if c in df_mun.columns]

    df_top = (df_mun[df_mun["riesgo_nivel"].isin(["Alto", "Muy alto"])]
              .sort_values("riesgo_score", ascending=False)
              .head(n_max)[cols_show]
              .reset_index(drop=True))

    if df_top.empty:
        return [Paragraph("No hay pacientes de alto riesgo en este municipio.",
                           estilos["normal"])]

    rename = {
        "id_paciente":    "ID",
        "riesgo_semaforo":"",
        "riesgo_score":   "Score",
        "riesgo_nivel":   "Nivel",
        "edad":           "Edad",
        "sexo":           "Sexo",
        "municipio":      "Municipio",
        "diabetes":       "DM",
        "hipertension":   "HTA",
        "dislipidemia":   "DLP",
        "obesidad":       "OB",
    }
    df_top = df_top.rename(columns={k: v for k, v in rename.items() if k in df_top.columns})

    # Cabecera
    encabezado = list(df_top.columns)
    filas = [encabezado]
    for _, row in df_top.iterrows():
        filas.append([str(v) if pd.notna(v) else "—" for v in row.values])

    n_cols = len(encabezado)
    col_w  = [16.8 / n_cols * cm] * n_cols   # distribuir ancho de página

    t = Table(filas, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  AZUL_INST),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("FONTSIZE",      (0, 0), (-1, 0),  7.5),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, GRIS_CLARO]),
        ("GRID",          (0, 0), (-1, -1), 0.3, GRIS_BORDE),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    return [
        Paragraph(f"Pacientes de alto riesgo (top {n_max})", estilos["seccion"]),
        t,
        Spacer(1, 4),
        Paragraph(
            "DM = Diabetes &nbsp; HTA = Hipertensión &nbsp; "
            "DLP = Dislipidemia &nbsp; OB = Obesidad &nbsp; Score = 0–100",
            estilos["pie"]
        ),
    ]


def _promedios_clinicos(df_mun: pd.DataFrame, estilos: dict) -> list:
    """Tabla compacta de promedios de indicadores clínicos."""
    indicadores = [
        ("HbA1c (%)",             "hba1c_pct"),
        ("Glucosa en ayuno (mg/dL)", "glucosa_ayunas"),
        ("Presión sistólica (mmHg)", "presion_sistolica"),
        ("Presión diastólica (mmHg)","presion_diastolica"),
        ("IMC (kg/m²)",           "imc"),
        ("LDL (mg/dL)",           "ldl"),
        ("Triglicéridos (mg/dL)", "trigliceridos"),
        ("HDL (mg/dL)",           "hdl"),
    ]
    filas = [["Indicador", "Promedio", "Mín", "Máx", "% con dato"]]
    for etiq, col in indicadores:
        if col not in df_mun.columns:
            continue
        s = pd.to_numeric(df_mun[col], errors="coerce").dropna()
        if s.empty:
            continue
        pct_dato = len(s) / len(df_mun) * 100
        filas.append([
            etiq,
            f"{s.mean():.1f}",
            f"{s.min():.1f}",
            f"{s.max():.1f}",
            f"{pct_dato:.0f}%",
        ])

    if len(filas) == 1:
        return []

    t = Table(filas, colWidths=[6*cm, 2.5*cm, 2*cm, 2*cm, 2.5*cm],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  AZUL_INST),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, GRIS_CLARO]),
        ("GRID",          (0, 0), (-1, -1), 0.3, GRIS_BORDE),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (0, -1),  6),
    ]))
    return [
        Paragraph("Promedios de indicadores clínicos", estilos["seccion"]),
        t,
    ]


def _pie_pagina(canvas_obj, doc):
    """Número de página y leyenda al pie."""
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.setFillColor(colors.HexColor("#888888"))
    canvas_obj.drawString(
        2 * cm, 1.2 * cm,
        "Secretaría de Salud de Guanajuato — Panel ECNT — Documento confidencial"
    )
    canvas_obj.drawRightString(
        19.5 * cm, 1.2 * cm,
        f"Página {doc.page}"
    )
    canvas_obj.restoreState()


# ---------------------------------------------------------------------------
# 4. CONSTRUCCIÓN DEL STORY POR MUNICIPIO
# ---------------------------------------------------------------------------

def _story_municipio(
    municipio: str,
    df_mun: pd.DataFrame,
    costos: Optional[dict],
    estilos: dict,
    es_ultimo: bool = True,
) -> list:
    """
    Genera la lista de flowables (story) para un municipio.
    Si es_ultimo=False, agrega un PageBreak al final para el PDF consolidado.
    """
    story = []
    n = len(df_mun)

    # --- Encabezado ---
    story.extend(_encabezado(municipio, n, estilos))
    story.append(Spacer(1, 6))

    # --- KPIs ---
    story.extend(_tabla_kpis(df_mun, estilos))
    story.append(Spacer(1, 10))

    # --- Gráficas fila 1: barras de prevalencia + pastel de riesgo ---
    story.append(Paragraph("Distribución epidemiológica", estilos["seccion"]))
    img_barras = _grafica_barras_enfermedades(df_mun, municipio)
    img_pastel = _grafica_pastel_riesgo(df_mun)

    fila_graficas = Table(
        [[img_barras, img_pastel]],
        colWidths=[13.5 * cm, 7.5 * cm],
    )
    fila_graficas.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(fila_graficas)
    story.append(Spacer(1, 8))

    # --- Gráfica fila 2: distribución edad/sexo ---
    story.append(_grafica_sexo_edad(df_mun))
    story.append(Spacer(1, 8))

    # --- Promedios clínicos ---
    story.extend(_promedios_clinicos(df_mun, estilos))
    story.append(Spacer(1, 10))

    # --- Tabla de alto riesgo ---
    story.extend(_tabla_alto_riesgo(df_mun, estilos))
    story.append(Spacer(1, 10))

    # --- Proyección de costos ---
    if costos:
        try:
            # Importamos compute_budget desde el módulo principal
            # para no duplicar la lógica de cálculo
            import importlib, sys
            if "seminario_Luisa" in sys.modules:
                mod = sys.modules["seminario_Luisa"]
                df_costos_mun = mod.compute_budget(df_mun, costos)
            else:
                df_costos_mun = _compute_budget_local(df_mun, costos)

            story.append(Paragraph("Proyección de costos", estilos["seccion"]))
            img_costos = _grafica_costos(df_costos_mun)
            if img_costos:
                story.append(img_costos)

            # Tabla resumen de costos por enfermedad a 10 y 20 años
            pivot_c = (df_costos_mun[df_costos_mun["anos"].isin([10, 20])]
                       .groupby(["enfermedad", "anos"])["costo_total"]
                       .sum().reset_index())
            if not pivot_c.empty:
                filas_c = [["Enfermedad", "Costo a 10 años (MXN)", "Costo a 20 años (MXN)"]]
                for enf in ["diabetes", "hipertension", "dislipidemia", "obesidad"]:
                    sub = pivot_c[pivot_c["enfermedad"] == enf]
                    c10 = sub[sub["anos"] == 10]["costo_total"].sum()
                    c20 = sub[sub["anos"] == 20]["costo_total"].sum()
                    if c10 > 0 or c20 > 0:
                        filas_c.append([
                            ETIQUETAS_ENF.get(enf, enf),
                            f"${c10:,.0f}",
                            f"${c20:,.0f}",
                        ])
                tc = Table(filas_c, colWidths=[5*cm, 6*cm, 6*cm])
                tc.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0),  AZUL_INST),
                    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                    ("FONTSIZE",      (0, 0), (-1, -1), 8),
                    ("ALIGN",         (1, 0), (-1, -1), "RIGHT"),
                    ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, GRIS_CLARO]),
                    ("GRID",          (0, 0), (-1, -1), 0.3, GRIS_BORDE),
                    ("TOPPADDING",    (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING",   (0, 0), (0, -1),  6),
                    ("RIGHTPADDING",  (-1, 0), (-1, -1), 6),
                ]))
                story.append(Spacer(1, 6))
                story.append(tc)
                story.append(Spacer(1, 4))
                story.append(Paragraph(
                    "* Proyección basada en costos unitarios del YAML de costos, "
                    "ajustados por inflación y tasa de descuento.",
                    estilos["pie"]
                ))
        except Exception as e:
            story.append(Paragraph(
                f"No se pudo calcular la proyección de costos: {e}",
                estilos["normal"]
            ))

    if not es_ultimo:
        story.append(PageBreak())

    return story


def _compute_budget_local(df: pd.DataFrame, costos: dict,
                           anos: list = [5, 10, 15, 20]) -> pd.DataFrame:
    """
    Versión local de compute_budget para cuando el módulo principal
    no está importado. Lógica idéntica a la de seminario_Luisa.py.
    """
    inflacion = costos.get("economia", {}).get("inflacion_g", 0.0)
    descuento = costos.get("economia", {}).get("descuento_r", 0.0)
    enfermedades = ["diabetes", "hipertension", "dislipidemia", "obesidad"]
    severidades  = ["leve", "moderada", "severa"]
    resultados   = []

    def _costo_pac(costos, enfermedad, severidad):
        if enfermedad not in costos:
            return 0.0
        enf_data  = costos[enfermedad]
        recursos  = costos.get("recursos", {})
        adherencia = costos.get("adherencia", {"alpha": 1.0, "beta": 1.0})
        alpha, beta = adherencia.get("alpha", 1.0), adherencia.get("beta", 1.0)
        sev    = enf_data.get("severidad", {}).get(severidad, {})
        q      = sev.get("q", {})
        eventos = sev.get("eventos", {})
        total  = 0.0
        for nombre, cantidad in q.items():
            if nombre in recursos:
                costo_u = recursos[nombre]["u"]
                if "mes" in nombre:
                    cantidad *= alpha
                total += costo_u * cantidad
        for ev, info in eventos.items():
            total += info.get("p", 0) * info.get("k", 0) * beta
        return total

    for enf in enfermedades:
        if enf not in df.columns:
            continue
        N = (df[enf] == 1).sum()
        if N == 0:
            continue
        for sev in severidades:
            costo_base = _costo_pac(costos, enf, sev)
            if costo_base == 0:
                continue
            for t in anos:
                factor    = ((1 + inflacion) ** t) / ((1 + descuento) ** t)
                resultados.append({
                    "enfermedad":    enf,
                    "severidad":     sev,
                    "anos":          t,
                    "pacientes":     int(N),
                    "costo_unitario": round(costo_base, 2),
                    "costo_total":   round(costo_base * factor * N, 2),
                })
    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# 5. GENERACIÓN DE PDF EN MEMORIA
# ---------------------------------------------------------------------------

def generar_pdf_municipio(
    municipio: str,
    df_filt: pd.DataFrame,
    costos: Optional[dict] = None,
) -> bytes:
    """
    Genera el PDF de un municipio específico y devuelve los bytes.

    Parámetros
    ----------
    municipio : nombre del municipio (debe existir en df_filt["municipio"])
    df_filt   : DataFrame ya filtrado y con columnas de score de riesgo
    costos    : diccionario del YAML de costos (opcional)
    """
    df_mun = df_filt[df_filt["municipio"] == municipio].copy()
    estilos = _estilos()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm,  bottomMargin=2 * cm,
        title=f"Reporte ECNT — {municipio}",
        author="Secretaría de Salud de Guanajuato",
    )
    story = _story_municipio(municipio, df_mun, costos, estilos, es_ultimo=True)
    doc.build(story, onFirstPage=_pie_pagina, onLaterPages=_pie_pagina)
    return buf.getvalue()


def generar_pdf_todos_municipios(
    df_filt: pd.DataFrame,
    costos: Optional[dict] = None,
    municipios: Optional[List[str]] = None,
) -> bytes:
    """
    Genera un PDF con todos los municipios (o los seleccionados),
    uno por página. Devuelve los bytes del PDF.

    Parámetros
    ----------
    df_filt    : DataFrame ya filtrado y con columnas de score de riesgo
    costos     : diccionario del YAML de costos (opcional)
    municipios : lista de municipios a incluir; si None, usa todos los del df
    """
    if municipios is None:
        municipios = sorted(df_filt["municipio"].dropna().unique().tolist())

    estilos = _estilos()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm,  bottomMargin=2 * cm,
        title="Reporte ECNT — Todos los municipios — Guanajuato",
        author="Secretaría de Salud de Guanajuato",
    )

    story = []
    for i, mun in enumerate(municipios):
        df_mun = df_filt[df_filt["municipio"] == mun].copy()
        if df_mun.empty:
            continue
        es_ultimo = (i == len(municipios) - 1)
        story.extend(_story_municipio(mun, df_mun, costos, estilos,
                                       es_ultimo=es_ultimo))

    doc.build(story, onFirstPage=_pie_pagina, onLaterPages=_pie_pagina)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 6. RENDER EN STREAMLIT
# ---------------------------------------------------------------------------

def render_seccion_pdf(
    df_filt: pd.DataFrame,
    costos: Optional[dict] = None,
) -> None:
    """
    Muestra la sección de generación de reportes PDF en Streamlit.

    Incluye:
      - Selector de municipio individual + botón de descarga
      - Selector múltiple para PDF consolidado + botón de descarga

    Uso (en seminario_Luisa.py):
        from reporte_pdf import render_seccion_pdf

        st.header("Reporte PDF ejecutivo")
        render_seccion_pdf(df_filt, costos)
    """
    import streamlit as st

    municipios_disp = sorted(df_filt["municipio"].dropna().unique().tolist())
    if not municipios_disp:
        st.warning("No hay municipios disponibles con los filtros actuales.")
        return

    # --- PDF individual ---
    st.subheader("Reporte por municipio")
    col1, col2 = st.columns([2, 1])
    with col1:
        mun_sel = st.selectbox(
            "Selecciona un municipio",
            options=municipios_disp,
            key="pdf_mun_sel",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generar PDF", key="btn_pdf_individual"):
            with st.spinner(f"Generando reporte de {mun_sel}…"):
                try:
                    pdf_bytes = generar_pdf_municipio(mun_sel, df_filt, costos)
                    st.download_button(
                        label=f"Descargar reporte — {mun_sel}",
                        data=pdf_bytes,
                        file_name=f"reporte_ecnt_{mun_sel.lower().replace(' ','_')}"
                                  f"_{datetime.today().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        key="dl_pdf_individual",
                    )
                    st.success("Reporte generado correctamente.")
                except Exception as e:
                    st.error(f"Error al generar el PDF: {e}")

    st.markdown("---")

    # --- PDF consolidado ---
    st.subheader("Reporte consolidado (todos los municipios)")
    muns_multi = st.multiselect(
        "Municipios a incluir (vacío = todos)",
        options=municipios_disp,
        key="pdf_muns_multi",
    )
    muns_final = muns_multi if muns_multi else municipios_disp
    st.caption(
        f"Se incluirán {len(muns_final)} municipios. "
        "La generación puede tomar algunos segundos."
    )

    if st.button("Generar PDF consolidado", key="btn_pdf_consolidado"):
        with st.spinner(f"Generando reporte consolidado ({len(muns_final)} municipios)…"):
            try:
                pdf_bytes = generar_pdf_todos_municipios(df_filt, costos,
                                                          municipios=muns_final)
                st.download_button(
                    label="Descargar reporte consolidado",
                    data=pdf_bytes,
                    file_name=f"reporte_ecnt_guanajuato_"
                              f"{datetime.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="dl_pdf_consolidado",
                )
                st.success(
                    f"Reporte consolidado con {len(muns_final)} municipios generado."
                )
            except Exception as e:
                st.error(f"Error al generar el PDF consolidado: {e}")
