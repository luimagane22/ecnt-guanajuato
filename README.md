# ECNT Epidemiological Surveillance Dashboard — Guanajuato

A public health intelligence platform developed as an undergraduate thesis project for the **B.Sc. in Computational Mathematics**. The system enables physicians, healthcare administrators, and public health authorities in the state of Guanajuato to monitor, prioritize, and follow up on patients with **Non-Communicable Chronic Diseases (NCDs)**, including diabetes, hypertension, dyslipidemia, and obesity.

## Overview

The application is an interactive dashboard built with **Streamlit** that centralizes clinical, epidemiological, and economic analyses for populations affected by NCDs. It is designed for two primary user profiles:

* **Physicians**: Access individual patient risk scores with detailed clinical explanations, automated alerts, and longitudinal monitoring.
* **Healthcare administrators and policymakers**: Access population-level risk prioritization tables, municipal distribution maps, cost projections, and downloadable PDF reports.

---

## Modules

| File                       | Responsibility                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| `seminario_Luisa.py`       | Main application: navigation, data loading, clustering, maps, cost analysis, and PDF generation |
| `score_riesgo.py`          | Cardiovascular and metabolic risk score computation (0–100 points)                              |
| `alertas.py`               | Automated clinical alerts based on ADA, JNC-8, and ATP-III guidelines                           |
| `seguimiento_pacientes.py` | Longitudinal patient monitoring dashboard                                                       |
| `conclusiones.py`          | Automatic natural-language insight generation                                                   |
| `normalizador_sis2024.py`  | Data cleaning and normalization for SIS-2024 Excel files                                        |

---

## Dashboard Pages

1. **Home** — Global KPIs, executive summary, and automated insights
2. **Risk Score** — Individual risk traffic-light indicator and population-wide prioritization table
3. **Clinical Alerts** — Automatic detection of critical clinical values
4. **Patient Follow-Up** — Longitudinal evolution of patient indicators
5. **Epidemiological Map** — Municipal choropleth visualization (Folium + GeoJSON)
6. **Patient Registry** — Complete patient database with CSV/Excel export
7. **Patient Clusters** — K-Means clustering with Elbow and Silhouette analysis
8. **Projected Costs** — Budget projections for 5, 10, 15, and 20 years
9. **PDF Report** — Downloadable municipality-level official reports

---

## Risk Scoring Model

A risk score ranging from 0 to 100 points based on six clinical factors, following **AHA/ACC** and **ESC** guidelines:

| Factor         | Maximum Weight | Primary Indicator                         |
| -------------- | -------------- | ----------------------------------------- |
| Diabetes       | 25 pts         | HbA1c (ADA thresholds) / fasting glucose  |
| Hypertension   | 25 pts         | Systolic/diastolic blood pressure (JNC-8) |
| Dyslipidemia   | 15 pts         | LDL, triglycerides, and low HDL (ATP-III) |
| Obesity        | 15 pts         | BMI and waist circumference               |
| Age            | 10 pts         | Independent risk factor                   |
| Multimorbidity | 10 pts         | Number of concurrent NCDs                 |

### Risk Categories

| Range  | Level        | Recommended Action                         |
| ------ | ------------ | ------------------------------------------ |
| 0–29   | 🟢 Low       | Annual follow-up                           |
| 30–54  | 🟡 Moderate  | Follow-up appointment within 3–6 months    |
| 55–74  | 🟠 High      | Medical consultation within the next month |
| 75–100 | 🔴 Very High | Priority care / urgent referral            |

> This score is intended as a clinical prioritization tool and does not replace validated 10-year cardiovascular risk models such as Framingham or SCORE2.

---

## Clinical Variables

The system validates and processes more than 35 patient variables, including:

### Demographic Variables

* Age
* Sex
* Municipality
* Indigenous origin

### Chronic Conditions

* Diabetes
* Hypertension
* Dyslipidemia
* Obesity

### Family Medical History

* Cardiovascular disease
* Hypertension
* Diabetes
* Dyslipidemia
* Obesity
* Cerebrovascular disease

### Personal Medical History

* Smoking
* Alcohol consumption
* Sedentary lifestyle
* Gestational diabetes
* Postmenopausal status
* HIV
* Tuberculosis

### Clinical Measurements

* Blood glucose
* HbA1c
* Blood pressure
* Weight
* BMI
* Waist circumference
* Total cholesterol
* LDL cholesterol
* HDL cholesterol
* Triglycerides

---

## Technologies

* **Python 3** / Streamlit
* `pandas`, `numpy` — Data processing
* `scikit-learn` — K-Means clustering, imputation, and scaling
* `plotly` — Interactive visualizations
* `folium` + `streamlit-folium` — Municipal choropleth mapping
* `statsmodels` — Time-series forecasting (Holt-Winters, ARIMA) *(optional)*
* `reportlab` — PDF report generation
* `pyyaml` — Healthcare cost configuration management

---

## Installation and Execution

```bash
pip install -r requirements.txt
streamlit run seminario_Luisa.py
```

### Expected Data Files (`data/` folder)

| File                     | Description                                      | Required          |
| ------------------------ | ------------------------------------------------ | ----------------- |
| `pacientes.xlsx`         | Patient database following the SIS-2024 schema   | Yes*              |
| `municipios_gto.geojson` | Guanajuato municipal boundaries                  | For mapping       |
| `costos.yaml`            | Disease-specific costs and projection parameters | For cost analysis |

* If `pacientes.xlsx` is not provided, the application automatically generates **1,500 synthetic patients** with clinically plausible distributions for demonstration purposes.

---

## Project Structure

```text
├── seminario_Luisa.py          # Main application
├── score_riesgo.py             # Risk scoring module
├── alertas.py                  # Clinical alert module
├── seguimiento_pacientes.py    # Longitudinal monitoring
├── conclusiones.py             # Automated insights
├── normalizador_sis2024.py     # Data normalization
├── reporte_pdf.py              # PDF report generation
├── data/
│   ├── pacientes.xlsx
│   ├── municipios_gto.geojson
│   └── costos.yaml
└── requirements.txt
```
