import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, dcc, html, Input, Output, State

# =========================
# Datos sintéticos (Telco)
# =========================
'''
np.random.seed(42)
df = pd.DataFrame({
    "Churn": np.random.choice(["Yes", "No"], size=7000, p=[0.26, 0.74]),
    "MonthlyCharges": np.random.normal(65, 20, 7000).clip(20, 120),
    "Tenure": np.random.randint(0, 72, 7000),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=7000, p=[0.55, 0.23, 0.22]),
})
'''
# =========================
# Datos REALES (Telco)
# =========================
import pandas as pd
import numpy as np

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['TotalCharges']   = pd.to_numeric(df['TotalCharges'],   errors='coerce')
df['tenure']         = pd.to_numeric(df['tenure'],         errors='coerce')

df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
mask_tc_nan = df['TotalCharges'].isna()
df.loc[mask_tc_nan, 'TotalCharges'] = df.loc[mask_tc_nan, 'MonthlyCharges'] * df.loc[mask_tc_nan, 'tenure']

cat_cols = ['gender','InternetService','StreamingTV','StreamingMovies',
            'Contract','PaymentMethod','PaperlessBilling','Churn']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

df['Tenure'] = df['tenure']
df = df[['Churn','MonthlyCharges','Tenure','Contract']].copy()

# Si existe endpoint real de predicción por lote, se pone la URL aquí (opcional)
API_URL_BATCH = "http://127.0.0.1:8000/predict_batch"  # p.ej. "http://localhost:5000/predict_batch" (payload: {"records":[...]}, resp: {"probas":[...]})

# =========================
# KPIs base
# =========================
def calc_kpis(data: pd.DataFrame):
    churn_rate = data["Churn"].eq("Yes").mean() * 100 if len(data) else 0.0
    avg_monthly = data["MonthlyCharges"].mean() if len(data) else 0.0
    avg_tenure = data["Tenure"].mean() if len(data) else 0.0
    total_customers = len(data)
    return churn_rate, avg_monthly, avg_tenure, total_customers

# =========================
# Helpers UI
# =========================
def kpi_card(title, value, color):
    return html.Div([
        html.H4(title, style={"color": "#666", "marginBottom": "0"}),
        html.H2(value, style={"color": color, "marginTop": "5px"})
    ], style={
        "background": "white",
        "borderRadius": "15px",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
        "padding": "18px",
        "textAlign": "center",
        "flex": "1",
        "minWidth": "200px"
    })

def fig_pie(data: pd.DataFrame):
    fig = px.pie(data, names="Churn", title="Distribución de Churn",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig

def fig_contract(data: pd.DataFrame):
    fig = px.histogram(data, x="Contract", color="Churn", barmode="group",
                       title="Churn por tipo de contrato",
                       color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40))
    return fig

def fig_monthly(data: pd.DataFrame):
    fig = px.histogram(data, x="MonthlyCharges", color="Churn", nbins=40, opacity=0.75,
                       title="Distribución de MonthlyCharges",
                       color_discrete_sequence=["#EF553B", "#00CC96"])
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40))
    return fig

def fig_box(data: pd.DataFrame):
    fig = px.box(data, x="Churn", y="Tenure", color="Churn",
                 title="Tenure por estado de churn",
                 color_discrete_sequence=["#00CC96", "#EF553B"])
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40))
    return fig

def fig_gauge(prob):
    p = float(np.clip(prob, 0.0, 1.0)) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p,
        number={'suffix': '%', 'valueformat': '.1f'},
        title={'text': "Probabilidad de Churn"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#EF553B"},
            'steps': [
                {'range': [0, 30], 'color': '#E5E7EB'},
                {'range': [30, 60], 'color': '#CBD5E1'},
                {'range': [60, 100], 'color': '#94A3B8'}
            ]
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ===== Scoring (demo) para visualizar "lo que hace" =====
def simple_fallback_scoring(contract: str, tenure: float, charges: float) -> float:
    base = 0.2
    if contract == "Month-to-month":
        base += 0.25
    elif contract == "One year":
        base += 0.10
    else:
        base += 0.05
    base += max(0, (70 - (tenure or 0))) / 140.0
    base += max(0, (charges or 0) - 60) / 200.0
    return float(np.clip(base, 0.01, 0.99))

def add_risk_column(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    # llamar y llenar Risk = proba si hay api por lote
    if API_URL_BATCH:
        try:
            payload = d[["Contract", "Tenure", "MonthlyCharges"]].to_dict(orient="records")
            r = requests.post(API_URL_BATCH, json={"records": payload}, timeout=8)
            r.raise_for_status()
            probs = r.json().get("probas", [])
            if len(probs) == len(d):
                d["Risk"] = np.clip(probs, 0, 1.0)
                return d
        except Exception:
            pass
    # Fallback determinístico para visualizar
    d["Risk"] = [
        simple_fallback_scoring(ct, tn, ch)
        for ct, tn, ch in zip(d["Contract"], d["Tenure"], d["MonthlyCharges"])
    ]
    return d

# ====== CORREGIDO: heatmap sin Interval (JSON-serializable) ======
def fig_riskmap(data: pd.DataFrame):
    d = data.copy()
    # Bins
    d["Tenure_bin"]  = pd.cut(d["Tenure"], bins=[0,6,12,24,36,48,60,72], include_lowest=True)
    d["Charges_bin"] = pd.cut(d["MonthlyCharges"], bins=[20,40,60,80,100,120], include_lowest=True)

    # Matriz de riesgo promedio (evita FutureWarning con observed=False)
    piv = d.pivot_table(
        index="Tenure_bin",
        columns="Charges_bin",
        values="Risk",
        aggfunc="mean",
        observed=False
    )

    # Convertir Interval -> string (clave para serializar)
    x_labels = [f"{c.left:.0f}-{c.right:.0f}" for c in piv.columns]
    y_labels = [f"{r.left:.0f}-{r.right:.0f}" for r in piv.index]
    z = piv.to_numpy()

    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        x=x_labels, y=y_labels, z=z,
        colorscale="Reds", zmin=0, zmax=1,
        colorbar=dict(title="Prob.")
    ))
    fig.update_layout(
        title="Mapa de riesgo (Tenure vs MonthlyCharges)",
        xaxis_title="MonthlyCharges ($)",
        yaxis_title="Tenure (meses)",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

def fig_risk_by_contract(data: pd.DataFrame):
    g = (data.groupby("Contract", as_index=False)
            .agg(Risk=("Risk","mean"), N=("Risk","size"))
            .sort_values("Risk", ascending=False))
    fig = px.bar(
        g, x="Contract", y="Risk",
        text=(g["Risk"]*100).round(1).astype(str)+"%",
        title="Riesgo promedio por tipo de contrato",
        color="Risk", color_continuous_scale="Reds", range_y=[0,1]
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=20,r=20,t=60,b=40), coloraxis_showscale=False)
    return fig

# =========================
# App
# =========================
app = Dash(__name__)
app.title = "Mock Telco Churn Dashboard"

tenure_min, tenure_max = int(df["Tenure"].min()), int(df["Tenure"].max())
charge_min, charge_max = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())

app.layout = html.Div([
    html.H1("Telco Customer Churn — Mock Dashboard",
            style={"textAlign": "center", "color": "#222", "fontFamily": "Segoe UI", "marginBottom": "8px"}),
    html.P("Arriba ves el comportamiento del RIESGO estimado (no solo el EDA). Abajo puedes filtrar.",
           style={"textAlign": "center", "color": "#888", "marginTop": "0"}),

    # KPIs
    html.Div(id="kpi-row", style={"display": "flex", "gap": "20px", "margin": "20px", "flexWrap": "wrap"}),

    # Dos columnas
    html.Div([
        # Izquierda: filtros + scoring individual
        html.Div([
            html.Div([
                html.H3("Controles", style={"marginTop": "0", "marginBottom": "10px", "color": "#222"}),

                html.Label("Contract", style={"fontWeight": 600}),
                dcc.Checklist(
                    id="flt-contract",
                    options=[{"label": c, "value": c} for c in df["Contract"].unique()],
                    value=list(df["Contract"].unique()),
                    inputStyle={"marginRight": "6px", "marginLeft": "4px"},
                    labelStyle={"display": "block", "marginBottom": "6px"}
                ),

                html.Label("Estado de Churn", style={"fontWeight": 600, "marginTop": "12px"}),
                dcc.Checklist(
                    id="flt-churn",
                    options=[{"label": s, "value": s} for s in ["Yes", "No"]],
                    value=["Yes", "No"],
                    inputStyle={"marginRight": "6px", "marginLeft": "4px"},
                    labelStyle={"display": "inline-block", "marginRight": "12px"}
                ),

                html.Label("Tenure (meses)", style={"fontWeight": 600, "marginTop": "12px"}),
                dcc.RangeSlider(
                    id="flt-tenure",
                    min=tenure_min, max=tenure_max, step=1, value=[tenure_min, tenure_max],
                    marks={0:"0", 12:"12", 24:"24", 36:"36", 48:"48", 60:"60", 72:"72"}
                ),

                html.Label("MonthlyCharges ($)", style={"fontWeight": 600, "marginTop": "12px"}),
                dcc.RangeSlider(
                    id="flt-charges",
                    min=round(charge_min, 0), max=round(charge_max, 0), step=1,
                    value=[round(charge_min, 0), round(charge_max, 0)],
                    marks={20:"20", 40:"40", 60:"60", 80:"80", 100:"100", 120:"120"}
                ),

                html.Hr(),
                html.Label("Gráfico principal", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="chart-type",
                    options=[
                        {"label": "Mapa de riesgo (2D)", "value": "riskmap"},
                        {"label": "Riesgo por contrato (Barras)", "value": "riskcontract"},
                        {"label": "Distribución de Churn (Pie)", "value": "pie"},
                        {"label": "Churn por Contrato (Barras)", "value": "contract"},
                        {"label": "MonthlyCharges (Hist)", "value": "monthly"},
                        {"label": "Tenure por Churn (Box)", "value": "box"},
                    ],
                    value="riskmap",
                    clearable=False
                ),

                html.Hr(),

                html.H3("Scoring (cliente individual)", style={"marginBottom": "8px", "color": "#222"}),
                html.Label("Contract", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="sc-contract",
                    options=[{"label": c, "value": c} for c in ["Month-to-month", "One year", "Two year"]],
                    value="Month-to-month",
                    clearable=False
                ),

                html.Label("Tenure (meses)", style={"fontWeight": 600, "marginTop": "10px"}),
                dcc.Input(
                    id="sc-tenure",
                    type="number",
                    min=0, max=72, step=1, value=6,
                    style={"width": "100%", "padding": "8px", "borderRadius": "8px", "border": "1px solid #e5e7eb"}
                ),

                html.Label("MonthlyCharges ($)", style={"fontWeight": 600, "marginTop": "10px"}),
                dcc.Input(
                    id="sc-charges",
                    type="number",
                    min=0, step=1, value=70,
                    style={"width": "100%", "padding": "8px", "borderRadius": "8px", "border": "1px solid #e5e7eb"}
                ),

                html.Button("Predecir Churn", id="btn-score",
                            style={"marginTop":"12px","padding":"10px 14px","borderRadius":"12px",
                                   "border":"1px solid #111827","background":"#111827","color":"white",
                                   "cursor":"pointer","fontWeight":600, "width":"100%"}),
                html.Div(id="score-msg", style={"color":"#6b7280","fontSize":"12px","marginTop":"6px"})
            ]),
        ], style={
            "background": "white", "borderRadius": "15px", "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            "padding": "18px", "flex": "1", "minWidth": "320px", "maxWidth": "420px"
        }),

        # Derecha: gráfico principal + gauge
        html.Div([
            dcc.Loading(
                dcc.Graph(id="main-graph", style={"height": "420px"}),
                type="circle"
            ),
            html.Div(id="sample-size", style={"textAlign":"right", "color":"#666", "fontSize":"12px"}),
            html.Hr(),
            dcc.Loading(
                dcc.Graph(id="score-gauge", style={"height": "260px"}),
                type="circle"
            )
        ], style={
            "background": "white", "borderRadius": "15px", "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            "padding": "12px", "flex": "2", "minWidth": "360px"
        })

    ], style={"display": "flex", "gap": "20px", "margin": "0 20px 20px 20px", "flexWrap": "wrap"}),

], style={"backgroundColor": "#f8f9fc", "padding": "16px"})

# =========================
# Callbacks
# =========================
def apply_filters(data: pd.DataFrame, contract_sel, churn_sel, tenure_rng, charge_rng):
    d = data.copy()
    if contract_sel:
        d = d[d["Contract"].isin(contract_sel)]
    if churn_sel:
        d = d[d["Churn"].isin(churn_sel)]
    if tenure_rng and len(tenure_rng) == 2:
        d = d[(d["Tenure"] >= tenure_rng[0]) & (d["Tenure"] <= tenure_rng[1])]
    if charge_rng and len(charge_rng) == 2:
        d = d[(d["MonthlyCharges"] >= charge_rng[0]) & (d["MonthlyCharges"] <= charge_rng[1])]
    return d

@app.callback(
    Output("kpi-row", "children"),
    Output("main-graph", "figure"),
    Output("sample-size", "children"),
    Input("flt-contract", "value"),
    Input("flt-churn", "value"),
    Input("flt-tenure", "value"),
    Input("flt-charges", "value"),
    Input("chart-type", "value")
)
def update_dashboard(contract_sel, churn_sel, tenure_rng, charge_rng, chart_type):
    d = apply_filters(df, contract_sel, churn_sel, tenure_rng, charge_rng)

    # Añadimos columna de riesgo para que el gráfico principal refleje "lo que hace"
    d_risk = add_risk_column(d)

    churn_rate, avg_monthly, avg_tenure, total_customers = calc_kpis(d)
    avg_risk = d_risk["Risk"].mean() * 100 if len(d_risk) else 0.0

    kpis = [
        kpi_card("Churn Rate", f"{churn_rate:.1f}%", "#EF553B"),
        kpi_card("Avg Predicted Risk", f"{avg_risk:.1f}%", "#F59E0B"),
        kpi_card("Avg Monthly Charges", f"${avg_monthly:.2f}", "#10B981"),
        kpi_card("Avg Tenure", f"{avg_tenure:.1f} meses", "#636EFA"),
    ]

    # Gráfico principal con opciones "de riesgo" y "EDA"
    if chart_type == "riskmap":
        fig = fig_riskmap(d_risk if len(d_risk) else d_risk.iloc[0:0])
    elif chart_type == "riskcontract":
        fig = fig_risk_by_contract(d_risk if len(d_risk) else d_risk.iloc[0:0])
    elif chart_type == "pie":
        fig = fig_pie(d if len(d) else df.iloc[0:0])
    elif chart_type == "contract":
        fig = fig_contract(d if len(d) else df.iloc[0:0])
    elif chart_type == "monthly":
        fig = fig_monthly(d if len(d) else df.iloc[0:0])
    else:
        fig = fig_box(d if len(d) else df.iloc[0:0])

    n_text = f"Muestra filtrada: {len(d):,} registros"
    return kpis, fig, n_text

# ===== Scoring individual (gauge) =====
@app.callback(
    Output("score-gauge", "figure"),
    Output("score-msg", "children"),
    Input("btn-score", "n_clicks"),
    State("sc-contract", "value"),
    State("sc-tenure", "value"),
    State("sc-charges", "value"),
    prevent_initial_call=True
)
def score_customer(n_clicks, ct, tn, ch):
    prob = simple_fallback_scoring(ct, float(tn or 0), float(ch or 0))
    return fig_gauge(prob), "Probabilidad estimada (demo). Conecta tu endpoint para valores reales."

if __name__ == "__main__":
    app.run(debug=True)
