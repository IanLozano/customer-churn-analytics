# model/predict.py
import pandas as pd
import mlflow.pyfunc

# Ajusta esta ruta a tu carpeta artifacts (usa r"" en Windows):
MODEL_PATH = r"mlruns/861949064364458891/models/m-f6764e0be71e4524bc1a9af2e6179144/artifacts"

# Carga el modelo guardado por MLflow
_model = mlflow.pyfunc.load_model(MODEL_PATH)

def make_prediction(input_data: pd.DataFrame):
    """
    Recibe un DataFrame con columnas de entrada (Contract, Tenure, MonthlyCharges, etc.)
    Devuelve: {"probas": [p_yes, p_yes, ...]} con probabilidades de churn=Yes (0..1).
    """
    preds = _model.predict(input_data)

    # Normaliza diferentes formatos de salida típicos
    if isinstance(preds, pd.DataFrame):
        # intenta columnas comunes
        for col in ["proba_yes", "prob_yes", "p_yes", "churn_proba", "Yes", "proba_Yes"]:
            if col in preds.columns:
                return {"probas": preds[col].astype(float).tolist()}
        # si solo tiene una columna, úsala
        if preds.shape[1] == 1:
            return {"probas": preds.iloc[:, 0].astype(float).tolist()}
        # si son 2 columnas tipo [p_no, p_yes], toma la 2
        if preds.shape[1] >= 2:
            return {"probas": preds.iloc[:, 1].astype(float).tolist()}

    if isinstance(preds, pd.Series):
        return {"probas": preds.astype(float).tolist()}

    if isinstance(preds, list):
        # lista directa de probabilidades
        return {"probas": [float(x) for x in preds]}

    # último recurso: convierte a DataFrame y usa la primera col
    preds_df = pd.DataFrame(preds)
    return {"probas": preds_df.iloc[:, 0].astype(float).tolist()}
