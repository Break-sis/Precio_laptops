"""Modelo de regresión lineal múltiple usando la ecuación normal.

    beta = (X^T X)^-1 X^T Y

Tema: predicción del precio (Bs) de una laptop usada.
"""
import numpy as np
import pandas as pd

COLUMNAS_X = ["ram_gb", "almacenamiento_gb", "anios_uso", "bateria_pct"]
COLUMNA_Y = "precio_bs"
ETIQUETAS_X = ["RAM (GB)", "Almacenamiento (GB)", "Años de uso", "Batería (%)"]
ETIQUETA_Y = "Precio (Bs)"
NOMBRES_BETA = ["Intercepto (b0)", "RAM (b1)", "Almacenamiento (b2)",
                "Años de uso (b3)", "Batería (b4)"]


def cargar_datos(origen):
    """Lee el CSV (ruta o archivo en memoria) y verifica las columnas necesarias."""
    df = pd.read_csv(origen)
    faltantes = [c for c in COLUMNAS_X + [COLUMNA_Y] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")
    df = df[COLUMNAS_X + [COLUMNA_Y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 10:
        raise ValueError("El CSV necesita al menos 10 filas válidas.")
    return df.reset_index(drop=True)


def dividir_datos(df, proporcion_prueba=0.2, semilla=42):
    """Separa los datos en entrenamiento (80%) y prueba (20%)."""
    rng = np.random.default_rng(semilla)
    indices = rng.permutation(len(df))
    n_prueba = max(1, int(len(df) * proporcion_prueba))
    return df.iloc[indices[n_prueba:]], df.iloc[indices[:n_prueba]]


def construir_matriz_X(valores):
    """Agrega una columna de unos (para el intercepto) a la matriz de datos."""
    valores = np.asarray(valores, dtype=float)
    return np.column_stack([np.ones(len(valores)), valores])


class RegresionLinealMultiple:
    def __init__(self):
        self.beta = None
        self.X = None
        self.Xt = None
        self.Y = None

    def entrenar(self, valores_x, valores_y):
        self.X = construir_matriz_X(valores_x)       # matriz de datos X
        self.Xt = self.X.T                           # matriz transpuesta
        self.Y = np.asarray(valores_y, dtype=float)  # vector de resultados Y
        XtX = self.Xt @ self.X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)            # si no es invertible
        self.beta = XtX_inv @ self.Xt @ self.Y       # beta = (XtX)^-1 Xt Y
        return self.beta

    def predecir(self, valores_x):
        if self.beta is None:
            raise RuntimeError("Primero debes entrenar el modelo.")
        valores_x = np.atleast_2d(np.asarray(valores_x, dtype=float))
        return construir_matriz_X(valores_x) @ self.beta


def calcular_metricas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_real - y_pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_real - y_real.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
