"""Genera un CSV con datos FICTICIOS de laptops usadas para entrenar el modelo."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RUTA_CSV = BASE / "datos" / "laptops_usadas.csv"


def generar(n=150, semilla=42):
    rng = np.random.default_rng(semilla)

    ram = rng.choice([4, 8, 16, 32], size=n, p=[0.20, 0.40, 0.30, 0.10])             # X1: RAM (GB)
    almac = rng.choice([128, 256, 512, 1024], size=n, p=[0.20, 0.35, 0.30, 0.15])     # X2: almacenamiento (GB)
    anios = rng.uniform(0.5, 6, n).round(1)                                            # X3: años de uso
    bateria = rng.uniform(40, 100, n).round(0)                                         # X4: estado de batería (%)
    ruido = rng.normal(0, 300, n)                                                      # variación aleatoria (Bs)

    precio = 2000 + 130 * ram + 2.2 * almac - 300 * anios + 14 * bateria + ruido
    precio = np.clip(precio, 800, None).round(0)

    return pd.DataFrame({
        "ram_gb": ram,
        "almacenamiento_gb": almac,
        "anios_uso": anios,
        "bateria_pct": bateria.astype(int),
        "precio_bs": precio.astype(int),
    })


if __name__ == "__main__":
    RUTA_CSV.parent.mkdir(parents=True, exist_ok=True)
    generar().to_csv(RUTA_CSV, index=False)
    print(f"Archivo creado: {RUTA_CSV}")
