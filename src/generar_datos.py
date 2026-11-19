"""Genera un conjunto de datos SINTETICO para el modelo de riesgo de salud mental.

No contiene personas reales. Semilla fija para que sea reproducible.
Uso: python src/generar_datos.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
N = 1200


def generar(n: int = N, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    df = pd.DataFrame(
        {
            "edad": rng.integers(18, 36, n),
            "sexo": rng.choice(["Femenino", "Masculino"], n),
            "estrato": rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.22, 0.28, 0.25, 0.14, 0.08, 0.03]),
            "ocupacion": rng.choice(
                ["Estudiante", "Empleado", "Independiente", "Desempleado"], n, p=[0.40, 0.30, 0.18, 0.12]
            ),
            "horas_sueno": np.clip(rng.normal(6.8, 1.3, n), 3, 10).round(1),
            "actividad_fisica": rng.choice(["Baja", "Media", "Alta"], n, p=[0.40, 0.38, 0.22]),
            "pantalla_horas": np.clip(rng.normal(6.5, 2.6, n), 1, 14).round(1),
            "sustancias": rng.choice(["Ninguno", "Ocasional", "Frecuente"], n, p=[0.60, 0.28, 0.12]),
            "estres_percibido": np.clip(rng.normal(19, 7, n), 0, 40).round(0).astype(int),
            "apoyo_social": np.clip(rng.normal(6.0, 2.2, n), 0, 10).round(1),
            "eventos_vitales": rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.30, 0.28, 0.20, 0.12, 0.07, 0.03]),
            "antecedentes_familiares": rng.choice(["Si", "No"], n, p=[0.28, 0.72]),
            "atencion_previa": rng.choice(["Si", "No"], n, p=[0.22, 0.78]),
        }
    )

    # Regla de puntaje (con ruido) que define la clase. Es ficticia: solo da senal para aprender.
    puntaje = (
        0.10 * df["estres_percibido"]
        + 0.40 * (7 - df["horas_sueno"]).clip(lower=0)
        - 0.25 * df["apoyo_social"]
        + 0.35 * df["eventos_vitales"]
        + df["sustancias"].map({"Ninguno": 0.0, "Ocasional": 0.4, "Frecuente": 1.0})
        + df["actividad_fisica"].map({"Baja": 0.5, "Media": 0.2, "Alta": 0.0})
        + 0.05 * (df["pantalla_horas"] - 6).clip(lower=0)
        + df["antecedentes_familiares"].map({"Si": 0.5, "No": 0.0})
        + rng.normal(0, 0.35, n)
    )
    corte_moderado, corte_alto = np.quantile(puntaje, [0.50, 0.82])
    df["riesgo"] = np.select(
        [puntaje >= corte_alto, puntaje >= corte_moderado],
        ["Alto", "Moderado"],
        default="Sin riesgo",
    )
    return df


if __name__ == "__main__":
    salida = Path(__file__).resolve().parent.parent / "data" / "dataset_sintetico.csv"
    datos = generar()
    datos.to_csv(salida, index=False)
    print(f"Guardado: {salida} ({len(datos)} filas)")
    print(datos["riesgo"].value_counts())
