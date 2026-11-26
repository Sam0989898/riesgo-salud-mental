"""Carga el modelo entrenado y clasifica un perfil de ejemplo.

Uso: python src/predecir_ejemplo.py
"""
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

perfil = pd.DataFrame([{
    "edad": 27, "sexo": "Masculino", "estrato": 2, "ocupacion": "Empleado",
    "horas_sueno": 6.5, "actividad_fisica": "Media", "pantalla_horas": 7.0,
    "sustancias": "Ocasional", "estres_percibido": 20, "apoyo_social": 5.5,
    "eventos_vitales": 2, "antecedentes_familiares": "No", "atencion_previa": "No",
}])

modelo = joblib.load(RAIZ / "models" / "random_forest.joblib")
probabilidades = modelo.predict_proba(perfil)[0]
print("Nivel de riesgo estimado:", modelo.predict(perfil)[0])
for clase, p in zip(modelo.classes_, probabilidades):
    print(f"  {clase}: {p:.0%}")
