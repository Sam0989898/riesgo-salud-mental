"""Entrena y compara J48 (arbol C4.5 aprox.), Random Forest y Naive Bayes.

Validacion cruzada estratificada de 10 pliegues. Guarda metricas y matrices de confusion.
Uso: python src/entrenar_modelos.py
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parent.parent
CLASES = ["Sin riesgo", "Moderado", "Alto"]
CATEGORICAS = ["sexo", "ocupacion", "actividad_fisica", "sustancias", "antecedentes_familiares", "atencion_previa"]


def construir_modelos(columnas):
    numericas = [c for c in columnas if c not in CATEGORICAS]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS)],
        remainder="passthrough",
    )
    return {
        "J48 (arbol C4.5)": Pipeline(
            [("pre", pre), ("clf", DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5, random_state=42))]
        ),
        "Random Forest": Pipeline(
            [("pre", pre), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))]
        ),
        "Naive Bayes": Pipeline([("pre", pre), ("clf", GaussianNB())]),
    }


def main():
    df = pd.read_csv(RAIZ / "data" / "dataset_sintetico.csv")
    X, y = df.drop(columns="riesgo"), df["riesgo"]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    filas, matrices = [], {}
    for nombre, modelo in construir_modelos(X.columns).items():
        pred = cross_val_predict(modelo, X, y, cv=cv)
        filas.append(
            {
                "Modelo": nombre,
                "Exactitud": accuracy_score(y, pred),
                "Precision": precision_score(y, pred, average="weighted", zero_division=0),
                "Exhaustividad": recall_score(y, pred, average="weighted"),
                "F1": f1_score(y, pred, average="weighted"),
                "Exhaustividad clase Alto": recall_score(y, pred, labels=["Alto"], average=None)[0],
            }
        )
        matrices[nombre] = pd.DataFrame(
            confusion_matrix(y, pred, labels=CLASES),
            index=[f"real_{c}" for c in CLASES],
            columns=[f"pred_{c}" for c in CLASES],
        )

    resultados = pd.DataFrame(filas).round(3)
    resultados.to_csv(RAIZ / "results" / "metricas.csv", index=False)
    with open(RAIZ / "results" / "matrices_confusion.txt", "w", encoding="utf-8") as f:
        for nombre, m in matrices.items():
            f.write(f"== {nombre} ==\n{m}\n\n")
    print(resultados.to_string(index=False))


if __name__ == "__main__":
    main()
