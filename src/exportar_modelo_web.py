"""Entrena los 3 modelos con todo el dataset sintetico y los exporta a JSON para la pagina web.

Uso: python src/exportar_modelo_web.py
Salida: docs/modelos.json
"""
import json

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from entrenar_modelos import CATEGORICAS, construir_modelos

RAIZ = Path(__file__).resolve().parent.parent


def exportar_arbol(est):
    t = est.tree_
    hojas, f, th, l, r = [], [], [], [], []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            hojas.append([round(float(x), 2) for x in v / v.sum()])
            f.append(-1)
            th.append(0)
            l.append(-1)
            r.append(len(hojas) - 1)
        else:
            f.append(int(t.feature[i]))
            th.append(round(float(t.threshold[i]), 3))
            l.append(int(t.children_left[i]))
            r.append(int(t.children_right[i]))
    return {"f": f, "t": th, "l": l, "r": r, "p": hojas}


def main():
    df = pd.read_csv(RAIZ / "data" / "dataset_sintetico.csv")
    X, y = df.drop(columns="riesgo"), df["riesgo"]
    modelos = construir_modelos(X.columns)
    for m in modelos.values():
        m.fit(X, y)

    (RAIZ / "models").mkdir(exist_ok=True)
    for nombre, archivo in [("J48 (arbol C4.5)", "j48_arbol.joblib"), ("Random Forest", "random_forest.joblib"), ("Naive Bayes", "naive_bayes.joblib")]:
        joblib.dump(modelos[nombre], RAIZ / "models" / archivo, compress=3)

    pre = modelos["Random Forest"].named_steps["pre"]
    nombres = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    clases = list(modelos["Random Forest"].named_steps["clf"].classes_)

    arbol = modelos["J48 (arbol C4.5)"].named_steps["clf"]
    rf = modelos["Random Forest"].named_steps["clf"]
    nb = modelos["Naive Bayes"].named_steps["clf"]

    # Importancia agregada por variable original
    imp = {}
    for n, v in zip(nombres, rf.feature_importances_):
        base = n
        for c in CATEGORICAS:
            if n.startswith(c + "_"):
                base = c
        imp[base] = imp.get(base, 0.0) + float(v)

    salida = {
        "clases": clases,
        "features": nombres,
        "j48": exportar_arbol(arbol),
        "rf": [exportar_arbol(e) for e in rf.estimators_],
        "nb": {
            "prior": [round(float(x), 6) for x in nb.class_prior_],
            "theta": [[round(float(x), 5) for x in fila] for fila in nb.theta_],
            "var": [[round(float(x), 6) for x in fila] for fila in nb.var_],
        },
        "importancia": {k: round(v, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])},
        "n_registros": int(len(df)),
        "distribucion": {k: int(v) for k, v in y.value_counts().items()},
    }
    # Muestra de verificacion: 200 filas con predicciones de sklearn
    muestra = X.sample(200, random_state=1)
    verificacion = {
        "filas": json.loads(muestra.to_json(orient="records")),
        "j48": [int(i) for i in modelos["J48 (arbol C4.5)"].predict_proba(muestra).argmax(axis=1)],
        "rf": [[round(float(x), 4) for x in p] for p in modelos["Random Forest"].predict_proba(muestra)],
        "nb": [[round(float(x), 4) for x in p] for p in modelos["Naive Bayes"].predict_proba(muestra)],
    }

    (RAIZ / "docs").mkdir(exist_ok=True)
    (RAIZ / "docs" / "verificacion.json").write_text(json.dumps(verificacion, separators=(",", ":")), encoding="utf-8")
    destino = RAIZ / "docs" / "modelos.json"
    destino.write_text(json.dumps(salida, separators=(",", ":")), encoding="utf-8")
    print(f"{destino}: {destino.stat().st_size/1024:.0f} KB, {len(nombres)} features, clases {clases}")


if __name__ == "__main__":
    main()
