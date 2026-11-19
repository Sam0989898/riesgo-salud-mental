# Modelo predictivo de riesgo de salud mental (prototipo académico)

Proyecto de aula de **Minería de Datos** — Fundación Universitaria Tecnológico Comfenalco, Cartagena de Indias.
Autor: Samuel Santiago Jiménez Acuña. Docente: Diana Paola Beetar Baraja.

Prototipo que clasifica a adultos jóvenes en **Sin riesgo / Moderado / Alto** a partir de 13 variables de hábitos y contexto psicosocial.

> **Aviso:** los datos son **sintéticos** (generados por script, sin personas reales) y el modelo es un ejercicio académico. No diagnostica ni reemplaza a un profesional de la salud mental.

## Estructura

```
riesgo-salud-mental/
├── data/dataset_sintetico.csv     # 1.200 registros simulados
├── src/generar_datos.py           # genera el dataset (semilla 42)
├── src/entrenar_modelos.py        # J48 (C4.5), Random Forest, Naive Bayes
├── results/metricas.csv           # exactitud, precisión, exhaustividad, F1
├── results/matrices_confusion.txt
└── requirements.txt
```

## Cómo ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generar_datos.py
python src/entrenar_modelos.py
```

## Metodología

CRISP-DM (6 fases), validación cruzada estratificada de 10 pliegues. Detalle completo en el Informe Metodológico.

## Resultados (datos sintéticos)

| Modelo | Exactitud | Precisión | Exhaustividad | F1 |
| --- | --- | --- | --- | --- |
| J48 (árbol C4.5) | 0,642 | 0,651 | 0,642 | 0,646 |
| Random Forest | 0,744 | 0,747 | 0,744 | 0,738 |
| Naive Bayes | 0,707 | 0,705 | 0,707 | 0,703 |

## Consideraciones éticas

Con datos reales aplicaría la Ley 1581 de 2012 (protección de datos personales), con autorización informada y tratamiento especial de datos sensibles.
