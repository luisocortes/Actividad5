# Actividad 5 - Gestión de proyectos de IA

## Objetivo
Comparar dos modelos de clasificación mediante GridSearchCV
y registrar sus resultados utilizando MLflow.

## Dataset
Breast Cancer Wisconsin, disponible en Scikit-learn.
569 registros, 30 características y una variable objetivo.

## Modelos
- Regresión logística
- Random Forest

## Metodología
1. Limpieza de datos.
2. División de entrenamiento y prueba (80/20).
3. Validación cruzada de 5 particiones.
4. Optimización mediante GridSearchCV.
5. Evaluación y seguimiento con MLflow.

## Reproducción
Instalar pandas, scikit-learn y mlflow.

Desde la carpeta Actividad5 ejecutar:
python fuentes/train.py

Los resultados se registran en la carpeta mlruns.
