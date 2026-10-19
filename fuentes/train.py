
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Cargar datos limpios
ruta = os.path.join("datos", "datos_limp", "dataset_limpio.csv")
df = pd.read_csv(ruta)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definir modelos y parámetros
modelos = {
    "Regresión logística": (
        Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", LogisticRegression(max_iter=2000))
        ]),
        {"modelo__C": [0.1, 1, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
    )
}

# Configurar MLflow
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("Actividad5_Clasificacion")

# Entrenar y registrar
for nombre, (modelo, parametros) in modelos.items():
    busqueda = GridSearchCV(
        modelo, parametros, cv=5, scoring="f1", n_jobs=-1
    )
    busqueda.fit(X_train, y_train)

    mejor = busqueda.best_estimator_
    pred = mejor.predict(X_test)
    prob = mejor.predict_proba(X_test)[:, 1]

    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(busqueda.best_params_)
        mlflow.log_metric("Accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("F1", f1_score(y_test, pred))
        mlflow.log_metric("ROC-AUC", roc_auc_score(y_test, prob))
        mlflow.log_metric("F1_CV", busqueda.best_score_)
        mlflow.sklearn.log_model(mejor, name="modelo", serialization_format="cloudpickle")

    print(nombre, "registrado correctamente.")
