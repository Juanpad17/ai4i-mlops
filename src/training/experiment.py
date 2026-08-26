# ============================================================
# EXPERIMENTACIÓN Y COMPARACIÓN DE MODELOS DE ANOMALÍAS
# Proyecto: AI4I Predictive Maintenance - MLOps
# ============================================================

# Importamos librerías estándar de Python.
import sys
from pathlib import Path

# Detecta automáticamente la raíz del proyecto y la inyecta en Python
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# Path permite construir rutas relativas al proyecto.
from pathlib import Path

# Pandas se utilizará para cargar los datos y crear
# una tabla resumen con los resultados de los experimentos.
import pandas as pd

# NumPy se utilizará en algunas operaciones numéricas.
import numpy as np

# MLflow permitirá registrar cada experimento de forma
# independiente para posteriormente compararlos.
import mlflow
import mlflow.sklearn

# Importamos los algoritmos de detección de anomalías
# que vamos a comparar.
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Pipeline permitirá unir:
#
# Feature Engineering
#       +
# Modelo de anomalías
#
# en un único objeto reproducible.
from sklearn.pipeline import Pipeline

# train_test_split permite separar los datos en subconjuntos.
from sklearn.model_selection import train_test_split

# Métricas utilizadas para evaluar los modelos.
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# Reutilizamos el Feature Engineering creado anteriormente.
#
# Esto evita duplicar lógica entre entrenamiento,
# experimentación y posteriormente producción.
from src.features.build_features import (
    build_features,
    normalize_column_names,
    split_features_target,
    build_preprocessor,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# Obtenemos la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ruta del dataset.
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i2020.csv"
)

# Carpeta donde almacenaremos artefactos generados
# durante los experimentos.
ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
)

# Creamos la carpeta si todavía no existe.
ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURACIÓN REPRODUCIBLE
# ============================================================

# Semilla fija para que las divisiones y modelos
# reproducibles entreguen los mismos resultados.
RANDOM_SEED = 42

# Identificador lógico de nuestra versión de datos.
DATA_VERSION = "ai4i_uci_601_v1"


# ============================================================
# CARGA DEL DATASET
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Carga el dataset AI4I.

    Returns
    -------
    pd.DataFrame
        Dataset listo para separar features y target.
    """

    # Verificamos que la etapa de ingesta haya generado
    # correctamente el archivo raw.
    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró el dataset en: {DATA_PATH}"
        )

    # Cargamos los datos.
    dataframe = pd.read_csv(
        DATA_PATH
    )

    return dataframe


# ============================================================
# DIVISIÓN TRAIN / VALIDATION / TEST
# ============================================================

def split_dataset(X, y):
    """
    Divide los datos en:

        60 % entrenamiento
        20 % validación
        20 % test

    La validación será utilizada para seleccionar el mejor
    modelo.

    El test se conservará para la evaluación final posterior.
    """

    # --------------------------------------------------------
    # PRIMERA DIVISIÓN
    # --------------------------------------------------------
    #
    # 80 % queda temporalmente para entrenamiento/validación.
    # 20 % se reserva como test final.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # --------------------------------------------------------
    # SEGUNDA DIVISIÓN
    # --------------------------------------------------------
    #
    # Del 80 % anterior tomamos:
    #
    # 75 % → entrenamiento
    # 25 % → validación
    #
    # Eso equivale finalmente a:
    #
    # 60 % train
    # 20 % validation
    # 20 % test
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# DEFINICIÓN DE EXPERIMENTOS
# ============================================================

def get_experiments():
    """
    Define los modelos y configuraciones que serán evaluados.

    Cada diccionario representa un experimento diferente
    dentro de MLflow.
    """

    experiments = [

        # ----------------------------------------------------
        # ISOLATION FOREST
        # ----------------------------------------------------

        {
            "run_name": "isolation_forest_100",
            "algorithm": "IsolationForest",

            "model": IsolationForest(
                n_estimators=100,
                contamination=0.035,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),

            "parameters": {
                "n_estimators": 100,
                "contamination": 0.035,
            },
        },

        {
            "run_name": "isolation_forest_200",
            "algorithm": "IsolationForest",

            "model": IsolationForest(
                n_estimators=200,
                contamination=0.035,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),

            "parameters": {
                "n_estimators": 200,
                "contamination": 0.035,
            },
        },

        {
            "run_name": "isolation_forest_300",
            "algorithm": "IsolationForest",

            "model": IsolationForest(
                n_estimators=300,
                contamination=0.035,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),

            "parameters": {
                "n_estimators": 300,
                "contamination": 0.035,
            },
        },


        # ----------------------------------------------------
        # LOCAL OUTLIER FACTOR
        # ----------------------------------------------------
        #
        # novelty=True es importante.
        #
        # Por defecto LOF está diseñado para detectar anomalías
        # solamente sobre los datos con los que se entrenó.
        #
        # Activando novelty=True podemos realizar predicciones
        # sobre datos nuevos, algo necesario para una API.
        {
            "run_name": "lof_neighbors_20",
            "algorithm": "LocalOutlierFactor",

            "model": LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.035,
                novelty=True,
                n_jobs=-1,
            ),

            "parameters": {
                "n_neighbors": 20,
                "contamination": 0.035,
            },
        },

        {
            "run_name": "lof_neighbors_35",
            "algorithm": "LocalOutlierFactor",

            "model": LocalOutlierFactor(
                n_neighbors=35,
                contamination=0.035,
                novelty=True,
                n_jobs=-1,
            ),

            "parameters": {
                "n_neighbors": 35,
                "contamination": 0.035,
            },
        },


        # ----------------------------------------------------
        # ONE-CLASS SVM
        # ----------------------------------------------------

        {
            "run_name": "one_class_svm_nu_0035",
            "algorithm": "OneClassSVM",

            "model": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=0.035,
            ),

            "parameters": {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": 0.035,
            },
        },

        {
            "run_name": "one_class_svm_nu_005",
            "algorithm": "OneClassSVM",

            "model": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=0.05,
            ),

            "parameters": {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": 0.05,
            },
        },
    ]

    return experiments


# ============================================================
# CONSTRUIR PIPELINE
# ============================================================

def build_pipeline(anomaly_model):
    """
    Construye el pipeline completo:

    datos
      ↓
    preprocesamiento
      ↓
    detector de anomalías
    """

    # Creamos un preprocesador nuevo para cada experimento.
    #
    # Es importante que cada modelo aprenda sus transformaciones
    # exclusivamente con los datos de entrenamiento.
    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                anomaly_model
            ),
        ]
    )

    return pipeline


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation
):
    """
    Evalúa un detector de anomalías sobre validation.

    Los algoritmos utilizados devuelven:

        1  = normal
        -1 = anomalía

    Mientras nuestro dataset utiliza:

        0 = normal
        1 = falla

    Por eso debemos convertir las predicciones.
    """

    # Predicción original del algoritmo.
    raw_predictions = model.predict(
        X_validation
    )

    # Convertimos:
    #
    # -1 → 1 (anomalía / falla)
    #  1 → 0 (normal)
    y_pred = (
        raw_predictions == -1
    ).astype(int)


    # --------------------------------------------------------
    # ANOMALY SCORE
    # --------------------------------------------------------

    # decision_function entrega valores más altos para
    # observaciones consideradas normales.
    #
    # Multiplicamos por -1 para que:
    #
    # score alto = más anomalía.
    anomaly_scores = -model.decision_function(
        X_validation
    )


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    precision = precision_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_validation,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        y_validation,
        anomaly_scores
    )


    # Retornamos todas las métricas en un diccionario.
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return metrics


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """
    Ejecuta todos los experimentos y los registra en MLflow.
    """

    print("\n==========================================")
    print("EXPERIMENTACIÓN DE MODELOS")
    print("==========================================\n")


    # --------------------------------------------------------
    # 1. CARGAR DATOS
    # --------------------------------------------------------

    dataframe = load_data()

    # Utilizamos la misma selección de features definida
    # anteriormente.
    X, y = split_features_target(
        dataframe
    )

    # Calculamos las features adicionales definidas en Feature Engineering.    
    X = build_features(X)

    # --------------------------------------------------------
    # 2. TRAIN / VALIDATION / TEST
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(
        X,
        y
    )

    print(
        f"Train:      {len(X_train)} registros"
    )

    print(
        f"Validation: {len(X_validation)} registros"
    )

    print(
        f"Test:       {len(X_test)} registros"
    )


    # --------------------------------------------------------
    # 3. ENTRENAMIENTO SOLO CON DATOS NORMALES
    # --------------------------------------------------------

    # Los detectores aprenderán el comportamiento considerado
    # normal de la maquinaria.
    normal_mask = y_train == 0

    X_train_normal = X_train.loc[
        normal_mask
    ]

    print(
        "\nRegistros normales utilizados para entrenamiento:"
    )

    print(
        len(X_train_normal)
    )


    # --------------------------------------------------------
    # 4. CONFIGURAR MLFLOW
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    mlflow.set_experiment(
        "AI4I_Anomaly_Detection_Comparison"
    )


    # --------------------------------------------------------
    # 5. OBTENER EXPERIMENTOS
    # --------------------------------------------------------

    experiments = get_experiments()

    # Aquí almacenaremos también los resultados localmente
    # para construir una tabla comparativa.
    experiment_results = []


    # ========================================================
    # 6. EJECUTAR CADA EXPERIMENTO
    # ========================================================

    for experiment in experiments:

        run_name = experiment[
            "run_name"
        ]

        algorithm = experiment[
            "algorithm"
        ]

        anomaly_model = experiment[
            "model"
        ]

        parameters = experiment[
            "parameters"
        ]


        print("\n------------------------------------------")
        print(f"Ejecutando: {run_name}")
        print("------------------------------------------")


        # Construimos un pipeline independiente.
        pipeline = build_pipeline(
            anomaly_model
        )


        # ----------------------------------------------------
        # INICIAR RUN
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=run_name
        ):

            # ------------------------------------------------
            # PARÁMETROS GENERALES
            # ------------------------------------------------

            mlflow.log_param(
                "algorithm",
                algorithm
            )

            mlflow.log_param(
                "random_seed",
                RANDOM_SEED
            )

            mlflow.log_param(
                "data_version",
                DATA_VERSION
            )

            mlflow.log_param(
                "feature_set",
                ",".join(X.columns)
            )

            mlflow.log_param(
                "training_strategy",
                "normal_samples_only"
            )


            # ------------------------------------------------
            # PARÁMETROS DEL MODELO
            # ------------------------------------------------

            for parameter_name, parameter_value in parameters.items():

                mlflow.log_param(
                    parameter_name,
                    parameter_value
                )


            # ------------------------------------------------
            # ENTRENAMIENTO
            # ------------------------------------------------

            pipeline.fit(
                X_train_normal
            )


            # ------------------------------------------------
            # EVALUACIÓN EN VALIDATION
            # ------------------------------------------------

            metrics = evaluate_model(
                pipeline,
                X_validation,
                y_validation
            )


            # Registramos cada métrica.
            for metric_name, metric_value in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value
                )


            # ------------------------------------------------
            # GUARDAR MODELO EN MLFLOW
            # ------------------------------------------------

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                skops_trusted_types=[
                    "sklearn.metrics._dist_metrics.EuclideanDistance64",
                    "sklearn.neighbors._kd_tree.KDTree",
                ],
            )


            # ------------------------------------------------
            # GUARDAR RESULTADOS PARA COMPARACIÓN
            # ------------------------------------------------

            result = {
                "run_name": run_name,
                "algorithm": algorithm,
                **metrics,
            }

            experiment_results.append(
                result
            )


            # ------------------------------------------------
            # MOSTRAR MÉTRICAS
            # ------------------------------------------------

            print(
                f"Precision : {metrics['precision']:.4f}"
            )

            print(
                f"Recall    : {metrics['recall']:.4f}"
            )

            print(
                f"F1-score  : {metrics['f1_score']:.4f}"
            )

            print(
                f"ROC-AUC   : {metrics['roc_auc']:.4f}"
            )

            print(
                f"PR-AUC    : {metrics['pr_auc']:.4f}"
            )


    # ========================================================
    # 7. CREAR TABLA COMPARATIVA
    # ========================================================

    results_dataframe = pd.DataFrame(
        experiment_results
    )

    # Ordenamos inicialmente por PR-AUC.
    #
    # Para problemas desbalanceados esta métrica suele ser
    # más informativa que Accuracy.
    results_dataframe = results_dataframe.sort_values(
        by="pr_auc",
        ascending=False
    )


    print("\n==========================================")
    print("COMPARACIÓN DE MODELOS")
    print("==========================================\n")

    print(
        results_dataframe.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # 8. GUARDAR TABLA
    # --------------------------------------------------------

    results_path = (
        ARTIFACTS_DIR
        / "model_comparison.csv"
    )

    results_dataframe.to_csv(
        results_path,
        index=False
    )


    print(
        "\nTabla comparativa guardada en:"
    )

    print(
        results_path
    )


    # --------------------------------------------------------
    # 9. MOSTRAR MEJOR CANDIDATO
    # --------------------------------------------------------

    best_model = results_dataframe.iloc[0]

    print("\n==========================================")
    print("MEJOR CANDIDATO SEGÚN PR-AUC")
    print("==========================================")

    print(
        f"Run: {best_model['run_name']}"
    )

    print(
        f"Algoritmo: {best_model['algorithm']}"
    )

    print(
        f"PR-AUC: {best_model['pr_auc']:.4f}"
    )

    print(
        f"Recall: {best_model['recall']:.4f}"
    )

    print(
        f"Precision: {best_model['precision']:.4f}"
    )

    print(
        f"F1: {best_model['f1_score']:.4f}"
    )


    print("\n==========================================")
    print("EXPERIMENTACIÓN FINALIZADA")
    print("==========================================")

    print(
        "\nEl conjunto TEST todavía no se utilizó "
        "para seleccionar el modelo."
    )

    # ============================================================
    # 10. MODEL REGISTRY (ciclo: Experiment -> Candidate -> Validation -> Production)
    # ============================================================
    #
    # Agregar este bloque al final de main(), después de imprimir
    # "EXPERIMENTACIÓN FINALIZADA". Requiere: from mlflow import MlflowClient
    # (o mlflow.tracking.MlflowClient según tu versión de MLflow).

    from mlflow import MlflowClient

    REGISTERED_MODEL_NAME = "ai4i_anomaly_detector"

    client = MlflowClient()

    # --------------------------------------------------------
    # 10.1 IDENTIFICAR LA RUN DEL MEJOR CANDIDATO
    # --------------------------------------------------------
    # best_model es la fila (Series) del results_dataframe con el
    # run_name ganador. Necesitamos su run_id real de MLflow para
    # poder registrar el modelo logueado en esa run.

    runs = mlflow.search_runs(
        experiment_names=["AI4I_Anomaly_Detection_Comparison"],
        filter_string=f"tags.mlflow.runName = '{best_model['run_name']}'",
    )

    best_run_id = runs.iloc[0]["run_id"]

    print("\n==========================================")
    print("MODEL REGISTRY")
    print("==========================================")

    print(f"Registrando run: {best_run_id} ({best_model['run_name']})")


    # --------------------------------------------------------
    # 10.2 CANDIDATE: registrar el modelo del mejor run
    # --------------------------------------------------------
    # Esto crea (o versiona, si ya existe) el modelo registrado.
    # Nace como "candidato" -- todavía no está en ninguna etapa oficial.

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    model_version = registered_model.version

    print(f"Modelo registrado como '{REGISTERED_MODEL_NAME}' versión {model_version}")

    # Dejamos explícito el criterio de selección como tag de la versión.
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="selection_criteria",
        value="pr_auc (mejor sobre validation)",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="pr_auc",
        value=f"{best_model['pr_auc']:.4f}",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="stage_custom",
        value="Candidate",
    )


    # --------------------------------------------------------
    # 10.3 VALIDATION: evaluar sobre el conjunto TEST (nunca usado antes)
    # --------------------------------------------------------
    # Este es el paso que tu script deja preparado pero no ejecuta:
    # usar X_test/y_test -- separados desde el inicio y jamás tocados --
    # como confirmación final e independiente del desempeño del candidato.

    loaded_model = mlflow.sklearn.load_model(model_uri)

    test_metrics = evaluate_model(loaded_model, X_test, y_test)

    print("\nMétricas finales sobre TEST (nunca usado para seleccionar el modelo):")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="test_pr_auc",
        value=f"{test_metrics['pr_auc']:.4f}",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        key="stage_custom",
        value="Validation",
    )


    # --------------------------------------------------------
    # 10.4 PRODUCTION: criterio explícito de promoción
    # --------------------------------------------------------
    # Solo promovemos a Production si el desempeño en TEST confirma
    # lo visto en validation (criterio explícito, ej. PR-AUC >= umbral).

    PRODUCTION_PR_AUC_THRESHOLD = 0.75  # ajustar según tu caso

    if test_metrics["pr_auc"] >= PRODUCTION_PR_AUC_THRESHOLD:

        # Nota: transition_model_version_stage está deprecado en MLflow
        # recientes a favor de alias. Usá el que corresponda a tu versión:

        # --- Opción A (MLflow >= 2.9, API de alias, recomendada) ---
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="production",
            version=model_version,
        )

        # --- Opción B (API de stages clásica, si tu versión aún la soporta) ---
        # client.transition_model_version_stage(
        #     name=REGISTERED_MODEL_NAME,
        #     version=model_version,
        #     stage="Production",
        # )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=model_version,
            key="stage_custom",
            value="Production",
        )

        print(f"\nModelo promovido a PRODUCTION (test_pr_auc={test_metrics['pr_auc']:.4f} >= {PRODUCTION_PR_AUC_THRESHOLD})")

    else:
        print(
            f"\nModelo NO promovido a producción "
            f"(test_pr_auc={test_metrics['pr_auc']:.4f} < {PRODUCTION_PR_AUC_THRESHOLD})"
        )


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()

