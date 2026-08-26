import sys
from pathlib import Path

# Detecta automáticamente la raíz del proyecto y la inyecta en Python
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# ============================================================
# EVALUACIÓN FINAL Y REGISTRO DEL MODELO SELECCIONADO
# Proyecto: AI4I Predictive Maintenance - MLOps
# ============================================================
import json

import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Reutilizamos exactamente el Feature Engineering definido
# previamente para garantizar consistencia entre entrenamiento
# y posterior producción.
from src.features.build_features import (
    build_preprocessor,
    split_features_target,
    build_features,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i2020.csv"
)

ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "final_model"
)

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Semilla utilizada durante todo el proyecto.
RANDOM_SEED = 42

# Versión lógica del dataset.
DATA_VERSION = "ai4i_uci_601_v1"

# Nombre que tendrá el modelo dentro de MLflow Model Registry.
REGISTERED_MODEL_NAME = "AI4I_LOF_Anomaly_Detector"


# ============================================================
# CARGA DE DATOS
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Carga el dataset y normaliza los nombres de columnas.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    return dataframe


# ============================================================
# DIVISIÓN FINAL DE DATOS
# ============================================================

def split_final_dataset(X, y):
    """
    Reconstruye exactamente la misma separación utilizada
    durante la experimentación.

    El 20 % reservado originalmente como TEST permanece como
    conjunto independiente para la evaluación final.
    """

    # Esta es exactamente la primera división utilizada
    # durante experiment.py.
    X_train_validation, X_test, y_train_validation, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=y,
        )
    )

    return (
        X_train_validation,
        X_test,
        y_train_validation,
        y_test,
    )


# ============================================================
# CONSTRUCCIÓN DEL MODELO GANADOR
# ============================================================

def build_final_model() -> Pipeline:
    """
    Construye el pipeline definitivo utilizando el modelo
    seleccionado mediante el conjunto de validación.
    """

    # Feature Engineering reutilizable.
    preprocessor = build_preprocessor()

    # Configuración ganadora obtenida durante experimentación.
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.035,

        # novelty=True permite realizar predicciones sobre
        # registros que no formaron parte del entrenamiento.
        # Esto será fundamental posteriormente para la API.
        novelty=True,

        n_jobs=-1,
    )

    # Guardamos preprocesamiento y modelo juntos.
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                lof,
            ),
        ]
    )

    return pipeline


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def save_confusion_matrix(y_true, y_pred) -> Path:
    """
    Genera la matriz de confusión correspondiente al TEST.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    axis.imshow(matrix)

    axis.set_title(
        "Matriz de confusión - LOF - TEST"
    )

    axis.set_xlabel("Predicción")
    axis.set_ylabel("Valor real")

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    axis.set_xticklabels(
        ["Normal", "Falla"]
    )

    axis.set_yticklabels(
        ["Normal", "Falla"]
    )

    for i in range(2):
        for j in range(2):
            axis.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    figure.tight_layout()

    output_path = (
        ARTIFACTS_DIR
        / "test_confusion_matrix.png"
    )

    figure.savefig(
        output_path
    )

    plt.close(figure)

    return output_path


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main() -> None:
    """
    Reentrena el modelo seleccionado, realiza la evaluación
    final sobre TEST y lo registra en MLflow Model Registry.
    """

    print("\n==========================================")
    print("MODELO FINAL - LOCAL OUTLIER FACTOR")
    print("==========================================\n")

    # --------------------------------------------------------
    # 1. CARGAR DATASET
    # --------------------------------------------------------

    dataframe = load_data()

    X, y = split_features_target(
        dataframe
    )

    # Aplicamos exactamente el mismo Feature Engineering
    # que se utilizó durante experimentación.
    X = build_features(X)


    # --------------------------------------------------------
    # 2. RECUPERAR TEST FINAL
    # --------------------------------------------------------

    (
        X_train_validation,
        X_test,
        y_train_validation,
        y_test,
    ) = split_final_dataset(
        X,
        y
    )

    print(
        f"Train + Validation: {len(X_train_validation)}"
    )

    print(
        f"Test final:         {len(X_test)}"
    )


    # --------------------------------------------------------
    # 3. ENTRENAR SOLAMENTE CON REGISTROS NORMALES
    # --------------------------------------------------------

    normal_mask = (
        y_train_validation == 0
    )

    X_normal = X_train_validation.loc[
        normal_mask
    ]

    print(
        "\nRegistros normales utilizados "
        "para entrenamiento final:"
    )

    print(
        len(X_normal)
    )


    # --------------------------------------------------------
    # 4. CONSTRUIR MODELO GANADOR
    # --------------------------------------------------------

    model = build_final_model()


    # --------------------------------------------------------
    # 5. CONFIGURAR MLFLOW
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    mlflow.set_experiment(
        "AI4I_Final_Model"
    )


    # --------------------------------------------------------
    # 6. INICIAR RUN FINAL
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="lof_neighbors_20_final"
    ):

        # Documentamos por qué este modelo fue seleccionado.
        mlflow.set_tag(
            "selection_reason",
            "Best validation PR-AUC and F1-score"
        )

        mlflow.set_tag(
            "model_status",
            "final_candidate"
        )


        # ----------------------------------------------------
        # 7. REGISTRAR PARÁMETROS
        # ----------------------------------------------------

        mlflow.log_param(
            "algorithm",
            "LocalOutlierFactor"
        )

        mlflow.log_param(
            "n_neighbors",
            20
        )

        mlflow.log_param(
            "contamination",
            0.035
        )

        mlflow.log_param(
            "novelty",
            True
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


        # ----------------------------------------------------
        # 8. ENTRENAMIENTO FINAL
        # ----------------------------------------------------

        print("\nEntrenando modelo final...")

        model.fit(
            X_normal
        )

        print(
            "Entrenamiento completado."
        )


        # ----------------------------------------------------
        # 9. PREDICCIÓN SOBRE TEST
        # ----------------------------------------------------

        raw_predictions = model.predict(
            X_test
        )

        # LOF:
        #
        #  1 = normal
        # -1 = anomalía
        #
        # Dataset:
        #
        # 0 = normal
        # 1 = falla
        y_pred = (
            raw_predictions == -1
        ).astype(int)


        # ----------------------------------------------------
        # 10. ANOMALY SCORE
        # ----------------------------------------------------

        anomaly_scores = (
            -model.decision_function(
                X_test
            )
        )


        # ----------------------------------------------------
        # 11. MÉTRICAS FINALES
        # ----------------------------------------------------

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            anomaly_scores,
        )

        pr_auc = average_precision_score(
            y_test,
            anomaly_scores,
        )


        # ----------------------------------------------------
        # 12. REGISTRAR MÉTRICAS
        # ----------------------------------------------------

        mlflow.log_metric(
            "test_precision",
            precision
        )

        mlflow.log_metric(
            "test_recall",
            recall
        )

        mlflow.log_metric(
            "test_f1_score",
            f1
        )

        mlflow.log_metric(
            "test_roc_auc",
            roc_auc
        )

        mlflow.log_metric(
            "test_pr_auc",
            pr_auc
        )


        # ----------------------------------------------------
        # 13. MOSTRAR RESULTADOS
        # ----------------------------------------------------

        print("\n==========================================")
        print("RESULTADOS FINALES EN TEST")
        print("==========================================")

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1-score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            f"PR-AUC    : {pr_auc:.4f}"
        )


        # ----------------------------------------------------
        # 14. MATRIZ DE CONFUSIÓN
        # ----------------------------------------------------

        confusion_path = save_confusion_matrix(
            y_test,
            y_pred,
        )

        mlflow.log_artifact(
            str(confusion_path),
            artifact_path="evaluation",
        )


        # ----------------------------------------------------
        # 15. CLASSIFICATION REPORT
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        report_path = (
            ARTIFACTS_DIR
            / "test_classification_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        mlflow.log_artifact(
            str(report_path),
            artifact_path="evaluation",
        )


        # ----------------------------------------------------
        # 16. FIRMA DEL MODELO
        # ----------------------------------------------------

        # Guardamos una pequeña muestra para que MLflow conozca
        # la estructura esperada por el modelo.
        input_example = (
            X_test.head(5)
        )

        example_prediction = model.predict(
            input_example
        )

        signature = infer_signature(
            input_example,
            example_prediction,
        )


        # ----------------------------------------------------
        # 17. MODEL REGISTRY
        # ----------------------------------------------------

        # Además de guardar el pipeline como artefacto,
        # solicitamos a MLflow que cree una versión dentro
        # del Model Registry.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,

            # LocalOutlierFactor utiliza internamente estas clases de scikit-learn.
            # MLflow usa "skops" para serializar el modelo de forma segura y,
            # por defecto, estas clases internas no están en su lista de confianza.
            #
            # Como conocemos el origen del modelo y sabemos que estas clases
            # pertenecen a scikit-learn, las declaramos explícitamente como
            # tipos confiables para permitir que MLflow guarde el modelo.
            skops_trusted_types=[
                "sklearn.metrics._dist_metrics.EuclideanDistance64",
                "sklearn.neighbors._kd_tree.KDTree",
            ],

        )


        print(
            "\nModelo registrado correctamente "
            "en MLflow Model Registry."
        )


    print("\n==========================================")
    print("MODELO FINAL REGISTRADO")
    print("==========================================")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()