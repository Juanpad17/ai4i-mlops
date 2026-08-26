# ============================================================
# ENTRENAMIENTO DEL MODELO DE DETECCIÓN DE ANOMALÍAS
# Proyecto: AI4I Predictive Maintenance - MLOps
# ============================================================

# Path permite manejar rutas del proyecto de forma portable.
from pathlib import Path

# Librerías estándar para guardar algunos resultados.
import json

# Pandas se utiliza para cargar y manipular los datos.
import pandas as pd

# Matplotlib permitirá guardar la matriz de confusión
# como artefacto del experimento.
import matplotlib.pyplot as plt

# MLflow será utilizado para registrar:
# - parámetros;
# - métricas;
# - artefactos;
# - modelo entrenado.
import mlflow
import mlflow.sklearn

# train_test_split permite separar datos de entrenamiento
# y prueba.
from sklearn.model_selection import train_test_split

# Isolation Forest es un algoritmo diseñado específicamente
# para detección de anomalías.
from sklearn.ensemble import IsolationForest

# Pipeline permite guardar en un único objeto:
#
# Feature Engineering + Modelo
from sklearn.pipeline import Pipeline

# Métricas utilizadas para evaluar qué tan bien las anomalías
# detectadas corresponden con las fallas reales.
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Reutilizamos EXACTAMENTE el Feature Engineering construido
# anteriormente. Esto evita tener una transformación diferente
# durante entrenamiento y posteriormente en producción.
from src.features.build_features import (
    build_preprocessor,
    normalize_column_names,
    split_features_target,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# Ruta raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dataset generado por nuestra etapa de ingesta.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

# Carpeta donde guardaremos artefactos temporales.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Creamos la carpeta si todavía no existe.
ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================

# Semilla fija para garantizar reproducibilidad.
RANDOM_SEED = 42

# Porcentaje reservado para evaluación.
TEST_SIZE = 0.20

# Parámetros iniciales de Isolation Forest.
N_ESTIMATORS = 200

# AI4I posee una proporción pequeña de fallas.
#
# contamination ayuda al modelo a determinar aproximadamente
# qué proporción de observaciones podría ser anómala.
CONTAMINATION = 0.035

# Versión lógica del dataset.
#
# Más adelante podremos mejorar este mecanismo utilizando hashes
# u otra estrategia de versionamiento.
DATA_VERSION = "ai4i_uci_601_v1"


# ============================================================
# CARGA DE DATOS
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Carga y normaliza el dataset utilizado por entrenamiento.
    """

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró el dataset en: {DATA_PATH}"
        )

    # Cargamos el CSV.
    dataframe = pd.read_csv(DATA_PATH)

    # Aplicamos la misma normalización centralizada
    # definida en Feature Engineering.
    dataframe = normalize_column_names(
        dataframe
    )

    return dataframe


# ============================================================
# CREACIÓN DEL MODELO
# ============================================================

def build_model() -> Pipeline:
    """
    Construye el pipeline completo:

        Datos
          ↓
        Feature Engineering
          ↓
        Isolation Forest

    Returns
    -------
    Pipeline
        Pipeline completo listo para entrenar.
    """

    # Construimos el preprocesador reutilizable.
    preprocessor = build_preprocessor()

    # Creamos el algoritmo de detección de anomalías.
    anomaly_detector = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # Unimos preprocesamiento y modelo.
    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                anomaly_detector
            ),
        ]
    )

    return model_pipeline


# ============================================================
# GUARDAR MATRIZ DE CONFUSIÓN
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred
) -> Path:
    """
    Genera y guarda una matriz de confusión.

    Este gráfico será registrado posteriormente como
    artefacto dentro de MLflow.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    # Mostramos la matriz como imagen.
    axis.imshow(matrix)

    axis.set_title(
        "Matriz de confusión - Isolation Forest"
    )

    axis.set_xlabel(
        "Predicción"
    )

    axis.set_ylabel(
        "Valor real"
    )

    axis.set_xticks(
        [0, 1]
    )

    axis.set_yticks(
        [0, 1]
    )

    axis.set_xticklabels(
        ["Normal", "Falla"]
    )

    axis.set_yticklabels(
        ["Normal", "Falla"]
    )

    # Añadimos los valores dentro de cada celda.
    for i in range(2):

        for j in range(2):

            axis.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    figure.tight_layout()

    output_path = (
        ARTIFACTS_DIR
        / "confusion_matrix.png"
    )

    figure.savefig(
        output_path
    )

    plt.close(figure)

    return output_path


# ============================================================
# ENTRENAMIENTO
# ============================================================

def main() -> None:
    """
    Ejecuta entrenamiento, evaluación y tracking con MLflow.
    """

    print("\n==========================================")
    print("ENTRENAMIENTO - ISOLATION FOREST")
    print("==========================================\n")

    # --------------------------------------------------------
    # 1. CARGAR DATOS
    # --------------------------------------------------------

    dataframe = load_data()

    # Separamos las features y el target utilizando la función
    # creada anteriormente.
    X, y = split_features_target(
        dataframe
    )

    print(f"Registros disponibles: {len(X)}")


    # --------------------------------------------------------
    # 2. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    # Utilizamos stratify para conservar aproximadamente
    # la misma proporción de fallas en train y test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(
        f"Registros entrenamiento: {len(X_train)}"
    )

    print(
        f"Registros prueba: {len(X_test)}"
    )


    # --------------------------------------------------------
    # 3. USAR SOLO COMPORTAMIENTO NORMAL PARA ENTRENAR
    # --------------------------------------------------------

    # Este punto es fundamental.
    #
    # Isolation Forest es utilizado como detector de anomalías.
    # Para establecer el patrón de comportamiento normal,
    # entrenaremos únicamente utilizando registros cuya
    # etiqueta Machine failure sea 0.
    #
    # Las etiquetas NO son entregadas al algoritmo.
    normal_mask = y_train == 0

    X_train_normal = X_train.loc[
        normal_mask
    ]

    print(
        "Registros normales utilizados para entrenar: "
        f"{len(X_train_normal)}"
    )


    # --------------------------------------------------------
    # 4. CONSTRUIR PIPELINE
    # --------------------------------------------------------

    model = build_model()


    # --------------------------------------------------------
    # 5. CONFIGURAR MLFLOW
    # --------------------------------------------------------

    # El servidor MLflow debe estar ejecutándose previamente:
    #
    # mlflow server --port 5000
    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    # Creamos o seleccionamos nuestro experimento.
    mlflow.set_experiment(
        "AI4I_Anomaly_Detection"
    )


    # --------------------------------------------------------
    # 6. INICIAR RUN DE MLFLOW
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="isolation_forest_baseline"
    ):

        # ----------------------------------------------------
        # REGISTRAR PARÁMETROS
        # ----------------------------------------------------

        mlflow.log_param(
            "algorithm",
            "IsolationForest"
        )

        mlflow.log_param(
            "n_estimators",
            N_ESTIMATORS
        )

        mlflow.log_param(
            "contamination",
            CONTAMINATION
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE
        )

        mlflow.log_param(
            "data_version",
            DATA_VERSION
        )

        mlflow.log_param(
            "feature_set",
            ",".join(X.columns)
        )


        # ----------------------------------------------------
        # 7. ENTRENAR
        # ----------------------------------------------------

        print("\nEntrenando Isolation Forest...")

        model.fit(
            X_train_normal
        )

        print("Entrenamiento completado.")


        # ----------------------------------------------------
        # 8. GENERAR PREDICCIONES
        # ----------------------------------------------------

        # Isolation Forest devuelve:
        #
        #  1  → observación normal
        # -1  → anomalía
        raw_predictions = model.predict(
            X_test
        )

        # Nuestro target utiliza:
        #
        # 0 → normal
        # 1 → falla
        #
        # Por eso convertimos la salida.
        y_pred = (
            raw_predictions == -1
        ).astype(int)


        # ----------------------------------------------------
        # 9. CALCULAR ANOMALY SCORE
        # ----------------------------------------------------

        # decision_function produce valores mayores para
        # observaciones consideradas normales.
        #
        # Multiplicamos por -1 para tener:
        #
        # mayor score = más anómalo.
        anomaly_scores = -model.decision_function(
            X_test
        )


        # ----------------------------------------------------
        # 10. MÉTRICAS
        # ----------------------------------------------------

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            anomaly_scores
        )

        pr_auc = average_precision_score(
            y_test,
            anomaly_scores
        )


        # ----------------------------------------------------
        # 11. REGISTRAR MÉTRICAS EN MLFLOW
        # ----------------------------------------------------

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.log_metric(
            "pr_auc",
            pr_auc
        )


        # ----------------------------------------------------
        # 12. MOSTRAR RESULTADOS
        # ----------------------------------------------------

        print("\n==========================================")
        print("RESULTADOS")
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
        # 13. MATRIZ DE CONFUSIÓN
        # ----------------------------------------------------

        confusion_path = save_confusion_matrix(
            y_test,
            y_pred
        )

        # Registramos la imagen como artefacto.
        mlflow.log_artifact(
            str(confusion_path),
            artifact_path="evaluation"
        )


        # ----------------------------------------------------
        # 14. CLASSIFICATION REPORT
        # ----------------------------------------------------

        # Guardamos también Precision, Recall y F1 detallados
        # por clase.
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_path = (
            ARTIFACTS_DIR
            / "classification_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        mlflow.log_artifact(
            str(report_path),
            artifact_path="evaluation"
        )


        # ----------------------------------------------------
        # 15. GUARDAR EL PIPELINE COMPLETO EN MLFLOW
        # ----------------------------------------------------

        # Registramos:
        #
        # preprocessing + modelo
        #
        # como un solo artefacto reproducible.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )


        print(
            "\nExperimento registrado correctamente "
            "en MLflow."
        )


    print("\n==========================================")
    print("ENTRENAMIENTO FINALIZADO")
    print("==========================================")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()