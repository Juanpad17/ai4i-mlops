from pathlib import Path
import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow  # <-- Nueva importación

# Configuración de Logs 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Valores por defecto configurables
DEFAULT_INPUT = "data/processed/validated.csv"
DEFAULT_TRAIN_OUT = "data/processed/train.csv"
DEFAULT_TEST_OUT = "data/processed/test.csv"
RANDOM_SEED = 42
TARGET_COLUMN = "Machine failure"
TEST_SIZE = 0.2

def split_stratified(df: pd.DataFrame, test_size: float, seed: int):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    return train_df, test_df

def split_normal_only(df: pd.DataFrame, test_size: float, seed: int):
    # Aplicando la corrección matemática para evitar alterar el tamaño del test_size
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    train_df = train_df[train_df[TARGET_COLUMN] == 0]
    return train_df, test_df

def split_data(
    input_path: str,
    train_out: str,
    test_out: str,
    strategy: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"No se encontró el dataset validado en {input_path}.")

    df = pd.read_csv(in_p)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COLUMN}' no está en el dataset.")

    # 1. Ejecutar la estrategia de split
    if strategy == "stratified":
        train_df, test_df = split_stratified(df, test_size, seed)
    elif strategy == "normal_only":
        train_df, test_df = split_normal_only(df, test_size, seed)
    else:
        raise ValueError(f"Estrategia '{strategy}' no reconocida.")

    # Guardar archivos físicamente
    for path_str, frame in [(train_out, train_df), (test_out, test_df)]:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False)

    # 2. INTEGRACIÓN CON MLFLOW
    # mlflow.start_run() asume que ya definiste el experimento antes en la consola o código
    with mlflow.start_run(run_name=f"data_split_{strategy}"):
        
        # Registrar parámetros (Configuración del experimento)
        mlflow.log_param("split_strategy", strategy)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("target_column", TARGET_COLUMN)
        
        # Registrar métricas del dataset (Dimensiones y proporciones)
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("test_rows", len(test_df))
        mlflow.log_metric("train_failure_rate", float(train_df[TARGET_COLUMN].mean()))
        mlflow.log_metric("test_failure_rate", float(test_df[TARGET_COLUMN].mean()))
        
        # Opcional: Registrar la firma o metadatos de los datos (MLflow Datasets API)
        # mlflow.data.log_inputs si estás usando versiones más recientes de MLflow

    logging.info(f"Split completado exitosamente con la estrategia: {strategy}")
    return train_df, test_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split reproducible con MLflow para AI4I 2020")
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--train-out", type=str, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--test-out", type=str, default=DEFAULT_TEST_OUT)
    parser.add_argument("--strategy", type=str, default="stratified", choices=["stratified", "normal_only"])
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Configuramos el nombre del experimento en MLflow (cámbialo si lo deseas)
    mlflow.set_experiment("AI4I_Predictive_Maintenance_Data_Prep")
    
    split_data(
        input_path=args.input_path,
        train_out=args.train_out,
        test_out=args.test_out,
        strategy=args.strategy,
        test_size=args.test_size,
        seed=args.seed
    )
