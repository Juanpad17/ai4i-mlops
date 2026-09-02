import pandas as pd

from src.data_quality.validate import load_params
from src.data_quality.quality_gates import (
    run_quality_gates,
    validate_and_gate,
)

#Para ejecutar este script, asegúrate de tener el dataset original en data/raw/ai4i2020.csv
#Mueve el dataset original a esa ruta si no lo tienes allí.
#Muevete el script a la raíz del proyecto y ejecuta: python prueba_quality_gate.py
# Abre el power shell y ejecuta: .\.venv\Scripts\Activate.ps1 para activar el entorno virtual con las dependencias del proyecto antes de ejecutar el script.
#Desde la raíz del proyecto, también puedes ejecutar: python -m src.data_quality.prueba_quality_gate


# Cargar el dataset original en memoria
df = pd.read_csv("data/raw/ai4i2020.csv")

# Crear una copia contaminada / cada linea comentada representa una regla de calidad que se puede probar
df_prueba = df.copy()
df_prueba.loc[0, "Rotational speed [rpm]"] = -500 #columnas con negativos

#df_prueba.loc[0, "Type"] = "X" #tipos_maquina_validos 

#df_prueba.loc[0, "Air temperature [K]"] = 500 #temperatura_aire_en_rango

#df_prueba.loc[0, "Air temperature [K]"] = 350 #proceso_mayor_igual_ambiente
#df_prueba.loc[0, "Process temperature [K]"] = 300 #proceso_mayor_igual_ambiente

#df_prueba.loc[0, "Machine failure"] = 2 #target_valores_validos

#df_prueba = df_prueba.drop(columns=["Torque [Nm]"]) #estructura_completa

#df_prueba = pd.concat(
#    [df_prueba] * 2,
#    ignore_index=True
#) #duplicados_bajo_control


# Cargar las reglas desde params.yaml
params = load_params()

print("=== 1. EJECUTANDO LAS REGLAS ===")

reporte = run_quality_gates(
    df_prueba,
    params,
    nombre_batch="prueba_valor_negativo",
)

print("Resultado general:", reporte["paso_general"])

print("\nReglas fallidas:")
for regla in reporte["reglas"]:
    if not regla["paso"]:
        print(f"- {regla['regla']}: {regla['detalle']}")

print("\n=== 2. VERIFICANDO EL BLOQUEO ===")

try:
    validate_and_gate(
        df_prueba,
        params,
        nombre_batch="prueba_valor_negativo_bloqueo",
    )
    print("ERROR: el dataset no fue bloqueado")
except ValueError as error:
    print("BLOQUEO CORRECTO")
    print(error)