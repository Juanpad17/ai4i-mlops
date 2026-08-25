# %%
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# VARIABLES DE ENTRADA SELECCIONADAS (DATASET REAL AI4I 2020)
# -----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Type",                   # Categoría de calidad del producto (L: Low, M: Medium, H: High)
    "Air temperature [K]",    # Temperatura ambiente generada por el entorno de la fábrica
    "Process temperature [K]",# Temperatura generada por la fricción/operación de la máquina
    "Rotational speed [rpm]", # Velocidad de rotación del husillo o herramienta
    "Torque [Nm]",            # Fuerza de torsión aplicada por el motor
    "Tool wear [min]",        # Minutos acumulados de desgaste de la herramienta actual
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de Feature Engineering para Mantenimiento Predictivo (AI4I 2020).
    
    Transforma las variables físicas crudas en indicadores de esfuerzo mecánico 
    y térmico. Esta función se ejecuta de forma idéntica en:
      1. El entrenamiento (Offline): Aplicado sobre data/processed/train.csv.
      2. La inferencia (Online/Producción): Aplicado sobre payloads JSON/API.
      
    Objetivo MLOps: Garantizar la paridad de los datos y evitar el 
    'Training-Serving Skew' (comportamiento dispar entre desarrollo y producción).
    """

    # 1. AISLAMIENTO DE DATOS:
    # Se trabaja sobre una copia profunda y estructurada para prevenir efectos
    # secundarios no deseados sobre el DataFrame original (SettingWithCopyWarning).
    data = df[FEATURE_COLUMNS].copy()

    # -------------------------------------------------------------------------
    # FEATURE 1: DIFERENCIA TÉRMICA (Delta T)
    # 
    # Justificación Física: En ingeniería, el gradiente térmico es crítico. Una 
    # disipación de calor deficiente eleva rápidamente la temperatura del proceso 
    # respecto al aire. Si esta diferencia se dispara, indica fricción anómala o 
    # problemas en los sistemas de enfriamiento, un predictor clave de fallas HDF 
    # (Heat Dissipation Failure).
    # -------------------------------------------------------------------------
    data["temperature_difference"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    # -------------------------------------------------------------------------
    # PASO INTERMEDIO: VELOCIDAD ANGULAR (omega = rad/s)
    #
    # Justificación Física: Las revoluciones por minuto (RPM) no son una unidad 
    # del Sistema Internacional (SI) apta para cálculos directos de energía.
    # Fórmula: omega = (RPM * 2 * pi) / 60
    # -------------------------------------------------------------------------
    angular_velocity = (
        data["Rotational speed [rpm]"]
        * 2
        * np.pi
        / 60
    )

    # -------------------------------------------------------------------------
    # FEATURE 2: POTENCIA MECÁNICA ESTIMADA (Watts)
    #
    # Justificación Física: La potencia consumida (P = Torque x Velocidad Angular) 
    # mapea directamente el esfuerzo real del motor.
    # En el dataset AI4I 2020, el torque y las RPM tienen una relación inversa perfecta. 
    # Las anomalías ocurren cuando rompen esta regla: ej. si el torque es muy alto 
    # para una velocidad alta, la potencia se dispara indicando sobrecarga del motor,
    # lo cual predice fallas PWF (Power Failure) u OSF (Overstrain Failure).
    # -------------------------------------------------------------------------
    data["mechanical_power"] = (
        data["Torque [Nm]"]
        * angular_velocity
    )

    # -------------------------------------------------------------------------
    # RECOMENDACIÓN DE MEJORA FUTURA (FEATURE OBLIGATORIA PARA MANTENIMIENTO)
    # 
    # Feature sugerida: "wear_strain" = mechanical_power * Tool wear [min]
    # Justificación: Una herramienta muy desgastada (high Tool Wear) sometida 
    # a alta potencia (high mechanical_power) fallará inminentemente por fatiga (TWF).
    # -------------------------------------------------------------------------

    return data



