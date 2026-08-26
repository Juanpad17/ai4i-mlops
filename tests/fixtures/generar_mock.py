# genera un archivo CSV de prueba con datos simulados para el dataset AI4I 2020, 
# con el fin de probar la etapa de split del pipeline sin necesidad de descargar el dataset real.
from pathlib import Path
import pandas as pd
import numpy as np

# Definir la ruta esperada por tu script de split
output_path = Path("data/processed/validated.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Configurar semilla para reproducibilidad
np.random.seed(42)
n_rows = 200

# Crear datos simulados con las columnas reales del dataset AI4I 2020
mock_data = {
    "UDI": range(1, n_rows + 1),
    "Product ID": [f"M{15000 + i}" for i in range(n_rows)],
    "Type": np.random.choice(["L", "M", "H"], size=n_rows, p=[0.6, 0.3, 0.1]),
    "Air temperature [K]": np.random.normal(300, 2, size=n_rows),
    "Process temperature [K]": np.random.normal(310, 1, size=n_rows),
    "Rotational speed [rpm]": np.random.normal(1500, 100, size=n_rows),
    "Torque [Nm]": np.random.normal(40, 10, size=n_rows),
    "Tool wear [min]": np.random.randint(0, 240, size=n_rows),
    # Forzamos un ~5% de fallas para que el split estratificado funcione
    "Machine failure": np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
}

# Convertir a DataFrame y guardar
df = pd.DataFrame(mock_data)
df.to_csv(output_path, index=False)
print(f"✅ Archivo de prueba creado exitosamente en: {output_path}")

