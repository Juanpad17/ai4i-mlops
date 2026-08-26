# script para confirmar que los gates SI fallan cuando el dataset
# tiene problemas, no solo cuando esta limpio. trabaja sobre copias,
# nunca toca el csv real en disco.

import pandas as pd
from quality_gates import run_quality_gates

df = pd.read_csv("data/raw/ai4i2020.csv")

print(">>> PRUEBA 1: dataset original (deberia pasar todo)")
reporte_ok = run_quality_gates(df.copy(), nombre_batch="prueba_original")
print(f"resultado: {'PASO' if reporte_ok['paso_general'] else 'FALLO'}")

print("\n>>> PRUEBA 2: nulos en el target (deberia fallar)")
df_con_nulos = df.copy()
df_con_nulos.loc[0, "Machine failure"] = None
reporte_nulos = run_quality_gates(df_con_nulos, nombre_batch="prueba_nulos_target")
for r in reporte_nulos["reglas"]:
    if r["regla"] == "target_sin_nulos":
        print(f"target_sin_nulos: {'PASO' if r['paso'] else 'FALLO (correcto, se esperaba esto)'}")

print("\n>>> PRUEBA 3: columna faltante (deberia fallar)")
df_sin_columna = df.copy().drop(columns=["Torque [Nm]"])
reporte_columna = run_quality_gates(df_sin_columna, nombre_batch="prueba_columna_faltante")
for r in reporte_columna["reglas"]:
    if r["regla"] == "estructura_completa":
        print(f"esquema_completo: {'PASO' if r['paso'] else 'FALLO (correcto, se esperaba esto)'}")

print("\n>>> PRUEBA 4: valor negativo en rpm (deberia fallar)")
df_negativo = df.copy()
df_negativo.loc[0, "Rotational speed [rpm]"] = -500
reporte_negativo = run_quality_gates(df_negativo, nombre_batch="prueba_valor_negativo")
for r in reporte_negativo["reglas"]:
    if r["regla"] == "variables_fisicas_no_negativas":
        print(f"variables_fisicas_no_negativas: {'PASO' if r['paso'] else 'FALLO (correcto, se esperaba esto)'}")