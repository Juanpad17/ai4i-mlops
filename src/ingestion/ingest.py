# %%
"""
Etapa de ingesta del pipeline.

Descarga un dataset desde una URL configurable y lo almacena en data/raw/, calculando un hash SHA256 para versionar los datos.
"""

# Importamos Path para manejar rutas de manera independiente
from pathlib import Path

# shutil permite copiar archivos (lo usamos al extraer el CSV del ZIP).
import shutil

# hashlib nos permitirá generar un hash del dataset.
import hashlib

# os nos permite leer variables de entorno (ej. DATASET_URL).
import os

# argparse nos permite pasar la URL como argumento al ejecutar el script.
import argparse

# tempfile y zipfile para poder descargar y extraer datasets en .zip
import tempfile
import zipfile

# requests para descargar el archivo desde internet.
import requests

# URL por defecto, usada solo si no se indica otra por argumento o variable de entorno.
DEFAULT_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

# Lugar oficial donde nuestro pipeline almacenará el dataset RAW.
RAW_PATH = Path("data/raw/ai4i2020.csv")

# Nombre del CSV que buscamos dentro del ZIP
EXPECTED_CSV_NAME = "ai4i2020.csv"

# Tiempo máximo de espera (segundos) para la descarga
TIMEOUT_SECONDS = 30


def calculate_hash(file_path: Path) -> str:
    """
    Esto permite identificar exactamente qué dataset se utilizó
    durante un experimento.
    """
    sha256 = hashlib.sha256()
    # Abrimos el archivo en modo binario.
    with open(file_path, "rb") as file:
        # Leemos el archivo por bloques para evitar cargar
        # archivos grandes completamente en memoria.
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


def resolve_url(cli_url: str | None) -> str:
    """
    Decide qué URL usar, en este orden de prioridad:
    1. Argumento --url pasado por línea de comandos.
    2. Variable de entorno DATASET_URL.
    3. DEFAULT_URL definida arriba.
    Esto permite cambiar la fuente de datos sin modificar el código.
    """
    if cli_url:
        return cli_url
    env_url = os.environ.get("DATASET_URL")
    if env_url:
        return env_url
    return DEFAULT_URL


def _fetch(url: str) -> requests.Response:
    """
    Hace la petición GET a `url` con manejo de errores claro.
    Separado de download_dataset para poder reutilizarlo tanto para
    CSV directos como para archivos ZIP.
    """
    try:
        # stream=True evita cargar el archivo completo en memoria de golpe.
        response = requests.get(url, stream=True, timeout=TIMEOUT_SECONDS)
        # Lanza una excepción si el código de estado HTTP indica error
        # (404, 500, etc.), en vez de guardar una página de error como CSV.
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Tiempo de espera agotado al descargar desde {url}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"No se pudo conectar a {url}. Verifica la URL o tu conexión."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"El servidor respondió con error para {url}: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Error inesperado al descargar {url}: {exc}"
        ) from exc


def _looks_like_zip(url: str, response: requests.Response) -> bool:
    """
    Determina si la respuesta es un archivo ZIP, ya sea por la extensión de la URL o por el Content-Type.
    """
    content_type = response.headers.get("Content-Type", "").lower()
    return url.lower().endswith(".zip") or "zip" in content_type


def download_dataset(url: str, destination: Path) -> None:
    """
    Descarga el archivo desde `url` y lo guarda en `destination`.
    """
    response = _fetch(url)

    # Creamos data/raw si todavía no existe.
    destination.parent.mkdir(parents=True, exist_ok=True)

    if _looks_like_zip(url, response):
        # Descargamos el ZIP a un archivo temporal antes de extraerlo,
        # ya que zipfile necesita poder buscar (seek) dentro del archivo.
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            for block in response.iter_content(chunk_size=8192):
                if block:
                    tmp_zip.write(block)
            tmp_zip_path = Path(tmp_zip.name)

        try:
            with zipfile.ZipFile(tmp_zip_path) as archive:
                # Buscamos primero el CSV esperado; si no está, usamos
                # el primer .csv que aparezca dentro del ZIP.
                csv_names = [
                    name for name in archive.namelist()
                    if name.lower().endswith(".csv")
                ]
                if not csv_names:
                    raise RuntimeError(
                        f"El ZIP descargado de {url} no contiene ningún archivo .csv"
                    )
                target_name = next(
                    (n for n in csv_names if n.endswith(EXPECTED_CSV_NAME)),
                    csv_names[0],
                )
                with archive.open(target_name) as csv_in_zip:
                    with open(destination, "wb") as out_file:
                        shutil.copyfileobj(csv_in_zip, out_file)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"El archivo descargado de {url} no es un ZIP válido"
            ) from exc
        finally:
            # Limpiamos el ZIP temporal, ya sea que todo haya salido bien o no.
            tmp_zip_path.unlink(missing_ok=True)
    else:
        # Descarga directa: escribimos el archivo por bloques (igual que
        # hacemos al leerlo para el hash), para no cargar datasets
        # grandes en memoria.
        with open(destination, "wb") as file:
            for block in response.iter_content(chunk_size=8192):
                if block:
                    file.write(block)


def ingest_data(url: str | None = None) -> None:
    """
    Ejecuta la etapa de ingesta del pipeline: descarga el dataset
    desde `url` y calcula su hash SHA256.
    """
    final_url = resolve_url(url)

    print(f"Descargando dataset desde: {final_url}")
    download_dataset(final_url, RAW_PATH)

    # Calculamos una identificación de esta versión de datos.
    data_hash = calculate_hash(RAW_PATH)

    print("Ingesta finalizada correctamente.")
    print(f"Dataset RAW: {RAW_PATH}")
    print(f"Data version SHA256: {data_hash}")


def parse_args() -> argparse.Namespace:
    """
    Define los argumentos de línea de comandos.
    Uso:
        python ingest.py
    """
    parser = argparse.ArgumentParser(
        description="Descarga un dataset desde una URL y lo guarda en data/raw/"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL del dataset a descargar. Si se omite, se usa DATASET_URL "
             "(variable de entorno) o la URL por defecto en el script.",
    )
    args, _unknown = parser.parse_known_args()
    return args


# Esto permite ejecutar directamente:
# python src/ingestion/ingest.py
# python src/ingestion/ingest.py
if __name__ == "__main__":
    args = parse_args()
    ingest_data(url=args.url)


