# 1. Imagen base oficial de Python
FROM python:3.12-slim

# 2. Directorio de trabajo en el contenedor
WORKDIR /app

# 3. Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el código del proyecto
COPY . .

# 5. Informar a Docker que la API escuchará en el puerto 8000
EXPOSE 8000

# 6. Comando para arrancar FastAPI usando Uvicorn
# "app:app" significa: archivo app.py, variable app = FastAPI()
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
ENV PYTHONPATH=/app
