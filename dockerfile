FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependências de runtime úteis para numpy/pandas/lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libopenblas0 libxml2 libxslt1.1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) instalar deps primeiro (cache melhor)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) copiar o código
COPY src/ /app/

# Porta do Uvicorn
EXPOSE 8000

# Comando padrão
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
