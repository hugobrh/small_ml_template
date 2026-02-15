FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

# Installation des dépendances sans cache pour alléger l'image
RUN uv sync --frozen --no-dev

# Exposer le port pour Cloud Run
EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.credit_model.api:app", "--host", "0.0.0.0", "--port", "8080"]