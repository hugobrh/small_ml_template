install-all:
    uv sync

install-prod:
    uv sync --no-dev

tests:
    uv run pytest tests/

train-model:
    uv run python src/credit_model/train.py

test-api:
    uv run python src/credit_model/api.py

lint:
    uv run ruff check . --fix

format:
    uv run ruff format .

train:
    uv run python src/credit_model/train.py

serve:
    uv run uvicorn src.credit_model.api:app --reload

docker-build:
    docker build -t credit-g-app .