COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: up down psql migrate test eval isolation

up:
	$(COMPOSE) up --detach --wait
	$(MAKE) migrate

down:
	$(COMPOSE) down

psql:
	$(COMPOSE) exec postgres psql -U efm_app -d efm

migrate:
	$(COMPOSE) exec postgres sh -c 'psql -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -f /docker-entrypoint-initdb.d/002-ingestion-search-metadata.sql'

test:
	uv run pytest

eval:
	uv run pytest eval

isolation:
	uv run pytest -m isolation
