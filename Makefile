COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: up down psql test eval isolation

up:
	$(COMPOSE) up --detach --wait

down:
	$(COMPOSE) down

psql:
	$(COMPOSE) exec postgres psql -U efm_app -d efm

test:
	uv run pytest

eval:
	uv run pytest eval

isolation:
	uv run pytest -m isolation
