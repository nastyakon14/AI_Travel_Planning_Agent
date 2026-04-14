.PHONY: test up down up-monitoring

test:
	python -m pytest tests/ -v

up:
	docker compose up --build

up-monitoring:
	docker compose --profile monitoring up --build

down:
	docker compose down
