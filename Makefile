.PHONY: start
start:
	uvicorn main:app --reload --port 9001

.PHONY: format
format:
	black .
	isort .