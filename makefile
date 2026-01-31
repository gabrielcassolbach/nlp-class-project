APP_DIR=05\ -\ Resultados\ finais/src/backend
APP_MODULE=app:app

run:
	uvicorn $(APP_MODULE) --app-dir $(APP_DIR) --reload

.PHONY: run
