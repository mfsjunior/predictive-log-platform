FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow>=2.11.1"

RUN mkdir -p /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "sqlite:///mlflow/mlflow.db", \
    "--default-artifact-root", "/models/mlflow-artifacts"]
