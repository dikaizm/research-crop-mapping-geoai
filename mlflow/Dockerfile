FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential git curl \
	&& rm -rf /var/lib/apt/lists/*

# Install MLflow (pin to a stable recent version) and optional extras
ARG MLFLOW_VERSION=2.14.3
RUN pip install --no-cache-dir mlflow==${MLFLOW_VERSION} boto3 psycopg2-binary

# Create a non-root user for better security
RUN useradd -ms /bin/bash mlflow
USER mlflow

WORKDIR /home/mlflow

# Default environment variables for MLflow server
ENV BACKEND_STORE_URI=sqlite:////home/mlflow/mlruns/mlflow.db \
	ARTIFACT_ROOT=/home/mlflow/mlruns/artifacts \
	MLFLOW_HOST=0.0.0.0 \
	MLFLOW_PORT=8080

# Create directories used by MLflow
RUN mkdir -p /home/mlflow/mlruns /home/mlflow/mlruns/artifacts

EXPOSE 8080

# Default command; using shell form so ENV vars are expanded
CMD ["sh", "-c", "mlflow server --backend-store-uri \"$BACKEND_STORE_URI\" --default-artifact-root \"$ARTIFACT_ROOT\" --host \"$MLFLOW_HOST\" --port \"$MLFLOW_PORT\""]

