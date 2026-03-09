FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential git curl \
	&& rm -rf /var/lib/apt/lists/*

# Install MLflow (pin to a stable recent version) and optional extras
ARG MLFLOW_VERSION=3.10.1
RUN pip install --no-cache-dir mlflow==${MLFLOW_VERSION} boto3 psycopg2-binary

# Create a non-root user for better security
RUN useradd -ms /bin/bash mlflow
USER mlflow

WORKDIR /home/mlflow

# Default environment variables for MLflow
ENV BACKEND_STORE_URI=sqlite:////home/mlflow/mlruns/mlflow.db \
	ARTIFACT_ROOT=/home/mlflow/mlruns/artifacts \
	MLFLOW_HOST=0.0.0.0 \
	MLFLOW_PORT=8080 \
	MLFLOW_ALLOWED_HOSTS="localhost,localhost:*" \
	MLFLOW_CORS_ALLOWED_ORIGINS=""

# MinIO compatibility: disable boto3 1.34+ automatic checksum headers
# boto3 ≥1.34 sends CRC32 checksums by default; MinIO rejects them with XAmzContentSHA256Mismatch.
# AWS_REQUEST_CHECKSUM_CALCULATION=when_required disables automatic checksum injection.
ENV AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
	AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

# Create directories used by MLflow server and ensure write access survives volume mounts
RUN mkdir -p /home/mlflow/mlruns /home/mlflow/mlruns/artifacts \
	&& chmod -R 777 /home/mlflow/mlruns

EXPOSE 8080

# Default command; using shell form so ENV vars are expanded
# --serve-artifacts: proxy artifact uploads through the server (clients need no S3 credentials)
CMD ["sh", "-c", "mlflow server --backend-store-uri \"$BACKEND_STORE_URI\" --artifacts-destination \"$ARTIFACT_ROOT\" --host \"$MLFLOW_HOST\" --port \"$MLFLOW_PORT\" --serve-artifacts --allowed-hosts \"$MLFLOW_ALLOWED_HOSTS\" --cors-allowed-origins \"$MLFLOW_CORS_ALLOWED_ORIGINS\""]