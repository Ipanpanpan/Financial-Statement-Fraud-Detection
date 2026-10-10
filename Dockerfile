# Production Dockerfile optimized for Google Cloud Run
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output for Cloud Logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install curl for container health check probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a separate layer for build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained ML artifacts
COPY app/ ./app/
COPY main.py .
COPY lr_fraud_model.joblib .
COPY scaler.joblib .

# Create non-root user for enterprise container security compliance
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Cloud Run injects the $PORT environment variable (default 8080)
EXPOSE 8080

# Execute uvicorn via shell to evaluate dynamic $PORT while passing UNIX signals
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]