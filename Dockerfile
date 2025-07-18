# Stage 1: Builder
FROM python:3.9-slim AS builder

# Install build tools untuk compile dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/python -r requirements.txt

COPY . .

# Stage 2: Distroless
FROM gcr.io/distroless/python3-debian11

WORKDIR /app

# Copy installed dependencies dari builder
COPY --from=builder /app /app
COPY --from=builder /python /python
ENV PYTHONPATH=/python/lib/python3.9/site-packages
ENV PATH="/python/bin:$PATH"

EXPOSE 8501

CMD ["/python/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]