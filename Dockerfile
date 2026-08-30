FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY . .

# Render sets PORT env var; default to 8000 for local testing
ENV PORT=8000

# Expose the port
EXPOSE ${PORT}

# Run the FastAPI app with uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
