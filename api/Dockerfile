FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (optional, for chronos)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (including your custom modules)
COPY src/ ./src/
COPY models/ ./models/
COPY api/app.py ./app.py

# Expose port for Render
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]