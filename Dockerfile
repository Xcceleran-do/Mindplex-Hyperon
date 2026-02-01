# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for hyperon and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY experiments/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Set PYTHONPATH to include the workspace root (for relative imports)
ENV PYTHONPATH=/app:$PYTHONPATH

# Set working directory to workspace root (so relative paths work)
WORKDIR /app

# Expose port (Cloud Run will set PORT env variable)
ENV PORT=8080
EXPOSE 8080

# Run the Flask app with production server (gunicorn)
RUN pip install gunicorn

# Command to run the application from the root directory
# This ensures relative imports like "experiments:pattern-miner" work correctly
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 experiments.mining_api:app
