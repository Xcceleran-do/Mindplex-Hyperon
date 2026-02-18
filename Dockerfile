FROM swipl:latest

ENV PYTHONUNBUFFERED=1

# Install Python and system build tools (SWI-Prolog is provided by base image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source (includes PeTTa submodule contents)
COPY . /app

# Install Python dependencies for the experiments service
RUN pip3 install --no-cache-dir -r experiments/requirements.txt

EXPOSE 5000

# Render will override CMD if a start command is configured
CMD ["python3", "experiments/mining_api.py"]
