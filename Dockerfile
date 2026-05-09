FROM swipl:10.0.2

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && swipl --version \
    && python3 --version

WORKDIR /app

COPY experiments/requirements.txt /app/experiments/requirements.txt
COPY PeTTa /app/PeTTa

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r experiments/requirements.txt \
    && python -c "import janus_swi; print('janus_swi import ok')" \
    && python -c "from petta import PeTTa; print('petta import ok')"

COPY . /app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "experiments.mining_api:create_app()"]
