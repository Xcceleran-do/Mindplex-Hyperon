FROM node:22-bookworm-slim AS mettascript

WORKDIR /opt/mettascript
COPY package.json package-lock.json ./
RUN npm ci

FROM swipl:10.0.2

ARG DEBIAN_MIRROR=http://ftp.us.debian.org/debian
ARG DEBIAN_SECURITY_MIRROR=http://security.debian.org/debian-security

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN printf 'Acquire::ForceIPv4 "true";\nAcquire::Retries "5";\nAcquire::http::Timeout "60";\nAcquire::https::Timeout "60";\n' > /etc/apt/apt.conf.d/99mindplex-retries \
    && sed -i "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g; s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && swipl --version \
    && python3 --version

WORKDIR /app

COPY --from=mettascript /usr/local/bin/node /usr/local/bin/node
COPY --from=mettascript /opt/mettascript/node_modules /app/node_modules

COPY experiments/requirements.txt /app/experiments/requirements.txt
COPY PeTTa /app/PeTTa

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && if [ ! -f /app/PeTTa/setup.py ]; then \
        echo "PeTTa source is missing from the Docker build context. Run: git submodule update --init --recursive PeTTa" >&2; \
        exit 1; \
    fi \
    && python -m pip install -r experiments/requirements.txt \
    && node --version \
    && node --import tsx --input-type=module -e "import { MeTTa } from '@metta-ts/hyperon'; const m = new MeTTa(); if (String(m.run('!(+ 1 1)')[0][0]) !== '2') process.exit(1)" \
    && python -c "import janus_swi; print('janus_swi import ok')" \
    && python -c "from petta import PeTTa; print('petta import ok')"

COPY . /app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "experiments.mining_api:create_app()"]
