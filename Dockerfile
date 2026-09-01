FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DATA_DIR=/data

WORKDIR /app

# LightGBM needs the OpenMP runtime (libgomp). python:3.12-slim doesn't ship
# it, so ml_filter.load() failed in the cluster with
# "libgomp.so.1: cannot open shared object file" — both the buy and short
# models silently degraded. Install it before pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY README.md ./
# Ship the deployment config (tokens + live flags) so the container is
# self-sufficient and trades live on restart. SANDBOX_API_KEY inside is blank;
# ArenaGo injects the real one via ENV (load_dotenv override=False respects it).
COPY .env ./
# Ship the v4 ML model inside the image so the bot works on a fresh /data
# volume; .dockerignore excludes data/logs|state|reports but keeps models/.
COPY data/models ./data/models

RUN mkdir -p /data

CMD ["python", "-m", "moex_agent"]
