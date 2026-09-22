# syntax=docker/dockerfile:1

# --- Stage 1: собрать PWA-клиент -------------------------------------------------
FROM node:20-alpine AS client-build
WORKDIR /client
COPY client/package.json client/package-lock.json* ./
RUN npm install
COPY client/ ./
RUN npm run build

# --- Stage 2: рантайм сервера -----------------------------------------------------
FROM python:3.11-slim AS server

# libgl1/libglib2.0-0 нужны OpenCV (даже headless-сборке) для декодирования кадров.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY server/app ./server/app
COPY config ./config
COPY --from=client-build /client/dist ./client/dist

ENV PYTHONPATH=/srv/server \
    PUZZLEVISION_CONFIG=/srv/config/config.yaml

EXPOSE 8443

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--app-dir", "/srv/server", \
     "--host", "0.0.0.0", "--port", "8443", \
     "--ssl-certfile", "/srv/certs/server.crt", \
     "--ssl-keyfile", "/srv/certs/server.key"]
