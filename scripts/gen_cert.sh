#!/usr/bin/env bash
# Генерирует самоподписанный TLS-сертификат для локального HTTPS-сервера.
#
# Chrome даёт доступ к камере (getUserMedia/ImageCapture) только по HTTPS
# (или на localhost), поэтому для доступа с телефона по Wi-Fi нужен
# сертификат — самоподписанный (этот скрипт) либо через Tailscale (см. README).
#
# Использование:
#   ./scripts/gen_cert.sh [IP_ИЛИ_ХОСТ ...]
#
# Пример (сервер виден в локальной сети по 192.168.1.50):
#   ./scripts/gen_cert.sh 192.168.1.50
#
# Без аргументов сертификат выпускается на localhost и 127.0.0.1 — подходит
# только для проверки на самом ПК, с телефона IP нужно передать явно.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p certs

EXTRA_HOSTS=("$@")
SAN_ENTRIES=("DNS:localhost" "IP:127.0.0.1")
for h in "${EXTRA_HOSTS[@]}"; do
  if [[ "$h" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    SAN_ENTRIES+=("IP:$h")
  else
    SAN_ENTRIES+=("DNS:$h")
  fi
done
SAN=$(IFS=,; echo "${SAN_ENTRIES[*]}")

echo "Выпускаю сертификат для: $SAN"

openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout certs/server.key \
  -out certs/server.crt \
  -days 825 \
  -subj "/CN=PuzzleVision Local/O=PuzzleVision/C=RU" \
  -addext "subjectAltName=$SAN"

chmod 600 certs/server.key

echo ""
echo "Готово: certs/server.crt, certs/server.key"
echo "На телефоне при первом заходе Chrome покажет предупреждение о"
echo "самоподписанном сертификате — это ожидаемо, нужно подтвердить переход"
echo "('Дополнительно' -> 'Перейти на сайт ...')."
