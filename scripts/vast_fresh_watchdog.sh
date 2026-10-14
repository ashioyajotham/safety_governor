#!/usr/bin/env bash
# Independent, per-instance paid-runtime backstop. Run under supervisor on Vast.
set -euo pipefail
set +x

source /opt/supervisor-scripts/utils/environment.sh

: "${CONTAINER_ID:?missing Vast container ID}"
: "${CONTAINER_API_KEY:?missing per-container Vast API key}"
: "${FRESH_EXPECTED_INSTANCE_ID:?missing approved target instance ID}"
: "${FRESH_MAX_SECONDS:?missing paid-runtime cap}"

if [[ "$CONTAINER_ID" != "$FRESH_EXPECTED_INSTANCE_ID" ]]; then
  echo "Watchdog target mismatch; refusing to arm" >&2
  exit 2
fi
if ! [[ "$FRESH_MAX_SECONDS" =~ ^[0-9]+$ ]] ||
   (( FRESH_MAX_SECONDS < 1 || FRESH_MAX_SECONDS > 36000 )); then
  echo "Watchdog deadline must be between 1 and 36000 seconds" >&2
  exit 2
fi
command -v vastai >/dev/null
vastai show instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >/dev/null

echo "Vast watchdog armed for instance $CONTAINER_ID, deadline in $FRESH_MAX_SECONDS seconds"
sleep "$FRESH_MAX_SECONDS"
echo "Vast watchdog deadline reached at $(date -u +%FT%TZ); stopping instance $CONTAINER_ID"
for attempt in 1 2 3 4 5; do
  if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
    exit 0
  fi
  sleep 30
done
echo "Vast stop API failed after five attempts; operator intervention required" >&2
exit 1
