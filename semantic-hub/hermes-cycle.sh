#!/usr/bin/env bash
# hermes-cycle: ciclo diario completo del hub de sesiones Hermes.
#   ingesta incremental → merge → restart del hub → smoke test
# Cron sugerido: 15 9 * * *  (06:15 Argentina, una vez por día).
# El ensure (*/10) cubre crash-recovery entre ciclos.
set -u
LOG_DIR="/mnt/d/WSL/orch"
HUB="/mnt/d/splatdb/hermes-hub"
SH="/mnt/d/splatdb/semantic-hub"
LOG="$LOG_DIR/hermes-hub.log"
mkdir -p "$LOG_DIR"

say() { echo "$(date -u +%FT%TZ) [cycle] $1" >> "$LOG"; }

say "=== inicio ciclo ==="

# 1. embed server tiene que estar vivo para embeber
bash /mnt/d/embedding-server/start-embed-server.sh >> "$LOG" 2>&1
for i in $(seq 1 150); do
    curl -s --max-time 2 http://127.0.0.1:8860/health | grep -q status && break
    sleep 2
done
if ! curl -s --max-time 2 http://127.0.0.1:8860/health | grep -q status; then
    say "ERROR: embed server no levantó — ciclo abortado"
    exit 1
fi

# 2. ingesta incremental de sesiones
if ! python3 "$SH/hermes_ingest.py" >> "$LOG" 2>&1; then
    say "ERROR: ingesta falló — ciclo abortado (sin merge ni restart)"
    exit 1
fi

# 2b. merge del part de sesiones → sidecar/bin del hub 8872
if ! python3 "$SH/hermes_merge.py" --parts hermes >> "$LOG" 2>&1; then
    say "ERROR: merge de sesiones falló — ciclo abortado"
    exit 1
fi

# 3. código: re-ingesta de repos (rebuild del part — los archivos cambian)
REPOS=(/mnt/d/splatdb /mnt/d/vuupta /mnt/d/vectorcraft /mnt/d/splatsdb-ui \
       /mnt/d/m2m-memory /mnt/d/kicad-mcp /mnt/d/pasturometro-app \
       /mnt/d/freerouting-ml /mnt/d/fusion360-mcp /mnt/d/career-ops \
       /mnt/d/ClinFusion /mnt/d/pulverizador-hpa /mnt/d/RTKBase_50k \
       /mnt/d/latent-space-language-diffusion-model)
ARGS=()
for r in "${REPOS[@]}"; do [ -d "$r" ] && ARGS+=(--repo "$r"); done
if ! python3 "$SH/ingest.py" --hub /mnt/d/splatdb/hermes-hub-code --name repos \
        --kind repo --cap 500 --batch 32 "${ARGS[@]}" >> "$LOG" 2>&1; then
    say "WARN: ingesta de código falló — el hub de código mantiene el índice anterior"
fi
if ! python3 "$SH/hermes_merge.py" --hub /mnt/d/splatdb/hermes-hub-code \
        --parts repos >> "$LOG" 2>&1; then
    say "WARN: merge del hub de código falló — mantiene el índice anterior"
fi

# 4. restart de ambos hubs para que carguen los bins nuevos
for H in "$HUB" "/mnt/d/splatdb/hermes-hub-code"; do
    if [ -f "$H/hub.pid" ] && kill -0 "$(cat "$H/hub.pid")" 2>/dev/null; then
        say "restart hub $(basename "$H") (pid $(cat "$H/hub.pid"))"
        kill "$(cat "$H/hub.pid")" 2>/dev/null
    fi
done
# Espera activa de muerte REAL (17-nov: el sleep 2 fijo alcanzaba de suerte;
# un hub lento en morir seguía respondiendo health OK y el ensure no relanzaba
# con los bins nuevos — índice congelado sin ningún FAIL visible).
STILL_UP=""
for i in $(seq 1 15); do
    STILL_UP=""
    curl -s --max-time 2 http://127.0.0.1:8872/health | grep -q status && STILL_UP="$STILL_UP 8872"
    curl -s --max-time 2 http://127.0.0.1:8873/health | grep -q status && STILL_UP="$STILL_UP 8873"
    [ -z "$STILL_UP" ] && break
    sleep 1
done
if [ -n "$STILL_UP" ]; then
    say "WARN: hubs$STILL_UP siguen vivos tras 15s — SIGKILL"
    for H in "$HUB" "/mnt/d/splatdb/hermes-hub-code"; do
        [ -f "$H/hub.pid" ] && kill -9 "$(cat "$H/hub.pid")" 2>/dev/null
    done
    sleep 2
fi
bash "$SH/hermes-hub-ensure.sh" >> "$LOG" 2>&1
for i in $(seq 1 60); do
    curl -s --max-time 2 http://127.0.0.1:8872/health | grep -q status \
      && curl -s --max-time 2 http://127.0.0.1:8873/health | grep -q status && break
    sleep 1
done

# 5. smoke: si falla, queda en el log con detalle
if python3 "$SH/hermes_smoke.py" >> "$LOG" 2>&1; then
    say "smoke OK"
else
    say "WARN: smoke falló — ver detalles arriba"
    exit 1
fi

# 6. validación de configuración del stack (14-nov): embed+hubs+hook+log
#   +crons en un comando. Fallo = exit 1 (cron notifica).
if python3 "$SH/hermes_stack_check.py" >> "$LOG" 2>&1; then
    say "stack_check OK"
else
    say "WARN: stack_check FAIL — algún check del stack en rojo"
    exit 1
fi

say "=== ciclo OK ==="
