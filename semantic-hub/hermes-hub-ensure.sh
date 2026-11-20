#!/usr/bin/env bash
# hermes-hub-ensure: asegura los hubs splatsdb de memoria Hermes.
#   :8872 sesiones (hermes-hub)  |  :8873 código (hermes-hub-code)
# Idempotente POR HUB (sin early-exit: un hub vivo no debe saltear el otro).
# No toca el 8871 (memorias m2m legacy, otro pipeline).
# Cron sugerido: */10 * * * *  (recuperación ante crash en <=10 min).
set -u
LOG_DIR="/mnt/d/WSL/orch"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/hermes-hub.log"
BIN="/mnt/d/splatdb/target/release/splatsdb"

# Exclusión mutua (17-nov: el ensure */10 podía solaparse con el ensure del
# ciclo 9:15; dos lanzadores en carrera envenenaban hub.pid con el pid perdedor).
exec 9>"$LOG_DIR/.hub-ensure.lock"
flock -n 9 || exit 0

ensure_hub() { # puerto, data-dir, etiqueta
    local PORT="$1" HUB="$2" LABEL="$3"
    if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"status"'; then
        return 0
    fi
    if [ ! -x "$BIN" ]; then
        echo "$(date -u +%FT%TZ) ERROR: binario splatsdb no encontrado en $BIN" >> "$LOG"
        return 1
    fi
    if [ ! -f "$HUB/data/vectors/default.bin" ]; then
        echo "$(date -u +%FT%TZ) skip $LABEL: sin $HUB/data/vectors/default.bin todavía (correr ingest+merge)" >> "$LOG"
        return 1
    fi
    # v2.8.0: trainer e-means para el fit jerárquico HRM2 (medido: −13.3%
    # startup serve, −0.56% loss, /search idéntico — aprobado Brian 13-nov).
    # v2.9.0: /recall nativo — el motor loggea sus fires al log canónico del
    # digest (schema idéntico al del hook legacy; en modo nativo el hook no
    # duplica la línea).
    SPLATSDB_RECALL_LOG="$LOG_DIR/hermes-recalls.jsonl" nohup "$BIN" --dim 768 serve --port "$PORT" --host 127.0.0.1 --data-dir "$HUB" --gpu --trainer emeans >> "$LOG" 2>&1 &
    # Solo confiar en el pid si el proceso sigue vivo tras 1s: un proceso que
    # pierde la carrera del puerto muere y hub.pid quedaría envenenado
    # (17-nov: el cycle mataría un pid inexistente para siempre).
    sleep 1
    if ! kill -0 $! 2>/dev/null; then
        echo "$(date -u +%FT%TZ) ERROR: hub $LABEL murió al arrancar (puerto $PORT ocupado o crash) — revisar log" >> "$LOG"
        return 1
    fi
    echo $! > "$HUB/hub.pid"
    echo "$(date -u +%FT%TZ) hub $LABEL lanzado pid $(cat "$HUB/hub.pid") (puerto $PORT)" >> "$LOG"
}

ensure_hub 8872 "/mnt/d/splatdb/hermes-hub" "sesiones"
ensure_hub 8873 "/mnt/d/splatdb/hermes-hub-code" "código"
exit 0
