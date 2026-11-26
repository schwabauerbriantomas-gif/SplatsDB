# Hermes Memory Hub — recall semántico de sesiones

Capa de memoria semántica para el agente Hermes: indexa los mensajes de
conversación de `state.db` en un SplatsDB dedicado y expone recall por
**significado** (no por keyword). Complementa el FTS5 nativo de Hermes,
que gana en identificadores literales pero falla cuando la pregunta usa
otras palabras que la sesión original.

## Arquitectura

```
Hermes state.db (SQLite, SOLO LECTURA)
        │  hermes_ingest.py   (incremental, dedup por hash)
        ▼
hermes-hub/parts/hermes.db ──► hermes_merge.py ──► hermes-hub/data/vectors/default.bin
        (part: text+vec+meta)         (master sidecar + shard bin)   + hermes-hub/sidecar.db
                                                                            │
embeddinggemma-2 (:8860, GPU 3090, 768d)          splatsdb serve (:8872) ◄──┘
        ▲                                                  │
        └──── hermes_recall.py (embed query → search → join sidecar)
              ↳ ≥ 2.9.0: POST /recall — pipeline completo EN el motor
                (hook = shim: FTS5 de state.db + formateo)

14 repos locales (código) ──► ingest.py --hub hermes-hub-code ──► hermes-hub-code/
                                                                      └─► splatsdb (:8873)
```

- **8871**: hub existente con corpus m2m legacy (26.273 chunks, oct-2026) — no se toca.
- **8872**: sesiones Hermes (hub puro), data-dir `hermes-hub/`.
- **8873**: código de 14 repos (~6.5k chunks), data-dir `hermes-hub-code/` — hub
  puro a propósito: embeddinggemma es sesgado a prosa y en un hub mixto los
  mensajes que *discuten* código le ganan al código mismo (code hits ~rank 25
  en top-30; en el hub puro, top-3).
- El estado vivo de Hermes **nunca se escribe**: la fuente se abre en modo RO.
- El sidecar (`sidecar.db`) es el JOIN post-búsqueda: el server devuelve
  índices; el texto y la metadata (`{message_id, session_id, role, title, date}`)
  viven client-side. `chunks.rowid = vector_index + 1`.
- Orden de merge: `hermes` primero — preserva los vector_idx de sesiones
  cuando se agregan parts nuevos.

## Filtros de ingesta

Entra un mensaje si: `role ∈ {user, assistant}` AND `active=1` AND
`length ≥ 60` AND no empieza con notificaciones de sistema
(`[IMPORTANT:`, `[ASYNC DELEGATION`, `[OUT-OF-BAND`, `[Background process`,
`[Command timed out`) AND no es blob JSON puro (`{` o `[` inicial).
Texto truncado a 1500 chars (consistente con el benchmark).
Dedup por blake2b-96 del texto: re-envíos idénticos no duplican vectores.

## Embeddings

- Modelo: `google/embeddinggemma-2` local (`/mnt/d/embedding-server`, venv
  `~/.venvs/embed-server`, puerto 8860, MRL truncado a 768).
- **Latencia (6-nov-2026)**: el pipeline ST completo corre en CUDA graphs por
  buckets de seq_len (16–512 tokens) — embed de query **103ms → 10.9ms mediana**
  (el forward eager en WSL2 era CPU-bound: ~218 matmuls/casts por batch=1).
  Fallback eager para batch>1 o >512 tokens. Test canónico:
  `/mnt/d/embedding-server/test_graph_ab.py` (cos ≥ 0.9999 vs encode oficial).
- **Arranque/recovery**: cache del modelo en ext4 (`~/.cache/huggingface`,
  HF_HOME solo del server) — carga en ~22s (NTFS frío: 220s). Cron
  `*/10 ensure-embed.sh` relanza el server si el health no responde.
- **Asimetría de prompts (crítico para la calidad medida)**: documentos sin
  prompt; queries con `prompt_name=Retrieval-query`. El smoke test de
  `faststart.sh` del hub 8871 embebe queries sin el prompt — por eso sus
  scores de demo salían degradados (~0.35) frente a los de este pipeline (~0.6).

## Números medidos (2026-11-05, corpus real)

Benchmark de falsificación (`cache/scratch/mem-bench/bench.py` del perfil
Hermes; corpus: 4.411 mensajes muestreados de state.db):

| Métrica | FTS5 (actual) | Vector (este hub) |
|---|---|---|
| Recall@10 sesión, queries paráfrasis (n=8) | 2/8 | **7/8** |
| MRR sesión | 0.27 | **0.60** |
| Recall@10 sesión, queries literales (n=3) | 3/3 | 3/3 |
| Latencia query | ~14 ms | ~115–220 ms end-to-end |

En 5 de 8 queries paráfrasis el FTS devolvió cero resultados relevantes y el
vector trajo la sesión correcta en rank 1–7. Advertencias honestas: 11 queries
armadas a mano por el agente (no es un dataset público), un solo corpus, sin
cross-validación; la magnitud del gap sostiene la dirección del resultado, no
el dígito exacto. Queries basura: FTS devuelve 105–11.594 matches irrelevantes
(sin rechazo); el vector ordena bien pero las similitudes absolutas son altas
incluso para basura (0.59–0.70) → sirve como top-k para leer, no como umbral
automático de rechazo.

Operación real: primera ingesta 5.585 docs en 81 s (~69 msg/s), bin 17.2 MB.
Smoke post-restart: 3 casos de paráfrasis en rank 5/2/1.

## Semántica del score (importante)

El `score` que devuelve el server es una **DISTANCIA: menor = más cerca**.
El server ya ordena best-first (ascendente); no re-invertir en el cliente.
Calibración medida sobre este corpus:

| tipo de query | sesiones (8872) | código (8873) |
| --- | --- | --- |
| frase literal de la fuente | 0.42–0.51 | — |
| paráfrasis / tema del corpus | 0.53–0.60 | 0.67–0.70 |
| pregunta válida fuera del corpus | — | 0.70–0.76 |
| basura sin relación | 0.77+ | 0.81+ |

Corte de relevancia: `--max-dist 0.60` (sesiones) / `≤ 0.70` (código).
El smoke incluye un check de sanidad (`literal < basura`) que detecta
inversiones de esta semántica. Lección: la hipótesis "no se puede rechazar
basura por score" del benchmark inicial era un artefacto de interpretar el
score como similitud — con la semántica correcta la separación es limpia.

## Stack de memoria de Hermes — configuración canónica (14-nov-2026)

Auditoría swarm (deleg_2f5f34ea, 27 hallazgos con probes) + replay de
datos pasados. Commits: 7b46184, 87d5d91, fa1e528. Todo lo de esta
sección está medido, no asumido.

### Los tres consumidores del hub (:8872)

| consumidor | trigger | qué hace | config clave |
|---|---|---|---|
| **hook de memoria** (`~/.hermes/agent-hooks/hermes-memory-recall.py`) | cada API-call (dispara varias veces por turno — bursts medidos hasta 29×) | inyecta bloque `[Memoria vectorial]` en el contexto | STRONG_SESS ≤0.55 → 5 hits + STRONG_CODE ≤0.70 → 1 (ventana 3→5 medida 21-nov en LongMemEval-S: recall@contexto 53.2→63.1%, MAX_CHARS 1300→2400; banda medium 0.55-0.62 ELIMINADA: uso 4% medido por replay, Fisher p=0.029) |
| **sac_recap** (`semantic-hub/sac_recap.py` → POST /recap) | a pedido (handoff/recall) | recap de sesión: qué pasó, LLM-free | extrae acciones de state.db él mismo; el hub solo rankea |
| **sac_recall_digest** (`semantic-hub/sac_recall_digest.py`) | a pedido (auditoría) | qué memoria se inyectó y qué mostró uso | turn-level (dedupe por session+qhash), cortes aparte |

### Embeddings (config exacta)

- `google/embeddinggemma-2` en :8860, CUDA graphs, **dims 768** (MRL),
  query con `prompt_name=Retrieval-query` (asimetría crítica — ver arriba).
- El hook y todos los clientes internos mandan SIEMPRE el embedding en
  /search: desde fa1e528 el hub responde **400 sin embedding** (el
  fallback text-hash producía ranking de ruido con 200 OK).

### Contrato /search

- `score` = distancia d² ascendente en AMBOS paths (GPU y CPU — unificado
  fa1e528; antes el fallback CPU daba d y desplazaba las bandas ×√2).
- `backend: "gpu"` presente en la respuesta. El hook hace **fail-closed**
  si backend ≠ gpu (bandas descalibradas = corrupción silenciosa; el FTS
  literal sigue cubriendo).
- top_k=0 → 0 resultados (antes coerción silenciosa a 1).

### Contrato /store (26-nov-2026)

- Sin `embedding` en el request, el motor embebe el texto con el MISMO
  servicio de `/recall` (:8860, `embeddinggemma-2`) y **sin prompt** — la
  asimetría de arriba aplicada a documentos. Responde `embedding_source: "real"`.
- Servicio de embeddings caído → **503 y no guarda nada**. No existe
  fallback text-hash en el hub (antes guardaba un hash del texto con `200 OK`).
- Con `embedding` en el request el vector se guarda tal cual y la respuesta
  no lleva `embedding_source`.

### Contrato /recap

- Stateless por diseño: el lookup de sesión NO existe en el hub —
  `session_id` se acepta y se devuelve (echo) solo para roundtrip check.
- `RecapAction.seq` default 0 (decorativo; el orden es el del array).
- `max_words` clamp **50..2000**: con <41 las secciones mínimas
  state-critical no cabían y el recap degeneraba a OBJECTIVE-only.

### Contrato /recall (NATIVO, motor ≥ 2.9.0 — 20-nov-2026)

**Ventana del gate (2.10.0 — 21-nov-2026)**: `RECALL_SESS_N_DEFAULT` 3→5,
medido en LongMemEval-S (500 Q, judge GLM-5.3-flash): recall@contexto con
`answer_session_ids` GT 53.2%→63.1% (multi-session +13.4 pts, preference
+23.4 pts, temporal +10.5 pts). El presupuesto del hook subió con ella
(`MAX_CHARS` 1300→2400 — con 1300 los hits 4-5 morían en el corte).
Rollback instantáneo por env `SPLATSDB_RECALL_SESS_N=3` (el handler lo lee
en cada fire). Recency-weighting (`dist*(1-w·strength)`) quedó DESCARTADO
por el sweep: diferencia ≈0 con GT (63.1 vs 62.8) — no entra sin beneficio
medido.

El pipeline completo del hook vive DENTRO del motor: embed (mismo servicio
:8860, prompt asimétrico) → /search interno (misma función, cero drift de
contrato) → join sidecar mode=ro → gate → log JSONL. El hook queda como
**shim** (flag `~/.hermes/agent-hooks/.recall-native`): fetch FTS5 de
state.db + formateo; cualquier fallo cae al pipeline legacy completo
(fail-open, cero comportamiento cambiado en la migración).

- `POST /recall {text, session_id?, top_k?, fts?: [{dist, sid, text, meta}]}`
- Los hits FTS viajan en el request y se merges PRE-GATE en el bucket de
  sesiones: hay UNA implementación del gate (en Rust), no dos.
- **Gate = bucket-best** (semántica del hook, 14-nov): el umbral aplica al
  BEST hit del bucket; los ranks 2..n viajan sin umbral propio. Per-hit
  gating inyecta estrictamente menos (A/B 20-nov: 4/78 queries divergían).
- Buckets por `kind` (`repo` → código; resto → sesiones). En producción
  los sidecars son homogéneos por hub → equivale al split por puerto del
  hook legacy (verificado: 7008×hermes/session, 6024×repo).
- Fail-closed: backend ≠ gpu inyecta nada con `fail_closed` seteado;
  embed caído → 503 (NUNCA SimCos, fa1e528); texto vacío → 400.
- `rowid = index+1` verificado contiguo en ambos sidecars; hits sin row
  se descartan (jamás inventar texto).
- Equivalencia medida (20-nov, mismo corpus + instante, 78 queries):
  **DIFF=0** (69 idénticas + 9 cortes consistentes). El hook legacy y el
  nativo producen contexto byte-idéntico en los payloads canónicos.
- El fire lo loggea el MOTOR (una sola línea por fire, con `kind` por
  hit — fts incluido): el shim en modo nativo no loggea. El ensure setea
  `SPLATSDB_RECALL_LOG=/mnt/d/WSL/orch/hermes-recalls.jsonl` para que el
  motor escriba al log canónico del digest; env vacía = log deshabilitado.

### Telemetría (JSONL, /mnt/d/WSL/orch/hermes-recalls.jsonl)

```json
{"ts": ..., "session_id": "...", "qhash": "sha1-12", "qlen": N,
 "n_injected": N, "arm": 0|-1, "backend": "gpu", "drops": N,
 "cut_dist": 0.585?, "hits": [{"kind","dist","sid","repo","path"}]}
```
- `arm=-1` + `cut_dist`: query cortada por la banda medium (falsificable).
- `backend` ≠ "gpu": falla de escala detectada.
- `drops` > 0: join index↔rowid roto (invariant roto — investigar merge).
- El log mide FIRES (del hook legacy o del motor nativo), no turnos: el
  digest dedupea por (session_id, qhash).

### Smoke y validación

- `hermes_smoke.py`: health x3 + consistencia POR HUB (el drift-check de
  sesiones era código muerto hasta 87d5d91) + recall-cases tolerantes a
  drift de corpus (WARN si en top-2k).
- **`hermes_stack_check.py` — validación de configuración en un comando**:
  embed (modelo/device/dims/prompt), hub health + backend gpu + escala,
  contratos del hook, log schema, state.db RO, digest, recap, crons.
  `PASS: 17/18` medido al deploy (WARN = hub viejo sin fa1e528, se
  resuelve con build+restart).

### Triage cuando la memoria parece muerta (en orden)

1. `curl :8860/health` — ¿embed vivo? (el ensure */10 lo relanza)
2. `hermes_stack_check.py` — localiza el check rojo
3. hook log: ¿líneas con backend≠gpu o drops>0?
4. digest: ¿los turnos recientes dan "sin veredicto" masivo? → match de
   timestamps roto o hook sin logging
5. `hermes-hub.log`: último ciclo — ¿smoke 8/8?

### Decisiones falsificadas/canceladas (no reintentar sin protocolo nuevo)

- Gate anti system-turns: REFUTADO (usan memoria MÁS que las queries:
  16.7% vs 7.7%).
- A/B top-1: CANCELADO (hits 2-3 con 15 usos únicos vs 6 del top con
  filtro DF — top-1 perdería ~71% del uso detectado).
- P2 re-recall (AUC 0.43) y P3 tool-lift (0.48): falsificados.
- P1 eco-cosine (AUC 0.78, IC ancho): en telemetría; re-eval con n≥300
  del log antes de usarlo como weight.
- KEV gate como filtro: desmontado (solapamiento en los datos).

## Operación (runbook)

```bash
# ciclo completo diario (sesiones + re-ingesta de código + merges + restarts + smoke)
bash /mnt/d/splatdb/semantic-hub/hermes-cycle.sh

# ensure idempotente AMBOS hubs (crash-recovery; cron cada 10 min)
bash /mnt/d/splatdb/semantic-hub/hermes-hub-ensure.sh

# smoke end-to-end (health x3 + consistencia x2 + 3 recalls sesión + 1 de código)
python3 /mnt/d/splatdb/semantic-hub/hermes_smoke.py

# ingesta manual / rebuild desde cero
python3 /mnt/d/splatdb/semantic-hub/hermes_ingest.py [--rebuild]
python3 /mnt/d/splatdb/semantic-hub/hermes_merge.py --parts hermes

# código: re-ingesta de los 14 repos + merge del hub de código
python3 /mnt/d/splatdb/semantic-hub/ingest.py --hub /mnt/d/splatdb/hermes-hub-code \
  --name repos --kind repo --cap 500 --repo /mnt/d/splatdb --repo /mnt/d/vuupta ...
python3 /mnt/d/splatdb/semantic-hub/hermes_merge.py --hub /mnt/d/splatdb/hermes-hub-code --parts repos
```

Cron del usuario (patrón existente de la máquina — sin sudo, sin systemd):

```
*/10 * * * * bash /mnt/d/splatdb/semantic-hub/hermes-hub-ensure.sh
15 9 * * *   bash /mnt/d/splatdb/semantic-hub/hermes-cycle.sh
```

Logs: `/mnt/d/WSL/orch/hermes-hub.log`.

### Lección operativa que motivó el watchdog

El server de embeddings estuvo caído 10 días (26-oct → 5-nov-2026) por un
venv roto mientras un trigger diario lo intentaba en silencio: el fallo solo
dejaba una línea en el log y ningún humano la leía. El ensure cada 10 min +
el ciclo diario con smoke que escribe PASS/FAIL al log existen para que eso
no se repita. Si algo falla, `hermes-cycle.sh` aborta **antes** del merge —
el hub sigue sirviendo el índice de ayer en vez de romperse.

## Gate de decisión (KEV / System One)

**Estado: NO shippable como filtro — server disponible para experimentos.**

Calibración con 48 mensajes reales del usuario (state.db): las distribuciones
de p(sí) se solapan — basura real 0.09-0.41 (mediana 0.23), mensajes de
proyecto 0.10-0.80 (mediana 0.38). Ningún threshold filtra basura sin
perder reales (th=0.12 filtra ~20% de la basura y pierde reales; th=0.40
filtra 18/21 pero pierde 14/27). La mitad de la "basura" son comandos
("Apaga la PC") con etiqueta genuinamente ambigua para la pregunta `noul
"¿recuperar memoria mejora la respuesta?"`. El 8/8 inicial fue overfit a
extremos elegidos a mano. El hook NO consulta el gate (fail-open total);
los tiers de distancia + FTS5 siguen siendo el único criterio.

Infraestructura desmontada del stack (9b61d0d, a pedido del usuario tras
la calibración): servicio muerto, autostart fuera del ensure, pidfile
borrada. El clon /mnt/d/kev y los caches HF quedan en disco. Para
reactivar: `cd /mnt/d/kev && HF_HOME=/mnt/d/huggingface
KEV_CUDA_GRAPHS=0 KEV_PREFIX_CACHE=0 KEV_DEVICE_PROBE_TIMEOUT=180
nohup uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b
--port 8008 --device cuda &` (requiere `flash-linear-attention` en el
venv — sin él el load no termina). Set de calibración en scratch
(`kev_calib_set.json`). El 0.8B no sirve ni como experimento (1/4).
Camino futuro: reformular la pregunta (comandos vs preguntas),
recolectar p en log-only, o Clef-Flash 9B si hay VRAM.

## Señales nativas para filtrado (auditoría 5-nov-2026)

Pregunta: ¿alguna función existente de splatdb filtra queries-basura en ms?
Medido sobre 48 mensajes reales del usuario (labels ruidosos documentados):

- **Confidence espacial (κ)**: el campo `confidence` de /search NUNCA
  existió en los hubs — requiere `field/` persistido en el data-dir
  (`field_cmds::persist_field`, invocado solo por el comando CLI `ingest`,
  NO por `/store` HTTP ni por `index`). El pipeline hermes nunca lo corrió
  → `state.field = None` → `enrich_topics` early-return. **No es un bug**:
  el código funciona (verificado: `ingest` CLI sobre copia del hub →
  field/ creado → confidence poblado). Pero **no separa**: junk
  0.012-0.063 vs reales 0.003-0.080, solapados (75 splats para 5.638
  chunks = celdas demasiado gruesas para OOD). Consistente con A5.
- **BM25 híbrido**: el módulo existe y está cableado en `semantic_memory`
  (fusión weight 0.6), pero el serve HTTP NO lo expone en /search — solo
  vive dentro del módulo de memoria MCP. Wire-able con un cambio en
  api_server.rs; el equivalente léxico ya existe en el hook (FTS5) y el
  conteo léxico tampoco separa (comandos reales comparten vocabulario
  técnico con el corpus).
- **Distancia vectorial ≤ 0.25**: ÚNICA señal nativa limpia — 0/21 falsos
  positivos, captura 16/27 reales. Es un aceptador positivo (fast-accept,
  ahorra FTS en el caso fuerte), no un filtro de basura.
- **`?rescore=gaussian`**: expuesto pero destruye el ranking en 768d
  (comentario propio en api_server.rs + A5) y no setea confidence.
  No usar para ranking ni para gating.

Veredicto: no hay señal nativa que filtre basura sin coste; la ambigüedad
está en los datos (mitad de la "basura" son continuaciones de trabajo).
El hook mantiene fast-path + tiers + FTS5.

## Limitaciones conocidas

- Mensajes con contenido thin (blobs JSON de health-checks) no son
  recuperables por significado — caso fallado del benchmark.
- La ingesta es append-only: un mensaje borrado/editado en state.db no
  desaparece del índice hasta el siguiente `--rebuild`.
- El restart diario del hub (~2 s) corta búsquedas en vuelo: está agendado
  a las 06:15 Argentina a propósito.
- Similitudes absolutas altas en embeddinggemma: no usar `--min-score` como
  filtro de calidad sin calibrar antes contra casos propios.

## Privacidad

El corpus son conversaciones privadas: vive en `hermes-hub/` (gitignored,
nunca commitea) y en la máquina local. Los tests usan datos sintéticos.
El CI no requiere GPU ni servicios: los tests mockean el server de embeddings.
