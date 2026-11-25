# Protocolo independiente M3 — predeclaración

Timestamp de predeclaración: 2026-11-25T20:31:54Z.
Estado: REGLAS FIJADAS ANTES DE CUALQUIER SALIDA NUEVA.

## 1. Independencia y alcance

- Evalúa la adaptación local de `m3-glm-protocol` @ `15795ca` con GLM-5.4 y GLM-5.3-Flash.
- La corrida anterior queda intacta y conserva su veredicto NO_GO. Este protocolo no la reinterpreta ni combina sus resultados.
- Directorio nuevo: `m3_eval_rerun/`. Se prohibe leer como entradas del cálculo `m3_eval/results.jsonl`, `m3_eval/calls.jsonl`, `m3_eval/full_context_tokens.jsonl` o sus métricas.
- Se reutilizan únicamente los artefactos fijos: preguntas LongMemEval-S cleaned, haystacks SQLite, embeddings y consultas ya construidas.
- Se ejecutarán 500 preguntas completas, ambas modelos y una baseline full-context nueva. No hay parada temprana, sustitución de preguntas ni selección de subconjuntos.
- Dos workers, orden fijo por qid. Nuevas llamadas con nuevos request IDs; no se reutilizan respuestas, juicios ni tokens.

## 2. Procedimiento fijo

- Recuperación, prompts, top-5 inicial, forced-commit, escalera, knowledge limits, `temperature=0.2`, topes M3 700/1000 y parser del repositorio sin cambios.
- Embeddings `embeddinggemma-2` 768D, `prompt_name=Retrieval-query`; SQLite/FTS5 por pregunta.
- Adaptaciones de API fijas desde el inicio: `thinking={"type":"disabled"}` para ambos modelos; tope del juez 64 tokens. Jueces cruzados con `JUDGE_PROMPT` del repositorio.
- Baseline full-context: mismo corpus y modelo, `FINAL_PROMPT` con todos los bloques, `max_tokens=1`; sólo se usa el conteo real de prompt tokens. Salida vacía es aceptable únicamente en esta etapa.
- Reintentos sólo para fallas transportistas 408/409/429/5xx/timeouts y respuestas vacías persistentes en generación/juez, máximo cinco intentos con backoff. Un content filter HTTP 400 código 1301 no se reintenta ni se elude cambiando prompt/modelo.
- Ningún prompt, parámetro, umbral, subset o regla puede modificarse después de observar salidas del rerun.

## 3. Clasificación de fallas

`provider_filtered` exige HTTP 400 con `contentFilter` y código `1301` del proveedor. Se registra por qid, modelo y etapa; se cuenta una vez por llamada lógica, no por reintentos transportistas.

Cualquier otro error es `nonfilter_error` y debe quedar resuelto antes del GO. No se imputan respuestas, juicios ni tokens.

## 4. Desenlace primario

Para cada modelo, razón entre:

- mediana del total de prompt tokens de todas las generaciones M3 por pregunta; y
- mediana de prompt tokens de la baseline full-context por pregunta.

GO primario exige razón de medianas ≤0,50 en AMBOS modelos y al menos **495/500 pares completos por modelo**. También se reportan la mayor prompt M3 por pregunta y la mediana de razones por pregunta, sin imputación.

## 5. Condiciones de validez y no-regresión

- Exactitud simétrica pareada: GLM-5.4 menos GLM-5.3-Flash ≥−0,02 sobre al menos 495 pares.
- Acuerdo entre jueces ≥0,90 y κ≥0,70 para cada answerer, con al menos 495 juicios pareados por answerer.
- Inválidos no derivados de content filter <1% por etapa.
- Content filters: ≤5/500 respuestas M3 por modelo; ≤5/500 llamadas baseline por modelo; ≤20/2000 juicios planificados en total.
- Cero errores no filtrados sin resolver. Filtrados de jueces reducen cobertura y no se convierten en etiquetas válidas.
- Sensibilidad intent-to-treat: sobre las 500 preguntas, una respuesta M3 faltante por filtro cuenta como incorrecta para ese modelo; una pregunta sin etiqueta válida por filtro de juez cuenta como incorrecta para el answerer afectado. La diferencia ITT también debe ser ≥−0,02. Tokens nunca se imputan.

## 6. Análisis y decisión

- Reportar IC de Wilson para exactitudes, McNemar para la comparación pareada, discordancias por qid, fases/rondas y uso real de tokens.
- No se afirma superioridad estadística a partir de una diferencia puntual.
- GO sólo si se cumplen simultáneamente cobertura, contexto, no-regresión, validez, filtros y sensibilidad ITT.
- Si alguna condición falla: NO_GO, con bloqueos y receipts. No hay GO parcial ni conversión posterior de la corrida anterior.
- El veredicto aplica exclusivamente a esta nueva corrida y a este corpus/configuración/proveedor.
