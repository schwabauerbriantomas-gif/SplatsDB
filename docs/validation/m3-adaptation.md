# Adaptación M3 evaluada con GLM-5.4 y GLM-5.3-Flash

## Veredicto predeclarado

**NO_GO por cobertura, no por magnitud observada.**

La reducción de contexto y la no-regresión de exactitud superan holgadamente los umbrales numéricos, pero el gate exigía 500 pares completos y ausencia de errores. Hay 498 pares para la comparación de tokens y una pregunta M3 sin respuesta por filtrado del proveedor. No se relajan los criterios ni se imputan valores después de observar los resultados.

## Diseño

- Fuente de verdad: repositorio privado `m3-glm-protocol` @ `15795ca`.
- Corpus: LongMemEval-S cleaned, 500 preguntas; 23.867 sesiones y 500 consultas.
- Recuperación y prompts sin cambios: top-5 inicial, acciones M3, forced-commit, escalera por rechazo, `temperature=0.2`.
- Embeddings: `embeddinggemma-2`, 768D, `prompt_name=Retrieval-query`; SQLite/FTS5 por pregunta.
- Modelos: GLM-5.4 y GLM-5.3-Flash; jueces cruzados con `JUDGE_PROMPT` del repositorio.
- Baseline: mismo corpus y modelo, `FINAL_PROMPT` con todos los bloques, `max_tokens=1`; sólo se usa el conteo real de tokens de entrada.
- Adaptaciones de transporte documentadas antes de la corrida reparada: `thinking={"type":"disabled"}` y tope del juez 64 tokens porque el proveedor puede emitir razonamiento aun con thinking desactivado. Los topes de generación M3 siguen siendo 700/1000.

## Resultados medidos

### Reducción de contexto

| Modelo | Baseline full-context, mediana | M3, mediana total por pregunta | Razón de medianas | Reducción | Pares |
|---|---:|---:|---:|---:|---:|
| GLM-5.4 | 70.241 tokens | 4.376,5 tokens | 0,0623069 | **93,77%** | 498/500 |
| GLM-5.3-Flash | 70.241 tokens | 4.378,5 tokens | 0,0623354 | **93,77%** | 498/500 |

La mediana de la razón por pregunta es 0,0617037 —93,83% de reducción— para ambos modelos. El mayor prompt M3 tiene la misma mediana que el total porque la enorme mayoría de las preguntas se resuelve en la fase inicial.

El umbral de ≤50% se cumple en magnitud. El subgate formal no pasa porque exige 500 pares, no 498.

### Exactitud y jueces

| Métrica | GLM-5.4 | GLM-5.3-Flash |
|---|---:|---:|
| Exactitud simétrica | **81,46%** | **79,56%** |
| Juez GLM-5.4 | 80,76% | 79,36% |
| Juez GLM-5.3-Flash | 82,16% | 79,76% |
| Acuerdo entre jueces | 95,39% | 98,40% |
| κ de Cohen | 0,8474 | 0,9507 |
| Respuestas inválidas | 1/500 | 1/500 |
| Juicios inválidos | 2/1000 | 2/1000 |

- Diferencia simétrica GLM-5.4 menos Flash: **+1,90 puntos porcentuales** sobre 499 pares.
- McNemar: 36 mejoras, 23 regresiones, **p=0,1175**. No se afirma superioridad estadística; sí se observa que la diferencia puntual queda por encima del margen de no-regresión de −2 pp.
- Acuerdo ≥90%, κ≥0,70 e inválidos <1%: condiciones numéricas cumplidas.

### Recall determinista de recuperación

Para las 500 preguntas: **97,0% top-3, 98,6% top-5 y 99,6% top-10**. Es recall por pregunta y no se mezcla con el recall global session-level de experimentos anteriores.

## Casos que impiden el GO

- `001be529`: content filter del proveedor, código 1301, en la generación M3 de ambos modelos. No hay respuesta ni juicios válidos.
- `gpt4_7f6b06db`: la baseline full-context es rechazada por content filter para ambos modelos; M3 sí produjo respuestas y juicios válidos.

Por eso hay 499 preguntas con exactitud evaluable y 498 pares completos para tokens. Los dos errores de la corrida M3 corresponden al primer caso.

## Receipts and reproducibility

The raw call logs, per-question results, baseline receipts, `REPORT.json` and the predeclared brief are retained locally and are not included in the public repository. The published chart is generated exclusively from the measured report; no raw conversations or credentials are published.

This report does not certify a reproduction of the paper or universal model equivalence. It evaluates the local adaptation on this corpus, configuration and provider.
