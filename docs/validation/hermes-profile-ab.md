# A/B final del perfil Hermes

## Veredicto

**GO_NO_REGRESSION.** El perfil Hermes conserva exactamente los rankings y mejora las cuatro medidas agregadas frente a `default` con la misma capacidad de 32.768 splats. No se afirma que sea óptimo universalmente: el veredicto aplica a este corpus, binario, hardware y configuración.

## Diseño predeclarado

- Binario: `splatsdb 2.11.0`.
- SHA-256: `2e6feb0f35b5d10b21e6f16fb387cd0889481058e96ef67599505a179554a94d`.
- Tres brazos: `default` con 100.000 splats, `default` con 32.768 y `hermes` con 32.768.
- Tres rondas por brazo: nueve servidores, 100 consultas top-30 por corrida, 900 consultas totales.
- Backend GPU, dimensión 768, trainer e-means, puerto aislado 18872.
- Aceptación: paridad de rankings en las nueve corridas; mediana Hermes p50 ≤110% de default, arranque ≤125% y RSS ≤105%.
- Se descartó el A/B anterior porque usó un binario anterior al perfil definitivo.

## Resultados

| Medida — mediana de 3 rondas | Hermes 32.768 | Default 32.768 | Cambio |
|---|---:|---:|---:|
| Arranque | 0,95017 s | 2,41546 s | **−60,66%** |
| p50 | 3,77405 ms | 3,94899 ms | **−4,43%** |
| p95 | 4,25914 ms | 4,32133 ms | **−1,44%** |
| RSS HWM | 136.064 KiB | 139.164 KiB | **−2,23%** |

- Paridad de rankings: **9/9**, SHA `9f144db3981a1eb400154d32735dbe39b0f8abba7360178f9c735265bb2b0cd9`.
- Todas las corridas usaron el mismo SHA de binario y completaron 100/100 consultas.
- Los cuatro umbrales de no-regresión se cumplen con margen.
- `preset-info --preset hermes` confirma `has_hnsw: true` y quantization/LSH/graph desactivados.

## Receipts and reproducibility

The nine-run report, per-run logs, historical A/B and harness are retained locally and are not included in the public repository. The published tables contain all values needed to audit the verdict without raw runtime data.
