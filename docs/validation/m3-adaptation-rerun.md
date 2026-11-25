# Adapted M3 evaluation — independent predeclared rerun

**Verdict: GO under the new predeclared protocol.** Protocol SHA-256:
`c6973edb93af2cdca2bab76a93418fc8f57892fab694fea3de176573087f4bf3`.

This is a new evaluation of the local adaptation of `m3-glm-protocol` @ `15795ca`, measured with GLM-5.4 and GLM-5.3-Flash. The exact [predeclared protocol](m3-adaptation-rerun-protocol.md) is included. This does not reinterpret the earlier run: that run remains **NO_GO under its stricter 500/500 coverage rule**.

## Independence and fixed procedure

- Five hundred LongMemEval-S cleaned questions; both models; a new full-context baseline.
- New response, judge and token receipts. The previous calls and metrics were excluded from computation. Only the fixed corpus, SQLite haystacks and embeddings were reused.
- Retrieval, prompts, top-5, escalation ladder, forced commit, `temperature=0.2`, generation caps 700/1000 and parser fixed before execution.
- `thinking={"type":"disabled"}` and crossed judges capped at 64 tokens were declared before the run.
- Provider code 1301 was recorded as `provider_filtered`; no prompt/model substitution, no imputation and no replacement questions.
- Protocol and runner hashes were verified before execution and after analysis. Raw JSONL was independently recomputed by `verify_m3_rerun.py`; all assertions passed.

## Primary outcome: context reduction

| Model | Complete token pairs | Baseline median | M3 median | Ratio of medians | Reduction | Paired-only ratio check |
|---|---:|---:|---:|---:|---:|---:|
| GLM-5.4 | 498/500 | 70,241 | 4,384.5 | 0.0624208 | 93.7579% | 0.0624221 |
| GLM-5.3-Flash | 498/500 | 70,241 | 4,383.5 | 0.0624066 | 93.7593% | 0.0624079 |

The report's baseline median uses its 499 completed calls; M3 uses the 498 paired questions. A second calculation using only the common 498 questions gives baseline median 70,239.5 and the paired-only ratios shown above. Both definitions pass the predeclared ≤0.50 threshold; neither involves imputation.

## Accuracy and non-regression

- Symmetric crossed-judge accuracy: **GLM-5.4 81.4629%, GLM-5.3-Flash 80.1603%** over 499 scored questions.
- Observed paired difference: **+1.3026 percentage points** for GLM-5.4. McNemar discordance: 30 positive, 21 negative; **p=0.2624**. No superiority claim is made.
- Intent-to-treat sensitivity over all 500 questions: **81.30% vs 80.00%**, delta **+1.30 pp**; missing/filtered labels count as incorrect. This gate also passes.
- Judge-specific accuracy and 95% Wilson intervals:
  - GLM-5.4 judged by GLM-5.4: 407/499, 81.5631% [77.9240%, 84.7200%].
  - GLM-5.4 judged by GLM-5.3-Flash: 406/499, 81.3627% [77.7111%, 84.5351%].
  - GLM-5.3-Flash judged by GLM-5.4: 401/499, 80.3607% [76.6487%, 83.6089%].
  - GLM-5.3-Flash judged by GLM-5.3-Flash: 399/499, 79.9599% [76.2248%, 83.2373%].

## Validity, filters and usage

- Judge agreement: **98.1964% / 97.5952%**; Cohen's κ **0.9403 / 0.9244**.
- Invalid judged answers: **2/1000 per model (0.2%)**, below 1%; both are the filtered M3 question.
- Coverage: **498/500 token pairs** and **499/500 scored answers**, above the predeclared 495 minimum.
- Filters: `001be529` in both M3 answer calls; `gpt4_7f6b06db` in both baseline calls. One per stage per model, within the ≤5 budget; zero judge filters; **zero non-filter errors**.
- Generation usage: GLM-5.4 517 calls, 2,226,273 prompt + 57,162 completion tokens; Flash 531 calls, 2,302,839 prompt + 66,014 completion tokens.
- Crossed-judge usage: 998 calls per model, 225,823 prompt tokens per model; 6,192 / 4,495 completion tokens.

## Interpretation and limits

This GO applies only to this corpus, configuration, provider and protocol. LLM judges are not human ground truth; LongMemEval accuracy is not production task success. Prompt-token reduction is not an equal reduction in total workflow cost: retrieval, embeddings, judge calls, output tokens and cache pricing are separate. The result does not prove exhaustive recall or remove the need for escalation to full context.

## Receipts and reproducibility

Local raw receipts: `m3_eval_rerun/{PROTOCOL.md,PROTOCOL.sha256,RUNNER.sha256,results.jsonl,calls.jsonl,full_context_tokens.jsonl,REPORT.json,VERIFICATION.json}`. Reproduce analysis with `run_m3_eval_rerun.py metrics` and `verify_m3_eval_rerun.py`. The public document and chart contain measured values only; raw logs remain local.
