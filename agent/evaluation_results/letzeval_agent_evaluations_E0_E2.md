# LëtzEval BAF Agent — Evaluation Results (E0–E2)

**Report date:** 5 November 2026  
**Scope:** 50-case agent evaluation suite; additional independent 15-case holdout evaluated for E1.  
**Status:** E0 and E1 reported; E2 OpenAI run manually reviewed. No E2 classical or Ollama results supplied.

## 1. Executive summary

The latest **E2 OpenAI** run achieved **46/50 correct (92%) after manual review**, with **2 partial (4%)**, **2 wrong (4%)**, no timeouts or errors, and an **average response latency of 11.17 seconds**. Its initial automated scoring was **30/50 correct (60%)**, **18 partial**, and **2 wrong**. Manual review reclassified 16 partial responses as correct.

These are encouraging results, but they should **not** be presented as proof that changing the OpenAI model alone improved accuracy. E2 also followed changes to the routing/interpretation implementation, and the reported E0/E1 scores have not been subjected to the same documented manual review as E2.

## 2. Experiment definitions

| Experiment | Implementation | Modes evaluated | Test cases | Outcome |
|---|---|---|---:|---|
| **E0** | Original agent implementation | Classical BAF, Ollama `llama3.1`, OpenAI `gpt-4o-mini` | 50 per mode | Baseline |
| **E1** | Refactored query-aware candidate retrieval and LLM interpretation | Classical BAF, Ollama `llama3.1:latest`, OpenAI `gpt-4o-mini` | 50 per mode | Post-refactor comparison |
| **E1 holdout** | E1 implementation on independent questions | Classical BAF, Ollama, OpenAI `gpt-4o-mini` | 15 per mode | Generalisation check |
| **E2** | Later agent implementation and OpenAI model change | OpenAI **GPT-5.1** (model identifier to verify; see note below) | 50 | Automated scoring plus manual review |

**Model identification note:** The latest user-provided description identifies E2 as **GPT-5.1**. Earlier terminal output in this conversation showed `LETZEVAL_LLM_MODEL=gpt-5.1` and an API response using `gpt-5.1-2025-11-13`. Before publication, verify the model actually used by the E2 evaluation from its recorded container environment or telemetry; this report does not assume these names are interchangeable.

## 3. Full 50-case evaluation results

### 3.1 Correctness summary

| Experiment | Mode / model | Correct | Partial | Wrong | Timeout | Error |
|---|---|---:|---:|---:|---:|---:|
| E0 | Classical BAF | 15/50 (**30%**) | 24 | 11 | 0 | 0 |
| E0 | Ollama `llama3.1` | 21/50 (**42%**) | 21 | 8 | 0 | 0 |
| E0 | OpenAI `gpt-4o-mini` | 16/50 (**32%**) | 22 | 12 | 0 | 0 |
| E1 | Classical BAF | 14/50 (**28%**) | 24 | 12 | 0 | 0 |
| E1 | Ollama `llama3.1:latest` | 28/50 (**56%**) | 18 | 4 | 0 | 0 |
| E1 | OpenAI `gpt-4o-mini` | 19/50 (**38%**) | 23 | 8 | 0 | 0 |
| E2 | OpenAI GPT-5.1 — **automatic scorer** | 30/50 (**60%**) | 18 | 2 | 0 | 0 |
| E2 | OpenAI GPT-5.1 — **after manual review** | 46/50 (**92%**) | 2 | 2 | 0 | 0 |

**Interpretation:** All rows represent 50 evaluation cases; the two E2 rows are *the same run scored two ways*, not two independent runs. E0/E1 values reflect the reported evaluation classifications; E2 reviewed values explicitly incorporate human judgement. Therefore, comparisons across E0/E1 versus E2 should be marked *descriptive, not strictly like-for-like*.

### 3.2 Latency information

| Experiment | Mode | Reported latency | Statistic |
|---|---|---:|---|
| E0 | Classical BAF | ~3.4 s | Mean (approximate; originally reported) |
| E0 | Ollama `llama3.1` | ~31.4 s | Mean (approximate; originally reported) |
| E0 | OpenAI `gpt-4o-mini` | 9.8 s | Mean (approximate; originally reported) |
| E1 | Classical BAF | 4.52 s | Median |
| E1 | Ollama `llama3.1:latest` | 47.27 s | Median |
| E1 | OpenAI `gpt-4o-mini` | 9.54 s | Median |
| E2 | OpenAI GPT-5.1 | **11.17 s** | **Mean** |

**Caution:** Mean and median are different measures. Do not present 11.17 s vs. 9.54 s as a direct latency comparison without computing the same statistic for both runs.

### 3.3 Changes in correct cases

| Mode | E0 | E1 | E0 → E1 change | E2 reviewed |
|---|---:|---:|---:|---:|
| Classical BAF | 15 (30%) | 14 (28%) | −1 case (−2 pp) | Not run/reported |
| Ollama | 21 (42%) | 28 (56%) | +7 cases (+14 pp) | Not run/reported |
| OpenAI | 16 (32%) — `gpt-4o-mini` | 19 (38%) — `gpt-4o-mini` | +3 cases (+6 pp) | **46 (92%) — GPT-5.1, manually reviewed** |

The OpenAI E1-to-E2 reported correct count increased by **27 cases (+54 percentage points)**, but this difference combines *model selection, code changes, and scoring/review differences*. It cannot be assigned to a single cause from these results.

## 4. E2 detailed outcome (OpenAI GPT-5.1)

### 4.1 Automated versus reviewed scores

| Outcome | Initial automated result | After manual review | Change |
|---|---:|---:|---:|
| Correct | 30 (60%) | **46 (92%)** | **+16** |
| Partial | 18 (36%) | **2 (4%)** | **−16** |
| Wrong | 2 (4%) | **2 (4%)** | 0 |
| Timeout | 0 | 0 | 0 |
| Error | 0 | 0 | 0 |
| **Total** | **50** | **50** | — |

The difference highlights the limitations of substring-based automated scoring: **16 responses initially classified as partial were judged correct on review**. Keep the original and reviewed CSVs separately for reproducibility.

### 4.2 Remaining wrong cases

| Case | User query | Observed agent response | Issue indicated |
|---|---|---|---|
| **22** | “Who scores highest overall?” | “No recorded results found for **All Total**.” | Incorrect/default metric selection for an overall-ranking request; inspect whether **Macro Average** should be used instead of `All Total` |
| **38** | “What does CEFR stand for?” | Begins: “This leaderboard presents the performance of large language models (LLMs) on the…” | Routed to a generic leaderboard explanation instead of defining the CEFR acronym |

The excerpt supplied for case 38 is truncated, so the full response has not been independently checked here. The supplied reviewed verdict is **Wrong**.

### 4.3 Unresolved partial results

Two E2 responses remain **Partial** after manual review. Their case identifiers and responses were not provided, so they cannot be characterized in this report.

## 5. Independent holdout results (15 cases, E1)

| Mode | Correct | Partial | Wrong |
|---|---:|---:|---:|
| Classical BAF | 5/15 (**33%**) | 8 | 2 |
| Ollama `llama3.1:latest` | 5/15 (**33%**) | 9 | 1 |
| OpenAI `gpt-4o-mini` | 4/15 (**27%**) | 9 | 2 |

**Important limitation:** Many holdout cases lacked a machine-verifiable expected-answer substring, so correct numerical answers could still be labelled **Partial**. These figures should not be presented as definitive model accuracy. An **E2 holdout run has not been supplied**.

## 6. Findings and next steps

1. **Address E2 case 22:** Validate the canonical metric mapping for “overall/highest” rankings. If `Macro Average` is the supported overall measure, ensure the LLM route and ranking handler use it, rather than querying a nonexistent `All Total` combination.
2. **Address E2 case 38:** Route acronym-definition requests such as “What does CEFR stand for?” to the metric/CEFR explanation state, with a direct definition rather than generic leaderboard prose.
3. **Review the two outstanding Partial E2 cases:** Assign final verdicts and document the rationale.
4. **Standardize scoring:** Apply the same manual-review rubric to E0/E1 outputs, or clearly publish automated and reviewed results in separate columns. Do not silently compare them as equivalent.
5. **Verify reproducibility:** Record the E2 commit SHA, database hash, exact OpenAI model identifier, frozen test-suite hash, agent mode, router prompt version, runtime parameters and date/time.
6. **Re-run the independent holdout for E2:** Report it separately from the previously used 50 cases, especially if the agent has been tuned on those questions.
7. **Report latency consistently:** Recalculate mean, median and preferably p95 from each run's per-case CSV, using the same methodology.

## 7. Data provenance and constraints

- E0/E1 aggregate values and E1 holdout values: figures supplied in the project conversation and the repository's `agent/docs/E0_E1_COMPARISON.md`.
- E2 aggregate results, latency and failed-case excerpts: supplied by the user from `E2/openai_gpt51_E2_reviewed.csv` summary; this CSV itself was not provided for independent inspection in this report.
- The report includes **no invented test-case responses, numerical scores, or latency measurements** beyond those supplied.
- **Correct / Partial / Wrong** reflect the experiment's scoring method and reviewer decisions; manual review of E2 does not establish that E0/E1 have been reviewed in the same way.

---

*Prepared 5 November 2026. Update the E2 model identifier and provenance metadata before using this report in a paper, presentation or media communication.*
