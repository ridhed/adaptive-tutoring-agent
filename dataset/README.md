# ASSISTments BKT Reference Dataset

## Why this dataset was created

The ASSISTments export records student interactions, including whether each response was correct, but it does not contain a direct measurement of the student's latent mastery. V1 of the Adaptive Tutoring Agent needs a mastery estimate to reason about a student's knowledge over time. We therefore replayed the interactions through V1's Bayesian Knowledge Tracing (BKT) model and saved the resulting probabilities as **model-derived reference labels**. The project calls the file `ground_truth.csv`, but its mastery values are estimates, not independently observed ground truth.

The output keeps three kinds of labels separate:

- `correct`: the observed ASSISTments response (0 or 1).
- `mastery_pre` and `mastery_post`: BKT mastery probabilities before and after the current response.
- `policy_action`: an action assigned by the dataset builder's explicit rule. It is not an expert annotation and is not evidence that the action improved learning.

## How it was created

1. Start with the authorized ASSISTments 2009-2010 Skill Builder export, stored locally as `dataset/skill_builder_data.csv`.
2. `dataset/build_dataset.py` validates the required student, skill, and binary correctness fields. Rows with missing/invalid required values are counted and skipped. It uses `order_id`, then `timestamp`, then input row order to determine sequence when available. For rows tagged with multiple skills, the first tag is used only if a delimiter is configured.
3. Group interactions by student and skill, sort them chronologically, and initialize the BKT state with `p_initial_know`.
4. For each response, record the current belief as `mastery_pre`. BKT uses the response likelihoods from the guess and slip parameters, applies Bayes' rule, then applies the learning and forgetting transition to produce `mastery_post`. That posterior becomes the next interaction's pre-response state.
5. Compute `policy_action` from `mastery_pre` and the preceding three responses: below 0.40 gives `TEACH_PRIOR`; from 0.40 to below 0.85 gives `HINT`; at/above 0.85 with at least two errors among the preceding three gives `ASK`; otherwise `ANSWER`.
6. Assign each student, rather than each row, to train, validation, or test using a deterministic SHA-256 split with seed 17 and proportions 80/10/10. This keeps each student's records in one split.

## Parameters and provenance

The builder's default BKT parameters match the values documented for V1:

| Parameter | Value |
| --- | ---: |
| Initial mastery | 0.591 |
| Guess | 0.245 |
| Slip | 0.116 |
| Learning | 0.152 |
| Forgetting | 0.000 |

The project calculated these global starting values with `analysis/prior_prob_cal.py` from the ASSISTments data. That script uses first-opportunity correctness and simple response-history heuristics; the estimates should not be presented as universal or as per-skill fitted parameters. The exact parameters, policy thresholds, split settings, input checksum, and row counts are recorded in `ground_truth.csv.metadata.json`.

## Output schema

`ground_truth.csv` contains one row per valid student-skill interaction:

- Identifiers/order: `student_id`, `problem_id`, `kc_id`, `sequence_index`
- Observed response and BKT state: `correct`, `mastery_pre`, `mastery_post`
- History and policy: `prior_errors`, `prior_attempts`, `policy_action`
- Available interaction features: `response_time_seconds`, `hint_count`, `attempt_count`
- Evaluation partition: `split` (`train`, `validation`, or `test`)

Use `mastery_pre` for pre-action modeling. Do not use `mastery_post` as a feature for a decision made before the response; it incorporates the current response.

## Reproduce the dataset

From the repository root, run:

```powershell
python dataset\build_dataset.py --input dataset\skill_builder_data.csv --output dataset\ground_truth.csv
```

To write metadata to a specific path or change settings, use `--metadata` or a JSON file with `--config`. The raw export is about 83 MB and should remain local unless you intentionally choose another distribution method. The current generated CSV is about 34.6 MB; it is a derived artifact and can be recreated with the command above.

## Limitations and evaluation

These are BKT-derived pseudo-labels made by the same model used in V1, so comparing V1's mastery estimate with them checks consistency with the model, not independent mastery accuracy. `policy_action` is a simple rule-based recommendation and is not the V1 agent's full action logic in every case (for example, V1 can use confidence). Agreement with this column measures imitation of the builder's rule, not tutoring effectiveness.

ASSISTments data are observational. Correctness is an imperfect proxy for knowledge, BKT assumes a simplified binary latent state, and the global parameters may not fit every skill or learner. Evaluate response prediction on held-out observed outcomes and assess tutoring impact with learning gains in a controlled evaluation. Keep the source data's access and redistribution terms in mind.