# Reproducible ASSISTments dataset construction

This folder turns an ASSISTments interaction export into a chronological, skill-level dataset for the Adaptive Tutoring Agent. It is a local, reproducible recipe; no source dataset is included because ASSISTments data have their own access and redistribution terms.

## What the labels mean

The generated rows deliberately separate three things:

* `correct` is the observed outcome in ASSISTments (1 or 0). It is not a mastery label.
* `mastery_pre` and `mastery_post` are BKT model estimates. They are soft, model-based reference values, not observed ground truth about a student's mind. `mastery_pre` uses only prior interactions for that student and skill. `mastery_post` incorporates this row's response and learning transition.
* `policy_action` is a deterministic teaching-policy recommendation generated using only pre-interaction mastery and prior history. It is not an expert annotation and is not inferred from this row's correctness. It is therefore suitable for policy imitation experiments, not a claim that the action helped.

## Input and provenance

Use an authorized ASSISTments 2009-2010 Skill Builder export (the source material references Assistment2009, DOI [10.57760/sciencedb.j00133.00253](https://doi.org/10.57760/sciencedb.j00133.00253)). Keep the untouched file outside version control and record its exact release, download date, source URL/DOI, and checksum in your experiment log. The generator accepts CSV columns corresponding to `user_id`, `problem_id`, `skill_id`, `correct`, and optionally `order_id` or `timestamp`, `ms_first_response`, `hint_count`, and `attempt_count`. Column names can be changed with a JSON config. Rows without student ID, skill ID, or binary correctness are skipped and counted; data are never silently assigned an outcome.

When a row has no timestamp/order field, input file order is used and this fact is recorded in the output metadata. For multi-skill rows, the first skill ID is used by default; configure a delimiter if the export encodes multiple tags. This is a simplification and should be reported.

## BKT parameters and sequential update

The default parameters match the values documented in `sources/V1 -  Bayesian Knowledge Tracing.md`: initial mastery 0.591, guess 0.245, slip 0.116, learning 0.152, forgetting 0.0. They are starting values from the existing project notes, not validated universal ASSISTments parameters. Override them in a JSON config. The pipeline does not claim to reproduce StanBKT: the referenced project notes cite Pradhan et al. (2026), but provide no paper-specific model, fitted parameters, or Stan code. This pipeline uses the standard binary BKT equations as transparent model-based reference labels. A StanBKT reimplementation or fitted parameter estimates should be added only after its exact model and fitting procedure are specified.

For each student-skill sequence, initialize with `p_initial_know`. Before each response, save `mastery_pre`. Compute the likelihood of the observed response given learned/not learned, apply Bayes' rule, then apply learning and forgetting transitions to obtain `mastery_post`. The next row's pre-state is that post-state. Sequences are independent across students and skills. Parameters are currently fixed/configured; there is no hidden fitting on the test set.

## Action policy

The simple, inspectable policy uses pre-action mastery and preceding history: below 0.40, `TEACH_PRIOR`; from 0.40 to below 0.85, `HINT`; at or above 0.85 with at least two recent prior errors, `ASK`; otherwise `ANSWER`. `ASK` checks uncertain or inconsistent understanding, `HINT` provides a small scaffold, `TEACH_PRIOR` addresses a likely prerequisite gap, and `ANSWER` confirms/provides a direct response when estimated mastery is high. Thresholds are configurable. BKT cannot diagnose a misconception or prerequisite gap by itself; these action names are operational policy labels.

## Splitting, leakage, and evaluation

The CLI assigns students, not rows, to train/validation/test using a seeded hash. A student's full interaction history belongs to one split, preventing identity and within-student sequence leakage across splits. BKT state is still generated chronologically within each student-skill sequence. If the research question is future performance for known students, use a separate forward-in-time evaluation design and describe it; do not mix that result with the default unseen-student split. Do not train a predictive model with `mastery_post`, since it incorporates the current outcome. `mastery_pre` and prior-history features are the appropriate pre-action inputs.

Report split sizes and student counts, class/action balance, parameter source, preprocessing settings, and dataset checksum. Evaluate outcome prediction with held-out log loss/Brier score/AUC where appropriate, and mastery calibration against later response evidence or an explicitly defined external criterion. Evaluate tutoring effectiveness with a controlled or randomized comparison of learning gains; agreement with `policy_action` only measures imitation of this rule. A high correctness rate alone does not establish that a policy caused learning.

## Usage

```sh
python dataset/build_dataset.py --input /path/to/skill_builder_data.csv --output /path/to/derived.csv --metadata /path/to/derived.metadata.json
```

Optional config:

```sh
python dataset/build_dataset.py --input raw.csv --output derived.csv --config dataset/config.example.json
```

The output is CSV. Metadata includes input/output row counts, skipped rows, order source, split seed/fractions, and BKT settings. Run the lightweight validation suite with `python -m unittest discover -s dataset/tests`.

## Limitations

ASSISTments interactions are observational and may contain retries, hints, selection effects, and inconsistent skill tags. Correctness is an imperfect measurement of knowledge; time and confidence are not available reliably for every row. Standard BKT assumes a binary latent state and simplified emission/transition probabilities. Current action labels are rule-generated, not human validated or causal. Dataset-derived mastery values must always be described as model estimates.
