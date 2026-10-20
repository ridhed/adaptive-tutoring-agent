# V0 vs V1: 40-case comparison

## Method

The comparison script sampled 40 interaction rows from `dataset/ground_truth.csv`, with 10 rows for each generated `policy_action` (`ANSWER`, `ASK`, `HINT`, and `TEACH_PRIOR`). Sampling used seed 42. Student identifiers were not copied into the comparison file. V0 used the input fields and FAST/SLOW latency conversion from its preprocessing rules. V1 used the row's pre-action mastery and prior error count.

Run again from the repository root:

```powershell
python analysis\compare_v0_v1_cases.py
```

Outputs:

- `analysis/v0_v1_case_comparison.csv`: the 40 cases, each agent's output, and the generated reference action.
- `analysis/v0_v1_case_summary.json`: agreement counts, rates, and action balance.

## Results

| Measure | Result |
| --- | ---: |
| V0 matches generated reference actions | 13 / 40 (32.5%) |
| V1 matches generated reference actions | 40 / 40 (100%) |
| V0 and V1 choose the same action | 13 / 40 (32.5%) |

## Interpretation

`policy_action` in `ground_truth.csv` is produced by the dataset builder using the same mastery thresholds and recent-error rule implemented by `V1/V1_bkt_agent.py`. V1's 40/40 match is therefore an implementation-consistency check by design, not an independent accuracy result. The 13/40 V0 match rate shows how often the simpler correctness/attempt/latency rules agree with the builder's reference policy on this sample. Neither result establishes that one agent improves learning more than the other. Review individual disagreements as the next step; action correctness requires an independently defined and reviewed educational rubric.