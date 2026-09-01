### Comprehensive Probability Decision Records (Scenarios A – D)

### **Scenario A: Correct Answer with High Confidence**

**The Situation:** An adaptive tutoring agent evaluates an average student baseline ($P(L_{t-1}) = 0.591$) who attempts a skill component. The hidden state is uncertain until evidence is gathered.  

| **Item**      | **Content**                                                                                                     |
| ------------- | --------------------------------------------------------------------------------------------------------------- |
| Evidence      | Student answered correctly (`correct=True`) and reported high confidence (`confidence=HIGH`)                    |
| Hidden states | (a) Mastered ($P(L_t) \ge 0.85$), (b) Uncertain ($0.40 < P(L_t) < 0.85$), (c) Knowledge Gap ($P(L_t) \le 0.40$) |
| Beliefs       | P(Mastered) = 0.591 (59.1%), P(Not Mastered) = 0.409 (40.9%) — sums to 100%                                     |
| Event         | True student mastery state of the target knowledge component                                                    |
| Actions       | `ANSWER`, `ASK`, `HINT`, `TEACH_PRIOR`                                                                          |
| Costs         | Advancing when unready causes frustration; unnecessary review wastes time                                       |
| Policy        | If posterior $\ge 0.85$ and Correct + High Confidence $\rightarrow$ `ANSWER`                                    |
| Decision      | Pending live evidence update                                                                                    |
| Audit data    | Timestamp: 2026-06-01, Data Version: Assistments2009, Model Version: v1-bkt                                     |

**Step-by-Step Update:**

1. **Prior probability:** $P(L_{t-1}) = 0.591$.
2. **New evidence:** Correct answer, High Confidence.
3. **Likelihood estimation:** Mastery likelihood = $0.884$, Non-Mastery likelihood = $0.245$, Total $P(Obs) = 0.6226$.
4. **Posterior calculation:** Bayes posterior = $0.8391$; Final posterior after learning transition ($P(T)=0.152$) = **$0.8635$**.
5. **Threshold comparison:** $0.8635 \ge 0.85$ mastery threshold.
6. **New action:** **`ANSWER`**.
### **Scenario B: Incorrect Answer with Low Confidence**

**The Situation:** An agent evaluates a baseline student ($P(L_{t-1}) = 0.591$) showing signs of struggle on a skill component.

|**Item**|**Content**|
|---|---|
|Evidence|Student answered incorrectly (`correct=False`) and reported low confidence (`confidence=LOW`)|
|Hidden states|Mastered, Uncertain, or Knowledge Gap|
|Beliefs|P(Mastered) = 0.591 (59.1%), P(Not Mastered) = 0.409 (40.9%) — sums to 100%|
|Event|True student knowledge state|
|Actions|`ANSWER`, `ASK`, `HINT`, `TEACH_PRIOR`|
|Costs|Failing to remediate severe gaps leads to downstream errors; false remediation slows progress|
|Policy|If posterior $\le 0.40 \rightarrow$ `TEACH_PRIOR`|
|Decision|Pending live evidence update|
|Audit data|Timestamp: 2026-06-01, Data Version: Assistments2009, Model Version: v1-bkt|

**Step-by-Step Update:**

1. **Prior probability:** $P(L_{t-1}) = 0.591$.
2. **New evidence:** Incorrect answer, Low Confidence.
3. **Likelihood estimation:** Mastery likelihood = $0.116$, Non-Mastery likelihood = $0.755$, Total $P(Obs) = 0.3774$.
4. **Posterior calculation:** Bayes posterior = $0.1817$; Final posterior after transition = **$0.3061$**.
5. **Threshold comparison:** $0.3061 \le 0.40$ lower gap threshold.
6. **New action:** **`TEACH_PRIOR`**.
### **Scenario C: Mastery State with Low Confidence (Careless Slip / Guess Check)**

**The Situation:** A student with a high historical mastery prior ($P(L_{t-1}) = 0.980$) interacts with an advanced problem.

|**Item**|**Content**|
|---|---|
|Evidence|Student answered correctly (`correct=True`) but reported low confidence (`confidence=LOW`)|
|Hidden states|Mastered, Uncertain, or Knowledge Gap|
|Beliefs|P(Mastered) = 0.980 (98.0%), P(Not Mastered) = 0.020 (2.0%) — sums to 100%|
|Event|True student knowledge state|
|Actions|`ANSWER`, `ASK`, `HINT`, `TEACH_PRIOR`|
|Costs|Ignoring low confidence on high-mastery states can mask uncalibrated guesses or conceptual slips|
|Policy|If posterior $\ge 0.85$ and Low Confidence $\rightarrow$ `ASK`|
|Decision|Pending live evidence update|
|Audit data|Timestamp: 2026-06-01, Data Version: Assistments2009, Model Version: v1-bkt|

**Step-by-Step Update:**

1. **Prior probability:** $P(L_{t-1}) = 0.980$.
2. **New evidence:** Correct answer, Low Confidence.
3. **Likelihood estimation:** Mastery likelihood = $0.884$, Non-Mastery likelihood = $0.245$, Total $P(Obs) = 0.8712$.
4. **Posterior calculation:** Bayes posterior = $0.9944$; Final posterior after transition = **$0.9952$**.
5. **Threshold comparison:** $0.9952 \ge 0.85$ mastery threshold, but low confidence triggers verification.
6. **New action:** **`ASK`**.
### **Scenario D: Uncertain / Partial Knowledge State**

**The Situation:** A struggling student starting from a low prior knowledge base ($P(L_{t-1}) = 0.300$) attempts a problem.

| **Item**      | **Content**                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------ |
| Evidence      | Student answered correctly (`correct=True`) with medium confidence (`confidence=MEDIUM`)                           |
| Hidden states | Mastered, Uncertain, or Knowledge Gap                                                                              |
| Beliefs       | P(Mastered) = 0.300 (30.0%), P(Not Mastered) = 0.700 (70.0%) — sums to 100%                                        |
| Event         | True student knowledge state                                                                                       |
| Actions       | `ANSWER`, `ASK`, `HINT`, `TEACH_PRIOR`                                                                             |
| Costs         | Skipping hints on partial knowledge stalls learning; giving answers outright hinders independent skill acquisition |
| Policy        | If $0.40 \le \text{posterior} \le 0.84 \rightarrow$ `HINT`                                                         |
| Decision      | Pending live evidence update                                                                                       |
| Audit data    | Timestamp: 2026-06-01, Data Version: Assistments2009, Model Version: v1-bkt                                        |

**Step-by-Step Update:**

1. **Prior probability:** $P(L_{t-1}) = 0.300$.
2. **New evidence:** Correct answer, Medium Confidence.
3. **Likelihood estimation:** Mastery likelihood = $0.884$, Non-Mastery likelihood = $0.245$, Total $P(Obs) = 0.4367$.
4. **Posterior calculation:** Bayes posterior = $0.6073$; Final posterior after transition = **$0.6670$**.
5. **Threshold comparison:** $0.6670$ falls inside the intermediate uncertainty band ($0.40$ to $0.84$).
6. **New action:** **`HINT`**.

experiments.json
``` json
[
    {
        "id": "A",
        "name": "Correct Answer with High Confidence",
        "prior": 0.591,
        "correct": true,
        "conf": "HIGH",
        "ev": 0.6226,
        "bayes": 0.8391,
        "final": 0.8635,
        "action": "ANSWER"
    },
    {
        "id": "B",
        "name": "Incorrect Answer with Low Confidence",
        "prior": 0.591,
        "correct": false,
        "conf": "LOW",
        "ev": 0.3774,
        "bayes": 0.1817,
        "final": 0.3061,
        "action": "TEACH_PRIOR"
    },
    {
        "id": "C",
        "name": "Mastery State with Low Confidence",
        "prior": 0.98,
        "correct": true,
        "conf": "LOW",
        "ev": 0.8712,
        "bayes": 0.9944,
        "final": 0.9952,
        "action": "ASK"
    },
    {
        "id": "D",
        "name": "Uncertain/Partial Knowledge State",
        "prior": 0.3,
        "correct": true,
        "conf": "MEDIUM",
        "ev": 0.4367,
        "bayes": 0.6073,
        "final": 0.667,
        "action": "HINT"
    }
]

```

Code output

```
[Running] python -u "V1_agent.py"

[SCENARIO_A] Correct Answer with High Confidence
  Inputs  -> Prior P(L): 0.591, Correct: True, Confidence: HIGH
  Results -> Evidence P(Obs): 0.6226 | Bayes Post: 0.8391 | Final P(L): 0.8635
  Action  -> Triggered: ANSWER

[SCENARIO_B] Incorrect Answer with Low Confidence
  Inputs  -> Prior P(L): 0.591, Correct: False, Confidence: LOW
  Results -> Evidence P(Obs): 0.3774 | Bayes Post: 0.1817 | Final P(L): 0.3061
  Action  -> Triggered: TEACH_PRIOR

[SCENARIO_C] Mastery State with Low Confidence (Careless Slip / Guess check)
  Inputs  -> Prior P(L): 0.98, Correct: True, Confidence: LOW
  Results -> Evidence P(Obs): 0.8712 | Bayes Post: 0.9944 | Final P(L): 0.9952
  Action  -> Triggered: ASK
  
[SCENARIO_D] Uncertain/Partial Knowledge State
  Inputs  -> Prior P(L): 0.3, Correct: True, Confidence: MEDIUM
  Results -> Evidence P(Obs): 0.4367 | Bayes Post: 0.6073 | Final P(L): 0.6670
  Action  -> Triggered: HINT

[Done] exited with code=0 in 0.34 seconds

```