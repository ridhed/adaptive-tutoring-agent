# 1. Input & Prior Probabilities
**Dataset:** [Assistment2009](https://doi.org/10.57760/sciencedb.j00133.00253 )

**Inputs:**
- `student_id`, `problem_id`, `kc_id`, `correct`, `time_seconds`, `hint_requested`, `attempt_number`
- `confidence` $\in$ {HIGH, MEDIUM, LOW}  [5] 

**Prior Probability (Learned from Dataset) [1], [3] :**
- $P(L_0)$ (Initial Knowledge) = 0.591
- $P(T)$ (Learn) = 0.152
- $P(F)$ (Forget) = 0.0
- $P(G)$ (Guess) = 0.245
- $P(S)$ (Slip) = 0.116

```text
[Running] python -u "prior_prob_cal.py"

Calculated Parameters from Dataset:
{
    'p_initial_know': np.float64(0.591),
    'p_guess': np.float64(0.245),
    'p_slip': np.float64(0.116),
    'p_learn': np.float64(0.152),
    'p_forget': 0.0
}

[Done] exited with code=0 in 2.004 seconds
```

# 2. Bayesian Updates 

## 2. Bayesian Knowledge Tracing (BKT)

**Bayesian Knowledge Tracing (BKT):** A hidden binary state representing whether a skill is `Learned` or `Not Learned`. BKT updates its belief about the student's knowledge based on observed evidence, such as correctness [2].

**1. Prior Belief ($P(L_{t-1})$):**  
The probability that the student knows the skill before the current interaction.

**2. Likelihood ($P(Obs \mid L)$):**

- If the observation is **Correct** ($Obs = 1$):
  - Master = $1 - P(S)$
  - Non-Master = $P(G)$

- If the observation is **Incorrect** ($Obs = 0$):
  - Master = $P(S)$
  - Non-Master = $1 - P(G)$

**3. Evidence / Marginal Probability ($P(Obs)$):**

For a correct observation:

$$
P(Obs=1) =
P(L_{t-1})(1-P(S)) +
(1-P(L_{t-1}))P(G)
$$

For an incorrect observation:

$$
P(Obs=0) =
P(L_{t-1})P(S) +
(1-P(L_{t-1}))(1-P(G))
$$

**4. Posterior Probability (Bayes' Theorem):**

$$
P(L_t \mid Obs) =
\frac{P(L_{t-1})P(Obs \mid L_{t-1})}
{P(Obs)}
$$

**5. Learning Transition ($P(L_t)$):**  
Accounts for the probability that the student acquires or forgets the skill during this step.

$$
P(L_t) =
P(L_t \mid Obs)
+
(1-P(L_t \mid Obs))P(T)
-
P(L_t \mid Obs)P(F)
$$

[4]

### 3. Hidden States & Actions

The hidden state represents the student's true knowledge, which cannot be observed directly. The tutor only observes evidence and uses it to infer the state.

- **Mastery:** $P(L_t) \ge 0.85$
- **Uncertain / Partial:** $0.40 < P(L_t) < 0.85$
- **Knowledge Gap:** $P(L_t) \le 0.40$

**Action Space:** {`ASK`, `HINT`, `TEACH_PRIOR`, `ANSWER`}.
# 4. Decision Table (Probability & Confidence Driven)

|**Posterior P(Lt​)**|**Correctness**|**Confidence**|**Inferred State**|**Action**|
|---|---|---|---|---|
|$\ge 0.85$|True|High|Mastery|ANSWER|
|$\ge 0.85$|True|Low|Guessing|ASK|
|$0.40 - 0.84$|Any|Any|Uncertain|HINT|
|$\le 0.40$|False|Any|Knowledge Gap|TEACH_PRIOR|
# 5. Architechture
<img width="723" height="812" alt="V1" src="https://github.com/user-attachments/assets/c55e124a-b63b-43bf-9d23-ac5c8deb0169" />

# 7. References

[1] Corbett, A. T., & Anderson, J. R. (1995). _Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge._ 

[2] Mislevy, R. J., Steinberg, L. S., & Almond, R. G. (2003). _On the Structure of Educational Assessments.

[3] Šarić-Grgić, I., Grubišić, A., & Gašpar, A. (2024). _Twenty-five years of Bayesian knowledge tracing: a systematic review._ 

[4] Pradhan, S., et al. (2026). _StanBKT: Rethinking Parameter Estimation in Bayesian Knowledge Tracing._ 

[5] Rus, V., & Ștefănescu, D. (2016). _Non-intrusive assessment of learners' prior knowledge in dialogue-based ITS._ 
# 8. What's Next

- **V2 - POMDP:** Optimize long-term learning gain, not just immediate correctness.
