# Adaptive Tutoring Agent

### Scenario: MCQ tutoring under hidden student knowledge

## 1. Project objective

- **Core Task** = An agent observes student MCQ interactions (choice, response time, attempt count, and self-reported confidence) and decides whether to `ANSWER`, `ASK`, `HINT`, or `TEACH_PRIOR`.
    
- **The Hidden Challenge** = True student mastery or understanding of a concept is latent and must be inferred from behavior on every single question.
    
- **Dataset:** Built iteratively using student-response data (`skill_builder_data.csv`) from the [SciDB Assistments Student Learning Records Dataset](https://www.scidb.cn/en/detail?dataSetId=b1c3986fc96d435e8b258a9b5c36cd7c).
  
## 2. Problem statement

>Student cognitive states (such as mastery, knowledge gaps, or guessing) are latent and unobservable. The agent must infer these hidden states from behavioral telemetry including correctness, response time, attempt counts, and confidence to dynamically select optimal pedagogical actions (`ANSWER`, `ASK`, `HINT`, `TEACH_PRIOR`) after every single question.

## 3. Decision-making architectures (versions)

| Version                 | Architecture                                          | What it adds                                                                                                                                                                                                                                            |
| ----------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **V0**                  | Expert System (if/elif rule)                    | A fixed decision table mapping (correctness, attempt number, response time) → hidden state → action. No memory across questions, no probabilities.                                                                                                      |
| **V1**                  | Bayesian Knowledge Tracing (BKT)                      | Replaces hard-coded rules with a probability P(Learned) that updates after every response using Bayes' rule, plus a self-reported confidence signal (added after Reddit feedback — see `discussion-record.md`). |

## 4. Evidence the agent uses (per interaction)

| Signal | Question it answers | Introduced in |
|---|---|---|
| **Correctness** | Did they get it right? | V0 |
| **Response time** (discretized FAST/SLOW vs. expected time) | Did they seem to rush (careless) or struggle (knowledge gap)? | V0 |
| **Attempt number** | Is this a first try, or have they already tried and failed on this item? | V0 |
| **Hint requests** | Did they need scaffolding to get here? | V1 |
| **Self-reported confidence** (HIGH/MEDIUM/LOW) | Do they *believe* they know it, independent of whether they're right? | V1, added after Reddit feedback |

## 5. Agent design (V1 - current)

| Part | Definition |
|---|---|
| **Input** | `student_id`, `problem_id`, `kc_id` (skill), `correct`, `time_seconds`, `hint_requested`, `attempt_number`, `confidence` |
| **Hidden state** | Learned / Not Learned (binary, BKT), surfaced as: Mastery (P(L) ≥ 0.85), Uncertain/Partial (0.40–0.84), Knowledge Gap (≤ 0.40) |
| **Belief** | P(Lₜ) — probability the student has learned the skill, starting from a global prior (P(L₀) = 0.591, learned from `skill_builder_data.csv` via `prior_prob_cal.py`) and updated with every new response |
| **Action** | **Answer** (give the answer / move on) · **Ask** (probe further before trusting the belief) · **Hint** (scaffold) · **Teach Prior** (drop back to a prerequisite concept) |
| **Cost** | Answering when the true state is a Knowledge Gap wastes a teaching moment the student needed. Asking a student who has clearly mastered a skill wastes time and risks frustration. Trusting a correct-but-low-confidence answer as mastery risks reinforcing a guess instead of catching it. |
| **Policy** | If P(Lₜ) ≥ 0.85 and correct and confidence = HIGH → Answer. If P(Lₜ) ≥ 0.85 but confidence = LOW or the item was missed → Ask. If 0.40 ≤ P(Lₜ) < 0.85 → Hint. If P(Lₜ) < 0.40 → Teach Prior. |
| **Feedback** | Each (belief, action, next-response outcome) triple can be used to re-estimate per-skill P(guess)/P(slip) instead of relying on one global prior forever — see `probability-decision-record.md` for a worked three-step example. |

- **Practitioner Insight:** Inspired by a Reddit discussion (`discussion-record.md`), noting that learners admit uncertainty more readily to an AI, the system adds self-reported confidence to catch lucky guesses.
    
- **Building Process:** Implemented as standalone Python modules (`V0_agent.py`, `V1_agent.py`) running against `skill_builder_data.csv`, with probabilistic parameters derived via `prior_prob_cal.py`.

## 6. Project files

| File | Purpose |
|---|---|
| `V0 - Expert System.md` / `V0_agent.py` | Rule-based baseline: decision table + implementation |
| `V1 - Bayesian Knowledge Tracing.md` / `V1_agent.py` | Probabilistic belief-updating version + implementation |
| `prior_prob_cal.py` | Learns P(L₀), P(guess), P(slip), P(learn), P(forget) from `skill_builder_data.csv` |
| `probability-decision-record.md` | Worked, step-by-step Bayesian update example across several interactions |
| `discussion-record.md` | External practitioner feedback (Reddit) and the design change it produced |
| `research-file.md` | Glossary of terms, search queries, and relevant communities used while researching this problem |
| `Literature_Review.md` | Annotated bibliography connecting each open design question to a specific paper |

## 7. What's next
- Incorporate hint-request count into the V1 policy table (currently collected as an input but not yet used in the decision rule).
- Move from a single global prior to per-skill, and eventually per-student, priors as more interaction data accumulates.
