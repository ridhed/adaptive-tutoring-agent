# Adaptive Tutoring Agent

### Scenario: MCQ tutoring under hidden student knowledge

## 1. Project objective

- **Core Task:** An agent observes student MCQ interactions (choice, response time, attempt count, and self-reported confidence) and decides whether to `ANSWER`, `ASK`, `HINT`, or `TEACH_PRIOR`.
    
- **The Hidden Challenge:** True student mastery or understanding of a concept is latent and must be inferred from behavior on every single question.
    
- **Dataset:** Student-response data (`skill_builder_data.csv`) from the [SciDB Assistments Student Learning Records Dataset](https://www.scidb.cn/en/detail?dataSetId=b1c3986fc96d435e8b258a9b5c36cd7c).
  
## 2. Problem statement

>Student cognitive states (such as mastery, knowledge gaps, or guessing) are latent and unobservable. The agent must infer these hidden states from behavioral telemetry including correctness, response time, attempt counts, and confidence to dynamically select optimal pedagogical actions (`ANSWER`, `ASK`, `HINT`, `TEACH_PRIOR`) after every single question.

## 3. Decision-making architectures (versions)

| Version | Architecture  | What it adds                                                                             |
| ------- | ------------- | ---------------------------------------------------------------------------------------- |
| **V0**  | Expert System | Fixed rules mapping responses → state → action; no memory or probabilities.              |
| **V1**  | BKT           | Tracks **P(Learned)** and updates it after each response, with self-reported confidence. |


## 4. Evidence the agent uses (per interaction)

| Inputs             | Terminology   | Question it answers                  | Introduced in |
| ------------------ | ------------- | ------------------------------------ | ------------- |
| **Correctness**    | Performance   | Did they get it right?               | V0            |
| **Response time**  | Latency       | Did they rush or struggle?           | V0            |
| **Attempt number** | Persistence   | Is this a first or repeated attempt? | V0            |
| **Hint requests**  | Scaffolding   | Did they need help?                  | V1            |
| **Confidence**     | Metacognition | Do they believe they know it?        | V1            |

## 5. Agent design (V1 - current)

| Features | How it is used in the workflow |
|---|---|
| **Input** | `student_id`, `problem_id`, `kc_id` (skill), `correct`, `time_seconds`, `hint_requested`, `attempt_number`, `confidence` |
| **Hidden state** | Learned / Not Learned (binary, BKT), Mastery (P(L) ≥ 0.85), Uncertain (0.40–0.84), Knowledge Gap (≤ 0.40) |
| **Belief** | P(Lₜ) = probability the student has learned the skill global prior, (P(L₀) = 0.591, learned from `skill_builder_data.csv` via `prior_prob_cal.py`) and updated with every new response |
| **Action** | **Answer**, **Ask**, **Hint**, **Teach Prior** |
| **Policy** | [P(Lₜ) ≥ 0.85 = Answer], [P(Lₜ) ≥ 0.85 = Ask], [0.40 ≤ P(Lₜ) < 0.85= Hint], [P(Lₜ) < 0.40 → Teach Prior.] |
| **Feedback** | Each (belief, action, next-response outcome) used to re-estimate per-skill P(guess)/P(slip) instead of relying on one prior probability. |

- **Practitioner Insight:** Inspired by a Reddit discussion (`discussion-record.md`), noting that learners admit uncertainty more readily to an AI, the system adds self-reported confidence to catch lucky guesses.
    
- **Building Process:** Implemented as standalone Python modules (`V0_agent.py`, `V1_agent.py`) running against `skill_builder_data.csv`, with probabilistic parameters derived via `prior_prob_cal.py`.

## 6. Project Files

```text
adaptive-tutoring-agent/
├── .git/
├── README.md
├── analysis/
│   ├── prior_prob_cal.py
│   └── probability-decision-record.md
├── design/
├── research/
│   ├── research-file.md
│   └── Research Papers/
├── socials/
│   ├── discussion-record.md
│   └── linkedin-posts.md
├── V0/
│   ├── V0 - Expert System.md
│   └── V0_agent.py
└── V1/
    ├── experiments.json
    ├── V1 -  Bayesian Knowledge Tracing.md
    └── V1_agent.py
```

## 7. What's next
- Incorporate hint-request count into the V1 policy table (currently collected as an input but not yet used in the decision rule).
- Move from a single global prior to per-skill, and eventually per-student, priors as more interaction data accumulates.
