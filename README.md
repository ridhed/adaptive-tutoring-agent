# Tutor Agent

### Scenario: MCQ tutoring under hidden student knowledge

## 1. Project objective

A tutoring agent watches a student answer multiple-choice questions — which option they picked, how long they took, how many attempts, and (as of V1) how confident they say they felt — and has to decide, after every single question: **answer it for them, ask them a follow-up, give a hint, or go back and teach a prerequisite concept.** The right choice depends on *why* the student got the question right or wrong, and that reason is never directly visible. Someone (or something) has to infer it from behavior, fast, on every question.

We're building this agent in versions of increasing sophistication, and testing each version's reasoning against real feedback from practitioners on Reddit and against a real student-response dataset (ASSISTments-style `skill_builder_data.csv`).

## 2. Problem statement

> The agent observes a student's response to an MCQ item: the option selected, correctness, response time, attempt number, and (from V1 onward) a self-reported confidence rating. It must select **ANSWER, ASK, HINT, or TEACH_PRIOR**, because whether the student is in a state of {Mastery, Uncertain/Partial, Knowledge Gap, Misconception, Careless, Guessing, Needs Prior} is a hidden state — not something the agent can read off directly.

### A concrete example to anchor everything

Student S-1084 sees a fraction-addition item for the first time. They answer correctly but say they weren't confident. Did they actually understand it, or guess right? The next item they get wrong. Did that mean they never knew it, or did they just slip up? The agent has to pick an action after *each* of these, without ever being told the true answer to "do they know this skill?"

## 3. Decision-making architectures (versions)

| Version | Architecture | What it adds |
|---|---|---|
| **V0** | Expert System (if/elif rule table) | A fixed decision table mapping (correctness, attempt number, response time) → hidden state → action. No memory across questions, no probabilities. |
| **V1** | Bayesian Knowledge Tracing (BKT) | Replaces hard-coded rules with a probability P(Learned) that updates after every response using Bayes' rule, plus a self-reported confidence signal (added after Reddit feedback — see `discussion-record.md`) to catch guesses that look like mastery. |
| **V2 (next)** | Partially Observable Markov Decision Process (POMDP) | Optimizes for long-term learning gain instead of reacting to just the current question — plans a sequence of actions under uncertainty rather than a single best response. |
| *(possible later work)* | Deep Knowledge Tracing (DKT) / Reinforcement Learning | Continuous hidden state via LSTM (DKT) and learned, rather than hand-specified, action policies (RL) — see `Literature_Review.md`. |

## 4. Evidence the agent uses (per interaction)

These are the observable signals that stand in for the six data-quality-style "checks" in this project — the raw numbers the agent turns hidden-state inference into:

| Signal | Question it answers | Introduced in |
|---|---|---|
| **Correctness** | Did they get it right? | V0 |
| **Response time** (discretized FAST/SLOW vs. expected time) | Did they seem to rush (careless) or struggle (knowledge gap)? | V0 |
| **Attempt number** | Is this a first try, or have they already tried and failed on this item? | V0 |
| **Hint requests** | Did they need scaffolding to get here? | V1 input, not yet used in the V1 policy table |
| **Self-reported confidence** (HIGH/MEDIUM/LOW) | Do they *believe* they know it, independent of whether they're right? | V1, added after Reddit feedback (see below) |

## 5. Agent design (V1 — current)

| Part | Definition |
|---|---|
| **Input** | `student_id`, `problem_id`, `kc_id` (skill), `correct`, `time_seconds`, `hint_requested`, `attempt_number`, `confidence` |
| **Hidden state** | Learned / Not Learned (binary, BKT), surfaced as: Mastery (P(L) ≥ 0.85), Uncertain/Partial (0.40–0.84), Knowledge Gap (≤ 0.40) |
| **Belief** | P(Lₜ) — probability the student has learned the skill, starting from a global prior (P(L₀) = 0.591, learned from `skill_builder_data.csv` via `prior_prob_cal.py`) and updated with every new response |
| **Action** | **Answer** (give the answer / move on) · **Ask** (probe further before trusting the belief) · **Hint** (scaffold) · **Teach Prior** (drop back to a prerequisite concept) |
| **Cost** | Answering when the true state is a Knowledge Gap wastes a teaching moment the student needed. Asking a student who has clearly mastered a skill wastes time and risks frustration. Trusting a correct-but-low-confidence answer as mastery risks reinforcing a guess instead of catching it. |
| **Policy** | If P(Lₜ) ≥ 0.85 and correct and confidence = HIGH → Answer. If P(Lₜ) ≥ 0.85 but confidence = LOW or the item was missed → Ask. If 0.40 ≤ P(Lₜ) < 0.85 → Hint. If P(Lₜ) < 0.40 → Teach Prior. |
| **Feedback** | Each (belief, action, next-response outcome) triple can be used to re-estimate per-skill P(guess)/P(slip) instead of relying on one global prior forever — see `probability-decision-record.md` for a worked three-step example. |

**Human-style reasoning added:** a Reddit conversation on r/AIEducation (`discussion-record.md`) raised that learners may be more willing to admit uncertainty to an AI than to a human teacher. That turned into a concrete design change: **self-reported confidence is now a second observation, alongside correctness**, used specifically to catch cases where a correct answer is actually a lucky guess rather than real mastery.

**How we're building it:** each version is implemented as a small, standalone Python module (`V0_agent.py`, `V1_agent.py`) that can be run directly against `skill_builder_data.csv`, with the probabilistic parameters for V1 learned from the same dataset (`prior_prob_cal.py`) rather than guessed.

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

- **V2 — POMDP:** stop optimizing for "get the current question right" and start optimizing for long-term learning gain — planning a sequence of actions under uncertainty, in the spirit of Rafferty (2014) and Kadir (2025) (see `Literature_Review.md`).
- Incorporate hint-request count into the V1 policy table (currently collected as an input but not yet used in the decision rule).
- Move from a single global prior to per-skill, and eventually per-student, priors as more interaction data accumulates.
