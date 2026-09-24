import pandas as pd
import numpy as np


class BayesianTutoringAgent:
    def __init__(self, p_init=0.59, p_guess=0.25, p_slip=0.12, p_learn=0.15, p_forget=0.0):
        self.params = {
            "p_init": p_init,
            "p_guess": p_guess,
            "p_slip": p_slip,
            "p_learn": p_learn,
            "p_forget": p_forget,
        }
        self.mastery_threshold = 0.85
        self.struggle_threshold = 0.40

    @classmethod
    def fit_from_telemetry(cls, df: pd.DataFrame):
        """Derives empirical priors directly from observational telemetry."""
        clean_df = df.dropna(subset=["user_id", "skill_id", "correct", "opportunity"]).copy()
        clean_df.sort_values(["user_id", "skill_id", "opportunity"], inplace=True)

        # Initial probability of correct on first opportunity
        p_init = clean_df[clean_df["opportunity"] == 1]["correct"].mean()

        # Prior performance lags
        clean_df["prev_correct"] = clean_df.groupby(["user_id", "skill_id"])["correct"].shift(1)
        clean_df["prev2_correct"] = clean_df.groupby(["user_id", "skill_id"])["correct"].shift(2)

        # Guess estimate: correct after two consecutive failures
        guess_mask = (clean_df["prev_correct"] == 0) & (clean_df["prev2_correct"] == 0)
        p_guess = clean_df[guess_mask]["correct"].mean() if guess_mask.sum() > 0 else 0.20

        # Slip estimate: incorrect after two consecutive successes
        slip_mask = (clean_df["prev_correct"] == 1) & (clean_df["prev2_correct"] == 1)
        p_slip = 1.0 - clean_df[slip_mask]["correct"].mean() if slip_mask.sum() > 0 else 0.10

        # Learning transition estimate
        raw_transition = clean_df[clean_df["prev_correct"] == 0]["correct"].mean()
        p_learn = max(0.02, raw_transition - p_guess) if not np.isnan(raw_transition) else 0.15

        return cls(
            p_init=round(p_init, 3),
            p_guess=round(p_guess, 3),
            p_slip=round(p_slip, 3),
            p_learn=round(p_learn, 3),
        )

    def update_belief(self, p_prior: float, is_correct: bool) -> tuple[float, float]:
        """Calculates Bayesian evidence and updated knowledge state."""
        p_guess = self.params["p_guess"]
        p_slip = self.params["p_slip"]
        p_learn = self.params["p_learn"]
        p_forget = self.params["p_forget"]

        # Likelihoods
        p_obs_given_known = (1 - p_slip) if is_correct else p_slip
        p_obs_given_unknown = p_guess if is_correct else (1 - p_guess)

        p_obs = (p_prior * p_obs_given_known) + ((1 - p_prior) * p_obs_given_unknown)

        # Posterior update
        post_given_obs = (p_prior * p_obs_given_known) / p_obs

        # Transition forward
        p_next = post_given_obs + (1 - post_given_obs) * p_learn - (post_given_obs * p_forget)
        return p_obs, p_next

    def select_action(self, p_know: float, is_correct: bool, response_time_sec: float) -> dict:
        """
        Policy function: selects action using probability state and interaction dynamics,
        without requiring human-annotated action labels.
        """
        is_fast = response_time_sec < 10.0

        if p_know >= self.mastery_threshold:
            if is_correct:
                action = "ADVANCE"
                state = "Mastered"
            else:
                # High mastery but incorrect response indicates a slip or misread
                action = "ASK_EXPLANATION"
                state = "Careless Slip"
        elif p_know >= self.struggle_threshold:
            action = "PROVIDE_HINT"
            state = "In Progress / Zone of Proximal Development"
        else:
            action = "TEACH_PREREQUISITE"
            state = "Knowledge Gap"

        return {
            "posterior_knowledge": round(p_know, 4),
            "inferred_state": state,
            "prescribed_action": action,
        }


# --- Demonstration on simulated / real interaction sequence ---
if __name__ == "__main__":
    # Initialize agent with calibrated baseline parameters
    agent = BayesianTutoringAgent(p_init=0.50, p_guess=0.25, p_slip=0.12, p_learn=0.15)

    # Simulated student sequence on a specific Knowledge Component
    interactions = [
        {"attempt": 1, "correct": False, "latency_s": 25.4},
        {"attempt": 2, "correct": True,  "latency_s": 14.2},
        {"attempt": 3, "correct": True,  "latency_s": 8.1},
        {"attempt": 4, "correct": True,  "latency_s": 5.0},
    ]

    p_current = agent.params["p_init"]
    print(f"Initial Prior P(L_0): {p_current}\n" + "-" * 60)

    for item in interactions:
        _, p_next = agent.update_belief(p_current, item["correct"])
        decision = agent.select_action(p_next, item["correct"], item["latency_s"])

        print(f"Step {item['attempt']}: Correct={item['correct']} | Latency={item['latency_s']}s")
        print(f"  -> Prior: {p_current:.4f} | Posterior: {decision['posterior_knowledge']}")
        print(f"  -> Inferred State: {decision['inferred_state']}")
        print(f"  -> Prescribed Action: {decision['prescribed_action']}\n")

        p_current = p_next