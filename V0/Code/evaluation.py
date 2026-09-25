import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from V0_agent import AgentV0
from V0_1_agent import AgentV01


def evaluate_agents_on_preprocessed(
    preprocessed_csv: str = "preprocessed_assistments.csv",
    sample_size: int = 100,
):
    df = pd.read_csv(preprocessed_csv)
    sample = df.groupby(["correct", "response_time", "attempt_number"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, sample_size // 8)), random_state=42),
        include_groups=False
    ).head(sample_size).copy()
    sample = df.loc[sample.index].copy()

    v0 = AgentV0()
    v01 = AgentV01()

    y_true = sample["annotated_action"].tolist()
    y_v0 = []
    y_v01 = []

    for _, r in sample.iterrows():
        # Predict V0
        _, act_0 = v0.predict(
            is_correct=bool(r["correct"]),
            attempt_number=int(r["attempt_number"]),
            response_time=str(r["response_time"])
        )
        y_v0.append(act_0)

        # Predict V0.1
        _, act_01 = v01.predict(
            is_correct=bool(r["correct"]),
            kc_success_history=int(r["kc_success_history"]),
            response_time=str(r["response_time"]),
            attempt_number=int(r["attempt_number"]),
            hint_count=int(r["hint_count"]),
            mistake_history=int(r["mistake_history"])
        )
        y_v01.append(act_01)

    labels = ["ANSWER", "ASK", "HINT", "TEACH_PRIOR"]

    print("\n" + "=" * 60)
    print(f"HEAD-TO-HEAD EVALUATION (Sample Size = {len(sample)})")
    print("=" * 60)
    print(f"Agent V0   Accuracy: {accuracy_score(y_true, y_v0)*100:.1f}% | Cohen's Kappa: {cohen_kappa_score(y_true, y_v0):.3f}")
    print(f"Agent V0.1 Accuracy: {accuracy_score(y_true, y_v01)*100:.1f}% | Cohen's Kappa: {cohen_kappa_score(y_true, y_v01):.3f}")
    print("=" * 60)

    print("\n--- Detailed Report: Agent V0 ---")
    print(classification_report(y_true, y_v0, labels=labels, zero_division=0))

    print("\n--- Detailed Report: Agent V0.1 ---")
    print(classification_report(y_true, y_v01, labels=labels, zero_division=0))


if __name__ == "__main__":
    evaluate_agents_on_preprocessed("preprocessed_assistments.csv", sample_size=100)