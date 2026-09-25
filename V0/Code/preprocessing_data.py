import pandas as pd
import numpy as np
import os

def assign_pedagogical_ground_truth(row) -> str:
    if row["correct"] == 1:
        return "ANSWER"
    if row["mistake_history"] >= 2 or row["attempt_number"] > 2 or row["hint_count"] > 1:
        return "TEACH_PRIOR"
    if row["attempt_number"] == 1 and row["kc_success_history"] >= 1 and row["response_time"] == "FAST":
        return "ASK"
    if row["kc_success_history"] == 0 and row["mistake_history"] == 0 and row["attempt_number"] == 1:
        return "ASK"
    return "HINT"


def preprocess_assistments(
    input_path: str = "skill_builder_data.csv",
    output_parquet: str = "preprocessed_assistments.parquet",
    output_csv: str = "preprocessed_assistments.csv",
):
    print(f"Reading raw data from '{input_path}'...")
    df = pd.read_csv(input_path, low_memory=False)

    col_mapping = {
        "skill_id": "kc_id",
        "skill_name": "kc_name",
        "attempt_count": "attempt_number",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    if "ms_first_response" in df.columns:
        raw_ms = pd.to_numeric(df["ms_first_response"], errors="coerce")
        df["time_seconds"] = (raw_ms / 1000.0).round(2).fillna(15.0)
    elif "time_seconds" not in df.columns:
        df["time_seconds"] = 15.0

    required = ["user_id", "kc_id", "correct"]
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    if "order_id" in df.columns:
        df = df.sort_values(["user_id", "kc_id", "order_id"]).reset_index(drop=True)
    elif "opportunity" in df.columns:
        df = df.sort_values(["user_id", "kc_id", "opportunity"]).reset_index(drop=True)

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)

    if "attempt_number" not in df.columns:
        df["attempt_number"] = 1
    else:
        df["attempt_number"] = pd.to_numeric(df["attempt_number"], errors="coerce").fillna(1).astype(int)

    if "hint_count" not in df.columns:
        df["hint_count"] = 0
    else:
        df["hint_count"] = pd.to_numeric(df["hint_count"], errors="coerce").fillna(0).astype(int)

    print("Computing cumulative historical features...")
    df["prev_correct"] = df.groupby(["user_id", "kc_id"])["correct"].shift(1).fillna(0)
    df["prev_incorrect"] = (1 - df.groupby(["user_id", "kc_id"])["correct"].shift(1)).fillna(0)

    df["kc_success_history"] = df.groupby(["user_id", "kc_id"])["prev_correct"].cumsum().astype(int)
    df["mistake_history"] = df.groupby(["user_id", "kc_id"])["prev_incorrect"].cumsum().astype(int)

    print("Computing skill-level latency thresholds...")
    kc_median = df.groupby("kc_id")["time_seconds"].transform("median").fillna(25.0)
    kc_median = kc_median.clip(lower=2.0)
    df["expected_time"] = kc_median.round(2)
    df["response_time"] = np.where(df["time_seconds"] < (df["expected_time"] * 0.3), "FAST", "SLOW")

    print("Assigning pedagogical ground-truth action labels...")
    df["annotated_action"] = df.apply(assign_pedagogical_ground_truth, axis=1)

    final_cols = [
        "user_id",
        "kc_id",
        "correct",
        "attempt_number",
        "time_seconds",
        "expected_time",
        "response_time",
        "hint_count",
        "kc_success_history",
        "mistake_history",
        "annotated_action",
    ]
    if "opportunity" in df.columns:
        final_cols.insert(2, "opportunity")

    clean_df = df[[c for c in final_cols if c in df.columns]].copy()

    # 8. Save artifacts
    clean_df.to_parquet(output_parquet, index=False)
    print(f" Saved Parquet: '{output_parquet}' ({os.path.getsize(output_parquet) / 1e6:.1f} MB)")

    clean_df.to_csv(output_csv, index=False)
    print(f" Saved CSV: '{output_csv}' ({os.path.getsize(output_csv) / 1e6:.1f} MB)")

    return clean_df


if __name__ == "__main__":
    preprocess_assistments()