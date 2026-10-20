"""Create StanBKT mastery estimates and derived tutoring-action reference labels.

Install StanBKT's requirements (Python 3.12+, pandas, CmdStan/C++ toolchain):
    python -m pip install stanbkt pandas

Set up CmdStan once if needed:
    python -c "from stanbkt.utils import setup_cmdstanpy; setup_cmdstanpy(n_cores=4)"

Run from the repository root:
    python dataset/ground_truth.py

StanBKT estimates filtered pre-response mastery (pKnow). It does not define
ASK/HINT/TEACH_PRIOR/ANSWER labels. Those are derived here by the explicit,
configurable rule in action_from_mastery; they are policy reference labels,
not observed ground truth. `correct` and `observed_help` remain observed data.
Paper: https://arxiv.org/abs/2605.23048
Package: https://github.com/SiddharthaPradhan/StanBKT
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "dataset" / "skill_builder_data.csv"
DEFAULT_OUTPUT = ROOT / "dataset" / "ground_truth.csv"


def student_split(student_id: str, seed: int, train_fraction: float) -> str:
    """Stable student-level holdout; all records from a student stay together."""
    token = hashlib.sha256(f"{seed}:{student_id}".encode("utf-8")).hexdigest()[:16]
    value = int(token, 16) / 16**16
    return "train" if value < train_fraction else "test"


def action_from_mastery(p_know: float, prior_errors: int) -> str:
    """Project policy over StanBKT's pre-response mastery estimate."""
    if p_know < 0.40:
        return "TEACH_PRIOR"
    if p_know < 0.85:
        return "HINT"
    if prior_errors >= 2:
        return "ASK"
    return "ANSWER"


def prepare_data(input_path: Path, top_kcs: int, min_sequence_length: int,
                 seed: int, train_fraction: float) -> tuple[pd.DataFrame, dict]:
    required = ["user_id", "problem_id", "skill_id", "correct", "order_id"]
    optional = ["hint_count", "attempt_count"]
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    frame = pd.read_csv(input_path, usecols=required + [c for c in optional if c in header])
    input_rows = len(frame)
    frame["source_row"] = frame.index
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce")
    valid = frame["user_id"].notna() & frame["skill_id"].notna() & frame["problem_id"].notna() & frame["correct"].isin([0, 1])
    dropped_invalid = int((~valid).sum())
    frame = frame.loc[valid].copy()

    frame["student_id"] = frame["user_id"].astype(str)
    frame["problem_id"] = frame["problem_id"].astype(str)
    frame["kc_id"] = frame["skill_id"].astype(str)
    frame["correct"] = frame["correct"].astype(int)
    frame["timestamp"] = pd.to_numeric(frame["order_id"], errors="coerce")
    dropped_order = int(frame["timestamp"].isna().sum())
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.sort_values(["student_id", "kc_id", "timestamp", "source_row"], kind="mergesort")

    # Mirror the paper's ASSISTments evaluation subset: most frequent KCs and
    # student-skill sequences with at least ten observations.
    selected = frame["kc_id"].value_counts().head(top_kcs).index
    frame = frame[frame["kc_id"].isin(selected)].copy()
    sequence_size = frame.groupby(["student_id", "kc_id"])["correct"].transform("size")
    short_sequences = int((sequence_size < min_sequence_length).sum())
    frame = frame[sequence_size >= min_sequence_length].copy()
    frame["split"] = frame["student_id"].map(lambda sid: student_split(sid, seed, train_fraction))
    if frame.empty:
        raise ValueError("No records remain after validity, KC, and sequence-length filtering.")
    if not (frame["split"] == "train").any() or not (frame["split"] == "test").any():
        raise ValueError("The student-level split produced an empty train or test set; change --seed or --train-fraction.")

    if "hint_count" not in frame:
        frame["hint_count"] = 0
    frame["hint_count"] = pd.to_numeric(frame["hint_count"], errors="coerce").fillna(0).astype(int)
    summary = {
        "input_rows": int(input_rows),
        "invalid_rows": dropped_invalid,
        "missing_or_invalid_order_rows": dropped_order,
        "selected_kcs": sorted(selected.tolist()),
        "short_sequence_rows_removed": short_sequences,
        "rows_for_model": int(len(frame)),
    }
    return frame.reset_index(drop=True), summary


def fit_predict(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    try:
        from stanbkt.fits import FitMethod
        from stanbkt.models import StandardBKT, StandardPriors
        from stanbkt.utils import ColumnNames, VerbosityLevel
    except ImportError as exc:
        raise RuntimeError(
            "StanBKT is not installed. Use Python 3.12+, install `stanbkt` and `pandas`, "
            "then set up CmdStan as described at the top of this file."
        ) from exc

    methods = {"mcmc": FitMethod.MCMC, "map": FitMethod.MAP}
    model = StandardBKT(fit_method=methods[method], verbose=VerbosityLevel.WARN)
    priors = StandardPriors(pi_know_mu=0, pi_know_std=2, guess_mu=0, guess_std=2)
    column_mapping = {
        ColumnNames.STUDENT_ID: "student_id",
        ColumnNames.PROBLEM_ID: "problem_id",
        ColumnNames.KC_ID: "kc_id",
        ColumnNames.CORRECTNESS: "correct",
        ColumnNames.ORDER: "timestamp",
    }

    model_columns = ["student_id", "problem_id", "kc_id", "correct", "timestamp"]
    train = frame.loc[frame["split"] == "train", model_columns].copy()
    model.fit(train, column_mapping=column_mapping, priors=priors)

    predictions = []
    for split in ("train", "test"):
        part = frame.loc[frame["split"] == split]
        pred = model.predict(part[model_columns].copy(), column_mapping=column_mapping)
        if len(pred) != len(part) or "pKnow" not in pred.columns:
            raise RuntimeError("StanBKT prediction output did not contain one pKnow estimate per input record.")
        predictions.append((part.index, pred.reset_index(drop=True)))

    result = frame.copy()
    result["pKnow"] = float("nan")
    result["pCorrectness"] = float("nan")
    for indexes, pred in predictions:
        result.loc[indexes, "pKnow"] = pred["pKnow"].to_numpy()
        if "pCorrectness" in pred.columns:
            result.loc[indexes, "pCorrectness"] = pred["pCorrectness"].to_numpy()
    if result["pKnow"].isna().any():
        raise RuntimeError("StanBKT returned missing mastery estimates.")
    return result


def create_labels(frame: pd.DataFrame) -> pd.DataFrame:
    prior_errors = frame.groupby(["student_id", "kc_id"], sort=False)["correct"].transform(
        lambda values: values.shift(1).eq(0).rolling(window=3, min_periods=1).sum()
    ).fillna(0).astype(int)
    frame["prior_errors"] = prior_errors
    frame["observed_help"] = frame["hint_count"].gt(0)
    frame["reference_action"] = [
        action_from_mastery(p, errors) for p, errors in zip(frame["pKnow"], prior_errors)
    ]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create StanBKT mastery estimates and derived tutoring-action labels.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=["map", "mcmc"], default="map",
                        help="MAP is faster; MCMC provides posterior sampling at higher cost.")
    parser.add_argument("--top-kcs", type=int, default=10,
                        help="Number of most frequent skills to model (paper uses ten).")
    parser.add_argument("--min-sequence-length", type=int, default=10,
                        help="Drop student-skill sequences shorter than this (paper uses ten).")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if args.top_kcs < 1 or args.min_sequence_length < 1 or not 0 < args.train_fraction < 1:
        parser.error("top-kcs and min-sequence-length must be positive; train-fraction must be between 0 and 1")

    data, summary = prepare_data(args.input, args.top_kcs, args.min_sequence_length, args.seed, args.train_fraction)
    labeled = create_labels(fit_predict(data, args.method))
    columns = ["source_row", "student_id", "problem_id", "kc_id", "timestamp", "correct", "pKnow",
               "pCorrectness", "prior_errors", "hint_count", "observed_help", "reference_action", "split"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled[columns].to_csv(args.output, index=False)
    print(f"Wrote {len(labeled):,} labeled records to {args.output}")
    print(f"Input: {summary['input_rows']:,}; invalid: {summary['invalid_rows']:,}; "
          f"missing/invalid order: {summary['missing_or_invalid_order_rows']:,}; "
          f"short sequences removed: {summary['short_sequence_rows_removed']:,}; "
          f"modeled: {summary['rows_for_model']:,}; skills: {len(summary['selected_kcs'])}")
    print("`pKnow` is StanBKT's filtered pre-response mastery estimate; `reference_action` is a derived policy label, not a paper-provided ground-truth action.")


if __name__ == "__main__":
    main()
