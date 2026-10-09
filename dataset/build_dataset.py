#!/usr/bin/env python3
"""Build chronological BKT reference labels from an authorized ASSISTments CSV."""
import argparse
import copy
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path

DEFAULT = {
    "columns": {"student": "user_id", "problem": "problem_id", "skill": "skill_id", "correct": "correct",
                "order": "order_id", "timestamp": "timestamp", "response_ms": "ms_first_response",
                "hints": "hint_count", "attempt": "attempt_count"},
    "multi_skill_delimiter": None,
    "bkt": {"p_initial_know": .591, "p_guess": .245, "p_slip": .116, "p_learn": .152, "p_forget": 0.0},
    "policy": {"mastery_threshold": .85, "gap_threshold": .40, "recent_error_window": 3, "ask_errors": 2},
    "split": {"seed": 17, "train": .8, "validation": .1, "test": .1},
}
OUT_FIELDS = ["student_id", "problem_id", "kc_id", "sequence_index", "correct", "mastery_pre", "mastery_post",
              "prior_errors", "prior_attempts", "policy_action", "response_time_seconds", "hint_count", "attempt_count", "split"]

def validate_config(cfg):
    for name, p in cfg["bkt"].items():
        if not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError(f"BKT parameter {name} must be between 0 and 1")
    sp = cfg["split"]
    if any(sp[k] < 0 for k in ("train", "validation", "test")) or not math.isclose(sum(sp[k] for k in ("train", "validation", "test")), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must be nonnegative and sum to 1")
    po = cfg["policy"]
    if not 0 <= po["gap_threshold"] < po["mastery_threshold"] <= 1 or po["recent_error_window"] < 1 or po["ask_errors"] < 1:
        raise ValueError("invalid policy thresholds")

def bkt_step(prior, correct, p):
    lm = 1-p["p_slip"] if correct else p["p_slip"]
    lnm = p["p_guess"] if correct else 1-p["p_guess"]
    evidence = prior*lm + (1-prior)*lnm
    posterior = prior*lm/evidence if evidence else prior
    return min(1.0, max(0.0, posterior + (1-posterior)*p["p_learn"] - posterior*p["p_forget"]))

def split_for(student, cfg):
    sp = cfg["split"]
    val = int(hashlib.sha256(f"{sp['seed']}:{student}".encode()).hexdigest()[:16], 16) / 16**16
    return "train" if val < sp["train"] else "validation" if val < sp["train"] + sp["validation"] else "test"

def action_for(pre, errors, cfg):
    po = cfg["policy"]
    if pre < po["gap_threshold"]: return "TEACH_PRIOR"
    if pre < po["mastery_threshold"]: return "HINT"
    if sum(errors) >= po["ask_errors"]: return "ASK"
    return "ANSWER"

def build(input_path, output_path, metadata_path=None, config=None):
    cfg = json.loads(Path(config).read_text(encoding="utf-8")) if config else copy.deepcopy(DEFAULT)
    # Allow partial config files while retaining documented defaults.
    if config:
        base = copy.deepcopy(DEFAULT)
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict): base[key] = {**base[key], **value}
            else: base[key] = value
        cfg = base
    validate_config(cfg)
    cols = cfg["columns"]
    required = ["student", "skill", "correct"]
    rows, skipped = [], 0
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [cols[k] for k in required if cols[k] not in (reader.fieldnames or [])]
        if missing: raise ValueError("input missing required columns: " + ", ".join(missing))
        for i, raw in enumerate(reader):
            try:
                student, skill = raw[cols["student"]].strip(), raw[cols["skill"]].strip()
                correct = int(raw[cols["correct"]])
                if not student or not skill or correct not in (0, 1): raise ValueError()
            except (KeyError, ValueError, AttributeError):
                skipped += 1; continue
            delim = cfg.get("multi_skill_delimiter")
            if delim and delim in skill: skill = skill.split(delim)[0].strip()
            order_col = cols.get("order")
            timestamp_col = cols.get("timestamp")
            order = raw.get(order_col, "") if order_col else ""
            stamp = raw.get(timestamp_col, "") if timestamp_col else ""
            rows.append({"student": student, "skill": skill, "correct": correct, "input_index": i,
                         "order": order, "timestamp": stamp,
                         "problem": raw.get(cols.get("problem", ""), "") or "",
                         "response_ms": raw.get(cols.get("response_ms", ""), "") or "",
                         "hints": raw.get(cols.get("hints", ""), "") or "",
                         "attempt": raw.get(cols.get("attempt", ""), "") or ""})
    def order_key(r):
        value = r["order"] or r["timestamp"]
        if value:
            try: return (0, float(value), r["input_index"])
            except ValueError: return (1, value, r["input_index"])
        return (2, r["input_index"], r["input_index"])
    grouped = defaultdict(list)
    for r in rows: grouped[(r["student"], r["skill"])].append(r)
    out = []
    for (student, skill), seq in grouped.items():
        seq.sort(key=order_key); mastery = cfg["bkt"]["p_initial_know"]; errors = deque(maxlen=cfg["policy"]["recent_error_window"]); prior_attempts = 0
        for n, r in enumerate(seq, 1):
            pre = mastery
            out.append({"student_id": student, "problem_id": r["problem"], "kc_id": skill, "sequence_index": n,
                        "correct": r["correct"], "mastery_pre": f"{pre:.8f}", "mastery_post": f"{bkt_step(pre, bool(r['correct']), cfg['bkt']):.8f}",
                        "prior_errors": sum(errors), "prior_attempts": prior_attempts,
                        "policy_action": action_for(pre, errors, cfg),
                        "response_time_seconds": _float_or_empty(r["response_ms"], 1000), "hint_count": r["hints"], "attempt_count": r["attempt"],
                        "split": split_for(student, cfg)})
            mastery = bkt_step(pre, bool(r["correct"]), cfg["bkt"]); errors.append(1-r["correct"]); prior_attempts += 1
    out.sort(key=lambda x: (x["split"], x["student_id"], x["kc_id"], int(x["sequence_index"])))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS); writer.writeheader(); writer.writerows(out)
    metadata = {"input": str(input_path), "output": str(output_path), "input_rows": len(rows)+skipped, "output_rows": len(out), "skipped_rows": skipped,
                "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "order_source": "configured order_id when populated, otherwise timestamp, otherwise input row order",
                "bkt_parameters": cfg["bkt"], "policy": cfg["policy"], "split": cfg["split"],
                "labels_note": "correct is observed; mastery_pre/post are BKT estimates; policy_action is a rule-generated recommendation."}
    mpath = Path(metadata_path) if metadata_path else Path(str(output_path)+".metadata.json")
    mpath.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return metadata

def _float_or_empty(value, divisor):
    try: return f"{float(value)/divisor:.6f}" if value.strip() else ""
    except (ValueError, AttributeError): return ""

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="authorized ASSISTments interaction CSV")
    ap.add_argument("--output", required=True, help="derived CSV destination")
    ap.add_argument("--metadata", help="metadata JSON destination (default: OUTPUT.metadata.json)")
    ap.add_argument("--config", help="optional JSON configuration")
    args = ap.parse_args()
    meta = build(args.input, args.output, args.metadata, args.config)
    print(f"Wrote {meta['output_rows']} rows; skipped {meta['skipped_rows']} invalid rows.")

if __name__ == "__main__": main()
