import csv

def load_v0_data(path="skill_builder_data.csv", limit=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if limit and i >= limit: break
            
            try:
                ms = float(r["ms_first_response"]) if r["ms_first_response"] else 0.0
                correct = int(r["correct"]) if r["correct"] else 0
                attempt = int(r["attempt_count"]) if r["attempt_count"] else 1
            except ValueError:
                continue

            rows.append({
                "student_id": r["user_id"],
                "kc_id": r["skill_id"],
                "correct": correct == 1,
                "attempt_number": attempt,
                "time_seconds": ms / 1000.0,
            })
    return rows

def discretize_time(time_seconds, expected_time=30.0):
    return "FAST" if time_seconds < (expected_time * 0.3) else "SLOW"

def evaluate_policy(correct, attempt_number, response_time):
    if not correct:
        if attempt_number > 1:
            return {"state": "misconception", "action": "ASK"}
        else:
            if response_time == "FAST":
                return {"state": "careless", "action": "ASK"}
            else:
                return {"state": "knowledge_gap", "action": "TEACH_PRIOR"}
    else:
        if attempt_number > 1:
            return {"state": "guessing", "action": "ASK"}
        else:
            return {"state": "mastery", "action": "ANSWER"}

def run_expert_system(dataset_path):
    data = load_v0_data(dataset_path, limit=10)
    for interaction in data:
        speed = discretize_time(interaction["time_seconds"])
        decision = evaluate_policy(interaction["correct"], interaction["attempt_number"], speed)
        
        print(f"Student: {interaction['student_id']} | Correct: {interaction['correct']} | "
              f"Attempt: {interaction['attempt_number']} | Speed: {speed}")
        print(f"-> Inferred State: {decision['state']} | Action: {decision['action']}\n")

if __name__ == "__main__":
    run_expert_system("skill_builder_data.csv")