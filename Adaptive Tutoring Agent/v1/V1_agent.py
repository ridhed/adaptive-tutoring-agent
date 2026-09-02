import json

def bkt_update(p_know, is_correct, confidence, params):
    p_guess = params['p_guess']
    p_slip = params['p_slip']
    p_learn = params['p_learn']
    p_forget = params['p_forget']

    if is_correct:
        likelihood_m = 1 - p_slip
        likelihood_nm = p_guess
    else:
        likelihood_m = p_slip
        likelihood_nm = 1 - p_guess
        
    prob_evidence = (p_know * likelihood_m) + ((1 - p_know) * likelihood_nm)
    
    post_given_ev = (p_know * likelihood_m) / prob_evidence
    new_p_know = post_given_ev + ((1 - post_given_ev) * p_learn) - (post_given_ev * p_forget)
    
    if new_p_know >= 0.85:
        if is_correct and confidence == "HIGH":
            action = "ANSWER"
        elif is_correct and confidence == "LOW":
            action = "ASK"
        else:
            action = "ASK" if not is_correct else "CONFIRM"
    elif new_p_know >= 0.40:
        action = "HINT"
    else:
        action = "TEACH_PRIOR"
        
    return prob_evidence, post_given_ev, new_p_know, action

def run_experiments_from_file(json_filepath="experiments.json"):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
        
    for item in scenarios:
        s_id = item["scenario_id"]
        name = item["name"]
        inp = item["input_parameters"]
        
        p_know = inp["p_know"]
        is_correct = inp["is_correct"]
        confidence = inp["confidence"]
        params = inp["params"]
        
        ev, post_ev, final_p, action = bkt_update(p_know, is_correct, confidence, params)
        
        print(f"[{s_id.upper()}] {name}")
        print(f"  Inputs  -> Prior P(L): {p_know}, Correct: {is_correct}, Confidence: {confidence}")
        print(f"  Results -> Evidence P(Obs): {ev:.4f} | Bayes Post: {post_ev:.4f} | Final P(L): {final_p:.4f}")
        print(f"  Action  -> Triggered: {action}\n")

if __name__ == "__main__":
    run_experiments_from_file("experiments.json")
