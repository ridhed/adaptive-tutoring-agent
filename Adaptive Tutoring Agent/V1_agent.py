def bkt_update(p_know, is_correct, confidence, params):
    p_guess = params['p_guess']
    p_slip = params['p_slip']
    p_learn = params['p_learn']
    p_forget = params['p_forget']

    # 1. Likelihood of the observation
    if is_correct:
        likelihood_m = 1 - p_slip
        likelihood_nm = p_guess
    else:
        likelihood_m = p_slip
        likelihood_nm = 1 - p_guess
        
    prob_evidence = (p_know * likelihood_m) + ((1 - p_know) * likelihood_nm)
    
    # 2. Posterior Given Evidence (Bayes' Rule)
    post_given_ev = (p_know * likelihood_m) / prob_evidence
    new_p_know = post_given_ev + ((1 - post_given_ev) * p_learn) - (post_given_ev * p_forget)
    
    # 3. Action Selection (Updated with Confidence Signal)
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

if __name__ == "__main__":
    params = {
        'p_initial_know': 0.591, 
        'p_guess': 0.245, 
        'p_slip': 0.116, 
        'p_learn': 0.152, 
        'p_forget': 0.0
    }

    # Scenario A: Correct Answer, High Confidence
    ev_correct, post_ev_correct, final_correct, action_correct = bkt_update(
        params['p_initial_know'], True, "HIGH", params
    )
    
    print("--- Scenario A: New Evidence is CORRECT (1), Conf=HIGH ---")
    print(f"Prior P(L):             {params['p_initial_know']:.4f}")
    print(f"Likelihood of Evidence: {ev_correct:.4f}")
    print(f"Posterior (Bayes):      {post_ev_correct:.4f}")
    print(f"Final Posterior P(L):   {final_correct:.4f}")
    print(f"Action Triggered:       {action_correct}\n")

    # Scenario B: Incorrect Answer, Low Confidence
    ev_incorrect, post_ev_incorrect, final_incorrect, action_incorrect = bkt_update(
        params['p_initial_know'], False, "LOW", params
    )

    print("--- Scenario B: New Evidence is INCORRECT (0), Conf=LOW ---")
    print(f"Prior P(L):             {params['p_initial_know']:.4f}")
    print(f"Likelihood of Evidence: {ev_incorrect:.4f}")
    print(f"Posterior (Bayes):      {post_ev_incorrect:.4f}")
    print(f"Final Posterior P(L):   {final_incorrect:.4f}")
    print(f"Action Triggered:       {action_incorrect}\n")

    # Scenario C: Triggering "ASK" via Confidence (High prior, Correct, Low Confidence)
    ev_ask, post_ev_ask, final_ask, action_ask = bkt_update(
        0.980, True, "LOW", params
    )

    print("--- Scenario C: Triggering ASK (Mastery, Correct, Low Conf) ---")
    print(f"Prior P(L):             0.9800")
    print(f"Likelihood of Evidence: {ev_ask:.4f}")
    print(f"Posterior (Bayes):      {post_ev_ask:.4f}")
    print(f"Final Posterior P(L):   {final_ask:.4f}")
    print(f"Action Triggered:       {action_ask}\n")

    # Scenario D: Triggering "HINT" (Low prior, Correct, Any Confidence)
    ev_hint, post_ev_hint, final_hint, action_hint = bkt_update(
        0.300, True, "MEDIUM", params
    )

    print("--- Scenario D: Triggering HINT (Uncertain/Partial, Conf=MEDIUM) ---")
    print(f"Prior P(L):             0.3000")
    print(f"Likelihood of Evidence: {ev_hint:.4f}")
    print(f"Posterior (Bayes):      {post_ev_hint:.4f}")
    print(f"Final Posterior P(L):   {final_hint:.4f}")
    print(f"Action Triggered:       {action_hint}")