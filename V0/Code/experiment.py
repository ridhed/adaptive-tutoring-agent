import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N = 5000
rng = np.random.default_rng(SEED)

STATES = ["mastery", "carelessness", "execution_error", "conceptual_difficulty"]
ACTIONS = ["ANSWER", "ASK", "HINT", "TEACH_PRIOR"]
ORACLE = {"mastery": "ANSWER", "carelessness": "ASK",
          "execution_error": "HINT", "conceptual_difficulty": "TEACH_PRIOR"}
PRIOR = np.array([0.35, 0.15, 0.15, 0.35])

# TRUE generative parameters (simulation only)
P_CORRECT = np.array([0.95, 0.80, 0.55, 0.20])
P_FAST    = np.array([0.50, 0.80, 0.20, 0.35])
P_ATTEMPT_GT1 = np.array([0.05, 0.10, 0.20, 0.50])
P_HINT    = np.array([0.05, 0.05, 0.10, 0.40])
P_PRIOR_SUCC = np.array([0.90, 0.85, 0.75, 0.40])   # kc_success_history >= 1
P_REPEAT_MIS = np.array([0.05, 0.15, 0.30, 0.70])   # mistake_history >= 2

# ASSUMED parameters used by the agent: true values perturbed by assumption error
perturb = np.random.default_rng(7).normal(0, 0.05, size=(6, 4))
def clip(x): return np.clip(x, 0.01, 0.99)
A = {
    "correct": clip(P_CORRECT + perturb[0]),
    "fast":    clip(P_FAST + perturb[1]),
    "att":     clip(P_ATTEMPT_GT1 + perturb[2]),
    "hint":    clip(P_HINT + perturb[3]),
    "succ":    clip(P_PRIOR_SUCC + perturb[4]),
    "mis":     clip(P_REPEAT_MIS + perturb[5]),
}

# Loss matrix: LOSS[action][state]  (assumed pedagogical costs)
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}
ST_IDX = {s: i for i, s in enumerate(STATES)}
LOSS = np.array([
    #  mastery carel exec  concept
    [0.0,   1.0,  1.0,  3.0],   # ANSWER
    [0.5,   0.0,  0.5,  0.5],   # ASK (mild cost for asking, plus interaction cost)
    [0.7,   1.0,  0.0,  1.0],   # HINT
    [1.2,   1.0,  1.0,  0.0],   # TEACH_PRIOR
])
C_ASK = 0.5

# ---- simulate students ----
true_state = rng.choice(4, size=N, p=PRIOR)
obs = {}
obs["correct"] = (rng.random(N) < P_CORRECT[true_state]).astype(int)
obs["fast"] = (rng.random(N) < P_FAST[true_state]).astype(int)
obs["att"] = (rng.random(N) < P_ATTEMPT_GT1[true_state]).astype(int)
obs["hint"] = (rng.random(N) < P_HINT[true_state]).astype(int)
obs["succ"] = (rng.random(N) < P_PRIOR_SUCC[true_state]).astype(int)
obs["mis"] = (rng.random(N) < P_REPEAT_MIS[true_state]).astype(int)
# diagnostic probe used when ASK is chosen: correct on a fresh item
probe = (rng.random(N) < P_CORRECT[true_state]).astype(int)
# reconstruct the V0/V0.1 style raw fields
response_time = np.where(obs["fast"] == 1, "FAST", "SLOW")
attempt_number = np.where(obs["att"] == 1, 2, 1)
hint_count = obs["hint"]
kc_success_history = np.where(obs["succ"] == 1, 1, 0)
mistake_history = np.where(obs["mis"] == 1, 2, 0)
is_correct = obs["correct"].astype(bool)

# ---- Agents ----
def agent_v0(c, att, rt):
    if not c:
        if att > 1: return "ASK"
        if rt == "FAST": return "ASK"
        return "TEACH_PRIOR"
    return "ASK" if att > 1 else "ANSWER"

def agent_v01(c, succ, rt, att, hint, mis):
    if c: return "ANSWER"
    if att > 1 or hint > 0 or mis >= 2: return "TEACH_PRIOR"
    if att == 1 and hint == 0 and succ >= 1:
        return "ASK" if rt == "FAST" else "HINT"
    return "ASK"

def posterior(c, fast, att, hint, succ, mis, prior=PRIOR):
    lk = np.ones(4)
    for val, p in [(c, A["correct"]), (fast, A["fast"]), (att, A["att"]),
                   (hint, A["hint"]), (succ, A["succ"]), (mis, A["mis"])]:
        lk *= p if val == 1 else (1 - p)
    post = prior * lk
    return post / post.sum()

def entropy(p):
    p = p[p > 0]; return float(-(p * np.log2(p)).sum())

def expected_loss(post):
    return LOSS @ post  # vector over actions

def probe_value(post):
    """Expected loss after an ASK probe (one binary diagnostic item)."""
    total_loss = 0.0; exp_H = 0.0
    for d in [1, 0]:
        pd = np.where(d == 1, A["correct"], 1 - A["correct"])  # P(d|state)
        joint = post * pd
        pdat = joint.sum()
        if pdat == 0: continue
        post_d = joint / pdat
        total_loss += pdat * expected_loss(post_d).min()
        exp_H += pdat * entropy(post_d)
    return total_loss, entropy(post) - exp_H

def agent_prob(o):
    post = posterior(*o)
    el = expected_loss(post)
    el[ACT_IDX["ASK"]] = C_ASK + probe_value(post)[0]
    return ACTIONS[int(np.argmin(el))], post

# ---- Run ----
preds = {"V0": [], "V0.1": [], "Prob+InfoSel": []}
posts = []; ent_before = []; info_gain = []
for i in range(N):
    preds["V0"].append(agent_v0(is_correct[i], attempt_number[i], response_time[i]))
    preds["V0.1"].append(agent_v01(is_correct[i], kc_success_history[i], response_time[i],
                                   attempt_number[i], hint_count[i], mistake_history[i]))
    a, post = agent_prob((obs["correct"][i], obs["fast"][i], obs["att"][i],
                          obs["hint"][i], obs["succ"][i], obs["mis"][i]))
    preds["Prob+InfoSel"].append(a)
    posts.append(post)
    ent_before.append(entropy(post))
    info_gain.append(probe_value(post)[1])

y_true = [ORACLE[STATES[s]] for s in true_state]
posts = np.array(posts)

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
res = {}
for name, p in preds.items():
    losses = [LOSS[ACT_IDX[pa], ST_IDX[STATES[s]]] for pa, s in zip(p, true_state)]
    res[name] = {
        "accuracy": accuracy_score(y_true, p),
        "kappa": cohen_kappa_score(y_true, p),
        "mean_loss": float(np.mean(losses)),
        "ask_rate": float(np.mean([x == "ASK" for x in p])),
        "confusion": confusion_matrix(y_true, p, labels=ACTIONS).tolist(),
    }
    # failure classes: (oracle action, predicted action) mismatches
    fails = {}
    for t, pa in zip(y_true, p):
        if t != pa: fails[f"{t}->{pa}"] = fails.get(f"{t}->{pa}", 0) + 1
    res[name]["failures"] = dict(sorted(fails.items(), key=lambda kv: -kv[1]))

# Informational summaries
res["_meta"] = {
    "N": N, "seed": SEED,
    "mean_entropy_before_action_bits": float(np.mean(ent_before)),
    "mean_info_gain_of_probe_bits": float(np.mean(info_gain)),
    "median_info_gain_bits": float(np.median(info_gain)),
}
# Analytic threshold for ANSWER vs TEACH_PRIOR (2-way case)
c_ans_wrong = LOSS[ACT_IDX["ANSWER"], ST_IDX["conceptual_difficulty"]]
c_teach_wrong = LOSS[ACT_IDX["TEACH_PRIOR"], ST_IDX["mastery"]]
res["_meta"]["threshold_answer_vs_teach"] = float(c_ans_wrong / (c_ans_wrong + c_teach_wrong))

# Worked example: one observation pattern
ex = dict(c=1, fast=0, att=1, hint=0, succ=1, mis=0)
ex_post = posterior(**ex)
res["_meta"]["worked_example_posterior"] = dict(zip(STATES, [float(x) for x in ex_post]))
ex_bad = dict(c=0, fast=1, att=1, hint=0, succ=1, mis=0)
res["_meta"]["worked_example_2_posterior"] = dict(zip(STATES, [float(x) for x in posterior(**ex_bad)]))
res["_meta"]["worked_example_2_probe"] = probe_value(posterior(**ex_bad))
res["_meta"]["worked_example_2_action"] = agent_prob(tuple(ex_bad.values()))[0]

# Per-state accuracy for prob agent
pa = np.array(preds["Prob+InfoSel"]); ts = np.array(y_true)
res["_meta"]["prob_per_state_acc"] = {
    STATES[k]: float(np.mean(pa[true_state == k] == ts[true_state == k])) for k in range(4)}
res["_meta"]["v01_per_state_acc"] = {
    STATES[k]: float(np.mean(np.array(preds["V0.1"])[true_state == k] == ts[true_state == k])) for k in range(4)}

json.dump(res, open("results.json", "w"), indent=2, default=float)

# Figure: accuracy & loss
names = list(preds.keys())
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
ax[0].bar(names, [res[n]["accuracy"] for n in names], color=["#999", "#666", "#222"])
ax[0].set_title("Accuracy vs oracle action"); ax[0].set_ylim(0, 1)
ax[1].bar(names, [res[n]["mean_loss"] for n in names], color=["#999", "#666", "#222"])
ax[1].set_title("Mean pedagogical loss"); 
for a in ax: a.tick_params(axis="x", labelsize=7)
plt.tight_layout(); plt.savefig("fig_results.pdf")
print(json.dumps({k: (v if k=="_meta" else {kk: vv for kk, vv in v.items() if kk!="confusion"}) for k, v in res.items()}, indent=2, default=float))
