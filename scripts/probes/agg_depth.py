"""Latent prediction quality by depth, per K arm: did a longer training rollout make the model better
at deep prediction, even where it made control worse?"""
import json
import statistics

RUNS = [("K=5", "breakout_se_nstep_500k"), ("K=10", "breakout_se_k10_500k"),
        ("K=20", "breakout_se_k20_500k"), ("K=30", "breakout_se_k30_500k")]
DEPTHS = [1, 3, 5, 10, 20, 30]


def m(run, depth, key):
    vals = []
    for s in range(3):
        try:
            d = json.load(open(f"runs/{run}/seed{s}/diagnostics.json"))
        except FileNotFoundError:
            continue
        block = d["latent_prediction_cosine_distance"].get(f"depth_{depth}")
        if block and key in block:
            vals.append(block[key])
    return statistics.mean(vals) if vals else None


def row(label, run, key):
    out = []
    for depth in DEPTHS:
        v = m(run, depth, key)
        out.append(f"{v:7.4f}" if v is not None else "      -")
    print(f"{label:6s} " + " ".join(out))


for key, title in [("predicted", "predicted-vs-real latent distance (lower = better model)"),
                   ("persistence", "persistence baseline (predict no change)"),
                   ("shuffled_actions", "same rollout with the action sequence shuffled")]:
    print(title)
    print(f"{'arm':6s} " + " ".join(f"{'d=' + str(d):>7s}" for d in DEPTHS))
    for label, run in RUNS:
        row(label, run, key)
    print()

print("action sensitivity at the root / movement probe")
for label, run in RUNS:
    a, p = [], []
    for s in range(3):
        try:
            d = json.load(open(f"runs/{run}/seed{s}/diagnostics.json"))
        except FileNotFoundError:
            continue
        a.append(d["action_sensitivity_at_root"]["mean_pairwise_cosine_distance_next_latent"])
        p.append(d["movement_probe"]["test_balanced_acc"])
    if a:
        print(f"{label:6s} action_dist {statistics.mean(a):.4f}   probe_balanced_acc {statistics.mean(p):.3f}")
