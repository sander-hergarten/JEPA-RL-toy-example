"""Preview of wave 18 from the in-pipeline evals (eps=0, 10 episodes), with matched-protocol
references computed the same way so the comparison is like-for-like."""
import json
import statistics

ROWS = [
    ("conv  K=5  live", "breakout_se_nstep_500k"),
    ("LMU   K=5  live", "breakout_se_lmu_500k"),
    ("LMU   K=5  frozen", "breakout_se_lmu_k5_offline_frozen_500k"),
    ("LMU   K=5  finetune", "breakout_se_lmu_k5_offline_finetune_500k"),
    ("conv  K=10 live", "breakout_se_k10_500k"),
    ("LMU   K=10 live", "breakout_se_lmu_k10_500k"),
    ("LMU   K=10 frozen", "breakout_se_lmu_k10_offline_frozen_500k"),
    ("LMU   K=10 finetune", "breakout_se_lmu_k10_offline_finetune_500k"),
    ("conv  K=30 live", "breakout_se_k30_500k"),
    ("LMU   K=30 live", "breakout_se_lmu_k30_500k"),
    ("LMU   K=30 frozen", "breakout_se_lmu_k30_offline_frozen_500k"),
    ("LMU   K=30 finetune", "breakout_se_lmu_k30_offline_finetune_500k"),
]


def cell(run, name):
    v = []
    for s in range(3):
        try:
            v.append(json.load(open(f"runs/{run}/seed{s}/{name}.json"))["return_mean"])
        except FileNotFoundError:
            pass
    if not v:
        return "     -    "
    m = statistics.mean(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    return f"{m:+6.2f}±{sd:4.1f}" + ("" if len(v) == 3 else f"({len(v)})")


print(f"{'arm':24s} {'Q (eps=0)':>12s} {'lookahead H=1':>14s}   [in-pipeline evals, eps=0]")
print("-" * 58)
for label, run in ROWS:
    print(f"{label:24s} {cell(run, 'eval_q'):>12s} {cell(run, 'eval_lookahead'):>14s}")

print()
print("diagnostics: latent smoothing and action sensitivity")
print(f"{'arm':24s} {'persist d1':>11s} {'pred d1':>9s} {'act dist':>9s} {'probe':>7s} {'reach':>7s}")
for label, run in ROWS:
    vals = {}
    for s in range(3):
        try:
            d = json.load(open(f"runs/{run}/seed{s}/diagnostics.json"))
        except FileNotFoundError:
            continue
        b = d["latent_prediction_cosine_distance"]["depth_1"]
        vals.setdefault("persist", []).append(b["persistence"])
        vals.setdefault("pred", []).append(b["predicted"])
        vals.setdefault("act", []).append(
            d["action_sensitivity_at_root"]["mean_pairwise_cosine_distance_next_latent"])
        vals.setdefault("probe", []).append(d["movement_probe"]["test_balanced_acc"])
        if "gradient_reach" in d:
            vals.setdefault("reach", []).append(d["gradient_reach"]["reach_root_over_deepest"])
    if not vals:
        continue
    g = lambda k, f="{:.4f}": f.format(statistics.mean(vals[k])) if k in vals else "   -"
    print(f"{label:24s} {g('persist'):>11s} {g('pred'):>9s} {g('act'):>9s} "
          f"{g('probe', '{:.3f}'):>7s} {g('reach', '{:.2f}x'):>7s}")
