"""K x H return table plus the model-quality diagnostics, including the capped-value-horizon arms."""
import json
import statistics

RUNS = [("K=5", "breakout_se_nstep_500k"),
        ("K=10", "breakout_se_k10_500k"),
        ("K=10/qd5", "breakout_se_k10_qd5_500k"),
        ("K=20", "breakout_se_k20_500k"),
        ("K=20/qd5", "breakout_se_k20_qd5_500k"),
        ("K=30", "breakout_se_k30_500k")]
HORIZONS = [1, 3, 5, 10, 20, 30]
SEEDS = range(3)


def vals(run, name):
    out = []
    for s in SEEDS:
        names = [name] + (["eval_lookahead_eps001"] if name == "eval_lookahead_h1_eps001" else [])
        for n in names:
            try:
                out.append(json.load(open(f"runs/{run}/seed{s}/{n}.json"))["return_mean"])
                break
            except FileNotFoundError:
                continue
    return out


def cell(v):
    if not v:
        return "     -    "
    m = statistics.mean(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    return f"{m:+6.2f}±{sd:4.1f}" + ("" if len(v) == 3 else f"({len(v)})")


head = f"{'arm':9s} {'Q':>12s} " + " ".join(f"{'H=' + str(h):>12s}" for h in HORIZONS)
print(head)
print("-" * len(head))
for label, run in RUNS:
    row = [cell(vals(run, "eval_q_eps001"))]
    row += [cell(vals(run, f"eval_lookahead_h{h}_eps001")) for h in HORIZONS]
    print(f"{label:9s} " + " ".join(f"{c:>12s}" for c in row))

print()
print(f"{'arm':9s} {'d=1':>7s} {'d=5':>7s} {'d=10':>7s} {'d=20':>7s} {'d=30':>7s}   action_dist  probe")
for label, run in RUNS:
    cells, act, probe = [], [], []
    for depth in [1, 5, 10, 20, 30]:
        v = []
        for s in SEEDS:
            try:
                d = json.load(open(f"runs/{run}/seed{s}/diagnostics.json"))
            except FileNotFoundError:
                continue
            b = d["latent_prediction_cosine_distance"].get(f"depth_{depth}")
            if b:
                v.append(b["predicted"])
        cells.append(f"{statistics.mean(v):7.4f}" if v else "      -")
    for s in SEEDS:
        try:
            d = json.load(open(f"runs/{run}/seed{s}/diagnostics.json"))
        except FileNotFoundError:
            continue
        act.append(d["action_sensitivity_at_root"]["mean_pairwise_cosine_distance_next_latent"])
        probe.append(d["movement_probe"]["test_balanced_acc"])
    tail = f"   {statistics.mean(act):.4f}      {statistics.mean(probe):.3f}" if act else ""
    print(f"{label:9s} " + " ".join(cells) + tail)
