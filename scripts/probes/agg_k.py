"""K x H table: mean +- population std over seeds, from the eps=0.01 evaluations."""
import json
import statistics
import sys

RUNS = [("K=5", "breakout_se_nstep_500k"), ("K=10", "breakout_se_k10_500k"),
        ("K=20", "breakout_se_k20_500k"), ("K=30", "breakout_se_k30_500k")]
HORIZONS = [1, 3, 5, 10, 20, 30]
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)


def load(run, name):
    vals = []
    for s in SEEDS:
        for n in ([name, "eval_lookahead_eps001"] if name == "eval_lookahead_h1_eps001" else [name]):
            try:
                vals.append(json.load(open(f"runs/{run}/seed{s}/{n}.json"))["return_mean"])
                break
            except FileNotFoundError:
                continue
    return vals


def cell(vals):
    if not vals:
        return "     -    "
    m = statistics.mean(vals)
    sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    return f"{m:+6.2f}±{sd:4.1f}" + ("" if len(vals) == len(list(SEEDS)) else f"({len(vals)})")

head = f"{'arm':6s} {'Q':>12s} " + " ".join(f"{'H=' + str(h):>12s}" for h in HORIZONS)
print(head)
print("-" * len(head))
for label, run in RUNS:
    row = [cell(load(run, "eval_q_eps001"))]
    row += [cell(load(run, f"eval_lookahead_h{h}_eps001")) for h in HORIZONS]
    print(f"{label:6s} " + " ".join(f"{c:>12s}" for c in row))

print()
print("latency ms / decision (mean over seeds)")
for label, run in RUNS:
    lat = []
    for h in HORIZONS:
        v = []
        for s in SEEDS:
            for n in ([f"eval_lookahead_h{h}_eps001"] + (["eval_lookahead_eps001"] if h == 1 else [])):
                try:
                    v.append(json.load(open(f"runs/{run}/seed{s}/{n}.json"))["controller_stats"]["latency_ms_mean"])
                    break
                except FileNotFoundError:
                    continue
        lat.append(f"{statistics.mean(v):6.2f}" if v else "     -")
    print(f"{label:6s} " + " ".join(lat))
