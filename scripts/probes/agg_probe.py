"""Delta-probe summary: mean-relative score (0 = the conditional-mean predictor, 1 = exact)."""
import json
import statistics

GROUPS = [("K=5  500k", "breakout_se_nstep_500k", [0, 1, 2]),
          ("K=30 500k", "breakout_se_k30_500k", [0, 1, 2]),
          ("K=5  2.5M", "breakout_long_n5", [1])]
KEYS = [("persistence", "persist"), ("iterated", "iterated"),
        ("direct_action_summary", "direct/act"), ("direct_none", "direct/none")]


def rows(run, seeds):
    out = {}
    for s in seeds:
        try:
            data = json.load(open(f"runs/{run}/seed{s}/delta_probe.json"))
        except FileNotFoundError:
            continue
        for r in data["rows"]:
            out.setdefault(r["delta"], []).append(r)
    return out


for label, run, seeds in GROUPS:
    data = rows(run, seeds)
    if not data:
        print(f"{label}: no probe data\n")
        continue
    n = max(len(v) for v in data.values())
    print(f"{label}   (n={n} seeds)")
    print(f"{'delta':>6} " + " ".join(f"{name:>12s}" for _, name in KEYS))
    for delta in sorted(data):
        cells = []
        for key, _ in KEYS:
            v = [r[f"score_{key}"] for r in data[delta]]
            m = statistics.mean(v)
            sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
            cells.append(f"{m:+6.3f}±{sd:4.2f}" if len(v) > 1 else f"{m:+6.3f}     ")
        print(f"{delta:>6} " + " ".join(f"{c:>12s}" for c in cells))
    print()
