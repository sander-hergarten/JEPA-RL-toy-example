import json
import statistics
import sys

RUNS = sys.argv[1:] or ["breakout_se_nstep_500k", "breakout_se_hjepa_detached_500k"]
FILES = ["eval_q_eps001", "eval_lookahead_h1_eps001", "eval_lookahead_h5_eps001"]

for run in RUNS:
    print(run)
    for f in FILES:
        per = {}
        for s in range(6):
            try:
                per[s] = json.load(open(f"runs/{run}/seed{s}/{f}.json"))["return_mean"]
            except FileNotFoundError:
                pass
        if not per:
            print(f"   {f:26s} MISSING")
            continue
        v = list(per.values())
        m = statistics.mean(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        seeds = " ".join(f"{s}:{x:.1f}" for s, x in per.items())
        print(f"   {f:26s} n={len(v)}  {m:+7.2f} +- {sd:5.2f}   {seeds}")
    print()
