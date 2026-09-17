import json
import statistics

# seeds 0-2 stored H=1 under the default name, seeds 3-5 under the explicit one
H1 = ["eval_lookahead_h1_eps001", "eval_lookahead_eps001"]


def series(run, names):
    out = {}
    for s in range(6):
        for n in names if isinstance(names, list) else [names]:
            try:
                out[s] = json.load(open(f"runs/{run}/seed{s}/{n}.json"))["return_mean"]
                break
            except FileNotFoundError:
                continue
    return out


def line(label, run, names):
    v = series(run, names)
    if not v:
        print(f"   {label:16s} n=0")
        return
    x = list(v.values())
    m = statistics.mean(x)
    sd = (sum((a - m) ** 2 for a in x) / len(x)) ** 0.5
    print(f"   {label:16s} n={len(x)}  {m:+7.2f} +- {sd:5.2f}   " + " ".join(f"{a:.1f}" for a in x))


for run in ["breakout_se_nstep_500k", "breakout_se_hjepa_detached_500k", "breakout_se_hjepa_500k"]:
    print(run)
    line("Q", run, "eval_q_eps001")
    line("lookahead H=1", run, H1)
    line("lookahead H=3", run, "eval_lookahead_h3_eps001")
    line("lookahead H=5", run, "eval_lookahead_h5_eps001")
    line("hierarchical", run, "eval_hierarchical_eps001")
    print()
