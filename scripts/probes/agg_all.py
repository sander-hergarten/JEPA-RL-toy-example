import json
import statistics

HORIZONS = [1, 5, 10, 30]


def vals(run, name):
    out = []
    for s in range(3):
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


def table(title, rows):
    print(title)
    head = f"{'arm':34s} {'Q':>12s} " + " ".join(f"{'H=' + str(h):>12s}" for h in HORIZONS)
    print(head)
    print("-" * len(head))
    for label, run in rows:
        cells = [cell(vals(run, "eval_q_eps001"))]
        cells += [cell(vals(run, f"eval_lookahead_h{h}_eps001")) for h in HORIZONS]
        print(f"{label:34s} " + " ".join(f"{c:>12s}" for c in cells))
    print()


table("SUCCESSOR FEATURES (wave 15)", [
    ("conv K=5 baseline", "breakout_se_nstep_500k"),
    ("+ successor features", "breakout_se_sf_500k"),
])
sf = vals("breakout_se_sf_500k", "eval_sf_eps001")
if sf:
    print(f"   sf controller (greedy on w.psi, no rollout): {cell(sf)}")
for h in (1, 5):
    b = vals("breakout_se_sf_500k", f"eval_lookahead_h{h}_sfboot_eps001")
    if b:
        print(f"   lookahead H={h} bootstrapping on w.psi:        {cell(b)}")
print()

table("LMU ROLLOUT SWEEP, LIVE (waves 16-17) vs the conv prior", [
    ("conv  K=5", "breakout_se_nstep_500k"),
    ("LMU   K=5", "breakout_se_lmu_500k"),
    ("conv  K=10", "breakout_se_k10_500k"),
    ("LMU   K=10", "breakout_se_lmu_k10_500k"),
    ("conv  K=20", "breakout_se_k20_500k"),
    ("LMU   K=20", "breakout_se_lmu_k20_500k"),
    ("conv  K=30", "breakout_se_k30_500k"),
    ("LMU   K=30", "breakout_se_lmu_k30_500k"),
])

table("LMU + OFFLINE ENCODER (wave 18, in progress)", [
    (f"LMU K={K} offline {m}", f"breakout_se_lmu_k{K}_offline_{m}_500k")
    for K in (5, 10, 30) for m in ("frozen", "finetune")
])
