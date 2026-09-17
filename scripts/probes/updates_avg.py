import json, glob, os

arms = {"K=5": "breakout_se_nstep_500k", "K=10": "breakout_se_k10_500k", "K=20": "breakout_se_k20_500k",
        "K=30": "breakout_se_k30_500k", "K=30 inverse": "breakout_se_k30_depth_inverse_500k",
        "K=30 discount": "breakout_se_k30_depth_discount_500k", "K=20 qd5": "breakout_se_k20_qd5_500k",
        "LMU K=30": "breakout_se_lmu_k30_500k"}

def jl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out

for name, arm in arms.items():
    print("=" * 90)
    print(name, arm)
    per_seed = {}
    for sd in sorted(glob.glob(f"runs/{arm}/seed*")):
        p = f"{sd}/updates.jsonl"
        if not os.path.exists(p):
            continue
        recs = jl(p)
        if not recs:
            continue
        last = recs[-1]
        n = min(40, len(recs))
        keys = [k for k in last if k.startswith(("jepa_d", "persist_d", "q_td_d", "grad_norm", "loss_jepa", "loss_q", "loss_continue", "loss_reward"))]
        avg = {}
        for k in keys:
            vals = [r[k] for r in recs[-n:] if isinstance(r.get(k), (int, float))]
            if vals:
                avg[k] = sum(vals) / len(vals)
        per_seed[sd] = avg
        print(" ", sd, "records", len(recs), "decisions", last.get("decisions"), "updates", last.get("updates"))
        # also record nearest to 436k decisions
        near = min(recs, key=lambda r: abs((r.get("decisions") or 0) - 436000))
        print("   nearest-436k decisions:", near.get("decisions"), "jepa_d1", round(near.get("jepa_d1", float('nan')), 4), "jepa_d2", round(near.get("jepa_d2", float('nan')), 4), "persist_d1", round(near.get("persist_d1", float('nan')), 4), "q_td_d0", round(near.get("q_td_d0", float('nan')), 4))
    if not per_seed:
        continue
    keys = sorted(set().union(*[set(a) for a in per_seed.values()]))
    def keyorder(k):
        import re
        m = re.match(r"([a-z_]+?)(\d+)$", k)
        return (m.group(1), int(m.group(2))) if m else (k, 0)
    keys.sort(key=keyorder)
    print("  seed-mean of last-40-record averages:")
    row = {}
    for k in keys:
        vals = [a[k] for a in per_seed.values() if k in a]
        row[k] = round(sum(vals) / len(vals), 4)
    for k in keys:
        print("   ", k, row[k])
