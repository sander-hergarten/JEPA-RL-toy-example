import json, glob, os, statistics as st

def load(p):
    with open(p) as f:
        return json.load(f)

def summ(e):
    keys = list(e.keys())
    eps = e.get("episodes")
    out = {k: e.get(k) for k in ("controller", "planning_horizon", "epsilon", "return_mean", "return_std") if k in e}
    if isinstance(eps, list) and eps and isinstance(eps[0], dict):
        ek = list(eps[0].keys())
        out["n"] = len(eps)
        out["ep_keys"] = ek
        for k in ("raw_return", "return", "clipped_return", "decisions", "length"):
            if k in eps[0]:
                vals = [x[k] for x in eps if x.get(k) is not None]
                out["mean_" + k] = round(st.mean(vals), 2)
    cs = e.get("controller_stats")
    if cs:
        out["disagree"] = round(cs.get("disagreement_with_q", float("nan")), 3)
    return out

print("### FINAL eps=0 evals written by the run itself (10 eps): inverse K=30 vs uniform K=30 vs K=5")
for arm in ("breakout_se_k30_depth_inverse_500k", "breakout_se_k30_500k", "breakout_se_nstep_500k",
            "breakout_se_k30_depth_discount_500k"):
    for sd in sorted(glob.glob(f"runs/{arm}/seed[0-2]")):
        for fn in ("eval_q.json", "eval_lookahead.json"):
            p = f"{sd}/{fn}"
            if os.path.exists(p):
                print(f"  {arm} {os.path.basename(sd)} {fn}: {summ(load(p))}")
    print()

print("### in-training eval.jsonl series for inverse arm (decisions, controller, return_mean)")
for sd in sorted(glob.glob("runs/breakout_se_k30_depth_inverse_500k/seed[0-2]")):
    recs = [json.loads(l) for l in open(f"{sd}/eval.jsonl") if l.strip()]
    print("  ", os.path.basename(sd), [(r.get("decisions"), r.get("controller"), round(r.get("return_mean", float('nan')), 1)) for r in recs])
print("### same for uniform K=30")
for sd in sorted(glob.glob("runs/breakout_se_k30_500k/seed[0-2]")):
    recs = [json.loads(l) for l in open(f"{sd}/eval.jsonl") if l.strip()]
    print("  ", os.path.basename(sd), [(r.get("decisions"), r.get("controller"), round(r.get("return_mean", float('nan')), 1)) for r in recs])

print("\n### config of inverse arm")
for line in open("runs/breakout_se_k30_depth_inverse_500k/seed0/config.yaml").read().splitlines():
    if any(s in line for s in ("depth_weight", "depth_gamma", "rollout_steps", "q_imagined", "n_step", "total_decisions", "epsilon", "horizon")):
        print("  ", line.strip())
print("### train.log tail of inverse arm seed0")
for l in open("runs/breakout_se_k30_depth_inverse_500k/seed0/train.log", errors="replace").read().splitlines()[-6:]:
    print("  ", l[:220])

print("\n### per-seed eps=0.01 evals K=5/10/20/30: Q, H1, H5 -- raw, clipped, decisions")
ARMS = {5: "breakout_se_nstep_500k", 10: "breakout_se_k10_500k", 20: "breakout_se_k20_500k", 30: "breakout_se_k30_500k"}
FILES = {"Q": "eval_q_eps001.json", "H1": "eval_lookahead_h1_eps001.json", "H3": "eval_lookahead_h3_eps001.json", "H5": "eval_lookahead_h5_eps001.json"}
first = True
for K, arm in ARMS.items():
    for sd in sorted(glob.glob(f"runs/{arm}/seed[0-2]")):
        row = []
        for tag, fn in FILES.items():
            p = f"{sd}/{fn}"
            if not os.path.exists(p):
                row.append(f"{tag}: missing")
                continue
            e = load(p)
            eps = e.get("episodes", [])
            if first and eps:
                print("  episode keys:", list(eps[0].keys()))
                first = False
            raw = [x.get("raw_return", x.get("return")) for x in eps]
            clip = [x.get("clipped_return") for x in eps]
            dec = [x.get("decisions") for x in eps]
            ok = all(v is not None for v in clip) and all(v is not None for v in dec)
            if ok:
                rate = 1000 * sum(clip) / sum(dec)
                row.append(f"{tag}: raw {st.mean(raw):5.1f} clip {st.mean(clip):5.1f} dec {st.mean(dec):6.0f} rate/1k {rate:5.1f}")
            else:
                row.append(f"{tag}: raw {st.mean(raw):5.1f} (no clip/dec fields)")
        print(f"  K={K:2d} {os.path.basename(sd)} | " + " | ".join(row))
