import json, math
import numpy as np

d = json.load(open("/tmp/decisive_probe.json"))
order = ["breakout_se_nstep_500k", "breakout_se_k10_500k", "breakout_se_k20_500k", "breakout_se_k30_500k"]

def se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")

print("=== decisive-state probe (catch oracle), pooled over 3 seeds; mean +- SE ===")
keys = ["chance", "actor_ok", "critic_ok", "model_ok", "beam5_ok", "tau_model_critic", "tau_actor_critic",
        "tau_critic_catch", "tau_model_catch", "tau_actor_catch", "tau_beam_catch", "cf_err",
        "v_true_range", "v_pred_range", "q_range"]
for name in order:
    recs = [r for s in d[name] for r in s["records"]]
    K = d[name][0]["summary"]["K"]
    print(f"\n[K={K}] n_decisive={len(recs)} (per seed {[len(s['records']) for s in d[name]]})")
    for k in keys:
        v = [r[k] for r in recs]
        print(f"  {k:18s} {np.mean(v):+.4f} +- {se(v):.4f}")
    # paired differences
    for a, b in (("model_ok", "critic_ok"), ("model_ok", "actor_ok"), ("beam5_ok", "actor_ok"), ("tau_model_catch", "tau_critic_catch")):
        v = [r[a] - r[b] for r in recs]
        print(f"  paired {a}-{b}: {np.mean(v):+.4f} +- {se(v):.4f}")
    # lead bins
    for lo, hi in ((3, 5), (6, 9), (10, 15), (16, 40)):
        rr = [r for r in recs if lo <= r["tau"] <= hi]
        if not rr:
            continue
        print(f"  lead {lo}-{hi}: n={len(rr)} chance={np.mean([r['chance'] for r in rr]):.2f} actor={np.mean([r['actor_ok'] for r in rr]):.2f} "
              f"critic={np.mean([r['critic_ok'] for r in rr]):.2f} model={np.mean([r['model_ok'] for r in rr]):.2f} beam5={np.mean([r['beam5_ok'] for r in rr]):.2f} "
              f"tau(model,critic)={np.mean([r['tau_model_critic'] for r in rr]):+.3f}")
    # ball low and moving down proxy: ball_y large (RAM 101 increases downward?) -- report distribution
    ys = [r["ball_y"] for r in recs]
    print(f"  ball_y quantiles 10/50/90: {np.percentile(ys, [10, 50, 90])}")

print("\n=== K=5 vs K=30 contrasts (difference of means, SE from pooled per-state variance) ===")
r5 = [r for s in d[order[0]] for r in s["records"]]
r30 = [r for s in d[order[3]] for r in s["records"]]
for k in ["tau_model_critic", "model_ok", "critic_ok", "actor_ok", "beam5_ok", "tau_model_catch", "tau_critic_catch"]:
    a = np.array([r[k] for r in r5]); b = np.array([r[k] for r in r30])
    diff = a.mean() - b.mean(); s = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"  {k:18s} K5={a.mean():+.4f} K30={b.mean():+.4f} diff(K5-K30)={diff:+.4f} SE={s:.4f} z={diff/s:+.2f}")
# diff-in-diff: (model-critic)_K5 - (model-critic)_K30
a = np.array([r["model_ok"] - r["critic_ok"] for r in r5]); b = np.array([r["model_ok"] - r["critic_ok"] for r in r30])
diff = a.mean() - b.mean(); s = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  diff-in-diff (model_ok-critic_ok): K5={a.mean():+.4f} K30={b.mean():+.4f} DiD={diff:+.4f} SE={s:.4f} z={diff/s:+.2f}")
a = np.array([r["tau_model_catch"] - r["tau_critic_catch"] for r in r5]); b = np.array([r["tau_model_catch"] - r["tau_critic_catch"] for r in r30])
diff = a.mean() - b.mean(); s = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  diff-in-diff (tau_model_catch-tau_critic_catch): K5={a.mean():+.4f} K30={b.mean():+.4f} DiD={diff:+.4f} SE={s:.4f} z={diff/s:+.2f}")
