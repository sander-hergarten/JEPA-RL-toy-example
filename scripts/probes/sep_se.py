import json, numpy as np
d = json.load(open('remote_json/lead2_probe.json'))
order = [('K5','breakout_se_nstep_500k'),('K10','breakout_se_k10_500k'),('K20','breakout_se_k20_500k'),('K30','breakout_se_k30_500k')]
rng = np.random.default_rng(1)
def bse(x, n=3000):
    x = np.asarray(x, float); return float(np.std([rng.choice(x, len(x)).mean() for _ in range(n)]))
T = 169
print('lead-2 (ball_y<169) V separation catch-miss, real successor (v_true_sep) and predicted (v_pred_sep), with bootstrap SE and per-seed means; plus critic/model accuracy SE')
for lab, name in order:
    R = [r for s in d[name] for r in s['records'] if r['ball_y'] < T]
    per = [np.mean([r['v_true_sep'] for r in s['records'] if r['ball_y'] < T]) for s in d[name]]
    perm = [np.mean([r['v_pred_sep'] for r in s['records'] if r['ball_y'] < T]) for s in d[name]]
    vt = [r['v_true_sep'] for r in R]; vp = [r['v_pred_sep'] for r in R]
    print(f'  {lab}: n={len(R)} v_true_sep {np.mean(vt):.3f} (SE {bse(vt):.3f}; seeds {["%.3f"%p for p in per]}) v_pred_sep {np.mean(vp):.3f} (SE {bse(vp):.3f}; seeds {["%.3f"%p for p in perm]}) | critic acc {np.mean([r["critic_ok"] for r in R]):.3f} (SE {bse([r["critic_ok"] for r in R]):.3f}) model acc {np.mean([r["model_ok"] for r in R]):.3f} (SE {bse([r["model_ok"] for r in R]):.3f}) | v_true_range {np.mean([r["v_true_range"] for r in R]):.3f} q_range {np.mean([r["q_range"] for r in R]):.3f}')
print()
print('Same for lead 1 (ball_y>=169):')
for lab, name in order:
    R = [r for s in d[name] for r in s['records'] if r['ball_y'] >= T]
    vt = [r['v_true_sep'] for r in R]
    print(f'  {lab}: n={len(R)} v_true_sep {np.mean(vt):.3f} (SE {bse(vt):.3f}) v_true_range {np.mean([r["v_true_range"] for r in R]):.3f} q_range {np.mean([r["q_range"] for r in R]):.3f}')
