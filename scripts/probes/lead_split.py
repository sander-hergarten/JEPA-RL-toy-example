import json, numpy as np
d = json.load(open('remote_json/lead2_probe.json'))
order = [('K5','breakout_se_nstep_500k'),('K10','breakout_se_k10_500k'),('K20','breakout_se_k20_500k'),('K30','breakout_se_k30_500k')]
rng = np.random.default_rng(0)
def boot_se(x, n=2000):
    x = np.asarray(x, float); return float(np.std([rng.choice(x, len(x)).mean() for _ in range(n)]))
pooled = {lab: [dict(r, seed=i) for i, s in enumerate(d[name]) for r in s['records']] for lab, name in order}
print('ball_y distribution per K (min/median/max, n):')
for lab in pooled:
    ys = np.array([r['ball_y'] for r in pooled[lab]]); print(f'  {lab}: {ys.min()} {np.median(ys):.0f} {ys.max()} n={len(ys)}  hist161-176:', np.bincount(ys-161, minlength=16).tolist())
print()
for T in (167, 168, 169, 170, 171):
    print(f'=== lead-1 := ball_y >= {T}; lead-2 := ball_y < {T}')
    for lab in pooled:
        R = pooled[lab]
        for grp, sel in (('lead2', [r for r in R if r['ball_y'] < T]), ('lead1', [r for r in R if r['ball_y'] >= T])):
            if not sel: continue
            m = lambda k: np.mean([r[k] for r in sel])
            print(f'  {lab:4s} {grp} n={len(sel):3d} chance {m("chance"):.3f} actor {m("actor_ok"):.3f} critic {m("critic_ok"):.3f} model {m("model_ok"):.3f} (SE {boot_se([r["model_ok"] for r in sel]):.3f}) beam5 {m("beam5_ok"):.3f} | v_true_sep {m("v_true_sep"):.3f} v_pred_sep {m("v_pred_sep"):.3f} q_sep {m("q_sep"):.3f} | |gap| {np.mean([abs(r["gap"]) for r in sel]):.1f} cf_err {m("cf_err"):.3f}')
    print()
T = 169
print('=== per-seed lead-2 model/actor/critic/beam5 accuracy (T=169) and overall per-seed:')
for lab, name in order:
    for i, s in enumerate(d[name]):
        R = s['records']; l2 = [r for r in R if r['ball_y'] < T]; l1 = [r for r in R if r['ball_y'] >= T]
        f = lambda sel, k: np.mean([r[k] for r in sel]) if sel else float('nan')
        print(f'  {lab} seed{i}: lead2 n={len(l2)} actor {f(l2,"actor_ok"):.3f} critic {f(l2,"critic_ok"):.3f} model {f(l2,"model_ok"):.3f} beam5 {f(l2,"beam5_ok"):.3f} | lead1 n={len(l1)} model {f(l1,"model_ok"):.3f} critic {f(l1,"critic_ok"):.3f}')
print()
print('=== disagreement analyses (T=169):')
for lab in pooled:
    R = pooled[lab]
    for grp, sel in (('all', R), ('lead2', [r for r in R if r['ball_y'] < T]), ('lead1', [r for r in R if r['ball_y'] >= T])):
        mc = [r for r in sel if r['model_ok'] != r['critic_ok']]
        bm = [r for r in sel if r['beam5_ok'] != r['model_ok']]
        print(f'  {lab:4s} {grp:5s} n={len(sel):3d} | model vs critic disagree n={len(mc):3d} model right {np.mean([r["model_ok"] for r in mc]) if mc else float("nan"):.3f} | beam5 vs model disagree n={len(bm):3d} beam5 right {np.mean([r["beam5_ok"] for r in bm]) if bm else float("nan"):.3f}')
print()
print('=== matched-|gap| reanalysis at lead 2 (T=169): bins of |gap|')
for lo, hi in ((0, 15), (15, 25), (25, 40), (40, 200)):
    line = f'  |gap| in [{lo},{hi}): '
    for lab in pooled:
        sel = [r for r in pooled[lab] if r['ball_y'] < T and lo <= abs(r['gap']) < hi]
        if sel:
            line += f'{lab} n={len(sel):3d} model {np.mean([r["model_ok"] for r in sel]):.2f} critic {np.mean([r["critic_ok"] for r in sel]):.2f} actor {np.mean([r["actor_ok"] for r in sel]):.2f} beam5 {np.mean([r["beam5_ok"] for r in sel]):.2f} | '
    print(line)
print()
print('=== overall (all leads) pooled means, to compare with hypothesis 0.857/0.835/0.802/0.784 etc.')
for lab in pooled:
    R = pooled[lab]; m = lambda k: np.mean([r[k] for r in R])
    print(f'  {lab}: n={len(R)} chance {m("chance"):.3f} actor {m("actor_ok"):.3f} critic {m("critic_ok"):.3f} model {m("model_ok"):.3f} beam5 {m("beam5_ok"):.3f} tau(model,critic) {m("tau_model_critic"):.3f} v_true_sep {m("v_true_sep"):.3f} v_pred_sep {m("v_pred_sep"):.3f}')
