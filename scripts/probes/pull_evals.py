import json, statistics as st, os

ARMS = ["breakout_se_nstep_500k", "breakout_se_k10_500k", "breakout_se_k20_500k", "breakout_se_k30_500k",
        "breakout_se_k20_qd5_500k", "breakout_se_k10_qd5_500k"]

print("=== mc_truth.json (full) ===")
print(open("/tmp/mc_truth.json").read())
print("=== mc_truth.log ===")
print(open("/tmp/mc_truth.log").read())

print("=== final evals (eps 0.01), per arm/controller, pooled over seeds ===")
for r in ARMS:
    for c in ("q", "lookahead_h1", "lookahead_h5"):
        rets, lens, per_seed = [], [], []
        for s in (0, 1, 2):
            p = f"runs/{r}/seed{s}/eval_{c}_eps001.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            sr, sl = [], []
            for e in d["records"]:
                sr.append(e["raw_return"]); sl.append(e["decisions"])
            rets += sr; lens += sl
            per_seed.append((round(st.mean(sr), 1), round(st.mean(sl))))
        if rets:
            print(f"{r:28s} {c:13s} n={len(rets):2d} ret={st.mean(rets):6.2f} len={st.mean(lens):6.0f} "
                  f"ret/100dec={100*sum(rets)/sum(lens):5.2f} dec/life={st.mean(lens)/5:5.0f} per_seed={per_seed}")

print("=== in-training eval (eval.jsonl): lookahead - q per checkpoint, per arm (mean over seeds) ===")
for r in ARMS:
    gains = {}
    for s in (0, 1, 2):
        p = f"runs/{r}/seed{s}/eval.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        byd = {}
        for row in rows:
            byd.setdefault(row["decisions"], {})[row["controller"]] = row["return_mean"]
        for dec, v in byd.items():
            if "q" in v and "lookahead" in v:
                gains.setdefault(dec, []).append((v["lookahead"] - v["q"], v["q"], v["lookahead"]))
    if gains:
        print(r)
        for dec in sorted(gains):
            g = gains[dec]
            print(f"   {dec:7d}: gain={st.mean(x[0] for x in g):6.2f}  q={st.mean(x[1] for x in g):6.2f}  la={st.mean(x[2] for x in g):6.2f}  (n_seeds={len(g)})")
