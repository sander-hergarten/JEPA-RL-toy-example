import json, glob, os, statistics as st, subprocess

print("### venv:", os.path.exists(".venv/bin/python"))
print("### discount arm train.log tails:")
for sd in sorted(glob.glob("runs/breakout_se_k30_depth_discount_500k/seed[0-2]")):
    p = f"{sd}/train.log"
    if os.path.exists(p):
        lines = open(p, errors="replace").read().splitlines()
        print("  ", os.path.basename(sd), "|", lines[-1][:160] if lines else "(empty)")

def load(p):
    with open(p) as f:
        return json.load(f)

print("\n### per-seed eps=0.01 evals K=5/10/20/30: Q, H1, H3, H5 -- raw, clipped, decisions, clipped-rate/1k decisions")
ARMS = {5: "breakout_se_nstep_500k", 10: "breakout_se_k10_500k", 20: "breakout_se_k20_500k", 30: "breakout_se_k30_500k"}
FILES = {"Q": "eval_q_eps001.json", "H1": "eval_lookahead_h1_eps001.json", "H3": "eval_lookahead_h3_eps001.json", "H5": "eval_lookahead_h5_eps001.json"}
first = True
pooled = {}
for K, arm in ARMS.items():
    for sd in sorted(glob.glob(f"runs/{arm}/seed[0-2]")):
        row = []
        for tag, fn in FILES.items():
            p = f"{sd}/{fn}"
            if not os.path.exists(p):
                row.append(f"{tag}: missing")
                continue
            e = load(p)
            if first:
                print("  top keys:", list(e.keys()))
                first = False
            recs = e.get("records") or e.get("episode_records") or []
            if not recs:
                row.append(f"{tag}: no records; return_mean {e.get('return_mean')}")
                continue
            raw = [x["raw_return"] for x in recs]
            clip = [x["clipped_return"] for x in recs]
            dec = [x["decisions"] for x in recs]
            rate = 1000 * sum(clip) / sum(dec)
            pooled.setdefault((K, tag), []).append((st.mean(raw), st.mean(clip), st.mean(dec), rate))
            row.append(f"{tag}: raw {st.mean(raw):5.1f} clip {st.mean(clip):5.1f} dec {st.mean(dec):6.0f} rate {rate:5.1f}")
        print(f"  K={K:2d} {os.path.basename(sd)} | " + " | ".join(row))
print("\n### seed-averaged: raw, clip, decisions, clipped rate per 1k decisions, raw/clip")
for (K, tag), v in sorted(pooled.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    raw = st.mean(x[0] for x in v); clip = st.mean(x[1] for x in v); dec = st.mean(x[2] for x in v); rate = st.mean(x[3] for x in v)
    print(f"  K={K:2d} {tag:2s}: raw {raw:5.1f} clip {clip:5.1f} dec {dec:6.0f} rate {rate:5.1f} raw/clip {raw/clip:4.2f} c=clip/(clip+5) {clip/(clip+5):.3f}")
