"""State decodability (ridge R^2 from latents to RAM), spectrum and temporal structure, per arm."""
import json
import statistics

ROWS = [("conv K=5", "breakout_se_nstep_500k"), ("conv K=10", "breakout_se_k10_500k"),
        ("conv K=20", "breakout_se_k20_500k"), ("conv K=30", "breakout_se_k30_500k"),
        ("LMU K=30", "breakout_se_lmu_k30_500k"), ("K=20 + qd5", "breakout_se_k20_qd5_500k"),
        ("H-JEPA attached", "breakout_se_hjepa_500k"), ("H-JEPA detached", "breakout_se_hjepa_detached_500k")]


def load(run):
    out = []
    for s in range(3):
        try:
            out.append(json.load(open(f"runs/{run}/seed{s}/embeddings.json")))
        except FileNotFoundError:
            pass
    return out


def m(ds, fn):
    v = [fn(d) for d in ds]
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else float("nan")


print("STATE DECODABILITY  (ridge R^2, held-out episodes)")
print(f"{'arm':16s} {'n':>2s} {'paddle_x':>9s} {'ball_x':>8s} {'ball_y':>8s} {'blocks':>8s} {'mean varying':>13s} {'#>0.5':>6s}")
for tag, run in ROWS:
    ds = load(run)
    if not ds:
        print(f"{tag:16s}  -"); continue
    lab = lambda k: m(ds, lambda d: d["state_decodability"]["labelled_ram"].get(k))
    print(f"{tag:16s} {len(ds):2d} {lab('player_x'):9.3f} {lab('ball_x'):8.3f} {lab('ball_y'):8.3f} "
          f"{lab('blocks_hit_count'):8.3f} {m(ds, lambda d: d['state_decodability']['mean_r2_over_varying']):13.3f} "
          f"{m(ds, lambda d: d['state_decodability']['targets_r2_above_0.5']):6.1f}")

print()
d0 = load(ROWS[0][1])[0]
print("spectrum keys:", list(d0["spectrum_online"].keys()))
print("temporal keys:", list(d0["temporal_structure"].keys()))
print()
print("SPECTRUM (online root latents)")
keys = [k for k, v in d0["spectrum_online"].items() if isinstance(v, (int, float))]
print(f"{'arm':16s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
for tag, run in ROWS:
    ds = load(run)
    if not ds:
        continue
    print(f"{tag:16s} " + " ".join(f"{m(ds, lambda d, k=k: d['spectrum_online'][k]):14.3f}" for k in keys))

print()
print("TEMPORAL STRUCTURE")
ts = d0["temporal_structure"]
for tag, run in ROWS:
    ds = load(run)
    if not ds:
        continue
    row = {}
    for k, v in ts.items():
        if isinstance(v, (int, float)):
            row[k] = m(ds, lambda d, k=k: d["temporal_structure"].get(k))
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    row[f"{k}.{kk}"] = m(ds, lambda d, k=k, kk=kk: d["temporal_structure"].get(k, {}).get(kk))
    print(f"{tag:16s} " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))
