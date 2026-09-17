import json
import statistics

RUNS = ["breakout_se_nstep_500k", "breakout_se_hjepa_detached_500k", "breakout_se_hjepa_500k"]


def load(run):
    return [json.load(open(f"runs/{run}/seed{s}/diagnostics.json")) for s in range(3)]


def mean(ds, fn):
    return statistics.mean(fn(d) for d in ds)


for run in RUNS:
    ds = load(run)
    a = mean(ds, lambda d: d["action_sensitivity_at_root"]["mean_pairwise_cosine_distance_next_latent"])
    rr = mean(ds, lambda d: d["action_sensitivity_at_root"]["mean_expected_reward_range"])
    print(run)
    print(f"   action: next-latent dist {a:.4f}   reward range {rr:.4f}")
    for dep in ["depth_1", "depth_5"]:
        keys = ["predicted", "persistence", "shuffled_actions", "random_actions"]
        vals = {k: mean(ds, lambda d, k=k, dep=dep: d["latent_prediction_cosine_distance"][dep][k]) for k in keys}
        pen = (vals["shuffled_actions"] - vals["predicted"]) / max(vals["predicted"], 1e-9)
        print(f"   {dep}: pred {vals['predicted']:.4f}  persist {vals['persistence']:.4f}  "
              f"shuffled {vals['shuffled_actions']:.4f}  shufPen {pen * 100:.0f}%")
    for side in ["online", "target"]:
        rl = ds[0]["root_latents"][side]
        num = {k: mean(ds, lambda d, k=k, side=side: d["root_latents"][side][k])
               for k, v in rl.items() if isinstance(v, (int, float))}
        print(f"   root/{side}: " + "  ".join(f"{k}={v:.3f}" for k, v in num.items()))
    for block in ["reward_prediction", "continuation_prediction", "q_td_error", "movement_probe"]:
        b = ds[0][block]
        keys = [k for k, v in b.items() if isinstance(v, (int, float))]
        print(f"   {block}: " + " ".join(
            f"{k}={mean(ds, lambda d, k=k, block=block: d[block][k]):.4f}" for k in keys))
    print()
