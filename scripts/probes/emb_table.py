import json, glob
rows=[]
for f in sorted(glob.glob("runs/emb_random_*.json")):
    r=json.load(open(f)); name=f.split("emb_random_")[1][:-5]
    d=r["state_decodability"]; lab=d.get("labelled_ram") or d.get("pong_labelled_ram") or {}
    s=r["spectrum_online"]; t=r["temporal_structure"]; rp=r["reward_event_probe"]
    pk="player_x" if "breakout" in name else "player_y"
    rows.append((name, s["effective_rank_entropy"], s["dims_for_90pct"], lab.get("ball_x"), lab.get(pk),
                 d["median_r2_over_varying"], rp.get("test_auc"), t["gap_1"]["mean_cosine_distance"], r["heldout"]["roots"]))
hdr = ("checkpoint (held-out data from a random policy)", "effrank", "90%dim", "ball_x", "paddle", "medR2", "rwdAUC", "d1", "roots")
print(f"{hdr[0]:46s} {hdr[1]:>8} {hdr[2]:>7} {hdr[3]:>7} {hdr[4]:>7} {hdr[5]:>7} {hdr[6]:>7} {hdr[7]:>6} {hdr[8]:>6}")
for n,er,d90,bx,pd,med,auc,d1,roots in sorted(rows):
    f=lambda v: "  n/a " if v is None else f"{v:6.2f}"
    print(f"{n:46s} {er:8.0f} {d90:7.0f} {f(bx):>7} {f(pd):>7} {f(med):>7} {f(auc):>7} {d1:6.3f} {roots:6d}")
