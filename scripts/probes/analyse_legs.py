"""Leg-by-leg analysis of raw closed-loop logs: rows = [t, action, q_action, bx, by, px, lives, r]."""
import json
import sys
from statistics import mean, median

PADDLE_Y = 179
RIGHT, LEFT = 2, 3
CENTRE = 4  # catch-window centre in (bx - px) units, refined from the data below


def legs_of(rows):
    """Split one episode into descending legs; each ends in a catch (vy flips up near the paddle) or a miss
    (a life is lost). Returns dicts with per-decision (ttc, offset, action, q_action, vy) and the outcome."""
    out = []
    cur = None
    prev_by = None
    for i, (t, a, qa, bx, by, px, lives, r) in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        vy = None if (prev_by is None or by == 0 or prev_by == 0) else by - prev_by
        life_lost = nxt is not None and nxt[6] < lives
        if vy is not None and vy > 0:
            if cur is None:
                cur = {"steps": [], "speed": vy, "outcome": None, "start_by": by}
            ttc = (PADDLE_Y - by) / vy
            cur["steps"].append({"ttc": ttc, "off": bx - px, "a": a, "qa": qa, "by": by, "vy": vy})
            # catch: next decision has the ball moving up again from near the paddle row (a re-serve after
            # a lost life also lowers by, so a life loss must be checked first)
            if not life_lost and nxt is not None and nxt[4] > 0 and nxt[4] < by and by >= 168:
                cur["outcome"] = "catch"; cur["end_off"] = bx - px
                out.append(cur); cur = None
        if life_lost:
            if cur is not None:
                # offset when the ball crossed the paddle row (last step with by <= 190)
                crossing = [s for s in cur["steps"] if 170 <= s["by"] <= 192]
                cur["outcome"] = "miss"; cur["end_off"] = (crossing[-1]["off"] if crossing else cur["steps"][-1]["off"])
                out.append(cur)
            cur = None
        if vy is not None and vy < 0 and cur is not None and cur["outcome"] is None:
            # ball going up without a detected catch (e.g. bounce detected late): close as catch if near paddle
            if cur["steps"] and cur["steps"][-1]["by"] >= 160:
                cur["outcome"] = "catch"; cur["end_off"] = cur["steps"][-1]["off"]; out.append(cur)
            cur = None
        prev_by = by
    if cur is not None and cur["outcome"] is None and cur["steps"] and cur["steps"][-1]["by"] >= 190:
        # the episode ended on this leg: the last life was lost
        crossing = [s for s in cur["steps"] if 170 <= s["by"] <= 192]
        cur["outcome"] = "miss"; cur["end_off"] = (crossing[-1]["off"] if crossing else cur["steps"][-1]["off"])
        out.append(cur)
    return out


def miss_type(leg):
    """near: ended within 12 units of the window centre; away: paddle moved away from the ball over the
    last 4 approach decisions; late: moved toward it but did not arrive."""
    if abs(leg["end_off"] - CENTRE) <= 12:
        return "near"
    last = [s for s in leg["steps"] if 0 <= s["ttc"] <= 4][-4:]
    if not last:
        return "late"
    rate = mean(toward(s["off"], s["a"]) for s in last)
    return "away" if rate < 0.5 else "late"


def toward(off, a):
    """Is action a moving the paddle toward the ball? off = bx - px (positive: ball to the right)."""
    err = off - CENTRE
    if abs(err) <= 6:
        return a not in (RIGHT, LEFT) or (a == RIGHT and err > 0) or (a == LEFT and err < 0)
    return (a == RIGHT) if err > 0 else (a == LEFT)


def summarize(legs):
    catches = [l for l in legs if l["outcome"] == "catch"]
    misses = [l for l in legs if l["outcome"] == "miss"]
    rec = {"legs": len(legs), "catches": len(catches), "misses": len(misses),
           "catch_rate": len(catches) / max(len(legs), 1)}
    if catches:
        rec["catch_end_off_median"] = median(l["end_off"] for l in catches)
        rec["catch_end_off_q10_q90"] = (sorted(l["end_off"] for l in catches)[len(catches) // 10],
                                        sorted(l["end_off"] for l in catches)[9 * len(catches) // 10])
    if misses:
        types = [miss_type(l) for l in misses]
        rec["miss_types"] = {k: types.count(k) for k in ("near", "late", "away")}
        rec["miss_end_abs_off_mean"] = mean(abs(l["end_off"] - CENTRE) for l in misses)
        rec["miss_end_offs"] = [l["end_off"] for l in misses]
        rec["miss_start_abs_off_mean"] = mean(abs(l["steps"][0]["off"] - CENTRE) for l in misses)
        rec["miss_speed"] = [l["speed"] for l in misses]
        rec["miss_len"] = [len(l["steps"]) for l in misses]
    if catches:
        rec["catch_start_abs_off_mean"] = mean(abs(l["steps"][0]["off"] - CENTRE) for l in catches)
        rec["catch_len_mean"] = mean(len(l["steps"]) for l in catches)
    # approach behaviour: correct-direction rate by ttc band, over all legs
    bands = {"ttc>8": (8, 99), "4<ttc<=8": (4, 8), "2<ttc<=4": (2, 4), "ttc<=2": (-99, 2)}
    for name, (lo, hi) in bands.items():
        steps = [s for l in legs for s in l["steps"] if lo < s["ttc"] <= hi]
        if steps:
            rec[f"toward_{name}"] = round(mean(toward(s["off"], s["a"]) for s in steps), 3)
            rec[f"abs_off_{name}"] = round(mean(abs(s["off"] - CENTRE) for s in steps), 1)
            rec[f"n_{name}"] = len(steps)
            rec[f"noop_{name}"] = round(mean(s["a"] not in (RIGHT, LEFT) for s in steps), 2)
            rec[f"disagree_{name}"] = round(mean(s["a"] != s["qa"] for s in steps), 2)
    # per-leg: how much of the initial error is removed by ttc=2 (closing fraction)
    fr = []
    for l in legs:
        s0 = l["steps"][0]
        late = [s for s in l["steps"] if s["ttc"] <= 2]
        if late and abs(s0["off"] - CENTRE) > 12:
            fr.append(1 - abs(late[0]["off"] - CENTRE) / abs(s0["off"] - CENTRE))
    if fr:
        rec["closing_fraction_by_ttc2"] = round(mean(fr), 3)
        rec["n_closing"] = len(fr)
    return rec


if __name__ == "__main__":
    txt = open(sys.argv[1]).read()
    data = json.loads(txt.split("JSON_BEGIN")[1].split("JSON_END")[0])
    pooled = {}
    for name, seeds in data.items():
        for s, d in seeds.items():
            K = d["K"]
            for ctl in ("q", "lookahead"):
                legs = [l for rows in d[ctl] for l in legs_of(rows)]
                pooled.setdefault((K, ctl), []).extend(legs)
                rec = summarize(legs)
                print(f"K={K:2d} seed{s} {ctl:9s} legs={rec['legs']:3d} catch={rec['catches']:3d} miss={rec['misses']:2d} rate={rec['catch_rate']:.2f} "
                      f"toward: >8 {rec.get('toward_ttc>8')} 4-8 {rec.get('toward_4<ttc<=8')} 2-4 {rec.get('toward_2<ttc<=4')} <=2 {rec.get('toward_ttc<=2')} "
                      f"| |off|: >8 {rec.get('abs_off_ttc>8')} 4-8 {rec.get('abs_off_4<ttc<=8')} 2-4 {rec.get('abs_off_2<ttc<=4')} <=2 {rec.get('abs_off_ttc<=2')} "
                      f"| miss end offs {rec.get('miss_end_offs')} speeds {rec.get('miss_speed')}")
    print("\n=== pooled over 3 seeds ===")
    for (K, ctl), legs in sorted(pooled.items()):
        rec = summarize(legs)
        print(f"K={K:2d} {ctl:9s} legs={rec['legs']:3d} catch={rec['catches']:3d} miss={rec['misses']:2d} rate={rec['catch_rate']:.3f} "
              f"| toward >8 {rec.get('toward_ttc>8')} 4-8 {rec.get('toward_4<ttc<=8')} 2-4 {rec.get('toward_2<ttc<=4')} <=2 {rec.get('toward_ttc<=2')} "
              f"| |off| >8 {rec.get('abs_off_ttc>8')} 4-8 {rec.get('abs_off_4<ttc<=8')} 2-4 {rec.get('abs_off_2<ttc<=4')} <=2 {rec.get('abs_off_ttc<=2')} "
              f"| closing {rec.get('closing_fraction_by_ttc2')} (n={rec.get('n_closing')}) | noop >8 {rec.get('noop_ttc>8')} <=2 {rec.get('noop_ttc<=2')} "
              f"| disagree >8 {rec.get('disagree_ttc>8')} <=2 {rec.get('disagree_ttc<=2')} "
              f"| catch end off med {rec.get('catch_end_off_median')} q10-90 {rec.get('catch_end_off_q10_q90')} | miss types {rec.get('miss_types')} miss |end off| {rec.get('miss_end_abs_off_mean')} start |off| miss {rec.get('miss_start_abs_off_mean')} catch {rec.get('catch_start_abs_off_mean')} | miss speeds {rec.get('miss_speed')} lens {rec.get('miss_len')}")
