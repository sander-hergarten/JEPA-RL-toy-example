"""Closed-loop history split on the existing raw logs (rows = [t, action, q_action, bx, by, px, lives, r]).

For every decision inside a descending leg with ttc <= 2 (the hypothesis's decisive window), classify the
history over the previous 3 decisions as 'reversed' (the horizontal command changed sign at least once,
comparing each nonzero command with the previous nonzero command) or 'consistent'. Report, per (K,
controller): n, toward-rate, NOOP rate, |offset|, and the leg outcome (catch fraction) for each split.
Also the fraction of ttc<=2 decisions that are 'reversed' (how often the history is 'planner-typical').
"""
import json
import sys
from statistics import mean

PADDLE_Y = 179
RIGHT, LEFT = 2, 3
CENTRE = 4


def cmd(a):
    return 1 if a == RIGHT else (-1 if a == LEFT else 0)


def toward(off, a):
    err = off - CENTRE
    if abs(err) <= 6:
        return a not in (RIGHT, LEFT) or (a == RIGHT and err > 0) or (a == LEFT and err < 0)
    return (a == RIGHT) if err > 0 else (a == LEFT)


def reversed_within(actions, i, w=3):
    """Did the horizontal command reverse sign at any of decisions i-w..i-1 (relative to the previous nonzero command)?"""
    last = None
    for j in range(max(0, i - w - 6), i):  # look a little further back to find the previous nonzero command
        c = cmd(actions[j])
        if c == 0:
            continue
        if last is not None and j >= i - w and c * last < 0:
            return True
        last = c
    return False


def legs_with_index(rows):
    """Yield (leg_steps) where each step has the row index, so history can be read off the raw rows."""
    out, cur, prev_by = [], None, None
    for i, (t, a, qa, bx, by, px, lives, r) in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        vy = None if (prev_by is None or by == 0 or prev_by == 0) else by - prev_by
        life_lost = nxt is not None and nxt[6] < lives
        if vy is not None and vy > 0:
            if cur is None:
                cur = {"steps": [], "outcome": None}
            cur["steps"].append({"i": i, "ttc": (PADDLE_Y - by) / vy, "off": bx - px, "a": a, "qa": qa, "by": by})
            if not life_lost and nxt is not None and nxt[4] > 0 and nxt[4] < by and by >= 168:
                cur["outcome"] = "catch"; out.append(cur); cur = None
        if life_lost:
            if cur is not None:
                cur["outcome"] = "miss"; out.append(cur)
            cur = None
        if vy is not None and vy < 0 and cur is not None and cur["outcome"] is None:
            if cur["steps"] and cur["steps"][-1]["by"] >= 160:
                cur["outcome"] = "catch"; out.append(cur)
            cur = None
        prev_by = by
    if cur is not None and cur["outcome"] is None and cur["steps"] and cur["steps"][-1]["by"] >= 190:
        cur["outcome"] = "miss"; out.append(cur)
    return out


def main(path):
    txt = open(path).read()
    data = json.loads(txt.split("JSON_BEGIN")[1].split("JSON_END")[0])
    pooled = {}
    for name, seeds in data.items():
        for s, d in seeds.items():
            K = d["K"]
            for ctl in ("q", "lookahead"):
                for rows in d[ctl]:
                    actions = [r[1] for r in rows]
                    for leg in legs_with_index(rows):
                        for st in leg["steps"]:
                            if st["ttc"] <= 2:
                                rev = reversed_within(actions, st["i"], 3)
                                pooled.setdefault((K, ctl), []).append({
                                    "rev": rev, "tw": toward(st["off"], st["a"]), "noop": st["a"] not in (RIGHT, LEFT),
                                    "absoff": abs(st["off"] - CENTRE), "catch": leg["outcome"] == "catch",
                                    "seed": s, "leg_id": id(leg)})
    print("K  ctrl      | n(ttc<=2) frac_rev | toward: rev / cons (diff) | noop: rev / cons | |off|: rev / cons | leg catch frac: rev / cons")
    for (K, ctl), recs in sorted(pooled.items()):
        rev = [r for r in recs if r["rev"]]; con = [r for r in recs if not r["rev"]]
        def f(xs, k):
            return mean(x[k] for x in xs) if xs else float("nan")
        print(f"K={K:2d} {ctl:9s} | {len(recs):4d} {len(rev)/len(recs):.2f} | {f(rev,'tw'):.3f} / {f(con,'tw'):.3f} ({f(rev,'tw')-f(con,'tw'):+.3f}) "
              f"| {f(rev,'noop'):.2f} / {f(con,'noop'):.2f} | {f(rev,'absoff'):5.1f} / {f(con,'absoff'):5.1f} | {f(rev,'catch'):.2f} / {f(con,'catch'):.2f}")
    print("\nper-seed toward-rate at ttc<=2, rev / cons:")
    for (K, ctl), recs in sorted(pooled.items()):
        parts = []
        for s in ("0", "1", "2"):
            rs = [r for r in recs if r["seed"] == s]
            rev = [r["tw"] for r in rs if r["rev"]]; con = [r["tw"] for r in rs if not r["rev"]]
            parts.append(f"s{s}: {mean(rev) if rev else float('nan'):.2f}/{mean(con) if con else float('nan'):.2f} (n {len(rev)}/{len(con)})")
        print(f"K={K:2d} {ctl:9s} " + " | ".join(parts))
    # configuration-matched: restrict to |off| in a band so the two splits face the same task
    print("\ntoward-rate at ttc<=2 within |off| bands, rev / cons (n):")
    for (K, ctl), recs in sorted(pooled.items()):
        parts = []
        for lo, hi in ((0, 12), (12, 30), (30, 999)):
            rs = [r for r in recs if lo <= r["absoff"] < hi]
            rev = [r["tw"] for r in rs if r["rev"]]; con = [r["tw"] for r in rs if not r["rev"]]
            parts.append(f"[{lo},{hi}): {mean(rev) if rev else float('nan'):.2f}/{mean(con) if con else float('nan'):.2f} ({len(rev)}/{len(con)})")
        print(f"K={K:2d} {ctl:9s} " + " | ".join(parts))


if __name__ == "__main__":
    main(sys.argv[1])
