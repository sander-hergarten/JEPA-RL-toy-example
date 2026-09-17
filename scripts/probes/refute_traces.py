"""In-window behaviour from the closed-loop traces: P(move AWAY from landing | already inside window) by lead,
plus P(correct: stay-or-toward | in window), and lead-1/lead-2 toward rates, per K and controller."""
import numpy as np
CENTRE, WINDOW, SPEED = 9, 12, 11
LEADS = [1, 2, 3, 4, 5, 6]
MEAN = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
move = np.array([0, 0, 1, -1])
d = np.load('/tmp/approach_traces.npz')

def descents(rows):
    ep, t, a, qa, px, bx, by, lives = rows.T
    out, n, i = [], len(rows), 0
    while i < n - 1:
        if ep[i + 1] == ep[i] and by[i + 1] > by[i]:
            j = i + 1
            while j < n - 1 and ep[j + 1] == ep[j] and by[j + 1] >= by[j] and lives[j + 1] == lives[j]:
                j += 1
            miss = j < n - 1 and ep[j + 1] == ep[j] and lives[j + 1] < lives[j]
            if j - i >= 3:
                out.append((i, j, bool(miss)))
            i = j + 1
        else:
            i += 1
    return out

res = {}
for key in d.files:
    name, seed, ctrl = key.rsplit('_', 2)
    K = {'breakout_se_nstep_500k': 5, 'breakout_se_k10_500k': 10, 'breakout_se_k20_500k': 20, 'breakout_se_k30_500k': 30}[name]
    rows = d[key]
    ep, t, a, qa, px, bx, by, lives = rows.T
    away_in = {L: [] for L in LEADS}; ok_in = {L: [] for L in LEADS}; toward_out = {L: [] for L in LEADS}
    n_in = {L: 0 for L in LEADS}; n_desc = 0; n_miss = 0
    # misses split by whether paddle was in window at lead 2 / lead 1
    miss_in2 = []; miss_in1 = []
    for i, j, miss in descents(rows):
        landing = bx[j]; n_desc += 1; n_miss += miss
        for k in range(i, j):
            lead = j - k
            if lead not in away_in: continue
            gap = landing - (px[k] + CENTRE)
            mv = move[a[k]]
            if abs(gap) <= WINDOW:
                n_in[lead] += 1
                away_in[lead].append(float(mv != 0 and mv == -np.sign(gap) if gap != 0 else mv != 0))
                ok_in[lead].append(float(not (mv != 0 and ((gap != 0 and mv == -np.sign(gap)) or gap == 0))))
            else:
                toward_out[lead].append(float(mv == np.sign(gap)))
        g2 = landing - (px[j - 2] + CENTRE) if j - 2 >= i else None
        g1 = landing - (px[j - 1] + CENTRE)
        if g2 is not None: miss_in2.append((abs(g2) <= WINDOW, miss))
        miss_in1.append((abs(g1) <= WINDOW, miss))
    r = res.setdefault((K, ctrl), {'away_in': {L: [] for L in LEADS}, 'toward_out': {L: [] for L in LEADS}, 'n_in': {L: 0 for L in LEADS},
                                   'desc': 0, 'miss': 0, 'miss_in2': [], 'miss_in1': []})
    for L in LEADS:
        r['away_in'][L] += away_in[L]; r['toward_out'][L] += toward_out[L]; r['n_in'][L] += n_in[L]
    r['desc'] += n_desc; r['miss'] += n_miss; r['miss_in2'] += miss_in2; r['miss_in1'] += miss_in1

print("P(move AWAY from landing | paddle already within window) by lead   [n in window]")
for (K, c), r in sorted(res.items()):
    print(f"K={K:2d} {c:2s} desc/miss {r['desc']}/{r['miss']} | " + " ".join(f"L{L}:{np.mean(r['away_in'][L]) if r['away_in'][L] else float('nan'):.2f}[{r['n_in'][L]}]" for L in LEADS))
print("\nP(move toward | outside window) by lead (re-derived)")
for (K, c), r in sorted(res.items()):
    print(f"K={K:2d} {c:2s} | " + " ".join(f"L{L}:{np.mean(r['toward_out'][L]) if r['toward_out'][L] else float('nan'):.2f}[{len(r['toward_out'][L])}]" for L in LEADS))
print("\nMiss rate given paddle in window at lead 2 / outside at lead 2; same for lead 1")
for (K, c), r in sorted(res.items()):
    a2 = np.array(r['miss_in2']); a1 = np.array(r['miss_in1'])
    f = lambda arr, inw: (arr[arr[:, 0] == inw][:, 1].mean() if (arr[:, 0] == inw).any() else float('nan'), int((arr[:, 0] == inw).sum()))
    print(f"K={K:2d} {c:2s} | L2 in-window miss {f(a2,1)[0]:.3f} (n={f(a2,1)[1]}) out miss {f(a2,0)[0]:.3f} (n={f(a2,0)[1]}) | L1 in miss {f(a1,1)[0]:.3f} (n={f(a1,1)[1]}) out miss {f(a1,0)[0]:.3f} (n={f(a1,0)[1]})")
