import json

a = [json.loads(l) for l in open("runs/breakout_se_nstep_500k/seed0/updates.jsonl")]
b = [json.loads(l) for l in open("runs/breakout_se_hjepa_detached_500k/seed0/updates.jsonl")]
print("rows", len(a), len(b))
print(f"{'update':>10} {'flat loss_q':>14} {'detached loss_q':>16} {'flat jepa':>12} {'det jepa':>12}")
for i in list(range(8)) + [len(a) // 4, len(a) // 2, len(a) - 1]:
    if i >= len(a) or i >= len(b):
        continue
    print(f"{a[i].get('update', i):>10} {a[i]['loss_q']:>14.8f} {b[i]['loss_q']:>16.8f} "
          f"{a[i]['loss_jepa']:>12.8f} {b[i]['loss_jepa']:>12.8f}")
