I now have every number the critique cites verified from disk (or corrected where the critique itself was slightly off, e.g. the discount arm is at ~137k, not 127k). Here is the revised synthesis.

# The K-sweep dissociation: what is established, what is ruled out, what is left

Sources: the evidence dossier (§ numbers below), the README, `losses.py` / `networks.py` / `planning.py`, and direct re-reads of the remote box and local scratchpad for this revision. Every number below was re-read from a file on disk for this version (dossier, `diagnostics.json`, `updates.jsonl`, `eval.jsonl`, `eval_*_eps001.json`, `train_episodes.jsonl`, `/tmp/lead2_probe.json` via `lead_split.py` and `sep_se.py`, `/tmp/approach_probe.log`, `/tmp/approach_probe2.log`, `/tmp/decisive_probe.log`, `/tmp/mc_truth.log`, the `/tmp/o2_*`, `/tmp/o3_*`, `/tmp/oracle*` logs, `/tmp/ae_*qd5*.json`, and the scratchpad's `ttc_probe_out.txt`, `ttc_probe_planner_out.txt`, `kin_out.txt`, `decode_out.txt`). Numbers quoted in earlier analyses that exist in no file on the box (per-depth encoder-gradient shares, a "+14.3 / +45.2" oracle pair, subspace ablations, a lag-1 residual autocorrelation, any output of `history_split_closed_loop.py`) are not used, including where the previous version of this document used them. Status of the pending arms: the K=30 inverse-weighted arm is complete (500k, 3 seeds); the discount arm is at ~137k/500k decisions (~33k updates) and is used only for matched-update trajectory comparisons. No `o4_fixed_oracle.py` run exists (`ls /tmp/o4*` is empty).

---

## 1. What is established

### 1.1 The control effect

| | K=5 (3 seeds) | K=5 (6 seeds) | K=10 | K=20 | K=30 |
|---|---|---|---|---|---|
| Q-policy | +19.1 ±1.6 | +22.1 ±3.7 | +19.9 ±5.4 | +17.6 ±1.3 | +16.2 ±0.4 |
| H=1 | +40.4 ±2.2 | +39.2 ±3.9 | +30.7 ±6.9 | +21.6 ±1.1 | +18.3 ±1.7 |
| H=3 / H=5 / H=10 | +45.6 / +65.1 ±1.1 / +62.0 | – / +60.6 ±6.9 / – | +47.7 / +61.8 ±9.1 / +59.3 | +39.4 / +49.1 / +47.0 | +26.7 / +34.0 / +36.5 |
| H=1 gain over Q | +21.3 | **+17.0** | +10.8 | +4.0 | +2.1 |
| H=5 gain over Q | +46.0 | **+38.5** | +41.9 | +31.5 | +17.8 |

(dossier §1, incl. the 6-seed K=5 check: "the 3-seed +65.07±1.1 was tight by luck".) Established: the H=1 gain falls monotonically with K and is nearly gone at K=30; the H≥3 gain is intact at K=10 and falls at K≥20; search depth beyond H=5 is flat at every K (no H>K penalty). With the 6-seed anchors the H=1 gain ratio is 0.64 / 0.24 / 0.12 at K=10/20/30 and the H=5 ratio 1.09 / 0.82 / 0.46 (with the 3-seed anchors: 0.51/0.19/0.10 and 0.91/0.68/0.39). The reading of this as "two components with different onsets" (an H=1-specific loss visible at K=10, a general attenuation only at K≥20) is an *interpretation*: the K=10 Q and H=1 spreads are ±5.4 and ±6.9, and the anchor choice moves the percentages by 10–20 points.

**Both losses are survival losses, in clipped-score terms.** From `eval_*_eps001.json` (30 episodes per arm and controller, none capped; decisions per life = decisions / 5 lives):

| | K=5 | K=10 | K=20 | K=30 |
|---|---|---|---|---|
| decisions per life, Q / H=1 / H=5 | 136 / 229 / 290 | 138 / 221 / 255 | 134 / 182 / 256 | 147 / 170 / 196 |
| clipped score per 1000 decisions, Q / H=1 / H=5 | 23.3 / 23.4 / 25.5 | 22.9 / 20.7 / 27.0 | 22.7 / 19.5 / 24.5 | 19.4 / 18.1 / 24.4 |

The Q controller's life length is flat in K; the planner's falls 229→170 (H=1) and 290→196 (H=5) while the H=5 clipped scoring rate is flat (25.5→24.4). Fresh closed-loop play with RAM logging (`/tmp/approach_probe2.log`, 3 seeds × 3 episodes) gives catch rate per descent Q / H=1 / H=5: K=5 0.80 / 0.88 / 0.90; K=10 0.77 / 0.86 / 0.89; K=20 0.82 / 0.83 / 0.89; K=30 0.79 / 0.81 / 0.86. A second closed-loop sample (`kin_out.txt`, 3 episodes/seed, a different catch detector) gives K=5 lookahead 0.97/1.00/0.96 vs Q 0.95/0.95/1.00 and K=30 lookahead 0.89/0.93/0.94 vs Q 0.96/0.97/1.00, i.e. the K=30 H=1 controller catching *less* often than its own Q — which the eval files (170 vs 147 decisions per life) and approach_probe2 (0.81 vs 0.79) contradict. Three 3-episode samples with three catch definitions; the direction "K=30 H=1 ≈ Q, K=5 H=1 ≫ Q" is common to all three, the sign of the K=30 H=1−Q difference is not.

### 1.2 Per-decision instruments at contact: the planner's edge over the actor shrinks with K on outcome-based instruments and not on a direction-based one

Catch oracle on actor-generated states, last two decisions before contact (`/tmp/lead2_probe.json`, ~300 decisive states per arm pooled over 3 seeds; splits recomputed with `lead_split.py` / `sep_se.py`):

| K | actor | critic (Q on REAL successor) | model (H=1) | beam5 | model right when disagreeing with critic | tau(model, critic) |
|---|---|---|---|---|---|---|
| 5 | 0.739 | 0.844 | **0.857** | 0.834 | 0.55 (n=40) | 0.49 |
| 10 | 0.798 | 0.814 | 0.835 | 0.860 | 0.57 (51) | 0.48 |
| 20 | 0.700 | 0.834 | 0.802 | 0.843 | 0.41 (54) | 0.46 |
| 30 | 0.738 | 0.807 | **0.784** | 0.824 | 0.44 (63) | 0.43 |

Per seed the model is 0.861/0.849/0.860 at K=5 and 0.770/0.807/0.768 at K=30 (`/tmp/lead2_probe.log`): every K=5 seed above every K=30 seed. Split by lead at ball_y<169 (SE 0.02–0.04):

| | lead 1: actor / critic / model / beam5 | lead 2: actor / critic / model / beam5 |
|---|---|---|
| K=5 | 0.798 / 0.908 / 0.926 / 0.926 | 0.674 / 0.771 / 0.778 / 0.729 |
| K=10 | 0.848 / 0.873 / 0.909 / 0.964 | 0.745 / 0.752 / 0.758 / 0.752 |
| K=20 | 0.761 / 0.902 / 0.865 / 0.914 | 0.633 / 0.760 / 0.733 / 0.767 |
| K=30 | 0.786 / 0.906 / 0.893 / 0.887 | 0.683 / 0.697 / 0.662 / 0.754 |

What this establishes: at lead 1 the critic is K-invariant and the model is non-monotone (K=20 is the minimum), with a model−actor edge of +0.13/+0.06/+0.10/+0.11 — intact at K=30. At lead 2 the model's and the **critic's** accuracy both fall at K=30 (model 0.778→0.662, critic 0.771→0.697), and both sit at the actor's level (0.683); beam5 is flat (0.73–0.77). "Model below its own actor" holds only at the ball_y<169 cut (at <170: model 0.709 vs actor 0.691; at <171: 0.720 vs 0.704) and is within one SE; the honest statement is model ≈ critic ≈ actor at lead 2 at K=30. The lead-2 critic−actor edge is +0.10/+0.01/+0.13/+0.01 across K, non-monotone, with K=10's strong actor breaking the pattern. On states where the paddle is already within 15 RAM units of the landing point (all leads): K=5 model 0.868 vs critic 0.849; K=20 0.733 vs 0.852, the model right on 23% of 30 disagreements; K=30 0.765 vs 0.818, right on 40% of 35 (at lead 2 only: K=5 0.80/0.80 n=69, K=20 0.62/0.75 n=53, K=30 0.60/0.69 n=45). Suggestive of the long-K planner un-positioning a placed paddle; n is small.

Four 60-step Monte-Carlo instruments (all with the sticky-context flaw of §1.8; per-arm n≈150–250 decisive states; regret SE roughly 0.03):

| instrument (states; stratum) | K=5: actor / critic / model | K=30: actor / critic / model |
|---|---|---|
| dossier §14 (actor-generated, all decisive) | −0.190 / −0.212 / **−0.173** | −0.240 / **−0.183** / −0.242 |
| `ttc_probe_out.txt` (actor-generated, imminent) | −0.175 / −0.141 / **−0.123** | −0.119 / −0.111 / **−0.091** |
| `ttc_probe_planner_out.txt` (planner-generated, imminent) | −0.156 / −0.140 / **−0.127** | −0.156 / **−0.152** / −0.164 |
| lead2 catch oracle (accuracy, above) | 0.739 / 0.844 / **0.857** | 0.738 / **0.807** / 0.784 |

At K=5 the model is the best estimator on all four. At K=30 it is best on one (actor-state ttc) and worst or below the critic on three. The model−actor advantage shrinks with K on all four (+0.017→−0.002; +0.052→+0.028; +0.029→−0.008; +0.118→+0.046). The planner-state ttc probe is also the only instrument that compares state distributions: at K=30 all three estimators are worse on planner-generated than on actor-generated states (actor −0.119→−0.156, critic −0.111→−0.152, model −0.091→−0.164, the model most), while at K=5 none is (−0.175→−0.156, −0.141→−0.140, −0.123→−0.127). That is 1–2 SE per cell and relevant to §2 item 16.

The counter-instrument: the closed-loop directional-correctness probe (`/tmp/approach_probe.log`; "correct" = moved toward the landing point, dead zone ±8, scored against greedy-Q on the same states). The H=1 controller's edge over its own Q action is **K-invariant**: overall 0.453/0.431 (K=5), 0.462/0.424 (K=10), 0.437/0.419 (K=20), 0.435/0.413 (K=30); at lead 1–2: 0.61/0.54, 0.60/0.53, 0.61/0.54, 0.56/0.51; on override decisions 0.372/0.338, 0.386/0.329, 0.367/0.340, 0.368/0.333. So whether the planner's per-decision advantage at contact collapses with K depends on the instrument: catch/return-based oracles say yes (by 0.05–0.07 in accuracy), the direction-based one says no. Both are on disk; neither is refuted.

### 1.3 Model metrics: better in absolute terms, worse in one-step direction at shallow depth

- Held-out absolute cosine error at d=1 falls 0.140→0.097 while persistence falls faster (0.650→0.246): ratio 0.216 / 0.255 / 0.332 / 0.395 (§2; `diagnostics.json`). The ratio at d=3 is 0.241 / 0.235 / 0.245 / 0.262, at d=5 0.275 / 0.256 / 0.254 / 0.268, at d=10 0.397 / 0.337 / 0.317 / 0.340: flat-to-slightly-worse at d=3–5, better at d≥10.
- The training delta-direction loss (`updates.jsonl`, last-40 means over 3 seeds) rises at shallow depth: `jepa_d1` 0.115 / 0.139 / 0.197 / 0.235. The K=30−K=5 excess is 0.120 (d1), 0.064 (d2), 0.053 (d3), 0.024 (d5); K=5 trains only to d5 and has no `jepa_d10` key, so the earlier "0.000 at d10" was K=30 vs K=10 (0.382 vs 0.383). The excess is steepest at d1 and tapers by d5; it is not confined to depths 1–2. LMU K=30 0.226, K=20+qd5 0.223.
- Counterfactual one-step error on real branched successors falls 0.148→0.099 (§12); model/real action effect 0.66x→0.78x (§6); shuffled-action penalty at d=1 32%→44% (`diagnostics.json`: 0.186/0.140 vs 0.140/0.097). Long-K arms are not action-blind.
- Delta probe (§5): every Δ improves with K; a directly-fitted head cannot beat the mean latent at Δ=100.
- `decode_out.txt` (linear probes, 3 seeds): ball velocity is decodable from the K=30 latent (vx R² 0.41/0.40/0.29, "all" stratum) and not from K=5's (−0.34/−0.13/+0.05); the model successor inherits it (0.40/0.41/0.25). In the *imminent* stratum ball_y R² is at or below zero for both K and the K=30 model successor's vy-sign accuracy is below the real successor's in 2 of 3 seeds (0.59 vs 0.62, 0.54 vs 0.66; third 0.62 vs 0.54), K=5 in 1 of 3 (0.59 vs 0.66). Velocity decodability is a property of the whole distribution, not of contact states.

### 1.4 The representation

Total latent change per transition 0.64→0.24 with the action-driven part ~constant, action share 16.5%→38% (§6); centered temporal distance gap-1 0.77→0.29, gap-2 0.75→0.51, gap-5 0.92→0.85 (§10); participation ratio 104→55, effective rank 333→140 (§10); "predict no change" beats the mean predictor at Δ=1 at K=30 (+0.61) and loses at K=5 (−0.07) (§5). Ball_x R² ~0.85 flat, paddle 0.69→0.78, ball_y 0.73/0.72/0.76/0.67, blocks_hit 0.64/0.55/0.68/0.82 (§10). LMU K=30 and K=20+qd5 have the same code: gap-1 0.288 / 0.312 (§10), persistence d1 0.246 / 0.266 (`diagnostics.json`), total change 0.247 (§6) / 0.272 (`/tmp/ae_breakout_se_k20_qd5_500k_seed{0,1,2}.json`: 0.271/0.307/0.237, model/real 0.75/0.71/0.85x, action share 0.31/0.37/0.33). K=10+qd5 sits with K=10 (`/tmp/ae_breakout_se_k10_qd5_500k_seed*.json`: total change 0.50/0.50/0.47, model/real 0.77/0.80/0.77x, share 0.25/0.23/0.25).

### 1.5 Gradients

Credit from the deepest term arrives at the root 5.4x amplified (LMU 4.8x; K=5 2.3x), saturating, never clipped; total grad norm 1.61 / 1.84 / 2.59 / 2.95 (§7a; `updates.jsonl`). Depth terms conflict: 0% / 0% / 3.3% / 8.4% opposing pairs, d15–d30 cosine −0.345; action-embedding credit of the deepest term at the root 0.915→0.415 (§7b). Per-step Jacobian spectrum (§7c, one state): a random direction is contracted 0.36–0.41/step at K=5 and 0.71/step at K=30 (the K=30 map is closer to the identity in the bulk, i.e. the persistence result seen from the dynamics side), with an expanding subspace of ≥12 dims (gains 1.8–3.6) in both; realized reach (1.06/step) is far below σ₁³⁰, so credit does not ride the top direction.

### 1.6 The Q head

Global measures are flat in K: across-action Q range on replay roots 0.135 / 0.171 / 0.140 / 0.124 (§11; K=30 is 8% below K=5); Q Huber on real roots 0.0136 / 0.0166 / 0.0144 / 0.0165 and on imagined d=1 0.0145–0.0189, non-monotone (§13); the alignment of the action-driven latent displacement with the Q head's input gradient is 0.020 / 0.017 / 0.018 / 0.017 against a random baseline of 0.014–0.015 at every K (§11) — what V reads out of an action's effect is a near-random-direction component in every arm; the Q-policy's return is nearly flat (19.1→16.2) and its life length flat (§1.1).

Contact-local measures (`sep_se.py` on lead2_probe.json): the Q head's catch-vs-miss margin on **real** successors at lead 2 (v_true_sep) is 0.126 / 0.124 / 0.118 / 0.077 (SE 0.019 / 0.019 / 0.015 / 0.012; per seed K=5 0.144/0.139/0.085, K=30 0.086/0.102/0.040) — a K=30-only, ~2.2-SE effect with per-seed overlap, not a monotone halving; at lead 1 it is non-monotone (0.239 / 0.182 / 0.248 / 0.171). The actor's own margin q_sep at lead 2 is 0.125 / 0.143 / 0.062 / 0.063 (falls at K≥20). The critic's lead-2 accuracy falls at K=30 (0.771 / 0.752 / 0.760 / 0.697, SE 0.035–0.039) while its lead-1 accuracy is K-invariant (§1.2).

### 1.7 Wave 19 inverse weighting (K=30, w_k ∝ 1/(k+1), w_1 = 7.5x, w_30 = 0.25x)

Final (eps 0.01, `/tmp/w19inv_eval_*.json`): Q +3.5/4.0/4.8, H=1 +5.3/4.0/4.5, H=5 +3.4/3.2/6.6; in-training evals (`eval.jsonl`, 3 episodes) never exceed +7 at any checkpoint. The pre-registered "control recovers toward K=5" failed. What the arm did (`diagnostics.json`, `updates.jsonl`, 3 seeds):

- De-smoothed the code past K=5: held-out persistence d1 0.778 (K=5 0.650, K=30 0.246). But it is **partially collapsed across states**: final `latent_pairwise_cos` 0.331/0.305/0.282 vs 0.119–0.168 for every uniform arm.
- Restored training d1 to 0.141 (K=10 level) — and broke the iterated map: training d2 0.384 (uniform arms 0.24–0.30). Held-out absolute error is worse than uniform K=30 at every depth below 20 (d1 0.205 vs 0.097; d3 0.234 vs 0.156; d5 0.255 vs 0.197; d10 0.287 vs 0.266) and better only at d20/d30 (0.329 vs 0.351; 0.356 vs 0.419).
- **Action-blind dynamics**: root action distance 0.0034 (per seed 0.0031/0.0039/0.0032) vs 0.066 / 0.120 / 0.097 / 0.070 for uniform K=5/10/20/30 and 0.0008 for H-JEPA-attached; shuffled-action penalty ≈0% at d1/d3/d5 (0.205 vs 0.205, 0.235 vs 0.234, 0.257 vs 0.255); movement probe 0.480 vs majority 0.486.
- A broken Q: loss_q 0.0585 (0.067/0.054/0.053) vs 0.0142–0.0239 for uniform arms; q_pred_mean 1.48 vs 2.4–2.6.
- Its trajectory diverges from every uniform arm from the start: at matched ~33k updates jepa_d1 0.17–0.19 (K=5 0.14–0.15, K=30 0.28–0.29) while d2 is 0.52–0.56 (K=5 0.28–0.29, K=30 0.34–0.36) and pairwise cos 0.65–0.69 (K=5 0.30–0.34, K=30 0.24–0.25); clipped training return at 100k–138k decisions 2.2–2.3 (K=5 12.0–13.9, K=30 10.4–11.8).

So the arm changed two things at once — removed deep pressure and multiplied a scale-free one-step term by 7.5x — and the second opened a degenerate shortcut (large per-step deltas that ignore the action). It shows that the smoothing is *reversible* by re-weighting; it does not establish that the deep terms cause it (a large d1 weight alone can force large per-step deltas), and it tests nothing about control. The clean test is the discount arm (§5.9). On the lead2 probe the inverse arm is at chance for every estimator (model 0.546, critic 0.611, actor 0.632, chance 0.49; `/tmp/lead2_probe_w19inv.json`).

### 1.8 The real-successor oracle controller, with its matched learned-model control

Acting on argmax_a [r_a + γ(1−term_a) max_b Q(f(x'_a))] with the **true** one-step transition, 10 episodes × 3 seeds, eps 0.01, two branch orders (`/tmp/o3_*`, `/tmp/o2_*`, `/tmp/oracle*`, `/tmp/oraclectl_*`):

| | K=5 asc | K=5 prev | K=30 asc | K=30 prev |
|---|---|---|---|---|
| learned-model H=1 under the same branching harness | +29.8 ±1.6 | +43.0 ±2.6 | **+15.5 ±1.9** | **+16.4 ±1.3** |
| true-successor oracle | +19.7 ±1.3 | +11.8 ±2.8 | +13.6 ±0.7 | +7.0 ±0.5 |
| official H=1 (no branching) / Q | +40.4 / +19.1 | | +18.3 / +16.2 | |

(K=10 oracle +17.5 / +10.6, K=20 +14.6 / +6.7; oracle fidelity 0.80–0.92.) Established: the harness is contaminated — ALE's clone does not restore the sticky-action context, so 3 of 4 branches run with a wrong first-frame context (p=0.25), and the same branching costs the K=5 model planner 10 points under `asc` but not under `prev` (+43.0 vs the official +40.4; the clone-only modes reproduce +40.43 exactly and report fidelity 0 because no branch frames are compared). Also established, under that caveat: the true successor read by the arm's own V scores **below the learned model at both K and in both orders** (K=5 by 10–31 points, K=30 by 2–9), and at or below the Q policy at every K. `o4_fixed_oracle.py` (restores the context before every branch) exists and has not been run; its docstring *predicts* that `model_fixed` reproduces +40.43 at fidelity 1.0 — that is a check to perform, not a result.

### 1.9 Matched everything else

Updates per decision (123,750 for every completed arm), replay root sampling, epsilon, eval protocol; no capped episodes (30/30 terminated per arm and controller, `eval_*_eps001.json`); the K=30 uniform and inverse arms' first six warmup episodes have identical (return, length) tuples (frames were not compared bitwise). In-training evals (`eval.jsonl`, 3 episodes): the K=5 H=1 gain grows through training (+4.7 / +11.3 / +17.3 / +18.2 / +18.8 at 100k…500k) and the K=30 gain never appears (+2.0 / +2.3 / +4.6 / −1.4 / −5.1), so this is not undertraining.

---

## 2. What has been ruled out, and by what

1. **Vanishing or decaying credit through the rollout.** §7a: 5.4x reach, no decay; LMU dynamics matches conv at every K (§3).
2. **Rank or variance collapse.** §2/§10: per-dim std 0.83, effective rank 140, PR 55. Compression, not collapse.
3. **Action blindness of the long-K arms.** §6: 0.78x of the real action effect; shuffled-action penalty 44% at d1. The genuinely blind arms (H-JEPA attached, offline-frozen, wave-19 inverse) fail differently and score ~+4.
4. **Loss of ball or paddle position from the latent.** §10.
5. **The value head's imagined-depth supervision as the H=1 channel.** §3: K=20+qd5 H=1 22.6 vs 21.6, K=10+qd5 34.3 vs 30.7. It survives as an H≥5 contributor of poorly determined size (§3 L6).
6. **Reward and continuation heads.** §11: across-action ranges 0.0006–0.005 vs a value-term range of 0.08–0.12 at every K.
7. **The dynamics core / rollout memory.** LMU matches conv at every K with the same code (§3, §10).
8. **H>K extrapolation penalty and compounding latent error past K.** §1: flat beyond H=5; K=30's 5-step error (0.197) beats K=5's (0.214).
9. **Long-range prediction as a training-coverage failure.** §5.
10. **The critic (Q as a state value on real successors) degrading with K — as a global explanation.** §14: the critic is the best estimator of the table at K=30; lead-1 critic accuracy K-invariant (0.908/0.873/0.902/0.906); Q Huber on real roots flat (§13); the oracle controller is at or below Q at every K (§1.8), which cannot carry a K-dependent gain. **Not ruled out at lead 2**, where the critic's accuracy falls at K=30 alongside the model's (§1.2, §1.6); see §3 L4 and §4.5.
11. **Collapse of the planner's preference magnitude, alignment, or override rate.** §11.
12. **Collapse of the model's ordering relative to its own critic on average.** §12: argmax agreement 0.40–0.46, tau 0.21–0.27 at every K; lead2 tau(model, critic) 0.49→0.43.
13. **"Per-decision quality intact; only accumulation of a temporally persistent bias fails."** Disfavoured, not ruled out: three outcome-based per-decision instruments show the K=30 model's edge over the actor at contact shrinking or gone (§1.2), which is a per-decision loss on actor-generated states; the direction-based instrument does not (§1.2). The earlier autocorrelation argument used numbers not on disk and is withdrawn.
14. **"The H=1 gain was never foresight (one paddle step is below V's resolution)."** lead2: the critic on the real successor catches 0.84 vs the actor's 0.74 overall and 0.91 vs 0.80 at lead 1 at K=5, with a catch/miss margin of 0.13–0.24 Q-units (§1.6). The oracle-controller numbers quoted for this claim were not on disk; the on-disk oracle swings 8 points with branch order (§1.8).
15. **A two-decision-exclusive locus.** K=5 gains +25 from H=1→H=5 while beam5 ≤ H=1 at lead ≤2 (0.834 vs 0.857); H=5 loses 31 points K=5→K=30 while beam5 is flat at lead 2. The contact-adjacent decisions are where the *H=1* effect shows; they are not the whole story.
16. **History-dependent (policy-dependent) latents misread on the planner's own histories.** **Not ruled out** (downgraded from the previous version, which relied on a script output that is not on disk). Against: the model's deficit appears on actor-generated states (lead2, §14); the K=30 planner's histories contain *fewer* reversals than K=5's (`kin_out.txt`: 0.25–0.30 vs 0.31–0.33). For: on planner-generated states the K=30 model's regret is 0.073 worse than on actor-generated states, versus 0.041 (critic) and 0.037 (actor), with no such shift at K=5 (§1.2); 1–2 SE. §5.10 is the direct test.
17. **The critic's gradient migrating into a slow subspace.** Same as 10; the value-term range does not shrink (§11).
18. **A graded "ball-motion SNR" law / lost successor rendering.** K=20+qd5 has K=30-like code statistics on every measure and plays H=5 55.3 vs K=30's 34.0; the K=30 successor renders ball velocity *better* on the whole distribution (`decode_out.txt`); at K=20 the model is the best MC estimator (§14).
19. **Loss dilution of the depth-1 term as sufficient.** The inverse arm restored d1 to 0.141 with no control benefit — confounded by its action-blindness (§1.7), so this is "not shown", not "refuted"; also the H≥3 attenuation lives at depths whose direction loss is nearly flat in K.
20. **Depth re-weighting as a straightforward fix.** Wave 19 inverse: both controllers at ~+4, action-blind (§1.7). The discount arm is pending.
21. **Evaluation and training-protocol artifacts.** §1.9.
22. **Planner-driven collection / behaviour–target mismatch.** §8 is a different failure; every K arm collects with its Q policy.

---

## 3. The best-supported mechanism

Stated as a chain from objective to score. "E" = established by a measurement on disk; "I" = inferred; falsifiers in §5.

**L1 (E).** With uniform depth weights the encoder's latent loss is the mean of K delta-cosine terms (`losses.py`: `masked_mean(dist * w_depth)`), K−1 of them deep; a residual LayerNorm chain delivers them at full strength (§7a), from K≈20 they pull against each other (§7b), and the K=30 map contracts random directions far less per step than K=5's (§7c).

**L2 (E).** The encoder's stationary point under that pressure is a temporally coherent, compressed code: per-step change 0.64→0.24 with the action's part preserved, gap-1 0.77→0.29, PR 104→55; the same for LMU and qd5 (so it is the objective, not the core or Q's depth). The K=30 code linearly carries ball velocity, the K=5 code does not. Down-weighting the deep terms with a 7.5x shallow weight reverses the smoothing (persistence 0.78) — but with a confound (§1.7), so "the deep terms cause the smoothing" is I until the discount arm reports.

**L3 (E as a fact; interpretation open).** In that code the one-step change is small and its *direction* is predicted worse, most at d1 and tapering by d5 (training d1 0.115→0.235; excess 0.120/0.064/0.053/0.024 at d1/d2/d3/d5; held-out d1 ratio 0.22→0.40, d3 0.24→0.26, d≥10 better). Whether this is content lost from g's one-step output, a scale-free metric reading a shrunken signal, or irrelevant to control has not been separated; its only causal test (the inverse arm) was confounded (§2 item 19). It is a *correlate* in this chain, not an established link.

**L4 (E for the pattern; I for any mechanism).** At contact-adjacent decisions the one-step planner's edge over the actor shrinks with K on four outcome-based instruments and not on the direction-based one (§1.2). The K-dependence is lead-specific: at lead 1 the model's edge is intact at K=30 (+0.11 over the actor) and the critic is K-invariant; at lead 2 the model, the critic and the actor all sit at ~0.68 at K=30, whereas at K=5 (and K=20) the model and critic sit ~0.10 above the actor. So at K=30 *no* successor read — predicted or real — adds anything at lead 2, and the Q head's catch/miss margin on real successors there is smaller (0.077 vs 0.126, ~2 SE). The previous version's inference ("mispredicted one-step direction applied to a steeper V flips the sign") is withdrawn as the primary account: it does not explain the critic's fall at lead 2, and §11 shows the action displacement is a near-random direction to V at every K. What survives as inference: the K=30 Q head resolves the catch/miss outcome one decision later than K=5's does (fine at lead 1, not at lead 2), and the model's lead-2 read inherits that; whether the compressed code or Q's supervision on deep imagined latents causes it is §5.1.

**L5 (E for the correlation; I for sufficiency).** The H=1 gain is survival (§1.1). Catch rates per descent order the same way as returns in every closed-loop sample (K=5 H=1 0.88 > K=30 H=1 0.81 ≈ K=5 Q 0.80 ≈ K=30 Q 0.79, approach_probe2), but no mapping from per-decision oracle accuracy to per-descent catch probability has been derived, and the three closed-loop samples disagree on the sign of K=30's H=1−Q difference (§1.1). The hybrid-controller test (§5.4) settles sufficiency.

**L6 (E for the pattern; the rest not located).** Search of depth ≥3 removes most of the H=1-specific component (H=5 gain ratio 0.46–0.53 at K=30 vs 0.10–0.12 for H=1). The remaining H≥3 attenuation is a survival loss with a flat clipped-score rate (§1.1). It is not at lead ≤2 (beam5 flat, §1.2); not in 3–5-step fidelity (flat, §1.3); the qd5 arms show that Q's imagined-depth supervision contributes at H=5 (+6.2 at K=20 on the score, +9 on the gain, ±3.3 seed spread — anywhere from a fifth to most of the K=5–K=20 gap; no K=30+qd5 arm exists). The low-power decisive probe (`/tmp/decisive_probe.log`, pooled lead 6–9, n=56/45) puts K=5 model/beam5 at 0.70/0.77 and K=30 at 0.64/0.69 with actor/critic 0.72/0.64 (K=5) vs 0.76/0.76 (K=30): at K=30 the actor and critic beat the planners at those leads; SE ≈0.07. Mean lead at which lost races were lost is 3.7–4.4 at K=30 vs 2.5–3.0 at K=5 (approach_probe2) — but that holds for the K=30 *Q controller* (4.43) as much as for H=1/H=5 (3.66/4.14), so it is a property of K=30 play, not of search.

### Three readings survive; they differ in what fixes it

- **A (map fidelity):** what is wrong is g's one-step output on the same encoder; a better one-step map on the frozen K=30 encoder and Q head restores the read. Support: the shallow-depth direction loss (L3) and the lead-2 disagreement split (model right 0.55→0.44, n=40–63). Against: §12 (argmax agreement and tau with the critic flat in K; counterfactual one-step error *lower* at K=30, 0.099 vs 0.148); the critic's own fall at lead 2 (§1.2); and the matched oracle table (§1.8): a *perfect* one-step map read by V scores below the learned model at both K in both branch orders. A survives only if the harness contamination is what sinks the oracle — §5.5.
- **B (composite co-adaptation):** the K=5 gain is largely a property of Q∘g as a jointly trained action-value (README's "second, differently-trained head"), and the code change destroys the composite's content without any single component being identifiably wrong. Support: the oracle anomaly as it stands (§1.8), the K=5 model beating the critic on all four instruments (§1.2), and the lead-2 pattern where predicted and real successors fail together at K=30.
- **C (the value head's contact resolution):** what degrades at K≥20 is the Q head's ability to separate catch from miss one step earlier (v_true_sep, q_sep, critic accuracy at lead 2), so any one-sample successor read at lead 2 is uninformative and the planner has nothing to add there. Support: §1.2/§1.6. Against: the effect is K=30-only and ~2 SE; the K=10 actor breaks the pattern; and the K=30 Q *policy* already moves toward the landing at leads 1–2 as often as the K=5 planner (approach_probe2, "P(move toward landing | outside window)": K=30 Q 0.66/0.65, K=5 H=1 0.73/0.64, K=5 Q 0.59/0.51) without catching more (0.79 vs 0.80), which fits neither C cleanly nor "the Q policy caught up".

They are not exclusive. A predicts that the swap-in head (§5.7) and the corrected oracle (§5.5) both move; B predicts neither moves without retraining Q; C predicts the qd5 arms' lead-2 critic margin is restored (§5.1) and that the corrected K=30 oracle stays at Q.

What this account does *not* claim, because it was tested and failed or was never measured: that the long-K model is action-blind, that credit decays, that Q's depth is the H=1 channel, that the loss is only cumulative, that the latent is policy-independent (open), or that the depth-1 direction loss is causal (open).

---

## 4. What remains genuinely unexplained

1. **Which coordinate of g(z,a) — if any — misleads Q at K≥20.** Cosine says the successor is closer; three outcome instruments say the decision is worse; one direction instrument says it is not. No measurement isolates the delta direction, the paddle–ball relation, or anything else at the decisive states.
2. **The H≥3 attenuation (onset K≥20, ~half the H=5 gain at K=30).** A survival loss (§1.1) that is not at lead ≤2, not in 3–5-step fidelity, only partly attributable to Q's imagined depth; the only lead 3–9 evidence is the low-power decisive probe, where the K=30 *actor and critic* beat the planners. No K=30+qd5 arm exists.
3. **The oracle-controller anomaly.** A true successor read by V is worse than the learned model at both K and no better than Q as a policy at any K (§1.8). Harness contamination is the leading suspect; the corrected harness has not been run. If it is real, no one-sample successor reproduces the K=5 planning gain and B gains weight.
4. **Four MC/oracle instruments disagree about the K=30 model at contact** (§1.2): best of three on actor-state ttc, worst or below the critic on the other three. State sampling and strata differ; all carry the sticky-context flaw. "K=30 model at chance" (§14) is one instrument's result. Two things are consistent across all four: the K=5 model is best everywhere, and the model−actor edge shrinks with K.
5. **Why the Q head's lead-2 catch/miss margin and lead-2 critic accuracy fall at K=30** while its lead-1 accuracy, TD error, action range and policy are flat. Candidates: 97% of its supervision on deep imagined latents (the qd5 arms would show it), or the compressed code itself. Unmeasured on the qd5 arms.
6. **Why the direction loss concentrates at shallow depth** (L3), and whether it matters for control at all.
7. **How the inverse-weighted arm went action-blind**, and whether any re-weighting decouples smoothing from that shortcut. Paddle R² and `action_effect` on those checkpoints were never run.
8. **The K=10 step.** With the 6-seed anchor a third of the H=1 gain is lost (17.0→10.8, spreads ±3.9/±6.9) while every representation statistic moves a little (gap-1 0.77→0.62, d1 loss 0.115→0.139, lead2 model 0.857→0.835, 0% depth-term conflict).
9. **Whether the planner-generated state distribution matters at K=30** (§2 item 16).
10. **Why direction-based and outcome-based per-decision instruments disagree on K-dependence** (§1.2). A resolution would say which of them the closed-loop catch rate follows.

---

## 5. Decisive next measurements, cheapest first

All but the last two run on existing checkpoints with no training.

1. **lead2_probe on K=20+qd5, K=10+qd5, LMU K=30 (~1 min/seed).** A: model accuracy tracks K (qd5 arms ~0.80/0.83, LMU K=30 ~0.78). C: the qd5 arms' lead-2 v_true_sep and critic accuracy are back at K=5 level (≥0.11; ≥0.76) while model accuracy stays K-like — then §4.5 is the value-depth channel and belongs to the H≥3 component. B: nothing at lead 2 moves.
2. **Paddle R² (embeddings.py) and action_effect on the inverse checkpoints (minutes).** Prerequisite for any further re-weighting arm. Either the paddle is gone from the encoder (H-JEPA-attached signature) or the dynamics ignore an encoded paddle; either says the 7.5x scale-free d1 term opened a shortcut.
3. **Delta-direction fidelity on the lead2 states (minutes).** For each state and action: cosine distance between g(z,a)−z and f(x'_a)−z, per K, split by whether the model was right. A predicts the mean rises K=5→K=30 in step with jepa_d1 *and* is larger on the model's wrong calls than its right calls within each arm; B and C predict no within-arm relation. This is the only cheap test of whether L3 is a link or a correlate.
4. **Hybrid controllers on K=5 (2 min/checkpoint; RAM byte 101 read-only):** planner only when the ball is descending with ball_y ≥ 161, Q elsewhere; and the converse. L5 predicts the first recovers ≥60% of the +17–21 and the converse ≤40%; the reverse outcome moves the locus away from contact and makes the direction-based instrument (§1.2) the one to trust.
5. **Run `o4_fixed_oracle.py` on K=5 and K=30 (~15 min/arm).** First check: `model_fixed` must reproduce +40.43 (40.0/43.3/38.0) at fidelity 1.0 — if it does not, the harness is still contaminated and nothing else in the run is interpretable. Then: A predicts the clean K=5 oracle above Q (≈+25–35) and the K=30 oracle ≈ Q; B predicts the K=5 oracle ≤ +22 even clean; C predicts the K=30 oracle ≈ Q regardless. If the clean K=5 oracle reaches ≈+40, the gain is pure foresight, B is dead, and the K=30 loss is in what V reads from a successor.
6. **lead2 extended to leads 3–6 with H=1 and beam5 estimators (~15 min), powered.** The existing decisive probe has lead 3–5 bins of n=1–4 and lead 6–9 of n≈50 per K; it hints that at K=30 the actor/critic beat both planners at leads 6–9 (0.76 vs 0.65/0.69). A powered version (≥150 states per lead bin per K) locates §4.2: if beam5 at K=30 is below K=5 at leads 3–6 it is a per-decision multi-step read loss; if flat there too the H≥3 loss is state-distributional and needs a closed-loop probe.
7. **Swap-in one-step head (minutes of GPU + one 10-episode eval per seed).** Prerequisite: compute the *held-out delta-cosine* d1 of the existing K=30 and K=10 dynamics with the training code path (`compute_losses` on held-out batches) — no such number is on disk; the diagnostics report absolute cosine and the training log reports the on-policy replay value. Then fit a `JumpHead` on the frozen K=30 encoder with the delta-cosine target at Δ=1 (action one-hot) until its held-out d1 matches or beats the K=10 dynamics' value, score it on lead2, and run H=1 with it in place of `model.dynamics`. A: lead2 model accuracy → ≥0.83 and H=1 ≥ +24 (from 18.3). B and C: no change; then retrain the Q head alone on the frozen encoder + new head and re-test (B predicts recovery only there; C predicts none).
8. **Powered MC (≥500 states/seed, ≥8 rollouts per branch with common random numbers, sticky context restored before every branch, stratified by lead) on K=5/20/30.** Resolves §4.4. All readings predict model−critic ≤ 0 at K=30 and > 0 at K=5 on contact-adjacent states with SE ≤ 0.015; A additionally predicts the K=20 model already below the critic on the |gap|<15 stratum.
9. **The pending discount arm (0.9^k, w_1 = 3.1x, w_30 = 0.15x; ~137k/500k).** At matched ~33k updates it sits between the uniform and inverse arms on the shallow terms and with the inverse arm on Q: jepa_d1 0.22–0.23 (K=5 0.14–0.15, K=30 0.28–0.29, inverse 0.17–0.19), jepa_d2 0.38–0.41 (0.28–0.29 / 0.34–0.36 / 0.52–0.56), loss_q 0.042–0.044 (0.018–0.019 / 0.029–0.031 / 0.048–0.056), q_pred_mean 1.94–2.04 (2.18–2.25 / 2.13–2.20 / 0.96–1.16); clipped training return over 100k–138k decisions 9.2–10.0 (K=5 12.0–13.9, K=30 10.4–11.8, inverse 2.2–2.3); 100k in-training eval (3 episodes) Q 7.0/9.0/8.3, H=1 9.0/4.3/5.7 (K=5 10.3/11.0/13.7 and 16.3/12.7/20.0; K=30 9.3/15.3/11.0 and 14.0/17.7/10.0). Pre-registered: **if** its final Q is ≥ +14 and its code lands between K=10 and K=20 (held-out persistence d1 0.35–0.50, training jepa_d1 0.14–0.18, root action distance ≥ 0.05), then all three readings predict H=1 ≥ +26 and H=5 ≥ +50; if it de-smooths with a healthy Q and H=1 stays ≤ +22, the smoothing premise shared by A, B and C is wrong and the cause is K itself (transitions per update, 30-depth heads); if Q ≤ +8 or root action distance < 0.01 it has repeated the inverse arm's shortcut and tests nothing. The elevated loss_q and low q_pred at 33k updates make the last outcome the one to watch for.
10. **lead2_probe and the ttc probe on planner-generated states at every K (minutes).** Tests §2 item 16 directly: if the K=30 model's (and only the model's) accuracy is lower on H=1-generated states than on Q-generated states by ≥0.05 while K=5's is not, history dependence is back in the mechanism; if all estimators shift equally, it is a state-difficulty effect.
11. **Training controls (later):** K=30 with the JEPA loss restricted to depths ≤5 and everything else unchanged (A/B/C: H=1 ≥ +33 unless it reproduces the inverse arm's shortcut, which item 2 will have characterized), and K=30+qd5 (all: H=1 ≤ +24; H=5 +40–48 sizes the qd5 share of §4.2 at K=30).