# Evidence dossier: the rollout-length / control dissociation

All runs: Breakout (ALE/Breakout-v5, sticky 0.25, 4-frame decisions, FIRE on reset and life loss),
500k decisions unless stated, 3 seeds, evaluation eps = 0.01, 10 episodes, spreads are population
std over seeds. "conv" = the standard world model: Nature-CNN encoder + LayerNorm -> latent [64,7,7],
residual conv dynamics z' = LayerNorm(z + delta(z, a)), Q head (Double DQN, n-step 5), reward head
(3-class), continuation head, JEPA latent loss with the **delta** target (predict the per-step change,
cosine distance) + motion channels, EMA target encoder (tau 0.99). K = replay.rollout_steps = the
number of steps the dynamics is unrolled and supervised over; all K losses (JEPA, reward, continuation,
imagined-Q) are masked means over the K steps, uniform weight per depth. Planning = beam search
(width 16) over the one-step model, bootstrapping on max_a Q at the leaf.

## 1. Control vs rollout length (the central dissociation)

| arm | Q | H=1 | H=3 | H=5 | H=10 | H=20 | H=30 |
|---|---|---|---|---|---|---|---|
| conv K=5  | +19.10±1.6 | +40.43±2.2 | +45.57±6.9 | +65.07±1.1 | +62.00±5.0 | +61.97±4.6 | +60.33±3.2 |
| conv K=10 | +19.90±5.4 | +30.73±6.9 | +47.67±3.3 | +61.83±9.1 | +59.33±4.4 | +57.70±3.3 | +63.80±3.2 |
| conv K=20 | +17.63±1.3 | +21.63±1.1 | +39.40±3.3 | +49.13±3.3 | +47.03±4.2 | +55.73±4.8 | +51.33±4.3 |
| conv K=30 | +16.20±0.4 | +18.30±1.7 | +26.73±4.2 | +34.03±1.4 | +36.47±0.3 | +34.83±1.0 | +36.90±4.0 |

- 6-seed check of conv K=5: Q +22.13±3.68, H=1 +39.17±3.90, H=5 +60.62±6.90 (3-seed +65.07±1.1 was tight by luck).
- Search depth beyond H=5 neither helps nor hurts at any K (flat). No H>K penalty exists.
- The damage is largest for shallow controllers (H=1: 40.4 -> 18.3) and shrinks with search depth.
- Model-free baseline (no world model) at 500k: about +8; at 2.5M: +18.9.
- Throughput: 288 / 176 / 60 decisions per second at K=5/10/30.

## 2. Model quality vs rollout length (diagnostics, held-out episodes, eps-greedy 0.1 policy)

Predicted-vs-real latent cosine distance at depth d (lower = better), plus other stats:

| arm | d=1 | d=5 | d=10 | d=20 | d=30 | root action dist | movement probe (bal. acc) |
|---|---|---|---|---|---|---|---|
| K=5  | 0.1401 | 0.2137 | 0.3292 | – | – | 0.0661 | 0.580 |
| K=10 | 0.1349 | 0.2006 | 0.2769 | 0.4132 | 0.5022 | 0.1204 | 0.641 |
| K=20 | 0.1192 | 0.1993 | 0.2606 | 0.3517 | 0.4307 | 0.0968 | 0.629 |
| K=30 | 0.0971 | 0.1970 | 0.2660 | 0.3509 | 0.4189 | 0.0701 | 0.656 |

Persistence baseline (predict no change) at d=1: 0.6497 (K5), 0.5298 (K10), 0.3594 (K20), 0.2460 (K30).
Shuffled-action penalty at depth 5 (relative increase of error when actions are permuted): 48% (K5),
97% (K10), 81% (K20), (K30 similar to K20).
Latent per-dim std stays ~0.83 (no variance collapse); movement probe majority baseline ~0.43-0.50,
chance 0.33. Effective rank of root latents ~100-200 for all delta-target arms (no rank collapse).
"root action dist" = mean pairwise cosine distance between model-predicted next latents for the 4
actions from the same root.

Monotone: every model-quality measure improves with K; every control measure degrades.

## 3. Interventions that did NOT rescue control at long K

- **q_imagined_depth cap** (train Q only on imagined depths 0..4 while keeping K=20 rollout):
  K=20+qd5: Q +14.83, H=1 +22.57 (vs +21.63 uncapped), H=5 +55.33 (vs +49.13). K=10+qd5: H=5 +66.27±12.7.
  Row maximum unchanged. K=20+qd5 has the best d=1 latent error of any arm (0.0997) and still plays
  10+ points below K=5. => the value head's supervision depth is not the channel.
- **LMU dynamics** (Legendre Memory Unit state threaded through the rollout, theta = K, fixed LDN
  basis, +12% params): matches conv at every K. Best-of-row: K=5 +68.2 (H=30, +-10), K=10 +64.2,
  K=20 +47.8, K=30 +36.2. Same slope.
- **Offline-pretrained encoder (MSE+SIGReg on frames from a trained agent), LMU dynamics**:
  frozen: +2.7/+3.4/+1.6 at K=5/10/30 (floor: that encoder's total change per transition is 0.016,
  it barely separates consecutive frames). Fine-tuned: H=5 +5.7 / +27.3(+-21) / +24.1.
- **Successor features** (psi(z,a)=E[sum gamma^k phi(z_{t+k+1})], phi frozen random projection,
  bootstrapped, detached): inert (Q/H1/H5 unchanged: +19.97/+39.90/+61.10) but useless as a value:
  greedy on w.psi scores +2.0. psi is identical across actions (pairwise cosine 1.0); action gap
  0.0195 vs Q head's 0.1504; argmax agrees with Q on 20% of decisions (chance 25%).
- **H-JEPA level 2** (jumpy 5-step macro dynamics conditioned on the action sequence): attached to the
  encoder it drives every loss down and control to +4.6 (H=5), root action dist 0.0661 -> 0.0008,
  shuffled-action penalty 48% -> 2%, movement probe 0.580 -> 0.493 (= majority baseline). Detached it
  is harmless (H=5 +57.3 at 6 seeds vs +60.6) and its own planner scores +29.2.

## 4. What DID help (for contrast)

- n-step returns (n=3..10 all similar; n=1 much worse): worth ~3x the samples. +28.3 -> +40.4 at H=1.
- The delta target itself (predict per-step change instead of absolute next latent): ~3x score at
  K=5; the absolute-cosine objective produced rank-collapsed, action-blind latents.
- 300k replay buffer + replay ratio 0.5 on top of n-step: +47.9 at H=1 (best 500k recipe).
- Beam search depth H=1 -> H=5: +40 -> +65 at K=5.

## 5. Delta probe (frozen encoder; head trained directly on (x_t, x_{t+delta}) pairs vs iterating the
one-step dynamics; score = 1 - d/d_mean so 0 = predicting the conditional-mean latent)

| delta | K=5 iterated | K=5 direct | K=30 iterated | K=30 direct | K=5 persistence | K=30 persistence |
|---|---|---|---|---|---|---|
| 1   | +0.770 | +0.835 | +0.846 | +0.922 | -0.067 | **+0.609** |
| 5   | +0.663 | +0.618 | +0.692 | +0.771 | -0.283 | -0.153 |
| 10  | +0.478 | +0.445 | +0.599 | +0.649 | -0.375 | -0.228 |
| 30  | +0.039 | +0.121 | **+0.462** | +0.315 | -0.392 | -0.261 |
| 100 | -0.159 | -0.068 | +0.083 | +0.023 | -0.366 | -0.264 |
| 300 | -0.165 | -0.113 | -0.143 | -0.099 | -0.388 | -0.361 |

- Direct supervision at delta=100 cannot beat the mean latent: the long-range wall is stochasticity
  (sticky actions, bounces), not training coverage.
- K=30's latents barely move between consecutive frames (persistence +0.609 at delta=1: "predict no
  change" beats the mean predictor by a lot); K=5's consecutive latents are nearly orthogonal in the
  centered sense (delta target explicitly strips the persistent component).
- Episode lengths: ~640 decisions (train), ~1400-1600 (good eval agent). gamma=0.99 -> effective
  value horizon ~100 decisions. Prediction dies at delta ~100 too.

## 6. Counterfactual action effect (branch from one cloned emulator state incl. RNG, take each action,
encode each real successor with the arm's own frozen encoder; same statistic as "root action dist")

| arm | real successor distance across actions | model-predicted | model/real | total change root->successor | action share |
|---|---|---|---|---|---|
| conv K=5  | 0.1055 | 0.0699 | 0.66x | 0.6376 | 16.5% |
| conv K=10 | 0.1472 | 0.1157 | 0.79x | 0.5285 | 28.0% |
| conv K=20 | 0.1201 | 0.0992 | 0.82x | 0.3488 | 34.3% |
| conv K=30 | 0.0917 | 0.0706 | 0.78x | 0.2402 | 38.1% |
| LMU K=30  | 0.1064 | 0.0875 | 0.82x | 0.2473 | 43.0% |
| H-JEPA attached | 0.0160 | 0.0008 | 0.05x | 0.5192 | 3.1% |
| LMU K=5 offline frozen | 0.0009 | 0.0001 | 0.08x | 0.0161 | 5.5% |

NOTE: "real" is measured in each arm's own latent space, so it is a property of that encoder, not of
the world. Long-K arms are NOT action-blind (0.78x of ceiling). What collapses with K is the TOTAL
latent change per transition (0.64 -> 0.24) while the action-driven part holds roughly constant; so
the action's SHARE rises. Between two decisions in Breakout the paddle moves (action-driven) and the
ball moves ~4 frames (not action-driven except at contact); the non-action change is mostly the ball.

## 7. Gradient measurements (frozen checkpoints, one backward pass each)

**7a. Gradient reach.** Backprop ONLY the deepest JEPA term L_K; report ||dL_K/dz_hat[k]|| relative
to its value at k=K:

| arm | k=0 | k=K/4 | k=K/2 | k=3K/4 | k=K-1 | k=K |
|---|---|---|---|---|---|---|
| conv K=30 | 5.42x | 5.16x | 3.93x | 3.10x | 1.30x | 1.00x |
| LMU K=30  | 4.78x | 4.55x | 4.00x | 3.24x | 1.29x | 1.00x |
| conv K=10 | 5.69x | 5.17x | 4.78x | 3.40x | 1.76x | 1.00x |
| conv K=5  | 2.34x | 2.32x | 1.95x | 1.54x | 1.29x | 1.00x |
| LMU K=10 offline frozen | 41.5x | | | | | |
| LMU K=5 offline finetune | 2.92x; K=10 finetune 8.09x | | | | | |

Credit grows going backward and saturates: per-step gain 1.30 (k=30->29), 1.13/step (29->22),
1.035/step (22->15), 1.007/step (7->0). No vanishing gradient. Grad-norm clipping (10.0) never binds
(total grad norm ~1.6-2.1 throughout training).

**7b. Depth-term gradient anatomy.** Backprop each depth term L_d separately; compare the root
gradients g_d = dL_d/dz_0 pairwise (cosine); "action share" = ||dL_d/d embed(a_k)|| / ||dL_d/dz_k||:

| arm | effective rank of {g_d} | rank/K | mean pairwise cos | fraction of pairs with cos<0 | mean cos of those | action share (d=K, k=0) | action share (d=K, k=K-1) |
|---|---|---|---|---|---|---|---|
| K=5  | 3.21 | 0.64 | +0.425 | 0.0% | 0 | 0.915 | 0.526 |
| K=10 | 4.13 | 0.41 | +0.478 | 0.0% | 0 | 0.696 | – |
| K=20 | 7.29 | 0.36 | +0.329 | 3.3% | -0.032 | 0.622 | – |
| K=30 | 9.37 | 0.31 | +0.278 | 8.4% | -0.066 | 0.415 | 0.739 |

Selected pairwise cosines at K=30 (seed 0): d1-d7 +0.113, d1-d15 +0.039, d1-d30 +0.015,
d7-d15 +0.367, d7-d30 -0.019, d15-d30 **-0.345**. At K=5: d1-d2 +0.575, d1-d5 +0.143, d2-d5 +0.273.
=> Deep terms are neither parallel (no rank collapse to 1) nor independent; they increasingly OPPOSE.
Action-embedding credit from the deepest term decays only ~2x over 30 steps.

**7c. Per-step Jacobian spectrum** (one state, seed 0):
- Random-direction gain (48 random probes): K=5 median 0.36-0.41, top-probe 0.57-0.67; K=30 median
  0.71, top-probe 0.87, max/min over probes 1.4. LMU K=30 similar.
- Power iteration (true top): K=5 sigma_1 = 3.31, sigma_12 = 1.95; K=30 sigma_1 = 3.64, sigma_12 =
  1.79. All 12 probed top directions > 1.
=> A contracting bulk (~0.7-0.87 per step in a random direction) and an expanding subspace of at least
~12 dims with gains 1.8-3.6. Backward credit lands in the expanding subspace (explains growth without
decay) but does not collapse to one direction (explains effective rank ~9-10 at K=30). The realized
reach (5.4x over 30 steps ~ 1.06/step) is far below sigma_1^30, so the gradient does not stay on the
top direction.

## 8. Earlier project history relevant to mechanism

- Original spec objective (absolute cosine on the whole map): latent effective rank 13-20 (Pong),
  4-5 (Breakout) out of 3136; dynamics action-invariant (shuffled-action penalty ~0%). Low loss,
  useless model. Fixed by the delta target (predict change), which tripled Breakout score.
- Pong: the delta target gave no gain; Pong's ball was already better represented and its optimal
  policy is largely reactive.
- Ball position decodability (ridge R^2 from latents to RAM) was the weakest state variable for the
  original objective (ball x R^2 0.36-0.56 vs 0.85+ for paddle); delta+motion runs reached ~0.85.
- Planner-driven data collection hurts (behaviour/target mismatch degrades the Q head).
- Better prediction has never implied better control anywhere in this project; the delta target is
  the only representation change that improved both.

## 9. Pending

- Wave 19: K=30 with depth-weighted JEPA loss ("inverse": w_k ∝ 1/(k+1); "discount": 0.9^k;
  weights normalized to mean 1). Predictions written before running: control recovers toward K=5,
  conflicting-pair fraction drops, delta=1 persistence falls back, long-horizon prediction stays
  better than K=5. Inverse arm at ~406k/500k at time of writing.
- State-decodability probes (ridge R^2 for paddle x, ball x, ball y from latents) across the K sweep,
  LMU K=30, K=20+qd5, H-JEPA attached/detached: running now, results to be appended below.

## 10. State decodability, spectrum, temporal structure (embeddings.py, held-out episodes, 3 seeds)

Ridge R^2 from the frozen online latent to ALE RAM (evaluation-only; RAM never seen by the agent):

| arm | paddle_x | ball_x | ball_y | blocks_hit | mean R^2 over 43 varying bytes | #bytes R^2>0.5 |
|---|---|---|---|---|---|---|
| conv K=5  | 0.690 | 0.854 | 0.734 | 0.636 | -0.053 | 11.0 |
| conv K=10 | 0.736 | 0.827 | 0.717 | 0.549 | 0.150 | 13.0 |
| conv K=20 | 0.765 | 0.852 | 0.756 | 0.676 | 0.101 | 15.3 |
| conv K=30 | 0.775 | 0.861 | 0.668 | 0.818 | -0.124 | 15.7 |
| LMU K=30  | 0.676 | 0.824 | 0.455 | 0.375 | -0.133 | 12.0 |
| K=20+qd5  | 0.679 | 0.730 | 0.428 | 0.641 | -0.069 | 10.0 |
| H-JEPA attached | **0.043** | **0.907** | **0.900** | 0.317 | -0.188 | 6.3 |
| H-JEPA detached | 0.661 | 0.849 | 0.744 | 0.721 | -0.324 | 11.0 |

=> The ball's POSITION is NOT lost at long K (ball_x flat ~0.85, ball_y 0.73 -> 0.67 modest); paddle
and block-count decodability IMPROVE with K. The attached H-JEPA lost the PADDLE (0.043) while
representing the ball best of all arms -- its "action blindness" is paddle blindness. (Ball VELOCITY
is not probed; the latent has motion channels so it could be encoded; unknown.)

Spectrum of online root latents (PCA):

| arm | participation ratio | effective rank (entropy) | top-1 var frac | top-10 var frac | dims for 90% | dims for 99% |
|---|---|---|---|---|---|---|
| conv K=5  | 104.1 | 332.6 | 0.055 | 0.244 | 468 | 1199 |
| conv K=10 |  86.3 | 255.7 | 0.056 | 0.269 | 375 | 1081 |
| conv K=20 |  68.6 | 181.2 | 0.062 | 0.306 | 252 |  907 |
| conv K=30 |  55.4 | 140.0 | 0.067 | 0.349 | 185 |  866 |
| LMU K=30  |  59.3 | 140.6 | 0.058 | 0.337 | 177 |  779 |
| K=20+qd5  |  72.6 | 176.6 | 0.051 | 0.298 | 235 |  909 |
| H-JEPA attached | 47.4 | 195.5 | 0.099 | 0.361 | 345 | 864 |
| H-JEPA detached | 93.9 | 311.1 | 0.062 | 0.254 | 445 | 1143 |

=> Latent dimensionality shrinks monotonically and substantially with K (effective rank 333 -> 140,
2.4x; dims for 90% variance 468 -> 185). Not a collapse (140 is still large) but a strong compression.

Temporal structure (centered cosine distance between latents t and t+gap within an episode; across
different episodes ~1.0 for all arms):

| arm | gap 1 | gap 2 | gap 5 | gap 10 | gap 20 | gap 50 |
|---|---|---|---|---|---|---|
| conv K=5  | 0.768 | 0.747 | 0.920 | 0.985 | 0.999 | 0.943 |
| conv K=10 | 0.617 | 0.753 | 0.917 | 0.964 | 0.983 | 0.950 |
| conv K=20 | 0.412 | 0.640 | 0.902 | 0.947 | 0.958 | 0.924 |
| conv K=30 | 0.285 | 0.513 | 0.854 | 0.911 | 0.926 | 0.855 |
| LMU K=30  | 0.288 | 0.507 | 0.817 | 0.866 | 0.888 | 0.882 |
| K=20+qd5  | 0.312 | 0.536 | 0.830 | 0.875 | 0.895 | 0.878 |
| H-JEPA attached | 0.616 | 0.615 | 0.970 | 1.011 | 1.029 | 0.931 |

=> With K, latents become temporally smooth at every gap: at K=5 a latent is essentially uncorrelated
with itself 10 steps later (0.985 ~ across-episode level); at K=30 it stays correlated out to gap 50.
The K=5 space is "memoryless" (each frame stack nearly orthogonal to the next in the centered sense);
the K=30 space carries slow episode-level structure.

## 11. The planner's action signal (512 replay roots per seed, 3 seeds)

score(a) = E[r|z,a] + gamma P(cont|z,a) max_b Q(g(z,a)). Across-action ranges:

| arm | Q(z,.) range | planner range | reward-term range | cont-term range | value-term range | action displacement / |z| | |dQ/dz| at g(z,a) | alignment of displacement with Q-gradient (random baseline) | planner argmax = Q argmax |
|---|---|---|---|---|---|---|---|---|---|
| K=5  | 0.135 | 0.082 | 0.0006 | 0.0002 | 0.084 | 0.374 | 0.245 | 0.020 (0.015) | 0.32 |
| K=10 | 0.171 | 0.108 | 0.0008 | 0.0002 | 0.109 | 0.475 | 0.252 | 0.017 (0.014) | 0.32 |
| K=20 | 0.140 | 0.116 | 0.0025 | 0.0002 | 0.117 | 0.466 | 0.274 | 0.018 (0.014) | 0.35 |
| K=30 | 0.124 | 0.108 | 0.0054 | 0.0003 | 0.107 | 0.407 | 0.300 | 0.017 (0.014) | 0.33 |
| K=20+qd5 | 0.112 | 0.096 | 0.0012 | 0.0002 | 0.096 | 0.380 | 0.325 | 0.014 (0.015) | 0.33 |
| LMU K=30 | 0.114 | 0.102 | 0.0037 | 0.0001 | 0.102 | 0.441 | 0.307 | 0.017 (0.015) | 0.31 |

=> The planner's preference MAGNITUDE does not collapse with K. The reward and continuation heads
contribute essentially nothing to action choice at any K (Breakout reward is a rare event); the
planner is entirely "Q evaluated at the predicted next latent". The planner overrides Q on ~2/3 of
decisions at every K. The action-driven latent displacement is nearly orthogonal to the Q head's
input gradient at every K (barely above a random direction).

**Key reframing:** the Q-controller itself barely degrades with K (+19.1 -> +16.2). What collapses is
the PLANNING GAIN, H=1 minus Q: +21.3 (K=5), +10.8 (K=10), +4.0 (K=20), +2.1 (K=30). The world model
stops adding anything on top of Q.

## 12. Counterfactual value ordering (300 cloned states per seed; the arm's OWN Q head throughout)

v_real[a] = max_b Q(f(x'_a)) on the REAL successor of each action; v_model[a] = max_b Q(g(z,a)).

| arm | argmax agree | Kendall tau | model one-step error on all 4 counterfactual successors | regret (in the arm's own Q units) |
|---|---|---|---|---|
| K=5  | 0.404 | +0.211 | 0.1483 | -0.0236 |
| K=10 | 0.457 | +0.273 | 0.1403 | -0.0234 |
| K=20 | 0.421 | +0.239 | 0.1253 | -0.0256 |
| K=30 | 0.430 | +0.240 | **0.0991** | -0.0224 |
| K=20+qd5 | 0.400 | +0.229 | 0.1063 | -0.0261 |

=> Relative to its own Q head, the K=30 model ranks actions at least as well as K=5's, and its
one-step prediction of the counterfactual successors is markedly better (0.099 vs 0.148). So the
model is not "worse at what the planner consumes" as judged by the arm's own critic. This leaves the
CRITIC (Q as a state-value function) or a compounding/sequential effect as the remaining candidates.

## 13. Head quality by depth (diagnostics, 3 seeds)

| arm | Q Huber (real root) | Q Huber (imagined d=1) | Q Huber (d=5) | reward CE d=1 | reward CE d=5 | continuation BCE d=1 |
|---|---|---|---|---|---|---|
| K=5  | 0.0136 | 0.0145 | 0.0250 | 0.0266 | 0.0526 | 0.0044 |
| K=10 | 0.0166 | 0.0184 | 0.0274 | 0.0335 | 0.0349 | 0.0067 |
| K=20 | 0.0144 | 0.0159 | 0.0240 | 0.0261 | 0.0376 | 0.0038 |
| K=30 | 0.0165 | 0.0189 | 0.0304 | 0.0361 | 0.0475 | 0.0090 |
| K=20+qd5 | 0.0121 | 0.0144 | 0.0275 | 0.0256 | 0.0273 | 0.0092 |

=> No head is grossly worse at shallow depth at long K; the differences are within ~30% and not
monotone. TD error is a self-consistency measure, not a ground-truth one.

## 14. Pending ground truth (running): Monte-Carlo return per counterfactual branch
From each cloned state, take each action for real, then follow the arm's own eps-greedy (0.05) Q
policy for 60 decisions with a shared RNG; discounted clipped return = ground-truth value of that
action now. Score three estimators against it by Kendall tau and by regret of following their argmax:
actor Q(z,a); critic r_a + gamma max_b Q(f(x'_a)) [real successor, no model]; model r_hat + gamma
c_hat max_b Q(g(z,a)) [the planner]. Results appended when done (file: mc_truth.json).

### 14 (results). Monte-Carlo ground truth per counterfactual branch (150 cloned states per seed,
3 seeds; "decisive" = states where the four branches' 60-step returns differ at all; the arm's own
eps-greedy(0.05) Q policy is the continuation policy; shared RNG across branches)

| arm | decisive states / probed | MC spread (max-min return) | tau actor Q(z,a) | tau critic r+γV(f(x'_a)) | tau model r̂+γĉV(g(z,a)) | regret actor | regret critic | regret model |
|---|---|---|---|---|---|---|---|---|
| K=5  | 83/150 | 0.432 | +0.041 | +0.022 | **+0.050** | -0.190 | -0.212 | **-0.173** |
| K=10 | 85/150 | 0.404 | +0.044 | +0.048 | +0.052 | -0.214 | -0.159 | -0.194 |
| K=20 | 76/150 | 0.434 | +0.006 | +0.061 | **+0.082** | -0.223 | -0.202 | **-0.180** |
| K=30 | 70/150 | 0.452 | +0.020 | **+0.101** | **-0.007** | -0.240 | **-0.183** | **-0.242** |

(regret = MC return of the estimator's argmax minus the best branch's MC return; 0 is perfect; the
spread is ~0.43 so a uniformly random pick would score about -0.2 to -0.25. n ~ 210-250 decisive
states per arm; per-state tau is coarse (6 pairs), so differences below ~0.04 in tau and ~0.03 in
regret are within noise.)

Reading:
- All three estimators are weak predictors of the 60-step outcome at every K (tau <= 0.10): one
  decision's consequence is mostly drowned by stochasticity and later decisions. Regret is the more
  usable number.
- At K=5 the MODEL-based estimate (Q at the predicted successor) is the best of the three (lowest
  regret, -0.173) -- consistent with lookahead adding +21 points. It is better than the critic on the
  REAL successor (-0.212): plausibly because g(z,a) is a learned expectation over the sticky/bounce
  noise while the real successor is one sample.
- At K=30 the same Q head applied to the REAL successor is the best estimator of any arm (tau +0.101,
  regret -0.183) -- the critic is not worse at long K, if anything better -- while Q applied to the
  PREDICTED successor collapses to chance (tau -0.007, regret -0.242, no better than the actor's
  -0.240). Consistent with lookahead adding +2 points.
- Combined with section 12 (the model's ordering agrees with the critic's ordering equally at K=5 and
  K=30, tau ~0.21-0.24, and the model's cosine error on the counterfactual successors is LOWER at
  K=30): the K=30 predicted successor is closer to the real one by cosine distance yet leads the
  value head to a worse decision than the real one does. The damage is at the model -> value
  interface: what the one-step prediction gets wrong at K=30 is precisely the part Q reads, even
  though it is a small part by the training metric.
- The actor Q(z,a) is roughly equally weak at every K (tau 0.006-0.044; regret -0.19 to -0.24),
  matching the near-flat Q-controller returns (+19 -> +16).
