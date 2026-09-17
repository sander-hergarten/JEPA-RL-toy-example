# atari-jepa: a JEPA latent world model for Atari (Pong and Breakout)

A small, inspectable PyTorch agent that learns an **action-conditioned latent world model** from Atari
pixels by predicting future *representations* (not pixels), and uses that model for action selection.
The question it is built to answer: does predicting future representations produce a world model that
supports better decisions than the same network's Q-policy?

**Headline (Breakout, 2.5M decisions, 3 seeds, ε = 0.01):**

| | Q-policy | One-step lookahead |
|---|---|---|
| Model-free Double DQN baseline | +18.87 ± 1.83 | – |
| World model + delta target + motion channels | +22.03 ± 2.57 | +57.70 ± 5.20 |
| **… + 5-step Double DQN targets** | **+32.00 ± 1.67** | **+70.93 ± 17.47** |

Planning with the learned model adds **+35.7 points** over the same checkpoint's own Q-policy and
reaches ~3.1× the model-free baseline. The margin grows with budget rather than washing out:

| Budget | Model-free | World model + lookahead |
|---|---|---|
| 500k | +8.10 | +25.53 |
| 1.5M | +13.67 | +38.77 |
| 2.5M | +18.87 | **+57.70** |

Offline "learn by observing" pretraining (1.5M) reaches +20.67 fine-tuned and +1.97 frozen — see
[that section](#learn-by-observing-offline-jepa-pretraining-then-re-attach-rl).

Planning with the learned model beats the same checkpoint's Q-policy **in every seed** (+9 to +15
points) and roughly triples the model-free baseline. The decisive change was not the planner, the
architecture or the budget, but **what the temporal objective is asked to predict**: with the
originally specified "predict the next latent" target, the latent collapsed into ~4 of 3,136 directions
and the dynamics ignored the action entirely; predicting the per-step *change* fixes both. On Pong the
same question stays unanswered — seed spreads of 5–6 points swamp every difference. See
[Results](#results) for the full path, including the failures that led here, and
[Limitations](#limitations) for what 3 seeds on 2 games does not support.

This is an experimental baseline, not a reproduction of a paper, and its scores are not comparable to
published Atari 100k results (preprocessing, interaction counting, resets and evaluation protocol have
not been matched). Influences: temporal latent prediction (SPR), latent planning with reward/value heads
(MuZero, EfficientZero), and action-conditioned rollout training (V-JEPA 2).

Contents: [Setup](#setup) · [Commands](#commands) · [Environment conventions](#environment-conventions) ·
[Model and tensor contracts](#model-and-tensor-contracts) · [Replay and masks](#replay-and-masks) ·
[Losses](#training-objective) · [Offline pretraining](#offline-pretraining-learn-by-observing) ·
[Planning](#planning) · [Diagnostics](#diagnostics) ·
[Experiments](#experiments) · [Results](#results) · [Videos](#videos) ·
[Checkpoints](#checkpoints-and-resume) · [Tests](#tests) · [Limitations](#limitations) ·
[Assumptions](#assumptions-and-decisions)

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). PyTorch is selected by an extra, so the
CPU-only machine does not download CUDA wheels:

```bash
uv sync --extra cpu        # CPU-only torch (development and tests)
uv sync --extra cu130      # CUDA 13.0 wheels (the results below; needed for RTX 50xx); --extra cu126 for CUDA 12.6
source .venv/bin/activate  # the commands below assume the venv is active (or prefix with `uv run --extra cpu`)
```

**ROMs.** Nothing in this repository contains a ROM. `ale-py` (≥ 0.9) ships the Atari ROMs inside its
wheel, so `uv sync` is all that is needed. If the environment cannot be created, `atari_jepa` raises a
`SetupError` saying which import or id failed. Check with
`python -c "import gymnasium, ale_py; gymnasium.register_envs(ale_py); gymnasium.make('ALE/Pong-v5')"`.

Versions used for the results: Python 3.14.7, torch 2.14.0+cu130 on an RTX 5090 (tests also run on
2.14.0+cpu), numpy 2.5.3, Gymnasium 1.3.0, ale-py 0.12.1, opencv-python-headless 5.0. Every run records
its versions in `metadata.json` and in the checkpoint.

## Commands

```bash
# unit + integration tests (CPU, ~40 s; synthetic env/models, plus real-ALE checks when ale-py is present)
python -m pytest -q

# ROM-free synthetic smoke (toy "catch" game with the same contract) -- does NOT validate Atari
python -m atari_jepa.train --config configs/synthetic_smoke.yaml

# real-Pong smoke: 1,500 decisions, tiny batch, capped eval (~20 s on CPU). Not a learning experiment.
python -m atari_jepa.train --config configs/pong_smoke.yaml

# "learn by observing": save frames from a trained agent, pretrain the JEPA offline, re-attach RL
python -m atari_jepa.collect  --checkpoint runs/breakout_q_500k/seed1/checkpoint.pt --out data/breakout --decisions 200000
python -m atari_jepa.pretrain --dataset data/breakout --out runs/pretrain_breakout --updates 100000 --latent-loss mse --anti-collapse sigreg
python -m atari_jepa.train --config configs/breakout/breakout_world_model_500k.yaml \
    --set train.init_from=runs/pretrain_breakout/checkpoint.pt --set train.freeze_encoder=true

# the headline configuration: world model + delta target + motion channels (Breakout, 500k)
scripts/run_ablation.sh configs/breakout/breakout_world_model_500k.yaml breakout_wm_delta_motion_500k 0 2 \
    loss.jepa_target=delta network.motion_channels=true

# Breakout (500k decisions; FIRE on reset and after each lost life)
python -m atari_jepa.train --config configs/breakout/breakout_world_model_inverse_real_500k.yaml --seed 0
MODE=parallel THREADS=2 CONFIGS="configs/breakout/breakout_world_model_inverse_real_500k.yaml configs/breakout/breakout_world_model_500k.yaml configs/breakout/breakout_temporal_jepa_500k.yaml configs/breakout/breakout_q_500k.yaml" scripts/run_matrix.sh

# learning experiments (100k decisions each)
python -m atari_jepa.train --config configs/pong_q.yaml --seed 0              # A: Q baseline
python -m atari_jepa.train --config configs/pong_temporal_jepa.yaml --seed 0  # B: temporal JEPA
python -m atari_jepa.train --config configs/pong_world_model.yaml --seed 0    # C: full world model

# evaluate one checkpoint with both controllers (same parameters, same reset seeds)
python -m atari_jepa.evaluate --checkpoint runs/pong_world_model/seed0/checkpoint.pt --controller q
python -m atari_jepa.evaluate --checkpoint runs/pong_world_model/seed0/checkpoint.pt --controller lookahead
python -m atari_jepa.evaluate --checkpoint RUN/checkpoint.pt --controller lookahead --horizon 3  # optional H-step beam search

# record gameplay video: one panel per checkpoint/controller, shared reset seed
python -m atari_jepa.record --out compare.mp4 \
    --panel runs/breakout_q_500k/seed1/checkpoint.pt q "A: Q baseline" \
    --panel runs/breakout_wm_delta_motion_500k/seed0/checkpoint.pt lookahead "C+delta+motion: lookahead"

# is the latent space representative? spectrum, temporal structure, state/reward probes
python -m atari_jepa.embeddings --checkpoint runs/pong_world_model_500k/seed0/checkpoint.pt

# held-out diagnostics on a frozen checkpoint (+ optional fixed-batch optimization check)
python -m atari_jepa.diagnostics --checkpoint runs/pong_world_model/seed0/checkpoint.pt
python -m atari_jepa.diagnostics --checkpoint RUN/checkpoint.pt --fit-fixed-batch 300 --fit-components jepa

# long-range probe: a head trained directly on (x_t, x_t+delta) pairs vs the iterated one-step rollout,
# both against the conditional-mean floor. Needs replay.save_with_checkpoint.
python -m atari_jepa.delta_probe --checkpoint RUN/checkpoint.pt --deltas 1,5,10,30,100,300

# where rollout credit goes: do the K depth terms agree, conflict, or duplicate each other?
python -m atari_jepa.gradient_anatomy --checkpoint RUN/checkpoint.pt

# how action-sensitive should the latent be? counterfactual branches from one emulator state
python -m atari_jepa.action_effect --checkpoint RUN/checkpoint.pt

# resume an interrupted run (or use --auto-resume with --config)
python -m atari_jepa.train --resume runs/pong_world_model/seed0

# the whole A/B/C x seeds {0,1,2} comparison, resumable, then the aggregate report
scripts/run_matrix.sh                      # CPU: two lanes; PYTHON=... THREADS=4 to override
nvidia-cuda-mps-control -d                      # GPU box: enable MPS first -- ~3.4x faster for parallel runs
MODE=parallel THREADS=2 scripts/run_matrix.sh   # GPU box: all nine runs at once (used for the results)
MODE=parallel THREADS=2 CONFIGS="configs/extended/pong_world_model_500k.yaml configs/extended/pong_temporal_jepa_500k.yaml configs/extended/pong_q_500k.yaml" scripts/run_matrix.sh   # 500k follow-up
python -m atari_jepa.report runs           # -> runs/report.md
```

Useful flags: `--seed`, `--device auto|cpu|cuda|mps`, `--run-dir`, and `--set section.key=value` for
any config field (e.g. `--set train.total_decisions=20000`). Configs are strict: unknown keys are errors,
and `env.sticky_action_prob` has no default and must be stated.

### What a run writes

`runs/<name>/seed<k>/`:

| File | Content |
|---|---|
| `config.yaml`, `metadata.json` | resolved config; versions, env/preprocessing metadata, action meanings, device, parameter counts, command |
| `train_episodes.jsonl` | per training episode: raw and clipped return, length (decisions), emulator frames incl. reset no-ops, termination/truncation, ε |
| `updates.jsonl` | every 250 updates: mean of each loss component, per-depth JEPA/persistence/Q-TD, λ weights, grad norm, latent std stats, counters |
| `eval.jsonl` | periodic in-training evaluation (separate env, fixed reset seeds), eval decisions counted separately |
| `checkpoint.pt` (+ `replay.npz`) | see [Checkpoints](#checkpoints-and-resume) |
| `eval_q.json`, `eval_lookahead.json` | final evaluation, one record per episode |
| `diagnostics.json` | held-out diagnostics |

Console output from the smoke run looks like:

```text
[pong_smoke s0 d0] variant C_world_model on cpu; env ALE/Pong-v5 (sticky=0.0); actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']; budget 1500 decisions, warmup until 300
[pong_smoke s0 d500] update 50: loss_q=0.1128 loss_jepa=0.0490 loss_reward=0.1594 loss_continue=0.0195 loss_var=0.0745 latent_std_mean=0.0256
[pong_smoke s0 d923] episode 1: return -20, 923 decisions
[pong_smoke s0 d1500] eval [q] return -6.00 ± 0.00 (1 episodes, 300 eval decisions)
[pong_smoke s0 d1500] eval [lookahead] return -7.00 ± 0.00 (1 episodes, 300 eval decisions)
[pong_smoke s0 d1500] done: 1500 decisions, 300 updates, 1 episodes, 6027 training frames, 600 eval decisions
```

(Smoke eval episodes are capped at 300 decisions, so these returns are partial games.)

## Environment conventions

One **timestep = one agent decision** (4 emulator frames). Symbols: `o_t` processed frame
`uint8[84,84]`; `x_t` stack of the 4 most recent processed frames `uint8[4,84,84]`; `a_t` action chosen
after observing `x_t`; `r_raw_t` game reward summed over the repeat; `r_t = sign(r_raw_t)` training
reward; `c_t = 1 - terminated_t`; `x_{t+1}` the actual history after the action, before any reset.

| Setting | Value |
|---|---|
| Game | `ALE/Pong-v5` (`NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE`) or `ALE/Breakout-v5` (`NOOP, FIRE, RIGHT, LEFT`); the action set is read from the env |
| Base env frame skip | 1 (`gym.make(..., frameskip=1)`), so frames are never skipped twice |
| Action repeat | 4, in `atari_jepa.envs.AtariEnv` |
| Preprocessing | max of the last two emulated frames → ALE grayscale → `cv2.INTER_AREA` resize to 84×84 |
| History | 4 frames; an incomplete starting history repeats the episode's first frame |
| Reset no-ops | uniform in [1, 30], NOOP looked up by meaning, RNG seeded by the reset seed |
| FIRE on reset | off for Pong (it serves automatically: under NOOP-only play the opponent scores every ~35 decisions), on for Breakout (nothing happens otherwise). The helper looks up `FIRE` by meaning and refuses to guess an id |
| FIRE on life loss | `fire_on_life_loss`, on for Breakout only. Breakout needs a serve after every lost life, and a greedy policy that never fires would idle to the frame limit: measured, a NOOP-only policy runs 1,200+ decisions with no reward and no terminal, versus 120 decisions and a proper terminal with the serve enabled. The press costs one emulator frame, counted in the budget and reported per step as `fire_frames` |
| Life-loss termination | disabled (the config rejects enabling it); a lost life is not an episode boundary, only a serve |
| Sticky actions | 0.25 for experiments, 0.0 for the smoke/debug configs, always explicit |
| Truncation | ALE's 108,000-frame limit, plus optional `max_episode_decisions`; evaluation caps episodes at 27,000 decisions |
| Rewards | raw return is reported; the clipped `sign(r)` is used for training |

The wrapper is custom rather than `gymnasium.wrappers.AtariPreprocessing` because the latter
hard-codes action 0 for no-ops and, when an episode ends mid-repeat, returns a max-pool of stale frames
instead of the actual final observation. A single, non-vectorized environment with no autoreset is used.
The collector stores the true final observation before it resets.

Accounting keeps agent decisions, emulator frames (including reset no-ops), training updates and
evaluation interactions separate. The training budget counts training decisions, warmup included.
Evaluation and diagnostics interactions are reported but never added to replay.

Four frames only approximate the state, and sticky actions make transitions stochastic. The
deterministic latent predictor can only approximate either.

## Model and tensor contracts

| Module | Signature | Architecture |
|---|---|---|
| Encoder `f` | `uint8[B,4,84,84] → [B,64,7,7]` | `/255` → Conv(32,8,s4)-ReLU → Conv(64,4,s2)-ReLU → Conv(64,3,s1)-ReLU → LayerNorm over (64,7,7). No BN or dropout. With `network.motion_channels` the 3 signed frame differences are appended first, so the input is `[B,7,84,84]` |
| Dynamics `g` | `([B,64,7,7], int64[B]) → [B,64,7,7]` | 16-d action embedding broadcast over 7×7, concat → Conv3×3(80→64)-ReLU → 2 residual blocks → Conv3×3 → `LayerNorm(z + delta)` |
| Q head | `[B,64,7,7] → [B,A]` | flatten → 512 ReLU → A |
| Reward head | `(z, a) → [B,3]` | [flatten(z), one_hot(a)] → 256 ReLU → logits over rewards {−1, 0, +1} |
| Continuation head | `(z, a) → [B]` | same as the reward head, 1 logit |
| Inverse head (optional) | `(z_t, z_{t+1}) → [B,A]` | [flatten(z_t), flatten(z_{t+1})] → 256 ReLU → action logits. Built only when `loss.inverse` ≠ `none` |
| Targets | EMA copies of encoder and Q head | `θ_t ← τ θ_t + (1−τ) θ`, τ = 0.99 after every optimizer step. No target dynamics |

Planning uses `E[r] = p(+1) − p(−1)` and `P(continue) = sigmoid(logit)`. The 3-class reward head
needs sign-clipped rewards: `reward_to_class` raises if a valid reward is outside {−1, 0, +1}, and the
config rejects any other `reward_transform`. Removing clipping needs a different reward head.

Parameters (Pong, A = 6): encoder 84k (90k with motion channels), dynamics 237k, Q head 1.61M, reward
head 805k, continuation head 805k, optional inverse head 1.61M. Variant A trains encoder + Q (1.69M), B adds dynamics (1.93M), C trains everything (3.54M).
The core modules are always built, so A/B/C checkpoints share one layout. The optional inverse head
(1.61M) exists only in configs that use it, so older checkpoints load unchanged. `metadata.json` lists
which modules receive gradients.

**Precision.** `network.conv_dtype: bfloat16` runs the encoder and dynamics conv stacks under bf16
autocast (CUDA). Parameters, Adam state, LayerNorm, the MLP heads and every loss stay fp32, because
latent cosine distances (≈ 1e-2) are below bf16's resolution near 1.0 (≈ 4e-3). On the RTX 5090 at
batch 32 it is **slower** than fp32 (8.2 vs 7.2 ms per C update). The convs are too small for bf16
tensor cores to pay off, fp32 convs already use TF32, and autocast adds casts. All reported runs
therefore use fp32. Some CPU backends (oneDNN on AVX2) cannot run bf16 conv backward, and training
refuses to start there with a clear error. An update fetches all its metrics in a single
device-to-host transfer; the finiteness and reward-range checks run after that transfer (≈ 5% faster
per update).

## Replay and masks

`atari_jepa.replay.SequenceReplay` is a ring buffer of processed frames (one slot per observation,
100k slots). An episode of T decisions takes T+1 consecutive slots, the last holding the real final
observation. Every slot stores its episode id, step index and absolute write index. Stacks are rebuilt
at sample time with the same padding rule as acting, and latents are never cached. Roots are sampled
uniformly and independently over valid roots: the root has a transition, and its whole history is still
in the buffer after eviction. A sequence stops at the first terminal or truncated transition, at the end
of an unfinished episode, or at a collection boundary. It never joins episodes.

Batch (K = 5): `observations [B,K+1,4,84,84]`, `actions/rewards/terminated/truncated/valid [B,K]`.
Padded entries are zero.

* `m_k = valid_k` masks reward, continuation and Q. The terminating transition itself is valid.
* `l_k = valid_k · (1 − terminated_k)` masks next-latent prediction (no latent target at true terminals).
* Truncation has continuation target 1 and **bootstraps from the stored final observation**. Nothing
  after termination or truncation is valid.

Every loss is `masked_mean`: `where(mask, x, 0).sum() / max(count, 1)`. It is normalized by its own
count, gives exactly 0 with zero gradient when empty, and padded values cannot enter it even if they are
NaN.

## Training objective

```text
z_hat[0] = encoder(x_0);  z_hat[k+1] = dynamics(z_hat[k], a_k)          # never detached, never replaced
target_z[k] = target_encoder(x_k)                                        # no_grad, each stack encoded alone

L_jepa     = masked_mean_k( 1 - cos(flatten z_hat[k+1], flatten target_z[k+1]), l_k )
L_reward   = masked_mean_k( CE(reward_head(z_hat[k], a_k), sign(r_k)+1), m_k )
L_continue = masked_mean_k( BCEWithLogits(cont_head(z_hat[k], a_k), 1 - terminated_k), m_k )
y_k        = r_k + γ (1 - terminated_k) Q_target(target_enc(x_{k+1}))[argmax_a Q(enc(x_{k+1}))]   # no_grad, real data
L_Q        = masked_mean_k( Huber(Q(z_hat[k])[a_k], y_k), m_k )          # k = 0 only for A and B
L_var      = mean_j relu(0.1 - sqrt(Var_batch(flatten z_hat[0])_j + 1e-4))
L_inverse  = masked_mean_k( CE(inverse_head(u_k, u_{k+1}), a_k), m_k )   # optional
             # inverse=real:      u_k = encoder(x_k)  (online, real observations; u_0 = z_hat[0])
             # inverse=predicted: u_k = z_hat[k]
L = λ_Q L_Q + λ_jepa L_jepa + λ_r L_reward + λ_c L_continue + λ_var L_var [+ λ_inv L_inverse]   (all enabled λ = 1)
```

Adam (lr 1e-4, eps 1.5e-4), global grad-norm clip 10, batch 32, γ = 0.99 per decision, Huber δ = 1.
Each component, its weight and the per-depth values go to `updates.jsonl`. An optional covariance
penalty is implemented through the B×B Gram matrix (no 3136² matrix). It is off.

Gradient audit (verified in `tests/test_gradients.py` and `tests/test_inverse_and_precision.py`):

| Loss / path | Encoder | Dynamics | Its head | Targets |
|---|---|---|---|---|
| Temporal prediction | ✓ | ✓ | – | ✗ |
| Reward/continuation on imagined states | ✓ | ✓ | ✓ | ✗ |
| Q on imagined states (C) | ✓ | ✓ | ✓ | ✗ |
| Q on real root | ✓ | – | ✓ | ✗ |
| Variance floor | ✓ | ✗ | – | ✗ |
| Inverse dynamics, `real` | ✓ | ✗ | ✓ | ✗ |
| Inverse dynamics, `predicted` | ✓ | ✓ | ✓ | ✗ |
| Bootstrap targets | ✗ | ✗ | ✗ | ✗ |

**Loop.** Random actions for 5,000 warmup decisions, then ε-greedy on the online Q with ε linear
1.0 → 0.1 over the first 50k decisions, for **every** variant, planning included. One update every 4
decisions: 100k decisions in the specified protocol (23,750 updates), 500k in the follow-ups (123,750
updates). The loop stores the transition and the true next
frame, updates when due, and resets only after the final observation has been stored. It evaluates and
checkpoints on schedule in a separately seeded environment. No augmentation.

## Offline pretraining ("learn by observing")

`atari_jepa.collect` saves the frames a trained agent generates (in the trainer's own replay format, so
masks and K-step sampling are identical online and offline), and `atari_jepa.pretrain` trains the
encoder and dynamics on that fixed dataset with **no reward, no value and no environment**:

```text
L = λ_jepa · masked_mean_k( MSE(z_hat[k+1], target_z[k+1]) , l_k )  + λ_anti · anti_collapse(z_root)
```

* **MSE instead of cosine** constrains the latent's scale as well as its direction — which makes a
  constant latent an optimum, so anti-collapse must be explicit.
* **`anti_collapse` = `sigreg`** (default), `vicreg`, `variance` or `none`. SIGReg draws random unit
  directions, projects the batch onto each, scales by **one global factor** (never per direction, or the
  test would be blind to a low-rank cloud whose individual projections are still Gaussian), and averages
  the Epps–Pulley normality statistic. An isotropic Gaussian embedding is its minimum; constant, low-rank
  and anisotropic embeddings are all penalized. Following the SIGReg idea in LeJEPA
  (Balestriero & LeCun, 2025); this is a plain Epps–Pulley sketch, not their full method.
* `train.init_from` loads the pretrained encoder/dynamics into an RL run, and `train.freeze_encoder`
  keeps the encoder fixed so RL learns only on top of it (the strict observation-only test). Frozen
  parameters are excluded from the optimizer, not merely zero-grad.

See [the results](#learn-by-observing-offline-jepa-pretraining-then-re-attach-rl) for how this compares
with training the JEPA jointly with RL.

## Planning

`--controller q`: `argmax_a Q(encoder(x_t))`.

`--controller lookahead` (one step, batched over all actions, `no_grad`):
`score(a) = E[r | z, a] + γ · P(continue | z, a) · max_b Q(g(z, a))[b]`, with `z = encoder(x_t)` of the
**real** current history. The controller re-encodes after every step and never carries a predicted
latent forward. Exact ties go to the lowest action index. The evaluation ε (0 by default) is applied the
same way to both controllers. Latency and the fraction of decisions where the planner disagrees with
`argmax Q` are logged.

Optional `--horizon H` runs beam search (`planning.beam_width`, exhaustive when the width is at least
`A^(H−1)`), scoring `J = Σ_k γ^k S_k E[R] + γ^H S_H max Q(z_H)` with `S_k = Π_{j<k} P(continue)`. It
executes the first action and replans. This expected-return score does not integrate over alternative
latent outcomes. Horizons above the training K = 5 are flagged as extrapolative.

## Diagnostics

`python -m atari_jepa.diagnostics` collects **new** trajectories: 4 episodes, reset seeds 20000+,
ε = 0.1 greedy-Q policy, never training replay. It evaluates every valid root with the frozen
checkpoint and its target encoder:

* latent cosine distance at depths 1, 3, 5, and 10 (10 is extrapolative for K = 5), against a
  **persistence** baseline (predict `z_hat[0]` at every depth) and against **shuffled** (batch-permuted)
  and **random** action sequences. Each is reported raw and **centered** (the mean held-out target
  latent subtracted). In Pong most of the latent is a static background, so raw distances are about
  0.01 for any pair of frames and cannot separate the model from persistence. The centered distance can.
* root-latent per-dimension std quantiles, the fraction below the variance floor, the fraction below
  1e-3, mean pairwise cosine, and input pixel variation (low input variation is reported, not treated as
  collapse),
* reward per depth: CE against the constant class-prior CE, per-class counts, and +1/−1 event
  precision/recall (accuracy is omitted because zero-reward steps dominate it),
* continuation per depth: BCE, Brier, prior BCE, terminal precision/recall and the terminal count,
* Q TD error (Huber and |TD|) on real roots and at each imagined depth,
* action sensitivity at a fixed state (pairwise next-latent distance, range of E[r] and P(continue)),
* a **movement probe**: an evaluation-only linear classifier for the horizontal movement class of
  `a_t` (none / RIGHT* / LEFT*, from the action meanings), trained on frozen online latents
  `(z_t, z_{t+1} − z_t)` from half of the held-out episodes and tested on the other half. It asks
  whether the encoder represents what the agent controls, and is reported against majority and chance
  balanced accuracy,
* for checkpoints with an inverse head, its held-out action and movement accuracy on real pairs
  `(f(x_t), f(x_{t+1}))` and on predicted pairs `(z_hat[0], z_hat[1])`,
* **gradient reach**: `‖∂L_K/∂z_hat[k]‖` at every rollout step for the *deepest* latent term alone,
  normalized by its value at step K. "Long rollouts must be losing gradient" is the intuitive
  explanation for the rollout-length results and it is checkable in one backward pass — here it is
  false. The dynamics is residual (`z' = LayerNorm(z + delta)`), whose Jacobian is near the identity,
  so credit does not attenuate: at K = 30 the depth-30 term reaches the root **5.4× stronger** than at
  the step that produced it (conv 5.42×, LMU 4.78×, and 2.3–5.7× across every K tested). The encoder is
  not starved of long-horizon signal, it is saturated with it,
* controller latency, and the held-out episodes' returns, decisions and frames.

`--fit-fixed-batch N [--fit-components ...]` optimizes a *copy* of the model on one held-out batch with
fixed targets. It checks the optimization path only.

### Is the latent space representative? (`atari_jepa.embeddings`)

```bash
python -m atari_jepa.embeddings --checkpoint runs/pong_world_model_500k/seed0/checkpoint.pt
python -m atari_jepa.embeddings --checkpoint runs/*/seed*/checkpoint.pt      # compare several
```

Low prediction loss does not make a representation good: a slow, nearly constant latent predicts itself
well. This read-only pass collects fresh episodes and measures four things on the frozen encoder,
writing `embeddings.json` next to the checkpoint:

* **Spectrum** of the root latents: participation ratio, entropy-based effective rank, variance in the
  leading components, dimensions holding 90/99% of the variance. This catches a low-rank latent that
  still passes the per-dimension variance floor.
* **Temporal structure**: centered cosine distance between latents `t` and `t + gap` within an episode,
  against pairs from different episodes.
* **State decodability** (evaluation-only): ridge from the latent to the emulator state (ALE RAM, or the
  toy game's true state), fit on half the held-out episodes and scored by R² on the others. Features are
  projected onto the training half's leading principal components first, since the latent has more
  dimensions than the probe has samples. The emulator state is never an agent input.
* **Task relevance**: linear probes for "a reward event occurs within the next 8 decisions" (AUC against
  its base rate) and for the movement class of the action.

RAM byte indices for the named Pong variables come from published annotations and are not verified
here. The Breakout ones (`player_x` 72, `ball_x` 99, `ball_y` 101) were verified here by correlating
every byte with pixel measurements over 3,000 random-policy frames (|r| = 0.94, 0.97, 0.79). The
unlabelled per-byte summary does not depend on any of these labels. Score-counter bytes can leave their
training range in the test episodes, so the *mean* R² over bytes is dragged negative by extrapolation
and the median is the number to read.

## Experiments

| Variant | Config | Training losses | Controller |
|---|---|---|---|
| A: Q baseline | `pong_q.yaml` | root Double DQN | Q |
| B: temporal representation | `pong_temporal_jepa.yaml` | root Q + K-step JEPA + variance floor | Q |
| C: full world model | `pong_world_model.yaml` | all of the above + reward + continuation + imagined-state Q | Q |
| C + planning | same checkpoint as C | – | one-step lookahead |

Later sections add three more, each a single switch on top of C (all tested individually first):

| Variant | Switch | Motivation |
|---|---|---|
| C + inverse `real` / `predicted` | `loss.inverse` | the dynamics were ignoring the action |
| C + motion channels | `network.motion_channels` | the ball was the worst-represented variable |
| **C + delta target** | `loss.jepa_target=delta` | the latent collapsed onto the static background |

All variants share the environment protocol (sticky 0.25), seeds, 100k-decision budget, encoder/Q
architecture and initialization (same seed gives the same initial encoder and Q weights), replay sampling,
exploration schedule and update schedule. B and C cost more compute per update (see the Results).
Evaluation: 10 episodes per training seed, reset seeds 10000–10009 for every checkpoint and controller.
Policy-dependent trajectories diverge even with matched seeds. Results are reported per training seed.

A second game, **Breakout**, runs the same comparison plus the grounded-inverse-dynamics variant
(`configs/breakout/*_500k.yaml`, 3 seeds, 500k decisions, same protocol). It is a harder
controllability test than Pong: the paddle is the only thing the agent moves, rewards are sparser and
come in 1/4/7 sizes (all clipped to +1), and the game needs an explicit serve.

Comparisons: A vs B asks whether temporal prediction helps Q-learning's representation. B vs C asks what
reward, continuation and imagined-state Q supervision add. C-Q vs C-lookahead asks whether using the
learned transitions helps at inference time. Planner-driven data collection is not implemented. It would
change the training distribution and is a separate experiment.

## Results

All numbers below come from runs that actually completed on an RTX 5090 machine (torch 2.14.0+cu130,
Gymnasium 1.3.0, ale-py 0.12.1, 24-core CPU). The nine runs trained concurrently on one GPU
(`MODE=parallel THREADS=2 scripts/run_matrix.sh`), taking about 13–16 min each. A lightweight snapshot of
every run is committed in `results/<experiment>/seed<k>/`: config, metadata, training/update/eval logs,
final evaluations and diagnostics. Checkpoints and replay buffers are left out. The generated tables
are in `results/report.md` (`python -m atari_jepa.report runs` rebuilds them from a local `runs/`).

### Summary of findings

The sections below are in the order the experiments were run, because each one was motivated by a
diagnostic failure in the previous one. In short:

| # | Experiment | Outcome |
|---|---|---|
| 1 | Specified protocol: A/B/C, Pong, 100k | Nothing learned; no variant separable. Diagnostics show the dynamics are **action-invariant** |
| 2 | Extended budget, Pong, 500k | All variants learn; B best (−12.9), C worst (−18.8). Planning helps C by +1 to +3.5 |
| 3 | Inverse dynamics (grounded), Pong 500k | Action sensitivity up 60–200×, rank 13 → 140–609; best average play but one seed collapses |
| 4 | Breakout, 500k | Ordering reverses: model-free wins (+8.1); B's latent collapses to **rank 3.9** |
| 5 | Ball sensitivity: delta target and motion channels, 100k, both games | Delta target fixes collapse and action-blindness at 1/5 the budget |
| 6 | Confirmation, 500k, both games | **Breakout: +25.5 with lookahead, ~3× model-free, every seed.** Pong stays inconclusive |
| 9 | Sample-efficiency matrix at a fixed 500k budget, then follow-ups | **n-step is worth ~3× the samples** (any n ≥ 3); the replay-ratio failure was the *buffer size*, and n-step + rr 0.5 + a 300k buffer gives **+47.9 at 500k**; planner-driven collection hurts even when switched in late |
| 10 | 2.5M with n-step | **+70.9 with lookahead**, 3.8× model-free |
| 11 | Planning depth sweep on the 2.5M n-step checkpoints | Depth pays to **+91.7 at H = 5**, then saturates (the apparent drop at H = 10 was seed noise; see the K sweep) |
| 12 | H-JEPA level 2 (jumpy dynamics) vs. multi-step search, 500k | Joint training makes the encoder **action-blind** (shuffled-action penalty 48% → 2%) and costs everything; detached it is harmless but its jumpy planner (+29.2) still loses to beam search over level 1 (+57.3) |
| 13 | Rollout horizon K ∈ {5, 10, 20, 30}, 500k | **A strictly better model that plays strictly worse**: K = 30 has the lowest latent error and the best movement probe and scores 36.9 against K = 5's 65.1. Capping the value horizon (`q_imagined_depth`) does not rescue it |
| 14 | Δ-probe: directly-supervised jump vs. iterated rollout on frozen latents | Long range is **not** a training-coverage failure — a head trained at Δ = 100 cannot beat a constant. Long rollouts do extend the model's horizon, by making the representation temporally smooth |
| 15 | Successor features (γ-model), 500k | Inert where it must be, useless as a value function (**+2.0**): ψ is identical across actions (pairwise cosine **1.0**), so its argmax is noise |
| 16 | Legendre Memory Unit dynamics, K ∈ {5,10,20,30} | Matches the conv core everywhere and does not change the K slope. Its premise — gradient decay — is false: a residual chain delivers credit to the root **5.4× stronger** than at the deepest step |
| 17 | Gradient anatomy and the counterfactual action reference | Depth terms **conflict** rather than duplicate (0% opposing pairs at K ≤ 10, 8.4% at K = 30). And the long-rollout arms are **not** action-blind: K = 30 captures 78% of the real action effect, against K = 5's 66% |
| 8 | Long runs: 1.5M (4 variants) and 2.5M (top 2), Breakout | Joint delta+motion reaches +38.8 at 1.5M and **+57.7 with lookahead at 2.5M** (3.1× model-free); offline fine-tuned +20.7; frozen stays flat at +2 |
| 7 | Offline "learn by observing": RL → frames → JEPA (MSE + SIGReg) → re-attach RL | SIGReg ends collapse (rank 100–198, no tuning). Frozen features never support control (Pong ≈ random); as an *initialization* with the matched delta+motion recipe it gives the best Pong result here, but stays far behind joint training on Breakout |

What held up across both games:

* **Action sensitivity is only readable against a ceiling.** `atari_jepa.action_effect` measures how
  much the choice of action really changes the next observation, by branching from one cloned emulator
  state. Against it, the live arms capture 66–82% of the real effect and the two genuinely broken ones
  capture 5–8%. Several claims in this file originally read a small absolute number as blindness; the
  corrections are marked where they occur.
* The originally specified objective ("predict the next latent", cosine on the whole map) **compresses
  the latent into very few directions** — effective rank 13–20 on Pong, 4–5 on Breakout out of 3,136 —
  and leaves the dynamics **action-blind**: shuffling the action sequence changed prediction error by
  ~0%. The per-dimension variance floor from the brief does not detect either failure; the spectrum and
  the shuffled-action diagnostic do.
* Two independent fixes work, and **they are not complementary** — stacking them is much worse than
  either alone. Predicting the per-step change (`delta`) is the stronger of the two and the only change
  that produced a model worth planning with.
* Low prediction loss never implied a useful model. C predicted rewards nearly perfectly (CE 0.004 vs a
  0.12 prior) while its planner was choosing between near-identical scores.

### Specified experiment: 100k decisions, sticky actions 0.25, seeds 0/1/2

Each run: 100,000 training decisions (≈ 401.6k emulator frames including reset no-ops), 5,000 random
warmup decisions and 23,750 updates. The in-training evaluations add another 9–21k evaluation decisions
per run, reported separately. The final evaluation uses the final checkpoint, 10 episodes, reset seeds
10000–10009, ε = 0.

| Variant | Controller | Per training seed: mean ± std over 10 episodes | Across seeds (mean ± std of seed means) |
|---|---|---|---|
| A: Q baseline | Q | s0 −19.9 ± 1.0 · s1 −21.0 ± 0.0 · s2 −20.8 ± 0.4 | −20.57 ± 0.48 |
| B: temporal JEPA | Q | s0 −20.7 ± 0.6 · s1 −19.7 ± 0.8 · s2 −21.0 ± 0.0 | −20.47 ± 0.56 |
| C: world model | Q | s0 −21.0 ± 0.0 · s1 −20.7 ± 0.5 · s2 −20.7 ± 0.5 | −20.80 ± 0.14 |
| C: world model | one-step lookahead (same checkpoints) | s0 −21.0 ± 0.0 · s1 −21.0 ± 0.0 · s2 −20.0 ± 0.9 | −20.67 ± 0.47 |

**No variant learned to play Pong within 100k decisions, and no difference between variants or
controllers is detectable** (−21 is the minimum score). Training-episode returns do not trend either:
over the last 10 training episodes they range from −19.7 to −20.9. This budget answers none of the three
comparison questions through returns. The diagnostics still separate the variants
(`results/*/seed*/diagnostics.json`, held-out episodes with reset seeds 20000+, 3.0k–5.8k roots per run):

| | A: Q baseline | B: temporal JEPA | C: world model |
|---|---|---|---|
| Centered latent distance, depth 1 (pred / persistence) | 0.35–0.40 / 0.18–0.29 (dynamics untrained) | 0.04 / 0.04, 0.06 / 0.07, **0.19 / 0.08** | 0.06 / 0.09, 0.09 / 0.11, 0.03 / 0.05 |
| Depth 5 (pred / persistence) | 0.86–0.98 / 0.67–0.97 | 0.09 / 0.15, 0.14 / 0.26, 0.31 / 0.34 | **0.17 / 0.44, 0.20 / 0.46, 0.09 / 0.27** |
| Depth 10, extrapolative (pred / persistence) | ≈ 1 / 0.87–1.06 | 0.14 / 0.22, 0.23 / 0.39, 0.40 / 0.56 | 0.29 / 0.67, 0.30 / 0.67, 0.18 / 0.42 |
| Root-latent dims below variance floor | 0.79–0.84 | 0.00, 0.00, 0.46 | 0.00 (all seeds) |
| Mean pairwise root cosine | 0.55–0.73 | 0.56, 0.56, 0.94 | 0.59–0.78 |
| Reward CE at the real root (class-prior CE) | untrained | untrained | **0.004–0.005 (0.12)**; −1 recall 0.98–1.00 (84 events); +1: 0–3 events |
| Reward CE at imagined depth 4 | – | – | 0.006–0.020, −1 recall 0.96–0.99 |
| Terminal prediction (4 held-out terminals) | – | – | precision 1.0 / recall 1.0, all seeds |
| Q TD Huber, real root / imagined depth 4 | 0.009–0.018 / 0.54–0.82 | 0.003–0.009 / 0.05–0.15 | 0.002–0.003 / 0.006–0.008 |
| **Next-latent distance between actions at a fixed state** | 0.017–0.022 (untrained) | 0.0001–0.0013 | **0.0002–0.0003** |
| **Depth-1 error with shuffled / random actions** (C) | – | – | 0.0641 / 0.0643 vs 0.0640 true (s0); equal to 3 decimals for all seeds |

What this shows:

1. **A vs B.** Temporal prediction plus the variance floor changes the representation a lot. In seeds
   0 and 1 essentially no dimensions sit below the floor (A: ~80%), and the dynamics beat persistence
   from depth 3 on. It does not change returns at this budget. Seed 2 of B is a warning case: its root latents
   are nearly collinear (pairwise cosine 0.94), 46% of dimensions sit below the floor, and its depth-1
   prediction is *worse* than persistence. That is a partial collapse, and it happened even with the
   EMA target and the variance penalty.
2. **B vs C.** Reward, continuation and imagined-state Q supervision make the world model consistent
   across seeds: C beats persistence at every depth in every seed, including the extrapolative depth 10.
   Its reward and terminal heads are accurate (the −1 events are the opponent scoring, which is
   predictable from the ball), and imagined-state Q errors are 10–20× smaller than B's (untrained)
   imagined values. Returns do not change.
3. **C vs C + lookahead.** One-step lookahead disagrees with argmax Q on **83%** of decisions and makes
   no difference to returns. The diagnostics explain why: **the learned dynamics are
   action-invariant**. Next latents for the 6 actions differ by a cosine distance of about 0.0003,
   100–300× below the one-step prediction error. Shuffling or randomizing the action sequence leaves
   prediction error unchanged to 3 decimals. The spread of E[r] across actions is 0.0004 and of
   P(continue) is 0. Lookahead scores therefore differ by amounts near numerical noise, and the planner's
   choice is effectively arbitrary. The model is a good *passive* predictor of ball and opponent
   motion, not an action-conditioned model of what the agent controls. Low latent loss plus accurate
   reward prediction did **not** give a model that supports decisions. This is the failure mode the
   brief warned about.

Latency (measured while other runs shared the GPU, so only indicative): Q-policy ≈ 0.3 ms per
decision, one-step lookahead ≈ 1.3 ms (C runs).

### Sanity check: can this training loop learn at all?

The ROM-free synthetic catch game (`results/sanity/`, 20k decisions, same code path, 10-episode evals,
optimal return +5, random ≈ −4) was trained with A-style and C-style losses. The Q baseline improves
from −4.0 to **+3.8 ± 2.1**, so the RL loop, replay and targets do learn. The C-style model reaches
−2.0 (Q) and −0.2 (lookahead) at 20k. That is slower, a single seed, and not evidence either way about
Pong.

The same check rules out a plumbing bug behind Pong's action-invariant dynamics. On catch, where the
agent's paddle is a large part of the frame, the C-style model's diagnostics show strongly
action-conditioned predictions. Centered depth-5 distance is 0.16 with the true actions, 0.31 with
shuffled and 0.38 with random actions, and next latents across actions differ by 0.031 (about 100× the
Pong value). The action path works when the action's effect is large in latent space. In Pong, with this
objective and budget, it is not learned.

### Follow-up (not part of the specified protocol): 500k decisions

No variant learned anything within 100k decisions, so the same three configurations were re-run with
5× the budget (`configs/extended/*_500k.yaml`). Everything else is identical: sticky 0.25, seeds 0/1/2,
100k-frame replay, ε schedule reaching 0.1 at 50k, 1 update per 4 decisions (123,750 updates),
≈ 2.006M training frames, and the same final evaluation protocol (10 episodes, reset seeds 10000–10009).
Wall time was ≈ 1.2–1.3 h per run with all nine sharing the GPU. This answers a different question from
the specified experiment: it is a longer budget, not a matched comparison with published results.

| Variant | Controller | s0 | s1 | s2 | Across seeds |
|---|---|---|---|---|---|
| A: Q baseline | Q | −16.0 ± 2.4 | −11.6 ± 2.9 | −16.2 ± 2.2 | −14.60 ± 2.12 |
| B: temporal JEPA | Q | −11.7 ± 2.7 | −11.5 ± 3.2 | −15.4 ± 2.5 | **−12.87 ± 1.79** |
| C: world model | Q | −18.7 ± 2.1 | −19.4 ± 1.3 | −18.3 ± 3.1 | −18.80 ± 0.45 |
| C: world model | one-step lookahead | −17.5 ± 1.4 | −18.4 ± 2.6 | −14.8 ± 3.6 | −16.90 ± 1.53 |

All variants now learn: training returns climb from about −20.6 at 100k to −13 to −19 over the last 20
training episodes. What the three comparisons show at this budget:

1. **A vs B: B is at least as good as A in every seed pairing** (+4.3, +0.1, +0.8 points), −12.9
   against −14.6 overall. The gap is within the spread between seeds, so three seeds do not establish
   it. Diagnostics: B's latents stay well spread (no dimensions below the floor), while A's partially
   degenerate (43–59% of dimensions below the floor). B's dynamics beat persistence from depth 3 on in
   every seed.
2. **B vs C: adding reward, continuation and imagined-state Q supervision made the Q-policy worse.**
   C's Q-policy is the weakest in every seed (−18.8, against −12.9 for B), and its training returns
   lag as well. This is the clearest negative result of the study. These runs cannot say which added
   loss is responsible. The natural next ablation is C without imagined-state Q, i.e. B plus reward
   and continuation.
3. **C vs C + lookahead: planning helps the same checkpoint in all three seeds, by a small amount.**
   Paired over identical reset seeds, lookahead minus Q is +1.2 (SE 0.9), +1.0 (SE 0.9) and +3.5
   (SE 2.0) points, with 17 wins, 4 ties and 9 losses over 30 episode pairs. Lookahead episodes last
   longer (1,990–2,894 vs 1,404–1,827 decisions). The sign is consistent, but no single seed is
   individually significant, and planning does not lift C above the model-free variants (−16.9 vs
   −14.6 / −12.9). It partly compensates for C's weaker Q-head.

The diagnostics show what changed between 100k and 500k. C's dynamics have started to use the action:
shuffled actions now raise the depth-5 prediction error by about 10% (centered 0.067 → 0.075, 0.071 →
0.076, 0.082 → 0.089), and next latents across actions differ by 0.0004–0.0011, up from 0.0002–0.0003.
The reward and continuation heads are still action-blind (E[r] spread 0.0002–0.0004 across actions,
P(continue) spread 0). So the lookahead gain comes from `max_b Q(g(z, a))` evaluated on
action-dependent predicted states, not from predicted rewards. C still predicts rewards very well at the
real root (CE 0.001–0.006 against a prior of 0.08–0.09; −1 recall 0.94–0.99) and terminals well
(precision 1.0, recall 0.75–1.0 on 4 events). Predicting well once more did not yield the best policy.

These are 3-seed results with 10 evaluation episodes each, from a single game, budget and set of
untuned prototype hyperparameters. Treat them as directions for the next experiment, not as
conclusions.

### Follow-up: inverse dynamics, to make the dynamics use the action (500k)

The 500k diagnostics said C's dynamics were action-invariant, so two forms of an auxiliary
inverse-dynamics loss were added (`configs/extended/pong_world_model_inverse_{real,pred}_500k.yaml`,
identical to the C-500k control except `loss.inverse`):

* **`real`** predicts `a_k` from online encodings of *real* consecutive observations
  `(f(x_k), f(x_{k+1}))`, so the *encoder* must keep what the agent controls.
* **`predicted`** predicts `a_k` from `(z_hat[k], z_hat[k+1])`, so the *dynamics* must make the action
  recoverable from its own output.

| Variant (500k) | Q-policy | One-step lookahead | Paired lookahead − Q, per seed |
|---|---|---|---|
| C (control) | −18.80 ± 0.45 | −16.90 ± 1.53 | +1.2, +1.0, +3.5 |
| C + inverse `predicted` | −18.10 ± 0.93 | **−14.60 ± 0.22** | **+5.0, +2.8, +2.7** (10/0/0, 8/0/2, 7/1/2 win/tie/loss) |
| C + inverse `real` | −13.83 ± 5.27 | −12.33 ± 6.63 | +0.0, +0.9, +3.6 |
| (A / B, no model) | −14.60 / −12.87 | – | – |

Per seed, `real` gives −21.0, −12.0, −8.5 (Q) and −21.0, −11.1, **−4.9** (lookahead). Its seed 2 with
lookahead is the best result in this repository, and its seed 0 is the worst: it reached −14 at 100k,
then decayed to −21 by 300k and stayed there. In that collapsed run the two controllers return
*identical* episodes while disagreeing on 53% of decisions, which happens when the policy never moves
the paddle (NOOP and FIRE differ as actions but not in effect). So `real` is much stronger on average
and much less stable; three seeds cannot separate "better" from "higher variance".

**The mechanism worked, and the diagnostics show it directly:**

| 500k diagnostic | C | + inverse `real` | + inverse `predicted` |
|---|---|---|---|
| Next-latent distance between actions | 0.0004–0.0011 | **0.043–0.087** | 0.002–0.005 |
| Depth-5 error penalty for shuffled actions | +7–13% | **+13%, +46%, +40%** | +4–9% |
| Inverse head accuracy, real pairs (chance 0.17) | – | 0.43–0.50 | 0.15–0.29 |
| Inverse head accuracy, predicted pairs | – | 0.53–0.56 | **0.96–1.00** |
| Latent effective rank (of 3,136) | 13 | **140, 609, 598** | 19, 21, 10 |
| Centered distance Δ1 | 0.05 | 0.35–0.74 | 0.04–0.06 |
| Reward-within-8 probe AUC | 0.96 | 0.79–0.91 | 0.94 |
| RAM R²: paddle / ball y / ball x | 0.94 / 0.89 / 0.56 | 0.91 / 0.85 / 0.41 | 0.93 / 0.87 / 0.51 |

`real` raises action sensitivity by 60–200×, and it undoes the over-compression: effective rank goes
from 13 to 140–609 of 3,136 dimensions, and consecutive latents are no longer nearly identical
(Δ1 0.05 → 0.35–0.74). The emulator state stays decodable, but the representation is now far less
smooth and its reward probe is weaker, which is the cost of dropping the "predict a slow background"
solution. The two high-rank seeds are the two that played well.

`predicted` does exactly what it was predicted to do: it solves its own objective perfectly (accuracy
0.96–1.00 on predicted pairs) while barely changing the encoder, the rank, or the real-pair accuracy
(0.15–0.29). The action is written into the prediction rather than grounded in observation. Yet it
produces the **largest and most consistent planning gain** in the project (+2.7 to +5.0, winning 25 of
30 paired episodes). That is worth stating carefully: with the action recoverable from `g(z, a)`,
`max_b Q(g(z, a))` becomes an action-dependent value that was trained by the imagined-state Q loss, so
the lookahead score is closer to a second, differently-trained action-value head than to foresight
about the future. It improves decisions without improving the world model.

**Cross-game summary — for the absolute (originally specified) target only.** The delta target
overturns the last sentence of this paragraph; see
[Confirmation at 500k](#confirmation-at-500k-the-delta-target-changes-the-conclusion-breakout) below. On Pong,
temporal prediction helped a little (B best, −12.9) and the world model hurt (−18.8); on Breakout the
ordering reverses and the plain baseline wins by a wide margin (+8.1 against +3.2 for B). What *is* consistent across both games is the mechanism: the temporal
objective compresses the latent into very few directions (effective rank 13–20 on Pong, 4–5 on
Breakout), the dynamics ignore the action, and a grounded inverse-dynamics term fixes both
(rank 96–600, action sensitivity up 60–1000×) while improving the model-based variant's play in both
games. With the absolute target, no model-based configuration beat the model-free baseline on either
game at this budget — the delta target later did, by 3× on Breakout.

**Answering the original question.** With the default absolute target, the answer was "the model
predicts well but does not help decisions". With the delta target on Breakout it becomes a clear yes:
the same checkpoint plays at +13.6 with its Q-policy and +25.5 with one-step lookahead, three times the
model-free baseline, and the gain holds in every seed. The deciding factor was not the planner, the
architecture or the budget, but *what the temporal objective is asked to predict*. On Pong the question
remains unanswered at this budget and seed count.

**The original Pong-only reading, kept for the record:** predicting future representations did
produce a model that supports better decisions than its own Q-policy (all three variants show a
positive lookahead gain, largest with `predicted`), but no model-based configuration beat the plain
model-free baselines at this budget (best model variant −12.3 versus B's −12.9 and A's −14.6, well
within seed spread). Grounded action-conditioning (`real`) helped play the most and helped the
representation the most, at the cost of stability.

### Second game: Breakout (500k decisions, 3 seeds, same protocol)

Breakout serves with FIRE (at reset and after each lost life, see the environment table), scores 1/4/7
per brick (all clipped to +1), and gives the agent exactly one thing to control. Because a
deterministic policy can fall into a bounce loop that never ends, evaluation is reported at ε = 0.01
as well as ε = 0; at ε = 0 several evaluations hit the 27,000-decision cap.

| Variant (Breakout 500k) | Q-policy, ε = 0.01 | Lookahead, ε = 0.01 | Q-policy, ε = 0 |
|---|---|---|---|
| A: Q baseline | **+8.10 ± 0.99** | – | +7.43 ± 0.92 |
| B: temporal JEPA | +3.23 ± 1.20 | – | +2.33 ± 0.66 |
| C: world model | +3.60 ± 0.43 | +4.17 ± 1.08 | +2.47 ± 1.03 |
| C + inverse `real` | +5.17 ± 0.12 | +4.57 ± 0.19 | +4.00 ± 1.10 |

(Everything in this section uses the absolute target. With the delta target the ordering changes
completely — see [Confirmation at 500k](#confirmation-at-500k-the-delta-target-changes-the-conclusion-breakout).)

**With the absolute target, the Breakout model-free baseline wins clearly, and temporal prediction
actively hurts.** A scores
more than twice B. This is the opposite ordering from Pong (where B was best and A worst), from the same
code and protocol, which is a useful reminder of how little a 3-seed, single-game result supports.

The embeddings show why, and it is the sharpest instance of the compression failure in this repository:

| Breakout diagnostic | A | B | C | C + inverse `real` |
|---|---|---|---|---|
| Latent effective rank (of 3,136) | 35 | **3.9** | 5.2 | **96** |
| Dimensions for 90% of variance | 61 | **4** | 6 | 169 |
| Median R² over varying RAM bytes | 0.05 | **−4.9** | 0.30 | 0.03 |
| R²: paddle x / ball x | 0.08 / 0.27 | **−5.2 / −44** | 0.83 / 0.41 | 0.45 / 0.45 |
| Centered distance Δ1 | 0.089 | 0.012 | 0.011 | 0.308 |
| Reward-within-8 probe AUC | 0.92 | 0.85 | 0.98 | 0.79 |
| Next-latent distance between actions | – | – | 0.0001 | **0.08–0.16** |
| Depth-5 error penalty for shuffled actions | – | – | +2–8% | **+72–100%** |
| Inverse head accuracy, real pairs (chance 0.25) | – | – | – | 0.66–0.78 |

**B collapses almost completely on Breakout**: 90% of its latent variance sits in *four* directions,
consecutive latents differ by 0.012, and the emulator state is no longer decodable at all (the negative
R² values mean the probe does worse than predicting the training mean). Breakout's screen is mostly a
static brick wall with one small ball, so "predict your own slow features" is an easy and nearly
useless solution. C resists this better because reward and continuation supervision force it to keep
score- and event-related information (reward AUC 0.98, paddle R² 0.83), and it plays slightly better
than B.

**The grounded inverse-dynamics loss again repairs exactly what the diagnostics flagged**, and more
strongly than on Pong: the effective rank goes from 5 to 96, action sensitivity of the dynamics rises
by roughly 1,000× (0.0001 → 0.08–0.16), shuffled actions now cost 72–100% extra prediction error
(control: 2–8%), and its inverse head reaches 0.66–0.78 accuracy on real pairs against a 0.25 chance
level. It is also the best model-based variant (+5.17 against C's +3.60), with much less seed spread
than on Pong (±0.12). It still does not reach the model-free baseline, and its state decodability and
reward probe are *worse* than C's, so the extra rank is not all task-relevant.

Planning helps C (+0.57 over its own Q-policy) but not C + inverse `real` (−0.60), so the one-step
lookahead gain is not consistent across games either.

### Making the encoder and objective sensitive to the ball (100k ablations, both games)

The probes said the ball is the worst-represented variable in every run, so two changes were added and
tested **separately before being combined** (`configs/ball/*.yaml`, `scripts/run_ablation.sh`, variant C,
100k decisions, 3 seeds, both games). Measurement first: the ball is *not* lost in preprocessing (in
Pong it is ~41 px at 210x160 with intensity 110, and its region still peaks at ~112/255 after the 84x84
resize), so this is about what the model keeps, not what the input has.

* **`network.motion_channels`**: append the 3 signed differences of consecutive frames in the stack to
  the encoder input (4 -> 7 channels). The ball is the only fast small mover, so differencing removes
  the static background from the *input*.
* **`loss.jepa_target`**: what the temporal loss compares. `absolute` (the latents, the default),
  `batch_centered` (minus the batch-mean target latent), or `delta` (the per-step *change*,
  D(z_hat[k+1] - z_hat[k], target_z[k+1] - target_z[k])), which removes the static background from the
  *target* so only what moves has to be predicted.

| Arm | Pong ball_x / ball_y R² | Pong rank | Pong Q / lookahead | Breakout ball_x / ball_y R² | Breakout rank | Breakout Q / lookahead |
|---|---|---|---|---|---|---|
| base (C) | 0.27 / 0.88 | 13 | −20.7 / −20.6 | −2.04 / −3.52 | 4 | 2.1 / 2.0 |
| motion only | 0.53 / 0.94 | 13 | −20.8 / −20.6 | −3.01 / −9.79 | 4 | 2.7 / 2.6 |
| centered only | −13.8 / −3.2 | 12 | −20.4 / −19.2 | −0.10 / −0.65 | 5 | 2.8 / 2.6 |
| delta only | 0.34 / 0.80 | 130 | −18.9 / **−15.8** | 0.77 / 0.62 | 98 | 9.0 / 10.5 |
| **delta + motion** | 0.48 / 0.85 | 218 | **−18.0** / −18.2 | **0.84 / 0.73** | 165 | **10.2 / 13.7** |

(Negative R² means the probe does worse than predicting the training mean, i.e. the variable is not
linearly decodable. These runs cap evaluation episodes at 5,000 decisions, so their returns are
conservative relative to the 500k tables above.)

**The delta target is the single most effective change made in this project.** It fixes every failure
the diagnostics had identified, at one fifth of the budget of the 500k runs:

* the latent stops collapsing (Breakout effective rank 4 -> 98, Pong 13 -> 130),
* the ball becomes decodable on Breakout (R² −2.0 -> 0.77) where it previously was not at all,
* the dynamics finally use the action *without* an inverse-dynamics term: shuffling actions costs
  125% extra prediction error on Breakout and 34% on Pong, versus ~0% for the baseline,
* returns improve from +2.1 to +9.0 (Breakout Q) and from −20.6 to −15.8 (Pong lookahead) at 100k
  decisions. For scale, plain C needed 500k decisions to reach −18.8 / −16.9 on Pong, and the best
  Breakout variant at 500k *at that point* (the model-free baseline) scored +8.1.

**Motion channels help the representation but not always the policy.** They consistently improve ball
decodability (Pong 0.27 -> 0.53 alone, 0.34 -> 0.48 on top of delta; Breakout 0.77 -> 0.84), and
combined with delta they give the best Breakout result here (+13.7 with lookahead, per-seed
17.9/10.7/12.6). On Pong they do not help alone, and the combination is worse for planning than delta
alone (−18.2 vs −15.8). So: take both on Breakout, take delta alone on Pong.

**The batch-centered variant is not worth keeping.** It barely changes the rank or the shuffled-action
penalty, and its Pong probes are wildly unstable (R² −13.8 ± 20). Centering removes a *global* mean;
the delta target removes the static component *per sample*, which is what actually matters.

These are 100k-decision, 3-seed results and the arms were not re-tuned (λ_jepa is still 1.0 against a
target whose loss is now ~10x larger). The obvious next run is delta (+ motion on Breakout) at 500k
against the same baselines.

### Confirmation at 500k: the delta target changes the conclusion (Breakout)

The delta target and motion channels were re-run at the full 500k budget, using the *same* config bases
as the earlier 500k runs so the numbers are directly comparable, plus a combination with the grounded
inverse-dynamics loss. (These runs were interrupted at 50k to enable CUDA MPS and resumed from
checkpoint; the collector restarts on resume, identically for all arms.)

**Breakout, 500k, 3 seeds, evaluation ε = 0.01:**

| Variant | Q-policy | One-step lookahead | Lookahead − Q, per seed | ball_x R² | rank | shuffled-action penalty |
|---|---|---|---|---|---|---|
| A: Q baseline | +8.10 ± 0.99 | – | – | 0.27 | 35 | ~0% |
| B: temporal JEPA | +3.23 ± 1.20 | – | – | −44 | 4 | 9% |
| C: world model | +3.60 ± 0.43 | +4.17 ± 1.08 | +0.6 | 0.41 | 5 | 4% |
| C + inverse `real` | +5.17 ± 0.12 | +4.57 ± 0.19 | −0.6 | 0.45 | 96 | 88% |
| **C + delta** | +13.70 ± 4.94 | **+25.57 ± 7.17** | +10.9, +9.4, +15.3 | 0.64 | 189 | 96% |
| **C + delta + motion** | +13.63 ± 1.09 | **+25.53 ± 1.68** | +13.0, +9.1, +13.6 | **0.85** | 250 | 50% |
| C + delta + motion + inverse | +6.10 ± 0.57 | +8.13 ± 0.54 | +2.2, +1.9, +2.0 | 0.67 | 320 | 56% |

This is the first configuration in which the world model is clearly worth having:

* **Planning beats the same checkpoint's Q-policy in every seed, by +9.1 to +15.3 points** (Q ≈ 13.6 →
  lookahead ≈ 25.5). Earlier the best planning gain was +1 to +3.5 and often within noise.
* **It beats the model-free baseline by ~3×** (+25.5 against +8.1). Before the delta target, no
  model-based configuration on either game beat plain DQN.
* **Motion channels mainly buy consistency and ball fidelity**: the same mean as delta alone but a much
  tighter spread (±1.7 vs ±7.2 on lookahead) and the best ball decodability of any run (R² 0.85).

**The two fixes are not complementary.** Combining delta with inverse dynamics is much *worse* than
either alone (+8.1 lookahead against +25.5), even though it produces the highest effective rank (320).
Both terms attack the same failure — an action-blind, over-compressed latent — and stacking them
appears to over-constrain the representation: its paddle R² drops to 0.23, the worst of the delta arms.
Pick one.

**Pong stays inconclusive**, as it has throughout: C+delta −18.40 ± 3.06 (Q) / −17.13 ± 3.33 (lookahead),
C+delta+motion −15.10 ± 4.88 / −12.30 ± 5.23, against B's −12.87 and A's −14.60. Per-seed spreads of 5–6
points swamp the differences, and the delta target does not reproduce its Breakout advantage here. Pong
rewards a mostly-reactive policy, and its ball was already the better-represented of the two games.

### "Learn by observing": offline JEPA pretraining, then re-attach RL

A suggested alternative to training everything jointly: **train the RL agent first, save its frames,
train the JEPA offline on that fixed dataset with MSE and an explicit anti-collapse term (SIGReg), then
re-attach the RL backend.** The argument is that it isolates representation collapse, which is easier to
debug than a moving data distribution with many interacting knobs. Implemented as three commands:

```bash
python -m atari_jepa.collect  --checkpoint runs/breakout_q_500k/seed1/checkpoint.pt --out data/breakout \
    --decisions 200000 --epsilons 1.0 0.1 0.01 --weights 0.2 0.6 0.2      # phase 2: save frames
python -m atari_jepa.pretrain --dataset data/breakout --out runs/pretrain_breakout_mse_sigreg \
    --updates 100000 --latent-loss mse --anti-collapse sigreg              # phase 3: observe only
python -m atari_jepa.train --config configs/breakout/breakout_world_model_500k.yaml \
    --set train.init_from=runs/pretrain_breakout_mse_sigreg/checkpoint.pt \
    --set train.freeze_encoder=true                                        # phase 4: re-attach RL
```

Datasets: 200k decisions per game from the trained model-free agent under a mixture of exploration
levels (ε ∈ {1.0, 0.1, 0.01}), giving 200k frames, 3.1–3.6k reward events, mean return +7.4 (Breakout)
and −14.4 (Pong). Pretraining: 100k updates, K = 5, MSE latent loss, **SIGReg** anti-collapse (random
unit directions, Epps–Pulley normality statistic per projection, one *global* scale so the test can see
anisotropy), EMA target, no reward/value/environment. A delta-target variant was pretrained too.

**Representations, measured before any RL** (`atari_jepa.embeddings`, held-out data from a *random*
policy so all encoders are compared on the same distribution):

| Encoder | Effective rank (of 3,136) | ball_x R² | paddle R² | Reward-event AUC |
|---|---|---|---|---|
| Breakout, live C (absolute target) | 6 | 0.66 | 0.86 | 1.00 |
| Breakout, live C + delta + motion | 159 | 0.59 | −0.16 | 0.96 |
| **Breakout, offline MSE + SIGReg** | **198** | 0.32 | −0.47 | **0.64** |
| Pong, live C (absolute target) | 15 | 0.53 | 0.92 | 0.97 |
| **Pong, offline MSE + SIGReg** | **100** | 0.15 | 0.68 | 0.87 |

**SIGReg solves collapse outright, and needed no tuning**: effective rank 100–198 against 6–15 for the
jointly-trained world models, at the first setting tried. That part of the advice is validated — with a
fixed dataset and no RL losses, collapse is a single, isolated, measurable problem.

**But the features are much less task-relevant.** On Breakout the offline encoder cannot linearly
predict the paddle it controls (R² −0.47) and barely predicts an imminent reward (AUC 0.64 against 0.96–1.00
for the live models; 0.5 is chance). The probe said so *before* any RL was run, and the RL then confirmed it:

| Variant (500k, 3 seeds) | Breakout Q | Breakout lookahead | Pong Q | Pong lookahead |
|---|---|---|---|---|
| Random policy (reference) | +0.80 | – | −20.40 | – |
| A: model-free baseline | +8.10 ± 0.99 | – | −14.60 ± 2.12 | – |
| C: live-trained JEPA (absolute) | +3.60 ± 0.43 | +4.17 ± 1.08 | −18.80 ± 0.45 | −16.90 ± 1.53 |
| **C: live + delta + motion** | **+13.63 ± 1.09** | **+25.53 ± 1.68** | −15.10 ± 4.88 | **−12.30 ± 5.23** |
| Offline absolute, **frozen** encoder | +2.17 ± 0.37 | +1.13 ± 0.69 | −20.97 ± 0.05 | −20.97 ± 0.05 |
| Offline absolute, fine-tuned | +3.90 ± 0.67 | +5.10 ± 1.27 | −18.10 ± 1.77 | −16.23 ± 0.87 |
| Offline delta, frozen | +1.93 ± 0.45 | +1.30 ± 0.57 | −21.00 ± 0.00 | −21.00 ± 0.00 |
| Offline **delta + motion**, frozen | +1.33 ± 0.17 | +2.27 ± 0.09 | −21.00 ± 0.00 | −20.90 ± 0.14 |
| Offline **delta + motion**, fine-tuned | +7.97 ± 1.95 | +9.33 ± 0.82 | **−12.77 ± 3.79** | **−11.33 ± 0.76** |

The last row matters: the first offline runs used the *originally specified* target, while the live
best uses delta + motion channels, so they were not a like-for-like comparison. Pretraining with the
same recipe as the live best closes much of the gap, and on Pong it is the best result in this
repository.

1. **Frozen "learn by observing" fails, under all three pretraining recipes.** On Pong no frozen variant
   beats a random policy (−20.9 to −21.0 against −20.40); on Breakout they reach +1.3 to +2.2 against
   +0.8 random and +8.1 for the model-free baseline. Purely observational features, in this setup, do
   not support control — and notably the delta recipe does not rescue them either.
2. **As an initialization, the recipe decides.** With the originally specified target it is roughly
   neutral (+3.9/+5.1 vs the live +3.6/+4.2 on Breakout). With the matched delta + motion recipe it is
   much stronger: **+7.97/+9.33 on Breakout** (double the absolute-target init) and **−12.77/−11.33 on
   Pong, the best Pong numbers here**, beating live delta+motion (−15.10/−12.30) with a quarter of the
   seed spread (±0.76 against ±5.23). On Breakout it still loses badly to joint training
   (+9.33 against +25.53).
3. **RL fine-tuning re-collapses the representation — but only for the absolute target.** Effective rank
   falls from 169 to 7 (Breakout) and 80 to 20 (Pong) when fine-tuning the absolute-target encoder,
   while the delta + motion encoder *keeps* its rank through RL (284 and 199). What protects a
   representation during re-attachment is the prediction target, not the offline regularizer.
4. **High rank is not the goal; the right prediction target is.** SIGReg maximizes spread, and spread on
   an Atari frame is mostly background and score digits. The delta target reached rank 250 *and* a
   decodable ball *and* triple the baseline score, by changing what is predicted rather than by
   regularizing harder.

Caveats, because this is one configuration per game: the dataset comes from a weak policy (+7.4 Breakout,
−14.4 Pong), so "observation" never sees good play; pretraining used a single untuned setting
(λ_anti = 1, 100k updates, 200k frames) whereas the live delta recipe emerged from several rounds of
diagnostics; and frozen transfer is the strictest possible test. A larger or more expert dataset, a
tuned anti-collapse weight, or a linear probe head instead of full RL might all change the picture.

### Long runs: 1.5M and 2.5M decisions, Breakout

Three times the previous budget, 3 seeds each, same protocol (evaluation ε = 0.01, 10 episodes per seed,
episodes capped at 10k decisions). `configs/long/`, launched with `scripts/run_long_breakout.sh`.

| Variant (1.5M) | Q-policy | One-step lookahead | at 500k (Q / lookahead) |
|---|---|---|---|
| Model-free Double DQN | +13.67 ± 2.10 | – | +8.10 / – |
| **Live delta + motion (joint)** | **+17.30 ± 3.06** | **+38.77 ± 3.19** | +13.63 / +25.53 |
| Offline delta + motion, fine-tuned | +11.57 ± 0.81 | +20.67 ± 2.32 | +7.97 / +9.33 |
| Offline delta + motion, frozen | +1.73 ± 0.21 | +1.97 ± 0.25 | +1.33 / +2.27 |

Per-seed lookahead for the live model: 43.2, 35.8, 37.3. Every seed beats every seed of every other
variant.

**Extended to 2.5M** for the two strongest arms (the offline ones stayed at 1.5M):

| Variant | 1.5M Q | 2.5M Q | 1.5M lookahead | 2.5M lookahead |
|---|---|---|---|---|
| Model-free | +13.67 ± 2.10 | +18.87 ± 1.83 | – | – |
| **Live delta + motion** | +17.30 ± 3.06 | **+22.03 ± 2.57** | +38.77 ± 3.19 | **+57.70 ± 5.20** |

Per-seed at 2.5M: Q 23.7 / 18.4 / 24.0, lookahead **64.9 / 55.4 / 52.8**. Neither variant has plateaued,
and the planning gain keeps growing with the model's quality (+12 points at 500k, +21 at 1.5M, +36 at
2.5M). The representation also keeps improving rather than degrading with more RL: effective rank
299 → 326, ball_x R² 0.84 → 0.87, paddle 0.67 → 0.74, reward-event AUC 0.92 → 0.96, while the
model-free encoder stays at rank 29 → 35 with the paddle still not linearly decodable (−0.29 → −0.07).

1. **The gap widens with budget.** Joint training with lookahead goes 25.5 → 38.8 while the model-free
   baseline goes 8.1 → 13.7, so the ratio holds at ~2.8× and the absolute margin grows from +17 to +25
   points. Planning adds +21.5 points over the same checkpoint's own Q-policy (17.3 → 38.8).
2. **Offline pretraining keeps improving but does not catch up.** Fine-tuned it more than doubles
   (9.3 → 20.7) yet stays ~18 points behind joint training and, on the Q-policy, slightly *below* the
   model-free baseline (11.6 vs 13.7).
3. **Frozen offline features never learn, at any budget.** Their in-training scores across the whole
   1.5M run are 1.4, 0.9, 1.2, 1.2, 0.9, 1.1 — flat from start to finish, with 3× the data. This is the
   clearest form of the result: representations trained purely by observation, then frozen, do not
   support control here.

The diagnostics track the returns exactly, and explain the ordering:

| 1.5M checkpoint | Effective rank | ball_x R² | paddle R² | Reward AUC | Shuffled-action penalty |
|---|---|---|---|---|---|
| Model-free | 29 | 0.38 | −0.29 | 0.83 | 0% |
| Live delta + motion | 299 | **0.84** | **0.67** | 0.92 | **53%** |
| Offline dm, fine-tuned | 284 | 0.75 | 0.45 | 0.95 | 5% |
| Offline dm, frozen | 152 | **0.05** | −0.06 | 0.67 | 12% |

The frozen encoder cannot locate the ball (R² 0.05) or the paddle (−0.06), which is exactly why nothing
built on it can play. And the fine-tuned offline model, despite a healthy rank and the *best* reward
probe, has an almost action-blind dynamics model (5% shuffled-action penalty against 53% for the live
one) — it predicts the future well without predicting *its own influence* on it, and that is the
difference between +20.7 and +38.8.

### Sample efficiency: which lever actually pays (500k budget, Breakout)

2.5M decisions is ~10M frames, far off the efficient frontier, so four candidate fixes were compared at
a **fixed 500k-decision budget** on top of the delta+motion world model (`configs/sample_eff/`, 3 seeds
each). Every arm trains from scratch and collects its own data; no arm ever sees another's samples.

| Arm | Q-policy | Lookahead | vs base | Optimizer updates | Wall time |
|---|---|---|---|---|---|
| base (the current recipe) | +15.10 ± 1.77 | +28.33 ± 2.41 | ref | 123,750 | 1.07 h |
| **+ n-step 5** | **+19.10 ± 1.56** | **+40.43 ± 2.19** | **+4.0 / +12.1** | 123,750 | 1.15 h |
| + replay ratio 0.5 (update every 2) | +7.40 ± 0.51 | +11.23 ± 1.25 | −7.7 / −17.1 | 247,500 | 1.59 h |
| + planner-driven collection | +2.87 ± 0.46 | +10.30 ± 1.24 | −12.2 / −18.0 | 123,750 | 1.18 h |
| + all three | +7.47 ± 0.78 | +31.30 ± 6.41 | −7.6 / +3.0 | 247,500 | 1.74 h |

**n-step returns are worth roughly 3× the samples.** Five-step targets reach **+40.4 at 500k**, which is
what the base recipe needed **1.5M** decisions to reach (+38.8), and they win on every seed for both
controllers. The change is a few lines, because the replay already returns masked K-step sequences. This
matches the diagnosis that value learning, not the model, was the bottleneck: the same world model with
better value propagation gains 12 points of planning performance.

**The other two hurt, and the reasons are instructive:**

* **Doubling the replay ratio made it much worse** (−17 points) despite twice the gradient steps —
  but only in combination with 1-step targets and the small buffer. The follow-up below shows the
  buffer was the real culprit: with n-step targets and a 300k-frame replay, the same ratio *helps*.
* **Collecting with the planner was the worst single change** (−18 points). Early in training the model
  is poor, so planner actions are close to noise, and the behaviour policy then diverges from the
  Q-head's own greedy policy, making its targets more off-policy. The controller that is *better at
  evaluation time* (+57.7 vs +22.0 at 2.5M) is not automatically a better *data* policy while it is
  still being learned. A late switch, once the model is good, is the experiment this suggests — not the
  from-scratch version tested here.

Combining all three recovers most of the lookahead loss (+31.3) but stays below n-step alone and has the
widest seed spread (±6.4), consistent with the two harmful factors partly cancelling the helpful one.

Two of these three predictions (mine, before running) were wrong in sign, which is the argument for
running the matrix rather than reasoning about it.

#### Follow-ups: how large should n be, and why did the other two fail?

**n-step sweep (500k).** The value barely matters, as long as it is not 1:

| n | Q-policy | Lookahead |
|---|---|---|
| 1 (base) | +15.10 ± 1.77 | +28.33 ± 2.41 |
| 3 | **+21.30 ± 1.53** | +39.03 ± 4.31 |
| 5 | +19.10 ± 1.56 | +40.43 ± 2.19 |
| 10 | +17.20 ± 3.51 | +40.63 ± 7.31 |

n = 3, 5 and 10 are indistinguishable for the planner (39–41) and n = 10 is the noisiest (±7.3).

**The two failures, retested on top of n-step 5:**

| Arm | Q-policy | Lookahead | vs n5 |
|---|---|---|---|
| n5 (control) | +19.10 ± 1.56 | +40.43 ± 2.19 | ref |
| + late planner switch at 250k | **+4.73 ± 1.10** | +35.43 ± 3.87 | −14.4 / −5.0 |
| + replay ratio 0.5 | +22.03 ± 2.75 | +38.83 ± 1.84 | +2.9 / −1.6 |
| + replay ratio 0.5 + Q-head resets | +20.83 ± 3.19 | +44.77 ± 5.56 | +1.7 / +4.3 |
| + replay ratio 0.5 + **300k buffer** | **+25.17 ± 3.47** | **+47.87 ± 10.20** | +6.1 / +7.4 |

* **The replay ratio was never the problem — the buffer was.** On the 1-step recipe, doubling the ratio
  cost 17 points; on the n-step recipe it is neutral (−1.6), with Q-head resets it gains +4.3, and with a
  3× larger buffer it gains **+7.4**, giving the best 500k result in this repository (+47.9 — more than
  the original recipe reached at 1.5M). The earlier negative result was an interaction between stale
  1-step targets and a buffer holding 20% of the run, not a property of replay ratio itself.
* **Planner-driven collection fails even when the model is good.** Switching at 250k, with the model
  already strong, the run tracks the control until the switch and then loses 14 points of *Q-policy*
  performance (+19.1 → +4.7) while the *planner* stays reasonable (+35.4). That is the signature of a
  behaviour/target mismatch rather than bad data quality: once the Q head's own greedy action is never
  the action taken, its targets go badly off-policy and it degrades, even though the data still
  supports the planner. Keep collecting with the policy you are training.

**The long run with n-step (2.5M):**

| Variant at 2.5M | Q-policy | Lookahead |
|---|---|---|
| Model-free baseline | +18.87 ± 1.83 | – |
| delta + motion, 1-step | +22.03 ± 2.57 | +57.70 ± 5.20 |
| **delta + motion + n-step 5** | **+32.00 ± 1.67** | **+70.93 ± 17.47** |

Per-seed lookahead 62.3 / 95.3 / 55.2 — 3.8× the model-free baseline, though the spread is now wide
enough (±17) that the mean should be read loosely.

#### How deep is the planning horizon worth searching?

Evaluation-only sweep on the same 2.5M n-step checkpoints (3 seeds, ε = 0.01, beam width 16, identical
parameters — only the search depth changes). Spreads are the population std over the three seeds:

| Search depth | Return | Per-seed | Latency / decision | Disagreement with Q |
|---|---|---|---|---|
| Q-policy (no search) | +32.00 ± 1.67 | 34.3 / 30.4 / 31.3 | 0.88 ms | – |
| H = 1 (one-step) | +70.93 ± 17.47 | 62.3 / 95.3 / 55.2 | 1.70 ms | 0.71 |
| H = 3 | +83.93 ± 3.45 | 87.9 / 79.5 / 84.4 | 1.98 ms | 0.72 |
| **H = 5** | **+91.73 ± 9.93** | 98.3 / 99.2 / 77.7 | 3.08 ms | 0.73 |
| H = 10 | +85.30 ± 5.05 | 80.6 / 92.3 / 83.0 | 6.84 ms | 0.73 |

**Depth pays up to about H = 5, then saturates.** Most of the gain is in the first step (+32 → +71);
going 1 → 5 adds another 21 points for 1.8× the latency; past that, nothing.

*I first read the 5 → 10 step as a reversal* ("−6 points, the model extrapolating past its K = 5
training horizon") and wrote that down as a finding. It does not survive. The per-seed columns already
overlap (98.3 / 99.2 / 77.7 against 80.6 / 92.3 / 83.0), and the K sweep below settles it: at a matched
500k budget the same recipe scores +65.1 / +62.0 / +62.0 / +60.3 at H = 5 / 10 / 20 / 30 — flat, not
falling. The honest statement is saturation, and a plausible-sounding mechanism ("compounding latent
error past K") made a noise-level difference look like a law. `evaluate.py` still prints a warning when
`--horizon` exceeds K, which is fair as a caution but is not evidence of a penalty.

Note the disagreement column: the planner overrides the greedy Q action on ~70% of decisions at every
depth, so the extra depth is not changing *how often* it disagrees, only *how well* it chooses.
`videos/n5_2p5m_h10_seed10001.mp4` shows H = 10 next to the same checkpoint's Q policy (+95 vs +28 on
that reset seed, close to both arms' 10-episode means).

### Hierarchy (H-JEPA level 2) against multi-step search (500k, Breakout)

Multi-step search costs H unrolls of the one-step model per candidate. The standard answer is temporal
abstraction: a *level 2* that predicts H steps ahead in one shot, conditioned on the whole action
sequence, so a candidate sequence is scored with one forward pass. `MacroDynamics` does that
(`loss.hierarchical`, `macro_horizon: 5`), with `MacroHead` predicting the discounted return collected
inside the window and whether the episode survives it; `HierarchicalController` scores candidate
sequences and executes the first action, replanning every step.

All arms are the n-step delta+motion recipe at 500k decisions, ε = 0.01; the configs differ *only* in
the `loss.hierarchical` block. The flat and detached arms have 6 seeds, the attached arm 3:

| Arm | Q-policy | Flat H = 1 | Flat H = 5 | Hierarchical planner |
|---|---|---|---|---|
| flat n-step (no level 2) | +22.13 ± 3.68 | +39.17 ± 3.90 | **+60.62 ± 6.90** | – |
| level 2, gradients **detached** | +22.05 ± 2.39 | +40.37 ± 6.96 | +57.33 ± 9.48 | +29.17 ± 3.26 |
| level 2, gradients **attached** | +13.03 ± 4.29 | +3.73 ± 0.66 | +4.63 ± 1.04 | +2.97 ± 0.73 |

**Attaching level 2 to the encoder destroys level 1.** Every loss improved when level 2 was trained
jointly — level-1 JEPA 0.242 → 0.228, reward CE 0.021 → 0.007 — while the agent became unable to play
at all (+4.6 with the search that gets +60.6 without the hierarchy). The diagnostics say why:

| | flat | detached | attached |
|---|---|---|---|
| next-latent distance between actions | 0.0661 | 0.0686 | **0.0008** |
| shuffled-action penalty, depth 5 | 48% | 51% | **2%** |
| movement probe, balanced accuracy | 0.580 | 0.598 | **0.493** |

Pulling the encoder toward features that survive a 5-step jump is pressure toward features that do not
change, and an action-invariant representation is the optimum of that objective. The movement probe is
at its majority-class baseline (0.49): the attached encoder's latents no longer encode which way
anything is moving. This is the same failure as the original cosine objective in experiment 1, reached
by a different route, and again the *losses* all looked better while it happened.

`loss.macro_detach` (default `true`) stops the level-2 loss at `z0.detach()`, so it reaches only
`macro_dynamics` and the macro heads. A parametrized test asserts the encoder gradient is exactly zero
when detached and nonzero when attached. With it, level 1 is fully preserved (columns 1 and 2 above are
indistinguishable) — and the hierarchy then neither helps nor hurts the flat controllers.

**But the jumpy planner is still much worse than unrolling.** On the same healthy checkpoints, the
hierarchical controller gets **+29.2** where 5-step beam search over level 1 gets **+57.3**. One forward
pass per candidate is ~5× cheaper and buys a substantially worse decision: level 2 has to represent the
effect of every action *sequence* in a single jump, while beam search re-grounds at each level and
prunes. For this problem, depth is better bought by searching than by abstraction.

*A correction worth recording.* At 3 seeds the detached arm looked 15 points *worse* than flat at H = 5
(+50.1 vs +65.1), which I could not explain — the detach is provably gradient-isolated and the level-1
diagnostics matched. Three more seeds per arm closed it (+57.3 vs +60.6, overlapping spreads): the
first three flat seeds happened to be both high and unusually tight (66.3 / 63.7 / 65.2, ±1.07), and
3 seeds underestimated the spread badly. The 6-seed spreads are ±6.9 and ±9.5.

### Latent space quality (Pong A/B/C, 500k checkpoints, 3 seeds each, 6,000 held-out roots per run)

Averages over seeds, from `results/pong_{q,temporal_jepa,world_model}_500k/seed*/embeddings.json`
(Breakout and the later variants have their own sections above):

| | A: Q baseline | B: temporal JEPA | C: world model |
|---|---|---|---|
| Effective rank (of 3,136 dims) | 70 | 20 | 13 |
| Variance in the top 10 components | 0.52 | 0.77 | 0.84 |
| Dimensions for 90% of variance | 125 | 26 | 18 |
| Median R² over varying RAM bytes | 0.53 | 0.60 | 0.61 |
| R²: player paddle / ball y / ball x | 0.89 / 0.91 / 0.44 | 0.91 / 0.85 / 0.36 | 0.94 / 0.89 / 0.56 |
| Centered distance Δ1 / Δ10 / across episodes | 0.17 / 0.51 / 1.01 | 0.03 / 0.18 / 1.01 | 0.05 / 0.26 / 1.01 |
| Reward within 8 decisions: AUC (base rate) | 0.95 (0.13) | 0.97 (0.13) | 0.96 (0.11) |
| Movement probe, balanced accuracy (chance 0.33) | 0.58 | 0.63 | 0.63 |

The inverse-dynamics variants are in the table in the previous section: `real` raises the effective
rank to 140–609, `predicted` leaves it at 10–21.

**The JEPA embeddings are not collapsed, and they are task-relevant.** The paddle and the ball's
vertical position are linearly decodable at R² ≈ 0.85–0.94, an imminent reward event is readable at
AUC 0.96–0.97 against a 0.11–0.13 base rate, and latent distance grows monotonically with the time gap
while staying well below the across-episode level (≈ 1.0), so the space has real temporal structure
rather than noise.

**But they are heavily compressed.** The temporal objective cuts the effective rank from 70 (A, no
JEPA) to 20 (B) to 13 (C), with 84% of the variance of C's latent in ten directions. The variance floor
is satisfied per dimension (0% of dimensions below it for B and C) while the cloud still lives in ~13
directions — per-dimension variance does not detect this, and the spectrum does. Compression is what
the prediction objective rewards: fewer, slower directions are easier to predict. It also tracks the
ordering of the Q-policy results at 500k (A −14.6, B −12.9, C −18.8): C compresses the most and plays
worst, which fits the picture of a representation optimized for predictability over control.

For these A/B/C checkpoints the weakest variable is the ball's **horizontal** position (R² 0.36–0.56,
versus 0.85+ for vertical); the delta+motion runs later reach 0.85 on it. In Pong, x is what determines *when* the ball arrives, and it moves fastest, so it is exactly
what a smoothness-rewarding objective discards first.

### The rollout horizon K: a better model that plays worse (500k, Breakout)

The depth sweep stops gaining at H = 5, which is `replay.rollout_steps` — the horizon the dynamics are
unrolled and supervised over. `configs/sample_eff/breakout_se_k{10,20,30}_500k.yaml` are identical to
`breakout_se_nstep_500k` except for K; every checkpoint is then scored at H = 1…30 through the same
evaluation path (`scripts/run_wave13.sh`, 3 seeds each, ε = 0.01).

| arm | Q | H=1 | H=3 | H=5 | H=10 | H=20 | H=30 |
|---|---|---|---|---|---|---|---|
| K=5 | +19.10 | **+40.43** | +45.57 | **+65.07** | +62.00 | +61.97 | +60.33 |
| K=10 | +19.90 | +30.73 | +47.67 | +61.83 | +59.33 | +57.70 | +63.80 |
| K=10 + `qd5` | +19.20 | +34.27 | +50.53 | +66.27 | +59.57 | +65.17 | +57.63 |
| K=20 | +17.63 | +21.63 | +39.40 | +49.13 | +47.03 | +55.73 | +51.33 |
| K=20 + `qd5` | +14.83 | +22.57 | +39.00 | +55.33 | +52.80 | +55.93 | +49.30 |
| K=30 | +16.20 | +18.30 | +26.73 | +34.03 | +36.47 | +34.83 | +36.90 |

Model quality over the same checkpoints (predicted-vs-real latent cosine distance, lower is better):

| arm | d=1 | d=5 | d=10 | d=20 | d=30 | root action dist | movement probe |
|---|---|---|---|---|---|---|---|
| K=5 | 0.1401 | 0.2137 | 0.3292 | – | – | 0.0661 | 0.580 |
| K=10 | 0.1349 | 0.2006 | 0.2769 | 0.4132 | 0.5022 | **0.1204** | 0.641 |
| K=20 | 0.1192 | 0.1993 | 0.2606 | 0.3517 | 0.4307 | 0.0968 | 0.629 |
| K=30 | **0.0971** | 0.1970 | 0.2660 | 0.3509 | 0.4189 | 0.0701 | **0.656** |

**Longer rollouts give a strictly better world model that plays strictly worse.** Monotone in both
directions: K = 30 has the lowest one-step latent error of any arm and the best movement probe, and
scores 36.9 where K = 5 scores 65.1. Two earlier results become one pattern — this is the same
dissociation as the attached H-JEPA level 2, reached by a different route. Note also that the H > K
"penalty" this sweep was built to test does not exist: K = 5 scores +65.1 / +62.0 / +62.0 / +60.3 at
H = 5 / 10 / 20 / 30, flat rather than falling.

**The value head is not the channel — my hypothesis, and it was wrong.** `q_imagined` supervises Q on
`z_hat[0..K-1]`, so a long-K arm trains its value head mostly on deep imagined latents. `loss.q_imagined_depth`
caps that independently of K; the prediction was that K = 20 + `qd5` would restore H = 1 toward +40. It
went +21.63 → +22.57, inside a seed spread. The cap is not inert (H = 5 gains 6 points) but the row
maximum is unchanged, and the dissociation survives it: K = 20 + `qd5` has the best d = 1 latent error
of any arm (0.0997) and still plays 10 points below K = 5. Whatever long rollouts break, it is not
reached through the Q-loss's depth. The reward and continuation heads are still trained at every
imagined depth and are queried by the planner at every search level; that is the untested suspect.

K costs throughput roughly linearly: 288 / 176 / 60 decisions per second at K = 5 / 10 / 30.

### Can the latent space support long-range prediction at all? (`atari_jepa.delta_probe`)

The sweep above leaves an asymmetric question. The dynamics are supervised on K-step unrolls and reach
depth 100 by *iterating* a one-step map 100 times, so a failure there is an extrapolation failure of
this training strategy — it says nothing about whether the latent space could support that range. And
Breakout is stochastic: at long range the best any *deterministic* predictor can do is the conditional
mean, which under a cosine metric is indistinguishable from collapse.

`delta_probe` supplies the missing control. With the encoder frozen it fits a head directly on pairs
`(x_t, x_{t+Δ})` for each Δ and compares it, at matched Δ, against the checkpoint's own iterated
dynamics, against persistence, and against **the mean latent** — the deterministic predictor's floor.
Scores are `1 − d/d_mean`, so 0 *is* the conditional-mean predictor. Pairs are drawn without touching
the span between them (a test asserts zero frame reads during selection), so memory is O(1) in Δ.

Δ = 1 is the calibration anchor: a directly-supervised head must at least match a dynamics model
trained for exactly one step, or the comparison measures the head's training budget rather than the
latent space. At 300 updates it did not; at 20k updates it does (+0.835 vs +0.770), and that is the
budget used throughout.

| Δ | K=5 iterated | K=5 direct | K=30 iterated | K=30 direct |
|---|---|---|---|---|
| 1 | +0.770 | **+0.835** | +0.846 | **+0.922** |
| 5 | +0.663 | +0.618 | +0.692 | **+0.771** |
| 10 | +0.478 | +0.445 | +0.599 | **+0.649** |
| 30 | +0.039 | +0.121 | **+0.462** | +0.315 |
| 100 | −0.159 | −0.068 | **+0.083** | +0.023 |
| 300 | −0.165 | −0.113 | −0.143 | −0.099 |

* **Direct supervision does not rescue long range.** At Δ = 100 the direct head — trained on exactly
  those pairs, on the same frozen latents — scores −0.068, i.e. worse than a constant. The long-range
  ceiling is *not* mainly a training-coverage problem, so a Δt-conditioned model or a latent ODE would
  meet the same wall. Breaking it needs a **distributional** model, not a better integrator.
* **Longer rollouts do buy real time-domain generalization.** At Δ = 30, K = 30 scores +0.462 against
  K = 5's +0.039, and at Δ = 100 it is the only arm still above the mean. The model genuinely improves;
  it is control that does not follow.
* **The persistence column shows the mechanism.** At Δ = 1, persistence scores −0.067 for K = 5 but
  **+0.609** for K = 30: consecutive latents are nearly orthogonal under the `delta` target, and barely
  move under long-rollout training. Long-horizon prediction pressure produces temporally smooth,
  slowly-varying features, and slow representations play worse.

  *Correction.* This paragraph originally continued "— the same thing that made the attached H-JEPA
  encoder action-blind", treating smoothing and action-blindness as one failure. The counterfactual
  measurement below shows they are not. The K = 30 encoder is **not** action-blind: it captures 78% of
  the real action effect, better than K = 5's 66%. What shrinks with K is the *total* change per
  transition (0.64 → 0.24), not the action's part of it — the action's share actually rises from 16.5%
  to 38.1%. Something else inside that shrinking change is what control loses.

Caveat: the direct head is deliberately handicapped (an action *summary* over the span, not the
sequence, since conditioning a jump on the full sequence is what the H-JEPA macro model failed at) and
is one `Dynamics`-sized block. Capacity is an unlikely explanation for failing to beat a constant, but
it is not excluded.

Two horizons land in the same place, which may or may not mean anything: prediction dies at Δ ≈ 100,
and the discount's effective horizon is 1/(1 − γ) = 100 decisions.

### Two architectural ideas that did nothing (500k, Breakout)

Both were motivated by a specific account of why long rollouts fail, and both are worth recording
because the *measurement* that killed each one is more useful than the arm itself.

**Successor features** (`loss.successor`). The Δ-probe showed the long-range wall is stochasticity, not
training coverage, and what survives at that range is a discounted average over futures — which is what
`ψ(z,a) = E[Σ γᵏ φ(z_{t+k+1})]` estimates, trained by bootstrapping so it never unrolls anything. φ is a
**frozen random projection**: ψ regresses a discounted sum of φ, so a learned φ makes the objective
circular with φ = constant as a trivial optimum.

| | Q | H=1 | H=5 | its own controller |
|---|---|---|---|---|
| conv K=5 baseline | +19.10 | +40.43 | +65.07 | – |
| + successor features | +19.97 | +39.90 | +61.10 | **+2.00** |

The head is inert where it must be, and useless as a value function: **+2.0, near random play**, with
lookahead bootstrapping on `w·ψ` reaching only +7.8 at H=5. The diagnostic says why in one line —
`ψ(z,a)` is identical across actions to five decimals (pairwise cosine similarity **1.0**), so `w·ψ` is
a constant plus noise and its argmax agrees with the Q head on 20% of decisions, below the 25% chance
rate. The action gap is 0.0195 against the Q head's 0.1504.

During implementation I reported `Q_sf ≈ 1.5` against the Q head's TD target of 1.8 as evidence the
construction was right. It was evidence the *magnitude* was right. A controller consumes the action
*gap*, not the magnitude, and a value function can match the true return in expectation while being
worthless. Check the gap before calling a value head validated.

**Legendre Memory Unit dynamics** (`network.dynamics_kind: lmu`). The rationale was gradient decay: with
K = 30 a deep term's gradient travels back through 30 applications of `g`, and a fixed Legendre Delay
Network basis should preserve what a memoryless chain attenuates. The LMU matches the conv core almost
exactly at every K (K=30: +36.2 vs +36.9 best-of-row; K=5: +68.2 vs +60.3 at H=30, within a ±10 spread)
and does not change the rollout-length slope at all.

The premise is false here, and one backward pass shows it — see the next section.

### What the gradient actually does through a rollout

Three measurements, none of which needed training, and all three contradicted the story I had been
telling about them.

**1. Credit does not decay backwards — it saturates.** `gradient_reach` (now part of `diagnostics`)
backpropagates only the deepest latent term and reports `‖∂L_K/∂z_hat[k]‖` at each step:

| arm | k=0 | k=K/2 | k=K−1 | k=K |
|---|---|---|---|---|
| conv K=30 | **5.42×** | 3.93× | 1.30× | 1.00× |
| LMU K=30 | 4.78× | 4.00× | 1.29× | 1.00× |
| conv K=5 | 2.34× | 1.95× | 1.29× | 1.00× |

The gradient arrives at the root **stronger** than at the step that produced it, at every K and for both
cores, because `z' = LayerNorm(z + delta)` is residual: its Jacobian is near the identity and LayerNorm
rescales upward. The per-step gain also falls toward 1.0 rootward (1.30× → 1.007×), so the profile
saturates rather than growing. Nothing here is starved of gradient, which is why the LMU had nothing to
fix — and it does not even improve reach.

**2. Deep terms conflict; they do not duplicate.** `atari_jepa.gradient_anatomy` backpropagates each
depth term *separately* and compares the root gradients they deliver (3 seeds per arm):

| arm | effective rank | rank/K | mean pairwise cos | **conflicting pairs** | action share (d=K, k=0) |
|---|---|---|---|---|---|
| K=5 | 3.21 | 0.64 | +0.425 | **0.0%** | 0.915 |
| K=10 | 4.13 | 0.41 | +0.478 | **0.0%** | 0.696 |
| K=20 | 7.29 | 0.36 | +0.329 | **3.3%** | 0.622 |
| K=30 | 9.37 | 0.31 | +0.278 | **8.4%** | 0.415 |

I predicted rank collapse (deep gradients becoming one repeated direction) and action-credit decaying
with depth. Both are wrong: effective rank *rises* with K, `d=15` and `d=30` reach cosine **−0.345**,
and the action-embedding share falls only ~2× over thirty steps. What does happen is **gradient
conflict** — at K = 5 and K = 10 no pair of depth terms opposes another; by K = 30, 8.4% do. Uniform
weighting treats a K-step rollout as K equally important tasks, and past K ≈ 20 they start fighting.
A feature that barely changes is the compromise they all agree on: predictable at depth 1 *and* depth
30, and useless for control. That is the smoothing the Δ-probe measured, arrived at from the gradients.

**3. Action sensitivity, finally calibrated** (`atari_jepa.action_effect`). Every action-sensitivity
number in this repository was uncalibrated: nobody had measured how much the choice of action actually
changes the next observation. From one identical emulator state (`cloneSystemState`, so branches share
the sticky-action draw and differ only in the action), take every action, encode each real successor,
and compute the same statistic the diagnostics compute on predicted latents:

| arm | real effect | model | model/real | total change | action's share |
|---|---|---|---|---|---|
| conv K=5 | 0.1055 | 0.0699 | **0.66×** | 0.6376 | 16.5% |
| conv K=10 | 0.1472 | 0.1157 | 0.79× | 0.5285 | 28.0% |
| conv K=20 | 0.1201 | 0.0992 | 0.82× | 0.3488 | 34.3% |
| conv K=30 | 0.0917 | 0.0706 | **0.78×** | 0.2402 | 38.1% |
| H-JEPA attached | 0.0160 | 0.0008 | **0.05×** | 0.5192 | 3.1% |
| LMU offline frozen | 0.0009 | 0.0001 | 0.08× | 0.0161 | 5.5% |

**The long-rollout arms are not action-blind.** K = 30 captures 78% of the real effect, *better* than
K = 5's 66%. "Action distance 0.0701" looked alarming only because the ceiling was unknown. The
genuinely blind arms are the two pathological ones, and they fail differently: H-JEPA-attached faces a
normal-sized real effect (0.0160 of a 0.52 total change) and has stopped representing it, while the
frozen-offline encoder barely distinguishes successive frames at all (total change 0.0161).

This also relocates the K story. The action effect holds roughly steady while the **total** change per
transition collapses 0.64 → 0.24, so the action's *share* rises. Long rollouts do not remove the action
from the latent; they remove much of everything else, and control loses whatever that was.

### Suggested next experiments (from the observed failures)

Ordered by what the current evidence most needs, not by size:

* **More statistical weight behind the Breakout result.** It is 3 seeds × 10 episodes. The effect is
  large (+25.5 vs +8.1) and consistent per seed, but the paired lookahead gain and the variant ordering
  both deserve more seeds and more evaluation episodes before they are quoted as facts.
* **Tune `lambda_jepa` for the delta target.** It is still 1.0, chosen for a loss that was ~10× smaller.
  The delta objective changed the loss scale, not the weighting, and nothing here has been re-tuned.
* **Explain why delta fails to transfer to Pong.** Breakout gains 3×, Pong gains nothing measurable.
  Pong's ball was already the better-represented of the two, and its optimal policy is largely
  reactive, so there may be little for a model to add — but this is a hypothesis, not a result.
  Per-seed spreads of 5–6 points mean Pong needs more seeds before any claim.
* **Do not stack the two fixes**; that is now measured (+8.1 vs +25.5). If more rank is wanted on top of
  delta, try the covariance penalty (implemented, off) rather than a second action-grounding term.
* **Untried encoder changes**, in order of expected value: spatial-softmax keypoints on an early,
  higher-resolution feature map; a finer latent grid (strides 2/2/1, ~14×14 instead of 7×7); CoordConv
  channels. The delta target attacked the *objective*; these attack the *representation's* spatial
  resolution, which is the remaining reason a 1–2 px ball is hard to localize.
* **Now worth running, finally:** planner-driven data collection and multi-step planning
  (`--horizon 3`, implemented but never evaluated). Both were pointless while action scores were ties;
  with delta checkpoints the planner's scores actually differ across actions.
* **Separate "better action-values" from "foresight".** Still open: compare the lookahead controller
  against an explicit `Q(z, a)` head trained on the same imagined-state targets. If that matches
  lookahead, part of the gain is a value-parameterization effect rather than planning.
* **Track effective rank during training**, not only the per-dimension variance floor, which missed
  every collapse in this project (rank 3.9 on Breakout while 0% of dimensions were below the floor).
* **A third game**, to see whether the delta result generalizes or is a property of brick-wall games
  with one small moving object.

## Videos

`atari_jepa.record` plays checkpoints and writes an mp4, one panel per checkpoint/controller, sharing
the reset seed and evaluation ε. Every *emulator* frame is captured (not only decision boundaries), so
60 fps playback runs at true game speed; the overlay shows the live score and decision count, and a
shorter episode freezes on its last frame while the longer one plays out.

```bash
python -m atari_jepa.record --out compare.mp4 --seed 10000 --epsilon 0.01 \
    --panel runs/breakout_q_500k/seed1/checkpoint.pt q "A: Q baseline (model-free), 500k" \
    --panel runs/breakout_wm_delta_motion_500k/seed0/checkpoint.pt lookahead "C + delta + motion: lookahead, 500k"
```

On that exact command (seed 10000, ε = 0.01, both 500k checkpoints) the baseline scores **+8** and ends
after 1,184 decisions, while the delta+motion world model with lookahead scores **+30** in 1,505
decisions. Both sit near their variants' 3-seed averages (+8.10 and +25.53), so the clip is
representative rather than a lucky episode. Videos are not committed (a few MB each); re-render them
with the command above.

Deep search on the 2.5M n-step checkpoint (`--horizon` also applies to `record`):

```bash
python -m atari_jepa.record --out h10.mp4 --seed 10001 --epsilon 0.01 --horizon 10 \
    --panel runs/breakout_long_n5/seed1/checkpoint.pt q         "2.5M n-step: Q policy (no planning)" \
    --panel runs/breakout_long_n5/seed1/checkpoint.pt lookahead "2.5M n-step: 10-step lookahead (beam 16)"
```

Q **+28** in 982 decisions against **+95** in 1,684 decisions for the 10-step planner — both within a
few points of their 10-episode means (+30.4 and +92.3 on that seed), so this clip is representative
too. Reset seeds 10000/10002/10003 give (26, 62), (43, 66) and (80, 74): the last one is the reminder
that single episodes swing widely and the table above is the result, not the video.

## Checkpoints and resume

`checkpoint.pt` holds the online and target networks, the Adam state, the full config, counters
(decisions, updates, episodes, frames, sampled transitions, eval interactions, wall time), exploration
schedule state, torch/python/numpy RNG states, and the env metadata (action meanings, preprocessing,
library versions). Writes are atomic. With `replay.save_with_checkpoint: true` (the learning configs),
`replay.npz` (about 700 MB) is written next to it and reused on resume. Without it, a resumed run starts
with an empty replay and **re-enters the random warmup**, and it logs that. Emulator and collector state
are not saved: a resume starts a new episode and never joins it to the unfinished one. That makes it a
valid training resume, not a bit-for-bit continuation. Loading checks the game id, action meanings,
observation shape and every preprocessing field (including the reward convention), warns on library
version changes, and evaluation always rebuilds the environment from the checkpoint's config.

## Tests

`python -m pytest -q` runs 69 tests in about 40 s. On a CPU without bf16 conv backward, the bf16
training-step test is skipped: 68 run locally, and all 69 pass on the CUDA machine. They use a synthetic env and toy models, plus
two real-ALE contract tests that are skipped without ale-py:

* replay: causal stacks, first-frame padding, eviction with absolute ordering, no sequence crossing a reset, partial episodes and collection boundaries, persistence,
* termination vs truncation: the terminal reward is trained without bootstrap; truncation bootstraps from the final frame, not the reset frame; padding is ignored,
* masks: changing padded observations/actions/rewards (even to invalid reward values) leaves the loss and every gradient unchanged; all-masked components are finite zeros,
* recursive gradients: a depth-K loss reaches the encoder and every dynamics application; targets get no gradient; intermediate predictions affect later ones; the audit table above,
* EMA initialization and the update convention; reward classes and expected reward; Double DQN selection/evaluation and discount masking,
* planning on a hand-built latent MDP: the higher-return action wins, continuation suppresses post-terminal value, ties break low, and exhaustive beam search matches brute force,
* checkpoint round trip (outputs, targets, optimizer, config, counters, schedule, RNG, replay), resume with and without replay, compatibility checks, and the evaluate CLI,
* fixed-batch fitting reduces JEPA and reward loss with fixed targets; diagnostics produce finite metrics and nonconstant latents,
* embeddings: the spectrum separates full-rank from low-rank latents and flags a constant one, temporal distance grows with the gap, and the evaluation-only state capture feeds probes that recover the toy game's true state,
* inverse dynamics: gradient routing for both forms, padding invariance, old configs and checkpoints loading unchanged; the movement probe separates a planted signal and fails on shuffled labels,
* bfloat16: forward matches fp32 with shared weights and keeps fp32 latents; a training step gives fp32 gradients (CUDA); the single-transfer update still rejects unclipped rewards,
* ball sensitivity: motion channels extend the input and vanish on a static scene; the centered and delta targets are invariant to a shared constant component while the absolute target is not; a no-change model scores the worst possible delta distance; both targets still train through the rollout,
* video: panels stack with the shorter one frozen, and the writer produces a readable file.

## Limitations

* **Deterministic dynamics under sticky actions.** With probability 0.25 per frame the previous action
  repeats, so the true next state is a mixture. `g` predicts a single latent and the planner scores
  that one outcome.
* **Partial observability.** Four frames give velocity but not the whole state. Pong is close to
  Markov with 4 frames, most games are not.
* **Sparse events.** Reward events are about 3% of Pong transitions and terminals about 0.1%. Reward CE
  close to the prior CE and low event recall are the expected failure mode, and the diagnostics compare
  against the prior explicitly.
* **Background-dominated latents.** Raw cosine JEPA distances on Pong are about 0.01 for any pair of
  frames, so the JEPA gradient is small next to reward CE and most latent dimensions sit below the
  variance floor. The floor (0.1) then pushes static features to vary. Both are logged rather than
  tuned.
* **Model exploitation.** Lookahead maximizes over learned reward/continuation/Q predictions at
  imagined states. Any optimistic error in `max_b Q(g(z,a))` is selected for. That is why planning is
  compared against the same checkpoint's Q-policy, not assumed to help.
* **Budget and statistical power.** At 100k decisions (≈ 400k frames) with one update per 4 decisions,
  no variant learned Pong. Everything here is 3 seeds × 10 evaluation episodes. On Pong that is far too
  little: the seed spread (5–6 points) swamps every difference between variants. On Breakout the
  delta-target effect is much larger than the spread (+25.5 ± 1.7 against +8.1 ± 1.0, and planning wins
  in every seed), but it is still 3 seeds on one game.
* Two games (Pong, Breakout), one environment instance per run; no MCTS, stochastic latents,
  augmentation or planner-driven collection. Multi-step planning (`--horizon`) is implemented but has
  not been evaluated.

## Assumptions and decisions

* The repository was a bare `uv init` scaffold. It is replaced by the self-contained `atari_jepa`
  package (src layout), and `requires-python` is relaxed to ≥ 3.12.
* Replay capacity counts stored frames (100k), so the one extra final frame per episode takes a slot.
* No-op count is uniform in [1, 30], following Gymnasium/baselines. FIRE-on-reset is off for Pong
  (verified above) and recorded in every run's metadata.
* Adam eps = 1.5e-4 (the Rainbow/SPR convention). The brief fixed only the learning rate.
* The dynamics output conv uses the default init (the model does not start as the identity/persistence map).
* The variance floor is computed on online root latents only, as specified, so it never reaches the dynamics.
* The in-training evaluation uses 3 episodes per controller, on the same reset seeds as the first 3
  final-evaluation episodes: every 25k decisions in the 100k configs, every 100k in the 500k ones. No checkpoint is selected on it: the final
  checkpoint is always evaluated.
* Diagnostic reward/continuation priors are fitted on the held-out depth-0 data itself, which is
  optimistic for the baseline.
* **Run many seeds concurrently, and enable CUDA MPS when you do.** This model is tiny (one update is
  ~20 GFLOP; batch 32, 84×84 convs, a 7×7×64 latent), so a single run reaches ~3% of an RTX 5090's fp32
  peak and the GPU is bound by kernel launches, not arithmetic. `nvidia-smi` then reports 99%
  "utilization" at 218 W of 600 W, because that field measures *time with a kernel resident*, not work
  done. Without MPS, concurrent runs time-slice the GPU instead of sharing it. Measured on 18 concurrent
  500k runs, before and after `nvidia-cuda-mps-control -d` (same runs, resumed from checkpoints):
  **112–114 → 31–35 ms per update (≈ 3.4× faster), 218 → 545 W, memory-bandwidth utilization 11 → 71%**.
  Clients must start *after* the daemon. This is the single largest throughput change found here —
  larger than any precision or batch-size choice.
* On an 8-core laptop CPU, two lanes × 4 torch threads gave the best throughput (≈ 85 min per C run).
  The reported runs used an RTX 5090 with all nine runs in parallel (≈ 30 ms per update including
  collection; 2–8 ms per update when a run has the GPU alone).
