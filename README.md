# atari-jepa: a JEPA latent world model for Atari (Pong)

A small, inspectable PyTorch agent that learns an **action-conditioned latent world model** from Atari
pixels by predicting future *representations* (not pixels), and uses that model for action selection.
The question it is built to answer: does predicting future representations produce a world model that
supports better decisions than the same network's Q-policy?

This is an experimental baseline, not a reproduction of a paper, and its scores are not comparable to
published Atari 100k results (preprocessing, interaction counting, resets and evaluation protocol have
not been matched). Influences: temporal latent prediction (SPR), latent planning with reward/value heads
(MuZero, EfficientZero), and action-conditioned rollout training (V-JEPA 2).

Contents: [Setup](#setup) · [Commands](#commands) · [Environment conventions](#environment-conventions) ·
[Model and tensor contracts](#model-and-tensor-contracts) · [Replay and masks](#replay-and-masks) ·
[Losses](#training-objective) · [Planning](#planning) · [Diagnostics](#diagnostics) ·
[Experiments](#experiments) · [Results](#results) · [Limitations](#limitations) · [Assumptions](#assumptions-and-decisions)

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
# unit + integration tests (CPU, ~20 s; synthetic env/models, plus real-ALE checks when ale-py is present)
python -m pytest -q

# ROM-free synthetic smoke (toy "catch" game with the same contract) -- does NOT validate Atari
python -m atari_jepa.train --config configs/synthetic_smoke.yaml

# real-Pong smoke: 1,500 decisions, tiny batch, capped eval (~20 s on CPU). Not a learning experiment.
python -m atari_jepa.train --config configs/pong_smoke.yaml

# learning experiments (100k decisions each)
python -m atari_jepa.train --config configs/pong_q.yaml --seed 0              # A: Q baseline
python -m atari_jepa.train --config configs/pong_temporal_jepa.yaml --seed 0  # B: temporal JEPA
python -m atari_jepa.train --config configs/pong_world_model.yaml --seed 0    # C: full world model

# evaluate one checkpoint with both controllers (same parameters, same reset seeds)
python -m atari_jepa.evaluate --checkpoint runs/pong_world_model/seed0/checkpoint.pt --controller q
python -m atari_jepa.evaluate --checkpoint runs/pong_world_model/seed0/checkpoint.pt --controller lookahead
python -m atari_jepa.evaluate --checkpoint RUN/checkpoint.pt --controller lookahead --horizon 3  # optional H-step beam search

# is the latent space representative? spectrum, temporal structure, state/reward probes
python -m atari_jepa.embeddings --checkpoint runs/pong_world_model_500k/seed0/checkpoint.pt

# held-out diagnostics on a frozen checkpoint (+ optional fixed-batch optimization check)
python -m atari_jepa.diagnostics --checkpoint runs/pong_world_model/seed0/checkpoint.pt
python -m atari_jepa.diagnostics --checkpoint RUN/checkpoint.pt --fit-fixed-batch 300 --fit-components jepa

# resume an interrupted run (or use --auto-resume with --config)
python -m atari_jepa.train --resume runs/pong_world_model/seed0

# the whole A/B/C x seeds {0,1,2} comparison, resumable, then the aggregate report
scripts/run_matrix.sh                      # CPU: two lanes; PYTHON=... THREADS=4 to override
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
| Game | `ALE/Pong-v5`, minimal action set: `NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE` (read from the env) |
| Base env frame skip | 1 (`gym.make(..., frameskip=1)`), so frames are never skipped twice |
| Action repeat | 4, in `atari_jepa.envs.AtariEnv` |
| Preprocessing | max of the last two emulated frames → ALE grayscale → `cv2.INTER_AREA` resize to 84×84 |
| History | 4 frames; an incomplete starting history repeats the episode's first frame |
| Reset no-ops | uniform in [1, 30], NOOP looked up by meaning, RNG seeded by the reset seed |
| FIRE on reset | off for Pong (it serves automatically: under NOOP-only play the opponent scores every ~35 decisions). A `fire_on_reset` helper looks up `FIRE` by meaning and refuses to guess an id |
| Life-loss termination | disabled (the config rejects enabling it) |
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
| Encoder `f` | `uint8[B,4,84,84] → [B,64,7,7]` | `/255` → Conv(32,8,s4)-ReLU → Conv(64,4,s2)-ReLU → Conv(64,3,s1)-ReLU → LayerNorm over (64,7,7). No BN or dropout |
| Dynamics `g` | `([B,64,7,7], int64[B]) → [B,64,7,7]` | 16-d action embedding broadcast over 7×7, concat → Conv3×3(80→64)-ReLU → 2 residual blocks → Conv3×3 → `LayerNorm(z + delta)` |
| Q head | `[B,64,7,7] → [B,A]` | flatten → 512 ReLU → A |
| Reward head | `(z, a) → [B,3]` | [flatten(z), one_hot(a)] → 256 ReLU → logits over rewards {−1, 0, +1} |
| Continuation head | `(z, a) → [B]` | same as the reward head, 1 logit |
| Inverse head (optional) | `(z_t, z_{t+1}) → [B,A]` | [flatten(z_t), flatten(z_{t+1})] → 256 ReLU → action logits. Built only when `loss.inverse` ≠ `none` |
| Targets | EMA copies of encoder and Q head | `θ_t ← τ θ_t + (1−τ) θ`, τ = 0.99 after every optimizer step. No target dynamics |

Planning uses `E[r] = p(+1) − p(−1)` and `P(continue) = sigmoid(logit)`. The 3-class reward head
needs sign-clipped rewards: `reward_to_class` raises if a valid reward is outside {−1, 0, +1}, and the
config rejects any other `reward_transform`. Removing clipping needs a different reward head.

Parameters (Pong, A = 6): encoder 84k, dynamics 237k, Q head 1.61M, reward head 805k, continuation
head 805k. Variant A trains encoder + Q (1.69M), B adds dynamics (1.93M), C trains everything (3.54M).
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
decisions, 100k decisions in total (23,750 updates). The loop stores the transition and the true next
frame, updates when due, and resets only after the final observation has been stored. It evaluates and
checkpoints on schedule in a separately seeded environment. No augmentation.

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
here; the unlabelled per-byte summary does not depend on them. Score-counter bytes can leave their
training range in the test episodes, so the *mean* R² over bytes is dragged negative by extrapolation
and the median is the number to read.

## Experiments

| Variant | Config | Training losses | Controller |
|---|---|---|---|
| A: Q baseline | `pong_q.yaml` | root Double DQN | Q |
| B: temporal representation | `pong_temporal_jepa.yaml` | root Q + K-step JEPA + variance floor | Q |
| C: full world model | `pong_world_model.yaml` | all of the above + reward + continuation + imagined-state Q | Q |
| C + planning | same checkpoint as C | – | one-step lookahead |

All variants share the environment protocol (sticky 0.25), seeds, 100k-decision budget, encoder/Q
architecture and initialization (same seed gives the same initial encoder and Q weights), replay sampling,
exploration schedule and update schedule. B and C cost more compute per update (see the Results).
Evaluation: 10 episodes per training seed, reset seeds 10000–10009 for every checkpoint and controller.
Policy-dependent trajectories diverge even with matched seeds. Results are reported per training seed.

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

### Latent space quality (500k checkpoints, 3 seeds each, 6,000 held-out roots per run)

Averages over seeds, from `runs/*/seed*/embeddings.json`:

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

The one consistently weak variable is the ball's **horizontal** position (R² 0.36–0.56, versus 0.85+ for
vertical). In Pong, x is what determines *when* the ball arrives, and it moves fastest, so it is exactly
what a smoothness-rewarding objective discards first.

### Suggested next experiments (from the observed failures)

* **Make the dynamics use the action.** The failure is controllability, not predictability. Options:
  an inverse-dynamics head `(z_t, z_hat_{t+1}) → a_t`, an action-contrastive term that pushes
  `g(z, a)` away from `g(z, a')` where the real next state differs, or JEPA on centered / whitened
  latents so the shared background and ball motion do not dominate the cosine. Re-run the
  shuffled-action and fixed-state action-sensitivity diagnostics as the acceptance test before looking
  at returns.
* **Check whether the encoder represents the agent's paddle at all**, with an evaluation-only linear
  probe for paddle and ball position. If it doesn't, no dynamics objective on these latents can be
  action-sensitive.
* **Guard against partial collapse and over-compression**: track the effective rank (not only the
  per-dimension variance floor) during training. At 500k the JEPA variants use ~13–20 of 3,136
  directions, and the variance floor does not see it. The optional covariance penalty is the cheapest
  thing to try, and the ball's horizontal position is the variable to watch.
* **Find which added loss slows C's Q-learning.** Ablate imagined-state Q (B + reward + continuation)
  and the reward/continuation weights, and compare Q-learning curves with B's. At 500k, C's Q-policy
  is the worst of the three.
* **More evaluation power for the planning effect.** The +1 to +3.5 paired lookahead gain at 500k needs
  more seeds and episodes (and later checkpoints) before it counts as a result.
* Only once the dynamics and heads are measurably action-sensitive: planner-driven collection and
  multi-step (`--horizon 3`) planning. Both are implemented but not yet evaluated, because action scores
  are still close to ties at 100k and only weakly separated at 500k.

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

`python -m pytest -q` runs 56 tests in about 25 s. On a CPU without bf16 conv backward, the
bf16 training-step test is skipped: 55 run locally, and all 56 pass on the CUDA machine. They use a synthetic env and toy models, plus
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
* bfloat16: forward matches fp32 with shared weights and keeps fp32 latents; a training step gives fp32 gradients (CUDA); the single-transfer update still rejects unclipped rewards.

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
* **Budget.** At 100k decisions (≈ 400k frames) with one update per 4 decisions, no variant learned
  Pong. The 500k follow-up shows learning but only 3 seeds × 10 episodes, and the differences between
  variants are within the spread between seeds except for C's slower Q-learning.
* Single environment and single game; no MCTS, stochastic latents, augmentation or planner-driven
  collection.

## Assumptions and decisions

* The repository was a bare `uv init` scaffold. It is replaced by the self-contained `atari_jepa`
  package (src layout), and `requires-python` is relaxed to ≥ 3.12.
* Replay capacity counts stored frames (100k), so the one extra final frame per episode takes a slot.
* No-op count is uniform in [1, 30], following Gymnasium/baselines. FIRE-on-reset is off for Pong
  (verified above) and recorded in every run's metadata.
* Adam eps = 1.5e-4 (the Rainbow/SPR convention). The brief fixed only the learning rate.
* The dynamics output conv uses the default init (the model does not start as the identity/persistence map).
* The variance floor is computed on online root latents only, as specified, so it never reaches the dynamics.
* The in-training evaluation uses 3 episodes per controller at 25k-decision intervals, on the same
  reset seeds as the first 3 final-evaluation episodes. No checkpoint is selected on it: the final
  checkpoint is always evaluated.
* Diagnostic reward/continuation priors are fitted on the held-out depth-0 data itself, which is
  optimistic for the baseline.
* On an 8-core laptop CPU, two lanes × 4 torch threads gave the best throughput (≈ 85 min per C run).
  The reported runs used an RTX 5090 with all nine runs in parallel (≈ 30 ms per update including
  collection; 2–8 ms per update when a run has the GPU alone).
