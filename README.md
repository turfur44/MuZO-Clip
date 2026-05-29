# MuZO-Clip

MuZO-Clip is an experimental research prototype for zeroth-order full-parameter
fine-tuning.

It combines three ideas:

- AdaMeZO-style SPSA projections with deterministic PRNG reconstruction
- Muon-style Newton-Schulz matrix updates for hidden 2D transformer weights
- optional QK-Clip stabilization from clean pre-softmax attention logits

This is not a drop-in "Muon optimizer for AdaMeZO". The optimizer keeps the
memory-free PRNG reconstruction idea from AdaMeZO, but intentionally replaces
Adam-style second-moment preconditioning with a Muon-style matrix update.

## Status

This repository contains the standalone `muzo_clip` Python package only. It is
intended to be imported by training scripts, experiments, or downstream forks.

Large-model training scripts, dashboards, datasets, and checkpoints are not
included in this public package.

This code is experimental. It is not proven to be stable or better than MeZO,
AdaMeZO, Muon, LoRA, or first-order fine-tuning. Test on small models first.

## Core Update

For each selected 2D weight matrix `W`, MuZO-Clip estimates SPSA scalar
projections:

```text
p_t = (L(w_t + eps * z_t) - L(w_t - eps * z_t)) / (2 * eps)
G_t^W = p_t * Z_t^W
```

It then reconstructs recent directions from seed history:

```text
M_t^W = sum_i beta^i * p_{t-i} * Z_{t-i}^W
U_t^W = NewtonSchulz(M_t^W)
U_t^W = U_t^W * sqrt(max(n, m)) * muon_scale
W <- W - lr * U_t^W
W <- W * (1 - lr * weight_decay)
```

The momentum matrix `M_t^W` is temporary and accumulated in fp32. It is not
stored persistently.

## Memory Constraints

MuZO-Clip does not:

- store model-size momentum buffers
- store model-size variance buffers
- store perturbation tensors
- call `loss.backward()`
- rely on `param.grad`
- use `torch.optim.Optimizer` as a normal first-order optimizer
- update embeddings, `lm_head`, norm layers, or biases by default

Persistent optimizer state is intentionally small:

- step seeds
- scalar projections
- scalar loss statistics
- optional small QK-Clip statistics

Temporary per-matrix or per-block tensors are allowed during update and then
released by normal Python/PyTorch lifetime.

## Deterministic PRNG

The critical invariant is:

```text
noise used during SPSA perturbation == noise reconstructed during update
```

`muzo_clip.prng.make_zo_noise_like(...)` derives deterministic param-wise random
directions from:

```text
hash64(global_step_seed, param_name, param_shape, block_index)
```

This avoids depending on global RNG order. The same function is used for plus
perturbation, minus perturbation, restore, and historical momentum
reconstruction.

Supported distributions:

- `normal`, default
- `rademacher`

## Parameter Selection

By default, MuZO-Clip updates hidden 2D transformer matrices whose names include:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

It excludes names containing:

```text
embed_tokens, lm_head, norm, bias, layernorm, ln_
```

Autograd is not required for selected parameters. The optimizer updates them
in-place under `torch.no_grad()`.

## QK-Clip

QK-Clip is optional and conservative.

It must use pre-softmax attention logits:

```text
Q K^T / sqrt(d_head)
```

Post-softmax attention probabilities are not valid for QK-Clip and are not used
as a substitute.

The current implementation captures Python-level `torch.nn.functional.softmax`
or `torch.softmax` calls while modules containing `q_proj` and `k_proj` execute.
This is a best-effort path for HuggingFace eager attention. If SDPA,
FlashAttention, Triton, or another fused kernel hides the logits, QK-Clip is
disabled and reports the reason instead of faking a metric.

For faster experiments, run the main training loop with SDPA/FlashAttention and
periodically do a clean eager probe:

```python
proj_stats = optimizer.estimate_projection(loss_closure)
step_stats = optimizer.step()

if step % qk_check_every == 0:
    qk_stats = optimizer.probe_and_apply_qk_clip(
        loss_closure,
        force_eager=True,
    )
```

Good signs:

```text
qk_smax_max > 0
disabled_reason is None
fallback_used is False, when exact per-head slicing is available
```

If the model architecture does not expose enough metadata for exact head
slicing, MuZO-Clip may use a whole-matrix fallback and report
`fallback_used=True`.

## Installation

From the repository root:

```bash
pip install -e .
```

Runtime dependency is currently only PyTorch:

```bash
pip install torch
```

Your training script will usually also need `transformers`, `accelerate`, and
dataset tooling.

## Minimal Usage

```python
from muzo_clip import MuZOClipOptimizer

optimizer = MuZOClipOptimizer(
    model,
    lr=1e-5,
    zo_eps=1e-3,
    horizon=8,
    min_history=4,
    beta_momentum=0.9,
    weight_decay=0.01,
    muon_scale=0.2,
    update_ratio_clip=0.01,
    enable_qk_clip=True,
)

def loss_closure():
    with torch.no_grad():
        outputs = model(**batch)
        return outputs.loss

proj_stats = optimizer.estimate_projection(loss_closure)
step_stats = optimizer.step()

if global_step % 50 == 0:
    qk_stats = optimizer.probe_and_apply_qk_clip(loss_closure, force_eager=True)
```

Training scripts must provide the loss closure, batching, checkpointing, logging,
and optional assistant-only label masking.

## Important Defaults

```text
lr = 1e-5
zo_eps = 1e-3
horizon = 8
min_history = 4
beta_momentum = 0.9
weight_decay = 0.01
muon_scale = 0.2
p_clip_value = 3.0
update_ratio_clip = 0.01
rollback = False
restore_exact = False
distribution = normal
```

`rollback=False` avoids copying selected parameters to CPU every step. That CPU
snapshot path is too expensive for large models.

When using fp16/bf16 parameters with `restore_exact=False`, perturb/restore is
not bit-exact because low precision arithmetic rounds the in-place updates. For
initial correctness checks, fp32 weights are easier to reason about.

## What To Watch

Useful training logs:

```text
loss_plus
loss_minus
p_raw
p_used
update_ratio_max
update_rms_mean
qk_smax_max
qk_clip_count
fallback_used
disabled_reason
gpu_memory_allocated
```

Sanity checks:

- `p_used` should not be glued to `+3` or `-3` forever
- `update_ratio_max` should respect `update_ratio_clip`
- QK-Clip should report a real `qk_smax_max` when enabled and logits are visible
- `disabled_reason` must be treated as QK-Clip not actually running

## Non-Goals

This repository does not claim:

- that MuZO-Clip is better than MeZO or AdaMeZO
- that QK-Clip works through fused attention kernels
- that this is production-ready
- that this is a complete trainer

The goal is to keep the optimizer logic small, inspectable, and reproducible.
