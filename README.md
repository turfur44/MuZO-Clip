# MuZO-Clip

MuZO-Clip is an experimental optimizer variant:

```text
AdaMeZO PRNG-based memory-free zeroth-order gradient reconstruction
+ Muon Newton-Schulz matrix update
+ optional QK-Clip stabilization
```

It intentionally replaces AdaMeZO's Adam-style second-moment update with a
Muon-style matrix update for selected hidden 2D weights.

This is not "Muon optimizer added to AdaMeZO".  It keeps the AdaMeZO idea of
reconstructing zeroth-order directions from small history, but changes the
update geometry:

```text
G_t^W = p_t * Z_t^W
M_t^W = sum_i beta^i * p_{t-i} * Z_{t-i}^W
U_t^W = NewtonSchulz(M_t^W)
U_t^W = U_t^W * sqrt(max(n, m)) * muon_scale
W <- W - lr * U_t^W
W <- W * (1 - lr * weight_decay)
```

This is experimental and should be tested first on small models.

## Safety Constraints

MuZO-Clip does not:

- store full model-size momentum buffers
- store full model-size variance buffers
- store perturbation tensors
- call `loss.backward()`
- rely on `param.grad`
- use `torch.optim.Optimizer` in the first-order optimizer path
- apply Muon updates to embeddings, `lm_head`, norms, or biases by default

Persistent optimizer history is small:

- step seeds
- scalar projections `p_t`
- scalar loss stats
- small QK-Clip per-head maxima when QK-Clip capture is enabled

Temporary per-matrix or per-block buffers are allocated during `step()` in fp32
and then released by normal Python/PyTorch lifetime.

## Deterministic PRNG

The critical invariant is:

```text
Z used during plus/minus SPSA perturbation == Z reconstructed during update
```

`muzo_clip/prng.py` uses param-wise deterministic seeds:

```text
hash64(global_step_seed, param_name, param_shape, block_index)
```

This avoids depending on global generator order.  The same
`make_zo_noise_like(...)` function is used for plus perturbation, minus
perturbation, restore, and historical momentum reconstruction.

Supported distributions:

- `normal`, default
- `rademacher`

## Parameter Selection

Default selected names include:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Default excluded names include:

```text
embed_tokens, lm_head, norm, bias, layernorm, ln_
```

Only 2D parameters matching the selected names and not matching excluded names
are updated.

## QK-Clip

QK-Clip is optional.  It is only applied from pre-softmax attention logits.
Post-softmax attention weights are never used.

The implementation captures the input to `torch.nn.functional.softmax` or
`torch.softmax` while a module with `q_proj` and `k_proj` is executing.  This is
intended for HuggingFace eager attention.

For speed, do not run eager attention on every training step.  Use the fast
training kernel normally, then periodically run one clean QK probe:

```python
proj_stats = optimizer.estimate_projection(loss_closure)
step_stats = optimizer.step()

if step % qk_check_every == 0:
    qk_stats = optimizer.probe_and_apply_qk_clip(
        loss_closure,
        force_eager=True,
    )
```

This keeps SDPA/FlashAttention on the main path and uses best-effort eager mode
only for the clean probe forward.  If the model cannot expose pre-softmax logits
even under eager mode, QK-Clip returns disabled instead of faking the metric.

If logits are not captured, QK-Clip reports:

```text
QK-Clip disabled because pre-softmax logits are unavailable. Use attn_implementation='eager'.
```

If per-head slicing is clear, rows of `q_proj.weight` and `k_proj.weight` are
scaled per head.  If exact head slicing is unclear, the conservative whole-matrix
fallback scales both matrices and reports `fallback_used=True`.

## Example

```bash
python scripts/train_muzo_clip.py \
  --model_name Qwen/Qwen3-0.6B \
  --data_path data/train.jsonl \
  --seq_len 512 \
  --steps 1000 \
  --lr 1e-5 \
  --zo_eps 1e-3 \
  --horizon 8 \
  --qk_clip_tau 100 \
  --qk_check_every 50 \
  --attn_implementation sdpa
```

For a very small smoke test, use a tiny model first:

```bash
python scripts/train_muzo_clip.py \
  --model_name sshleifer/tiny-gpt2 \
  --data_path data/smoke.txt \
  --seq_len 64 \
  --steps 10 \
  --lr 1e-5 \
  --zo_eps 1e-3 \
  --horizon 2 \
  --disable_qk_clip
```

## Tests

From the repository root:

```bash
python -m pytest muzo_clip/tests -q
```

The tests cover:

- deterministic PRNG reconstruction
- no `param.grad` or large persistent tensor state
- Newton-Schulz shape and zero/bf16 safety
- one projection and one update on a tiny model without backward

## Notes

The AdaMeZO source under `MeZO/` is used only as a reference.  MuZO-Clip lives in
this folder and the standalone script lives at `scripts/train_muzo_clip.py`.
