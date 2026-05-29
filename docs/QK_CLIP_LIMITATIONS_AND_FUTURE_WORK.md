# QK-Clip Limitations and Future Work

This document defines the current claim boundary for QK-Clip in MuZO-Clip.

MuZO-Clip currently provides QK-Clip through a best-effort Python-level capture
path. It records the input tensor passed to `torch.nn.functional.softmax` or
`torch.softmax` while attention modules containing `q_proj` and `k_proj` are
executing. This can work for eager HuggingFace-style attention implementations
where the pre-softmax attention logits are visible at Python level.

This capture mechanism is intentionally conservative. It does not treat
post-softmax attention probabilities as a valid substitute for pre-softmax
logits. If the model uses SDPA, FlashAttention, Triton attention, or another
fused attention kernel that does not expose pre-softmax logits to Python,
QK-Clip is disabled and reports the reason instead of fabricating a metric.

The current QK-Clip backend should be understood as a correctness-oriented
research path, not as a production fused-kernel integration.

## Why This Matters

QK-Clip requires a per-head scalar:

```text
Smax[h] = max attention logit for head h
```

The Kimi K2 report describes QK-Clip as a post-update weight rescaling method
using max attention logits computed during forward execution. In the MuonClip
algorithm, QK-Clip is applied after the Muon optimizer step by checking whether
each head's max logit exceeds a threshold `tau`, then rescaling the
corresponding query/key weights.

The important point is that QK-Clip does not need the full attention matrix to
be stored permanently. It only needs a small per-head statistic.

In production fused attention kernels, a related statistic may already exist
internally because online softmax implementations compute row-wise maxima as
part of numerically stable softmax. Standard public APIs usually return only the
final attention output, not the internal per-head max-logit statistics needed
for exact QK-Clip.

## Current Limitation

The current MuZO-Clip QK-Clip path has these limitations:

1. It depends on eager attention or visible Python softmax calls.
2. It does not work directly through standard fused SDPA, FlashAttention, or
   Triton kernels.
3. It may require an additional clean eager probe forward if the main training
   loop uses fused attention.
4. It currently assumes exposed `q_proj` and `k_proj` style attention modules.
5. It does not yet provide a dedicated fused-kernel path for collecting exact
   per-head max logits.
6. MLA-style architectures require architecture-specific handling, because
   query/key components may be split into shared and head-specific parts.

## Planned Direction

The preferred future direction is to decouple QK-Clip measurement from Python
softmax capture.

Instead of capturing pre-softmax logits from the attention implementation,
MuZO-Clip can add a dedicated QK max-logit probe backend:

```text
qk_probe backend:
1. Capture Q and K projection outputs with forward hooks.
2. Reshape them into [batch, heads, sequence, head_dim].
3. Compute only the per-head maximum of QK^T / sqrt(d).
4. Do not materialize or store the full attention matrix persistently.
5. Apply QK-Clip after the optimizer step using the resulting Smax[h].
```

This can first be implemented with a small PyTorch reference path for
correctness:

```python
scores = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float()) * scale
smax = scores.amax(dim=(0, 2, 3))
```

The reference path is not memory-efficient and is intended only for tests and
validation. A production implementation should use a tiled Triton or CUDA kernel
that computes per-head maxima without materializing the full score tensor.

## Possible Backends

### 1. Periodic Eager Probe

Run the main training loop with SDPA or FlashAttention. Every `N` steps, run one
clean eager forward on a small or representative batch to collect QK-Clip
statistics and apply clipping.

This is simple and safe, but it is not exact every step.

### 2. Dedicated QK Max Probe Kernel

Add a Triton or CUDA kernel that takes Q and K tensors and returns per-head max
logits. This keeps the main attention path fused and fast while making QK-Clip
independent of Python softmax capture.

This is likely the best medium-term implementation path for MuZO-Clip.

### 3. Fused Attention Auxiliary Output

Modify or wrap a fused attention kernel so it optionally returns per-head
max-logit statistics during forward execution.

This is the most efficient path, but it requires kernel-level integration and
careful maintenance across FlashAttention, PyTorch, CUDA, and hardware versions.

### 4. Approximate Softmax-LSE Signal

Some fused attention implementations expose softmax log-sum-exp statistics.
This can be useful as a diagnostic or approximate upper-bound-related signal,
but it is not the same as the true max logit.

For log-sum-exp:

```text
max(logits) <= logsumexp(logits) <= max(logits) + log(sequence_length)
```

Using LSE as a QK-Clip signal can over-clip, especially with aggressive `tau`
values. It must not be presented as exact QK-Clip unless the mathematical
relationship and clipping behavior are explicitly validated.

## QK Probe Design Sketch

A future exact `qk_probe` backend could use this flow:

1. Register forward hooks on `q_proj` and `k_proj`.
2. Capture Q/K projection outputs during a clean no-grad probe forward.
3. Infer reshape metadata:
   `[batch, sequence, hidden] -> [batch, heads, sequence, head_dim]`.
4. Compute:
   `smax[h] = max_{batch,i,j}(q[b,h,i] dot k[b,kv_head,j]) / sqrt(head_dim)`.
5. Support causal masking by excluding `j > i`.
6. Support GQA/MQA with:
   `kv_head = q_head // group_size`.
7. Return `exact_logits=True` only for exact pre-softmax max-logit signals.
8. Apply clipping after the optimizer step.

For standard MHA/GQA, QK-Clip scaling would be:

```text
if smax[h] > tau:
    gamma_h = tau / smax[h]
    q_head_rows *= gamma_h ** alpha
    k_head_or_group_rows *= gamma_h ** (1 - alpha)
```

MLA-style models need a separate adapter. Some architectures split query/key
components into shared and head-specific parts, so blindly slicing `q_proj` and
`k_proj` rows is not necessarily correct.

## Validation Requirements

Before claiming fused-kernel QK-Clip support, the following tests should exist:

1. Compare `qk_probe` per-head Smax against explicit eager QK matmul on tiny
   tensors.
2. Test causal masking.
3. Test non-causal masking.
4. Test GQA/MQA mapping from query heads to key-value heads.
5. Test fp32, bf16, and fp16 behavior.
6. Test that QK-Clip never runs when the max-logit signal is missing.
7. Test that QK-Clip reports whether the statistic is exact, approximate, or
   unavailable.
8. Test that applying QK-Clip changes only the intended Q/K head slices.
9. Test that the optimizer still works when QK-Clip is disabled.
10. Benchmark overhead for periodic probing versus every-step probing.

## Intended Claim Boundary

Until a dedicated QK max-logit probe or fused-kernel auxiliary-output path is
implemented, MuZO-Clip should only claim:

```text
QK-Clip is supported for eager attention or explicit clean probe paths where
pre-softmax logits are visible.
```

It should not claim full fused-kernel QK-Clip support yet.

The long-term goal is:

```text
Train with fused attention while collecting exact per-head QK max-logit
statistics through a lightweight auxiliary kernel or modified fused attention
backend.
```

