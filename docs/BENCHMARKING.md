# Benchmarking

MuZO-Clip defaults are unchanged by the benchmark scripts. The reference path is still:

```bash
--fast_path_backend torch
```

The fused path is opt-in and requires Triton:

```bash
pip install -e ".[fast,train,profile]"
```

## Fast-Path Microbenchmark

Run correctness checks and CUDA-event timing for perturb and momentum reconstruction:

```bash
python scripts/benchmark_fastpath.py \
  --jsonl outputs/benchmarks/fastpath.jsonl \
  --csv outputs/benchmarks/fastpath.csv \
  --warmup 5 \
  --iters 20
```

The script compares `fused_rademacher` against a Torch counter-Rademacher reference. It intentionally does not compare against `torch.Generator.randint`, because the fused path uses its own deterministic counter rule:

```text
base_seed + stable param_hash + block_index + linear element offset -> sign in {-1, +1}
```

Correctness checks run before speed reporting. A mismatch exits non-zero.

Default shapes:

```text
1024x1024
4096x4096
4096x11008
11008x4096
```

Default history lengths for reconstruct:

```text
H = 1, 4, 8
```

Default dtypes for perturb:

```text
fp32, bf16 when supported
```

## Training Variant Benchmark

Start with dry-run mode. It prints the exact commands without launching training:

```bash
python scripts/benchmark_train_variants.py \
  --mode dry_run \
  --model_name /path/to/model \
  --data_path /path/to/final_train_v3_200m_text_only.parquet \
  --output_root outputs/train_bench \
  --seq_len 1024 \
  --batch_size 4 \
  --max_steps 50
```

Run all variants:

```bash
python scripts/benchmark_train_variants.py \
  --mode run \
  --model_name /path/to/model \
  --data_path /path/to/final_train_v3_200m_text_only.parquet \
  --output_root outputs/train_bench \
  --seq_len 1024 \
  --batch_size 4 \
  --max_steps 50
```

Variants:

```text
torch_normal_256
torch_rademacher_256
torch_rademacher_1024
fused_rademacher_1024
fused_rademacher_auto
fused_rademacher_auto_sparse2
```

All variants use:

```text
qk_clip_mode none
same model/data/batch/seq/max_steps
no profiling by default
save_every 0
no final save
```

Use `profile_one` for one short profiled run:

```bash
python scripts/benchmark_train_variants.py \
  --mode profile_one \
  --variant fused_rademacher_auto \
  --model_name /path/to/model \
  --data_path /path/to/data.parquet \
  --output_root outputs/train_bench_profile \
  --seq_len 1024 \
  --batch_size 4 \
  --max_steps 10
```

## Summarize Training Logs

```bash
python scripts/summarize_benchmarks.py outputs/train_bench \
  --csv outputs/train_bench_summary.csv \
  --json outputs/train_bench_summary.json
```

The summary includes:

```text
tokens/sec
steps/sec
forward_time_ms
update_time_ms
gpu_memory_peak
final loss_mean
skipped steps
```

## Profiling vs Throughput

`--profile_phases` uses CUDA events and synchronizes at phase boundaries. That is correct for phase attribution, but it distorts normal training throughput. Use profile numbers to find bottlenecks; use unprofiled variant runs for actual tokens/sec comparisons.

`--torch_profile` has the same caveat. It is for timeline inspection, not final throughput.

## Comparison Order

Compare in this order:

1. `torch_normal_256` vs `torch_rademacher_256`
2. `torch_rademacher_256` vs `torch_rademacher_1024`
3. `torch_rademacher_1024` vs `fused_rademacher_1024`
4. `fused_rademacher_1024` vs `fused_rademacher_auto`
5. `fused_rademacher_auto` vs `fused_rademacher_auto_sparse2`

The key scientific comparison for the fused kernel is:

```text
torch_rademacher_* vs fused_rademacher_*
```

Do not compare `torch_normal` directly against `fused_rademacher` as if only the kernel changed; the distribution changed too.
