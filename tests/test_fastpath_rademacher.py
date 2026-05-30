from __future__ import annotations

import importlib.util

import pytest
import torch

from muzo_clip.fastpath import (
    fused_momentum_reconstruct_rademacher,
    fused_momentum_reconstruct_rademacher_batched,
    fused_perturb_inplace_rademacher,
    param_hash64,
    rademacher_counter_reference,
    require_supported_backend,
)
from muzo_clip.muzo_optimizer import MuZOClipOptimizer


def _has_cuda_triton() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


pytestmark = pytest.mark.skipif(not _has_cuda_triton(), reason="fused_rademacher requires CUDA and Triton")


def test_fused_perturb_restore_fp32() -> None:
    weight = torch.zeros((17, 19), device="cuda", dtype=torch.float32)
    initial = weight.clone()
    param_hash = param_hash64("layers.0.q_proj.weight", tuple(weight.shape))

    fused_perturb_inplace_rademacher(
        weight,
        base_seed=123,
        param_hash=param_hash,
        block_index=0,
        scale=1e-3,
    )
    fused_perturb_inplace_rademacher(
        weight,
        base_seed=123,
        param_hash=param_hash,
        block_index=0,
        scale=-1e-3,
    )
    torch.cuda.synchronize()

    assert torch.max(torch.abs(weight - initial)).item() == 0.0


def test_fused_perturb_deterministic() -> None:
    shape = (32, 32)
    param_hash = param_hash64("layers.0.q_proj.weight", shape)
    a = torch.zeros(shape, device="cuda", dtype=torch.float32)
    b = torch.zeros_like(a)
    c = torch.zeros_like(a)

    fused_perturb_inplace_rademacher(a, base_seed=999, param_hash=param_hash, block_index=0, scale=1.0)
    fused_perturb_inplace_rademacher(b, base_seed=999, param_hash=param_hash, block_index=0, scale=1.0)
    fused_perturb_inplace_rademacher(c, base_seed=999, param_hash=param_hash, block_index=1, scale=1.0)
    torch.cuda.synchronize()

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_fused_perturb_large_seed_hash_matches_counter_reference() -> None:
    shape = (257,)
    cases = [
        ((1 << 63) - 123, (1 << 63) - 456, 7),
        (0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 12345),
        (1234567890123456789, 9876543210987654321, 999999),
    ]
    for seed, param_hash, block_index in cases:
        out = torch.zeros(shape, device="cuda", dtype=torch.float32)
        fused_perturb_inplace_rademacher(
            out,
            base_seed=seed,
            param_hash=param_hash,
            block_index=block_index,
            scale=1.0,
        )
        torch.cuda.synchronize()
        expected = rademacher_counter_reference(
            shape,
            base_seed=seed,
            param_hash=param_hash,
            block_index=block_index,
            device="cuda",
        )
        assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_perturb_low_precision_compiles(dtype: torch.dtype) -> None:
    weight = torch.zeros((16, 16), device="cuda", dtype=dtype)
    fused_perturb_inplace_rademacher(
        weight,
        base_seed=77,
        param_hash=param_hash64("layers.0.k_proj.weight", tuple(weight.shape)),
        block_index=0,
        scale=0.25,
    )
    torch.cuda.synchronize()
    assert set(weight.float().unique().cpu().tolist()) == {-0.25, 0.25}


def test_fused_perturb_rejects_non_contiguous() -> None:
    weight = torch.zeros((8, 8), device="cuda", dtype=torch.float32).t()
    with pytest.raises(RuntimeError, match="contiguous"):
        fused_perturb_inplace_rademacher(
            weight,
            base_seed=1,
            param_hash=param_hash64("layers.0.v_proj.weight", tuple(weight.shape)),
            block_index=0,
            scale=1.0,
        )


@pytest.mark.parametrize("history_len", [1, 4, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_fused_reconstruct_matches_counter_reference(history_len: int, normalize: bool) -> None:
    shape = (7, 11)
    param_hash = param_hash64("layers.0.down_proj.weight", shape)
    seeds = torch.tensor([101 + i * 17 for i in range(history_len)], device="cuda", dtype=torch.int64)
    coeffs = torch.tensor(
        [((-1.0) ** i) * (0.25 + 0.1 * i) for i in range(history_len)],
        device="cuda",
        dtype=torch.float32,
    )
    out = torch.empty(shape, device="cuda", dtype=torch.float32)
    normalizer = float(sum(0.9**i for i in range(history_len)))
    kernel_coeffs = coeffs / normalizer if normalize else coeffs

    fused_momentum_reconstruct_rademacher(
        out,
        seeds=seeds,
        coeffs=kernel_coeffs,
        param_hash=param_hash,
        block_index=3,
    )
    torch.cuda.synchronize()

    expected = torch.zeros(shape, device="cuda", dtype=torch.float32)
    for seed, coeff in zip(seeds.cpu().tolist(), coeffs.cpu().tolist()):
        expected.add_(
            rademacher_counter_reference(
                shape,
                base_seed=int(seed),
                param_hash=param_hash,
                block_index=3,
                device="cuda",
            ),
            alpha=float(coeff),
        )
    if normalize:
        expected.div_(torch.tensor(normalizer, device="cuda", dtype=torch.float32))

    if normalize:
        torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-7)
    else:
        assert torch.equal(out, expected)


def test_fused_reconstruct_sparse_coeff_zero_case() -> None:
    shape = (5, 6)
    param_hash = param_hash64("layers.0.up_proj.weight", shape)
    out = torch.empty(shape, device="cuda", dtype=torch.float32)
    fused_momentum_reconstruct_rademacher(
        out,
        seeds=torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64),
        coeffs=torch.zeros(3, device="cuda", dtype=torch.float32),
        param_hash=param_hash,
        block_index=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("history_len", [1, 4, 8])
def test_fused_batched_reconstruct_matches_counter_reference(history_len: int) -> None:
    shape = (3, 5, 7)
    param_hash = param_hash64("layers.0.gate_proj.weight", (15, 7))
    seeds = torch.tensor([501 + i * 19 for i in range(history_len)], device="cuda", dtype=torch.int64)
    coeffs = torch.tensor(
        [((-1.0) ** i) * (0.2 + 0.05 * i) for i in range(history_len)],
        device="cuda",
        dtype=torch.float32,
    )
    out = torch.empty(shape, device="cuda", dtype=torch.float32)
    fused_momentum_reconstruct_rademacher_batched(
        out,
        seeds=seeds,
        coeffs=coeffs,
        param_hash=param_hash,
        block_start_index=2,
    )
    torch.cuda.synchronize()

    expected = torch.empty_like(out)
    for batch_index in range(shape[0]):
        ref = torch.zeros(shape[1:], device="cuda", dtype=torch.float32)
        for seed, coeff in zip(seeds.cpu().tolist(), coeffs.cpu().tolist()):
            ref.add_(
                rademacher_counter_reference(
                    shape[1:],
                    base_seed=int(seed),
                    param_hash=param_hash,
                    block_index=2 + batch_index,
                    device="cuda",
                ),
                alpha=float(coeff),
            )
        expected[batch_index].copy_(ref)

    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-6)


class TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 8, bias=False)


class TailLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 10, bias=False)
        self.down_proj = torch.nn.Linear(8, 10, bias=False)


def test_optimizer_fast_path_smoke() -> None:
    torch.manual_seed(5)
    model = TinyLinearModel().cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    opt = MuZOClipOptimizer(
        model,
        seed=11,
        horizon=2,
        min_history=1,
        distribution="rademacher",
        fast_path_backend="fused_rademacher",
        block_rows=None,
    )

    def loss() -> torch.Tensor:
        return model.q_proj.weight.float().pow(2).mean() + model.down_proj.weight.float().pow(2).mean()

    opt.estimate_projection(loss)
    stats = opt.step()
    torch.cuda.synchronize()

    assert stats["skipped"] is False
    assert stats["updated_param_count"] > 0
    assert all(torch.isfinite(param).all().item() for param in model.parameters())
    assert any(not torch.equal(param, before[name]) for name, param in model.named_parameters())


def test_optimizer_fast_path_with_gpu_stats_update_smoke() -> None:
    torch.manual_seed(6)
    model = TinyLinearModel().cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    opt = MuZOClipOptimizer(
        model,
        seed=12,
        horizon=2,
        min_history=1,
        distribution="rademacher",
        fast_path_backend="fused_rademacher",
        update_fast_path="gpu_stats",
        block_rows=None,
    )

    def loss() -> torch.Tensor:
        return model.q_proj.weight.float().pow(2).mean() + model.down_proj.weight.float().pow(2).mean()

    opt.estimate_projection(loss)
    stats = opt.step()
    torch.cuda.synchronize()

    assert stats["skipped"] is False
    assert stats["updated_param_count"] > 0
    assert torch.isfinite(torch.tensor(float(stats["update_rms_mean"]))).item()
    assert torch.isfinite(torch.tensor(float(stats["update_ratio_max"]))).item()


def test_optimizer_batched_blocks_matches_block_loop() -> None:
    torch.manual_seed(8)
    model_a = TinyLinearModel().cuda()
    model_b = TinyLinearModel().cuda()
    model_b.load_state_dict(model_a.state_dict())
    for param in model_a.parameters():
        param.requires_grad_(False)
    for param in model_b.parameters():
        param.requires_grad_(False)

    opt_a = MuZOClipOptimizer(
        model_a,
        seed=13,
        horizon=2,
        min_history=1,
        distribution="rademacher",
        fast_path_backend="fused_rademacher",
        update_fast_path="gpu_stats",
        block_rows=4,
    )
    opt_b = MuZOClipOptimizer(
        model_b,
        seed=13,
        horizon=2,
        min_history=1,
        distribution="rademacher",
        fast_path_backend="fused_rademacher",
        matrix_update_mode="batched_blocks",
        block_rows=4,
    )

    def loss_a() -> torch.Tensor:
        return model_a.q_proj.weight.float().pow(2).mean() + model_a.down_proj.weight.float().pow(2).mean()

    def loss_b() -> torch.Tensor:
        return model_b.q_proj.weight.float().pow(2).mean() + model_b.down_proj.weight.float().pow(2).mean()

    opt_a.estimate_projection(loss_a)
    opt_b.estimate_projection(loss_b)
    stats_a = opt_a.step()
    stats_b = opt_b.step()
    torch.cuda.synchronize()

    assert stats_a["skipped"] is False
    assert stats_b["skipped"] is False
    assert stats_a["updated_param_count"] == stats_b["updated_param_count"]
    assert torch.allclose(model_a.q_proj.weight, model_b.q_proj.weight, atol=1e-6, rtol=0.0)
    assert torch.allclose(model_a.down_proj.weight, model_b.down_proj.weight, atol=1e-6, rtol=0.0)
    assert torch.isfinite(torch.tensor(float(stats_b["update_rms_mean"]))).item()
    assert torch.isfinite(torch.tensor(float(stats_b["update_ratio_max"]))).item()


def test_optimizer_batched_blocks_tail_smoke() -> None:
    torch.manual_seed(10)
    model = TailLinearModel().cuda()
    opt = MuZOClipOptimizer(
        model,
        seed=15,
        horizon=2,
        min_history=1,
        distribution="rademacher",
        fast_path_backend="fused_rademacher",
        matrix_update_mode="batched_blocks",
        block_rows=4,
    )

    def loss() -> torch.Tensor:
        return model.q_proj.weight.float().pow(2).mean() + model.down_proj.weight.float().pow(2).mean()

    opt.estimate_projection(loss)
    stats = opt.step()
    torch.cuda.synchronize()

    assert stats["skipped"] is False
    assert stats["updated_param_count"] == 6
    assert torch.isfinite(torch.tensor(float(stats["update_rms_mean"]))).item()
    assert torch.isfinite(torch.tensor(float(stats["update_ratio_max"]))).item()


def test_optimizer_fast_path_rejects_cpu_selected_param() -> None:
    model = TinyLinearModel()
    with pytest.raises(RuntimeError, match="q_proj.weight"):
        MuZOClipOptimizer(
            model,
            distribution="rademacher",
            fast_path_backend="fused_rademacher",
        )


def test_optimizer_fast_path_rejects_noncontiguous_selected_param() -> None:
    model = TinyLinearModel().cuda()
    model.q_proj.weight = torch.nn.Parameter(torch.empty(8, 8, device="cuda").t())
    assert not model.q_proj.weight.is_contiguous()
    with pytest.raises(RuntimeError, match="q_proj.weight"):
        MuZOClipOptimizer(
            model,
            distribution="rademacher",
            fast_path_backend="fused_rademacher",
        )


def test_fused_backend_raises_clear_error_when_triton_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import muzo_clip.fastpath as fastpath

    monkeypatch.setattr(fastpath, "triton", None)
    monkeypatch.setattr(fastpath, "tl", None)

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "triton" or name == "triton.language":
            raise ImportError("mock missing triton")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="muzo-clip\\[fast\\]"):
        require_supported_backend("fused_rademacher", "rademacher")
