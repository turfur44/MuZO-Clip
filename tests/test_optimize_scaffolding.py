from __future__ import annotations

import inspect

import torch
import pytest

from muzo_clip.block_config import resolve_block_rows
from muzo_clip.fastpath import require_supported_backend
from muzo_clip.muzo_optimizer import MuZOClipOptimizer
from muzo_clip.profiling import PhaseProfiler
from muzo_clip.sparse_schedule import names_hash, select_active_parameters
from muzo_clip.parameter_filter import SelectedParameter


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)


def _loss(model: TinyModel) -> torch.Tensor:
    return model.q_proj.weight.float().pow(2).mean() + 0.5 * model.k_proj.weight.float().pow(2).mean()


def test_phase_profiling_preserves_default_update() -> None:
    torch.manual_seed(123)
    model_a = TinyModel()
    model_b = TinyModel()
    model_b.load_state_dict(model_a.state_dict())

    opt_a = MuZOClipOptimizer(model_a, seed=7, horizon=1, min_history=1)
    profiler = PhaseProfiler(enabled=True)
    opt_b = MuZOClipOptimizer(model_b, seed=7, horizon=1, min_history=1, phase_profiler=profiler)

    opt_a.estimate_projection(lambda: _loss(model_a))
    opt_b.estimate_projection(lambda: _loss(model_b))
    stats_a = opt_a.step()
    stats_b = opt_b.step()

    assert stats_a["skipped"] is False
    assert stats_b["skipped"] is False
    assert torch.allclose(model_a.q_proj.weight, model_b.q_proj.weight)
    assert torch.allclose(model_a.k_proj.weight, model_b.k_proj.weight)
    assert profiler.current_summary()


def test_gpu_stats_update_fast_path_matches_torch_path() -> None:
    torch.manual_seed(321)
    model_a = TinyModel()
    model_b = TinyModel()
    model_b.load_state_dict(model_a.state_dict())

    opt_a = MuZOClipOptimizer(model_a, seed=9, horizon=1, min_history=1, block_rows=2)
    opt_b = MuZOClipOptimizer(
        model_b,
        seed=9,
        horizon=1,
        min_history=1,
        block_rows=2,
        update_fast_path="gpu_stats",
    )

    opt_a.estimate_projection(lambda: _loss(model_a))
    opt_b.estimate_projection(lambda: _loss(model_b))
    stats_a = opt_a.step()
    stats_b = opt_b.step()

    assert stats_a["skipped"] is False
    assert stats_b["skipped"] is False
    assert torch.allclose(model_a.q_proj.weight, model_b.q_proj.weight)
    assert torch.allclose(model_a.k_proj.weight, model_b.k_proj.weight)
    assert float(stats_b["update_rms_mean"]) > 0.0
    assert float(stats_b["update_ratio_max"]) > 0.0


def test_update_fast_path_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported update_fast_path"):
        MuZOClipOptimizer(TinyModel(), update_fast_path="bad")  # type: ignore[arg-type]


def test_gpu_stats_apply_helper_has_no_item_sync() -> None:
    source = inspect.getsource(MuZOClipOptimizer._apply_update_gpu_stats)
    assert ".item(" not in source


def test_auto_block_rows_heuristic() -> None:
    assert resolve_block_rows(torch.empty(512, 4096), "auto") is None
    assert resolve_block_rows(torch.empty(2048, 4096), "auto") == 1024
    assert resolve_block_rows(torch.empty(2048, 8192), "auto") == 512
    assert resolve_block_rows(torch.empty(2048, 8192), 256) == 256
    assert resolve_block_rows(torch.empty(2048, 4096), "auto_full", full_block_max_elements=8_388_608) is None
    assert resolve_block_rows(torch.empty(4096, 4096), "auto_full", full_block_max_elements=8_388_608) == 1024


def test_sparse_round_robin_covers_all_params() -> None:
    params = [
        SelectedParameter("q_proj.weight", torch.nn.Parameter(torch.empty(2, 2))),
        SelectedParameter("k_proj.weight", torch.nn.Parameter(torch.empty(2, 2))),
        SelectedParameter("v_proj.weight", torch.nn.Parameter(torch.empty(2, 2))),
        SelectedParameter("o_proj.weight", torch.nn.Parameter(torch.empty(2, 2))),
    ]
    seen: set[str] = set()
    for step in range(2):
        active = select_active_parameters(params, mode="round_robin", groups=2, step_index=step)
        seen.update(item.name for item in active)
    assert seen == {item.name for item in params}
    assert names_hash([item.name for item in params]) == names_hash([item.name for item in params])


def test_fused_rademacher_backend_is_opt_in() -> None:
    require_supported_backend("torch", "normal")
    with pytest.raises(ValueError):
        require_supported_backend("fused_rademacher", "normal")
    try:
        require_supported_backend("fused_rademacher", "rademacher")
    except ImportError:
        pytest.skip("Triton is not installed")
