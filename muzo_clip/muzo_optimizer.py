"""MuZO-Clip optimizer core.

This is a standalone experimental optimizer.  It intentionally does not
subclass or use torch.optim.Optimizer because it never consumes param.grad and
never calls backward.
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import torch

from .block_config import BlockRows, resolve_block_rows
from .fastpath import (
    FastPathBackend,
    fused_momentum_reconstruct_rademacher,
    fused_perturb_inplace_rademacher,
    param_hash64,
    require_supported_backend,
)
from .newton_schulz import zeropower_via_newtonschulz5
from .parameter_filter import (
    DEFAULT_FROZEN_SUBSTRINGS,
    DEFAULT_TRAINABLE_SUBSTRINGS,
    SelectedParameter,
    select_muzo_parameters,
)
from .profiling import PhaseProfiler, null_phase_profiler
from .prng import NoiseDistribution, iter_param_blocks, make_zo_noise_like
from .qk_clip import QKClipApplyStats, QKClipController
from .sparse_schedule import SparseUpdateMode, names_hash, select_active_parameters

logger = logging.getLogger(__name__)

UpdateFastPath = Literal["torch", "gpu_stats"]


@dataclass(frozen=True)
class HistoryItem:
    seed: int
    p: float
    p_raw: float
    loss_plus: float
    loss_minus: float
    active_param_names: tuple[str, ...] | None = None


@dataclass
class ProjectionStats:
    seed: int | None
    loss_plus: float | None
    loss_minus: float | None
    p_raw: float | None
    p_used: float | None
    skipped: bool
    skip_reason: str | None


@dataclass
class StepStats:
    lr: float
    update_rms_mean: float
    update_ratio_max: float
    updated_param_count: int
    active_param_count: int
    active_param_names_hash: str
    skipped: bool
    skip_reason: str | None


def _as_float(value: torch.Tensor | float | object) -> float:
    if hasattr(value, "loss"):
        value = getattr(value, "loss")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            value = value.mean()
        return float(value.detach().float().cpu().item())
    return float(value)  # type: ignore[arg-type]


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().pow(2).mean().sqrt()


class MuZOClipOptimizer:
    """Muon-style zeroth-order optimizer with PRNG reconstruction."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 1e-5,
        zo_eps: float = 1e-3,
        horizon: int = 8,
        beta_momentum: float = 0.9,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        p_clip_value: float = 3.0,
        p_ema_beta: float = 0.95,
        update_ratio_clip: float = 0.01,
        distribution: NoiseDistribution = "normal",
        block_rows: BlockRows = None,
        seed: int = 1,
        rollback: bool = False,
        restore_exact: bool = False,
        min_history: int = 4,
        loss_spike_ratio: float = 2.0,
        trainable_substrings: Iterable[str] = DEFAULT_TRAINABLE_SUBSTRINGS,
        frozen_substrings: Iterable[str] = DEFAULT_FROZEN_SUBSTRINGS,
        normalize_momentum: bool = True,
        muon_scale: float = 0.2,
        enable_qk_clip: bool = False,
        qk_capture_clean_forward: bool = False,
        qk_clip_tau: float = 100.0,
        qk_clip_alpha: float = 0.5,
        phase_profiler: PhaseProfiler | None = None,
        fast_path_backend: FastPathBackend = "torch",
        update_fast_path: UpdateFastPath = "torch",
        sparse_update_mode: SparseUpdateMode = "off",
        sparse_update_groups: int = 1,
        full_block_max_elements: int = 8_388_608,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if min_history <= 0:
            raise ValueError("min_history must be positive")
        if min_history > horizon:
            raise ValueError("min_history must be less than or equal to horizon")
        if zo_eps <= 0:
            raise ValueError("zo_eps must be positive")
        if update_ratio_clip <= 0:
            raise ValueError("update_ratio_clip must be positive")
        if muon_scale <= 0:
            raise ValueError("muon_scale must be positive")
        if distribution not in ("normal", "rademacher"):
            raise ValueError(f"Unsupported distribution: {distribution}")
        if sparse_update_groups <= 0:
            raise ValueError("sparse_update_groups must be positive")
        if update_fast_path not in ("torch", "gpu_stats"):
            raise ValueError(f"Unsupported update_fast_path: {update_fast_path}")
        if full_block_max_elements <= 0:
            raise ValueError("full_block_max_elements must be positive")
        require_supported_backend(fast_path_backend, distribution)

        self.model = model
        self.lr = float(lr)
        self.zo_eps = float(zo_eps)
        self.horizon = int(horizon)
        self.beta_momentum = float(beta_momentum)
        self.ns_steps = int(ns_steps)
        self.weight_decay = float(weight_decay)
        self.p_clip_value = float(p_clip_value)
        self.p_ema_beta = float(p_ema_beta)
        self.update_ratio_clip = float(update_ratio_clip)
        self.distribution = distribution
        self.block_rows = block_rows
        self.fast_path_backend = fast_path_backend
        self.update_fast_path = update_fast_path
        self.full_block_max_elements = int(full_block_max_elements)
        self.rollback = bool(rollback)
        self.restore_exact = bool(restore_exact)
        self.min_history = int(min_history)
        self.loss_spike_ratio = float(loss_spike_ratio)
        self.normalize_momentum = bool(normalize_momentum)
        self.muon_scale = float(muon_scale)
        self.qk_capture_clean_forward = bool(qk_capture_clean_forward)
        self.phase_profiler = phase_profiler or null_phase_profiler()
        self.sparse_update_mode = sparse_update_mode
        self.sparse_update_groups = int(sparse_update_groups)

        self.selected_params: list[SelectedParameter] = select_muzo_parameters(
            model,
            trainable_substrings=trainable_substrings,
            frozen_substrings=frozen_substrings,
        )
        self._param_hashes = {
            selected.name: param_hash64(selected.name, tuple(selected.param.shape)) for selected in self.selected_params
        }
        if not self.selected_params:
            logger.warning("MuZO-Clip selected no parameters with the current filter")
        elif not self.restore_exact and self._has_low_precision_selected_params():
            logger.warning(
                "MuZO-Clip selected fp16/bf16 parameters while restore_exact=False; "
                "+eps/-2eps/+eps perturb restore is not numerically exact in low precision. "
                "This avoids CPU snapshots and is the intended LLM default."
            )
        if self.rollback:
            logger.warning("rollback=True snapshots selected params to CPU; avoid this for LLM-scale runs.")
        if self.restore_exact:
            logger.warning("restore_exact=True snapshots selected params to CPU; avoid this for LLM-scale runs.")
        if fast_path_backend == "fused_rademacher":
            self._validate_fused_rademacher_layout()

        self.history: deque[HistoryItem] = deque(maxlen=self.horizon)
        self.p_abs_ema: float = 0.0
        self.loss_ema: float | None = None
        self.bad_step_count: int = 0
        self.lr_scale: float = 1.0
        self._rng = random.Random(int(seed))
        self._has_pending_update = False
        self._last_step_updated = False
        self._projection_index = 0
        self._pending_active_params: list[SelectedParameter] = self.selected_params
        self._pending_active_names: tuple[str, ...] | None = None

        self.qk_clip: QKClipController | None = None
        if enable_qk_clip:
            self.qk_clip = QKClipController(model, tau=qk_clip_tau, alpha=qk_clip_alpha)

    @property
    def current_lr(self) -> float:
        return self.lr * self.lr_scale

    def selected_parameter_names(self) -> list[str]:
        return [item.name for item in self.selected_params]

    def active_parameter_names(self) -> list[str]:
        return [item.name for item in self._pending_active_params]

    def persistent_tensor_state(self) -> list[torch.Tensor]:
        """Return persistent non-parameter tensors owned by the optimizer."""

        tensors: list[torch.Tensor] = []
        if self.qk_clip is not None:
            tensors.extend(self.qk_clip.persistent_tensors())
        return tensors

    @torch.no_grad()
    def estimate_projection(self, loss_closure: Callable[[], torch.Tensor | float | object]) -> dict[str, object]:
        """Run the SPSA plus/minus projection and append scalar history."""

        seed = self._sample_seed()
        self._clear_qk_clip_capture()
        self._projection_index += 1
        active_params = select_active_parameters(
            self.selected_params,
            mode=self.sparse_update_mode,
            groups=self.sparse_update_groups,
            step_index=self._projection_index - 1,
        )
        self._pending_active_params = active_params
        active_names = tuple(item.name for item in active_params)
        self._pending_active_names = None if active_params is self.selected_params else active_names
        snapshot = self._snapshot_selected_params() if self.rollback or self.restore_exact else None
        displacement = 0

        try:
            with self._phase("perturb_plus"):
                self._perturb_selected(seed, scaling_factor=1.0, selected_params=active_params)
            displacement = 1
            with self._phase("forward_plus"):
                loss_plus = self._call_loss(loss_closure)

            with self._phase("perturb_minus"):
                self._perturb_selected(seed, scaling_factor=-2.0, selected_params=active_params)
            displacement = -1
            with self._phase("forward_minus"):
                loss_minus = self._call_loss(loss_closure)

            with self._phase("restore"):
                if self.restore_exact and snapshot is not None:
                    self._restore_snapshot(snapshot)
                else:
                    self._perturb_selected(seed, scaling_factor=1.0, selected_params=active_params)
            displacement = 0
        except Exception:
            self._restore_after_failed_projection(seed, displacement, snapshot)
            raise

        p_raw = (loss_plus - loss_minus) / (2.0 * self.zo_eps)
        skip_reason = self._projection_skip_reason(loss_plus, loss_minus, p_raw)
        if skip_reason is not None:
            if snapshot is not None:
                self._restore_snapshot(snapshot)
            if self.qk_clip is not None:
                self.qk_clip.clear()
            self.bad_step_count += 1
            self.lr_scale = max(self.lr_scale * 0.5, 1e-4)
            self._has_pending_update = False
            return ProjectionStats(
                seed=seed,
                loss_plus=loss_plus,
                loss_minus=loss_minus,
                p_raw=p_raw,
                p_used=None,
                skipped=True,
                skip_reason=skip_reason,
            ).__dict__

        abs_p = abs(float(p_raw))
        if self.p_abs_ema == 0.0:
            self.p_abs_ema = abs_p
        else:
            self.p_abs_ema = self.p_ema_beta * self.p_abs_ema + (1.0 - self.p_ema_beta) * abs_p
        p_norm = float(p_raw) / (self.p_abs_ema + 1e-12)
        p_used = max(-self.p_clip_value, min(self.p_clip_value, p_norm))

        self.history.append(
            HistoryItem(
                seed=int(seed),
                p=float(p_used),
                p_raw=float(p_raw),
                loss_plus=float(loss_plus),
                loss_minus=float(loss_minus),
                active_param_names=self._pending_active_names,
            )
        )
        self._has_pending_update = True
        self._update_loss_ema(0.5 * (float(loss_plus) + float(loss_minus)))
        if self.qk_clip is not None and self.qk_capture_clean_forward:
            with self.qk_clip.capture():
                self._call_loss(loss_closure)

        return ProjectionStats(
            seed=seed,
            loss_plus=loss_plus,
            loss_minus=loss_minus,
            p_raw=p_raw,
            p_used=p_used,
            skipped=False,
            skip_reason=None,
        ).__dict__

    @torch.no_grad()
    def step(self) -> dict[str, object]:
        """Reconstruct historical ZO directions and apply Muon-style updates."""

        if not self._has_pending_update:
            self._last_step_updated = False
            self._clear_qk_clip_capture()
            return StepStats(
                lr=self.current_lr,
                update_rms_mean=0.0,
                update_ratio_max=0.0,
                updated_param_count=0,
                active_param_count=0,
                active_param_names_hash="",
                skipped=True,
                skip_reason="no successful pending projection",
            ).__dict__
        if not self.history:
            self._has_pending_update = False
            self._last_step_updated = False
            self._clear_qk_clip_capture()
            return StepStats(
                lr=self.current_lr,
                update_rms_mean=0.0,
                update_ratio_max=0.0,
                updated_param_count=0,
                active_param_count=0,
                active_param_names_hash="",
                skipped=True,
                skip_reason="empty projection history",
            ).__dict__
        if len(self.history) < self.min_history:
            self._has_pending_update = False
            self._last_step_updated = False
            self._clear_qk_clip_capture()
            return StepStats(
                lr=self.current_lr,
                update_rms_mean=0.0,
                update_ratio_max=0.0,
                updated_param_count=0,
                active_param_count=0,
                active_param_names_hash="",
                skipped=True,
                skip_reason=f"waiting for min_history={self.min_history}",
            ).__dict__

        lr = self.current_lr
        update_rms_values: list[float] = []
        update_rms_tensors: list[torch.Tensor] = []
        update_ratio_tensors: list[torch.Tensor] = []
        update_ratio_max = 0.0
        updated_param_count = 0
        active_params = self._pending_active_params
        active_names = [item.name for item in active_params]
        active_names_set = set(active_names)
        if not active_params:
            self._has_pending_update = False
            self._last_step_updated = False
            self._clear_qk_clip_capture()
            return StepStats(
                lr=lr,
                update_rms_mean=0.0,
                update_ratio_max=0.0,
                updated_param_count=0,
                active_param_count=0,
                active_param_names_hash="",
                skipped=True,
                skip_reason="no active selected parameters",
            ).__dict__
        fused_history_seeds: torch.Tensor | None = None
        fused_shared_coeffs: torch.Tensor | None = None
        if self.fast_path_backend == "fused_rademacher":
            history_seeds = [int(item.seed) for item in reversed(self.history)]
            fused_history_seeds = torch.tensor(history_seeds, device=active_params[0].param.device, dtype=torch.int64)
            if self.sparse_update_mode == "off":
                coeffs = [float((self.beta_momentum**age) * item.p) for age, item in enumerate(reversed(self.history))]
                if self.normalize_momentum:
                    coeff_sum = sum(self.beta_momentum**age for age, _item in enumerate(reversed(self.history)))
                    if coeff_sum > 0:
                        coeffs = [value / coeff_sum for value in coeffs]
                fused_shared_coeffs = torch.tensor(coeffs, device=active_params[0].param.device, dtype=torch.float32)

        for selected in active_params:
            param = selected.param
            fused_seeds: torch.Tensor | None = None
            fused_coeffs: torch.Tensor | None = None
            if self.fast_path_backend == "fused_rademacher":
                fused_seeds = fused_history_seeds
                if fused_shared_coeffs is not None:
                    fused_coeffs = fused_shared_coeffs
                else:
                    coeffs: list[float] = []
                    coeff_sum = 0.0
                    for age, item in enumerate(reversed(self.history)):
                        coeff = self.beta_momentum**age
                        if item.active_param_names is None or selected.name in item.active_param_names:
                            coeffs.append(float(coeff * item.p))
                            coeff_sum += coeff
                        else:
                            coeffs.append(0.0)
                    if coeff_sum <= 0:
                        continue
                    if self.normalize_momentum:
                        coeffs = [value / coeff_sum for value in coeffs]
                    fused_coeffs = torch.tensor(coeffs, device=param.device, dtype=torch.float32)

            block_rows = resolve_block_rows(
                param.data,
                self.block_rows,
                full_block_max_elements=self.full_block_max_elements,
            )
            for block_index, _, block in iter_param_blocks(param.data, block_rows):
                with self._phase("muzo_reconstruct"):
                    M = torch.empty(block.shape, device=block.device, dtype=torch.float32)
                    if self.fast_path_backend == "fused_rademacher":
                        assert fused_seeds is not None and fused_coeffs is not None
                        fused_momentum_reconstruct_rademacher(
                            M,
                            seeds=fused_seeds,
                            coeffs=fused_coeffs,
                            param_hash=self._param_hashes[selected.name],
                            block_index=block_index,
                        )
                    else:
                        M.zero_()
                        coeff_sum = 0.0
                        for age, item in enumerate(reversed(self.history)):
                            if item.active_param_names is not None and selected.name not in item.active_param_names:
                                continue
                            coeff = self.beta_momentum**age
                            noise = make_zo_noise_like(
                                block,
                                item.seed,
                                selected.name,
                                block_index=block_index,
                                distribution=self.distribution,
                            ).float()
                            M.add_(noise, alpha=coeff * item.p)
                            coeff_sum += coeff
                        if self.normalize_momentum and coeff_sum > 0:
                            M.div_(coeff_sum)

                with self._phase("newton_schulz"):
                    U = zeropower_via_newtonschulz5(M, steps=self.ns_steps)

                with self._phase("apply_update"):
                    if self.update_fast_path == "gpu_stats":
                        update_rms_tensor, ratio_tensor = self._apply_update_gpu_stats(block, U, lr)
                        update_rms_tensors.append(update_rms_tensor)
                        update_ratio_tensors.append(ratio_tensor)
                    else:
                        U.mul_(math.sqrt(max(int(block.shape[0]), int(block.shape[1]))) * self.muon_scale)
                        update_rms = _rms(U) * lr
                        weight_rms = _rms(block)
                        ratio = update_rms / (weight_rms + 1e-12)
                        ratio_float = float(ratio.item())
                        if ratio_float > self.update_ratio_clip:
                            U.mul_(self.update_ratio_clip / ratio_float)
                            update_rms = _rms(U) * lr
                            ratio_float = float((update_rms / (weight_rms + 1e-12)).item())

                        block.add_(U.to(dtype=block.dtype), alpha=-lr)
                        if self.weight_decay:
                            block.mul_(1.0 - lr * self.weight_decay)
                        update_rms_values.append(float(update_rms.item()))
                        update_ratio_max = max(update_ratio_max, ratio_float)
                updated_param_count += 1

                del M, U

        self._has_pending_update = False
        self._last_step_updated = updated_param_count > 0
        if self.update_fast_path == "gpu_stats" and update_rms_tensors:
            mean_update_rms = float(torch.stack(update_rms_tensors).mean().item())
            update_ratio_max = float(torch.stack(update_ratio_tensors).max().item())
        else:
            mean_update_rms = sum(update_rms_values) / len(update_rms_values) if update_rms_values else 0.0
        return StepStats(
            lr=lr,
            update_rms_mean=mean_update_rms,
            update_ratio_max=update_ratio_max,
            updated_param_count=updated_param_count,
            active_param_count=len(active_names_set),
            active_param_names_hash=names_hash(active_names),
            skipped=False,
            skip_reason=None,
        ).__dict__

    def _apply_update_gpu_stats(
        self,
        block: torch.Tensor,
        U: torch.Tensor,
        lr: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        U.mul_(math.sqrt(max(int(block.shape[0]), int(block.shape[1]))) * self.muon_scale)
        update_rms = _rms(U) * lr
        weight_rms = _rms(block)
        ratio = update_rms / (weight_rms + 1e-12)
        clip = ratio.new_tensor(self.update_ratio_clip)
        should_clip = ratio > clip
        scale = torch.where(should_clip, clip / ratio.clamp_min(1e-12), torch.ones_like(ratio))
        scaled_update_rms = torch.where(should_clip, clip * (weight_rms + 1e-12), update_rms)
        scaled_ratio = torch.where(should_clip, clip, ratio)
        U.mul_(scale)
        block.add_(U.to(dtype=block.dtype), alpha=-lr)
        if self.weight_decay:
            block.mul_(1.0 - lr * self.weight_decay)
        return scaled_update_rms.detach(), scaled_ratio.detach()

    @torch.no_grad()
    def apply_qk_clip_if_available(self) -> dict[str, object]:
        if self.qk_clip is None:
            return QKClipApplyStats(
                enabled=False,
                exact_logits=False,
                fallback_used=False,
                disabled_reason="QK-Clip controller is not enabled",
                qk_smax_max=0.0,
                qk_clip_count=0,
            ).__dict__
        if not self._last_step_updated:
            self.qk_clip.clear()
            return QKClipApplyStats(
                enabled=False,
                exact_logits=False,
                fallback_used=False,
                disabled_reason="QK-Clip skipped because the last MuZO step did not update weights",
                qk_smax_max=0.0,
                qk_clip_count=0,
            ).__dict__
        self._last_step_updated = False
        return self.qk_clip.apply().__dict__

    @torch.no_grad()
    def probe_and_apply_qk_clip(
        self,
        clean_forward_closure: Callable[[], torch.Tensor | float | object],
        *,
        force_eager: bool = True,
    ) -> dict[str, object]:
        """Periodically measure clean QK logits and apply QK-Clip.

        Use this from a fast SDPA/Flash training loop every N steps. The normal
        MuZO projection/update path remains on the fast attention kernel; this
        method runs one extra clean forward under a best-effort eager attention
        override so pre-softmax logits can be captured. If the model still does
        not expose logits, the returned stats clearly report QK-Clip disabled.
        """

        if self.qk_clip is None:
            return QKClipApplyStats(
                enabled=False,
                exact_logits=False,
                fallback_used=False,
                disabled_reason="QK-Clip controller is not enabled",
                qk_smax_max=0.0,
                qk_clip_count=0,
            ).__dict__

        def wrapped_forward() -> float:
            return self._call_loss(clean_forward_closure)

        with self._phase("qk_probe"):
            return self.qk_clip.probe(wrapped_forward, force_eager=force_eager).__dict__

    def _sample_seed(self) -> int:
        return self._rng.randrange(0, (1 << 63) - 1)

    def _has_low_precision_selected_params(self) -> bool:
        return any(selected.param.dtype in (torch.float16, torch.bfloat16) for selected in self.selected_params)

    def _validate_fused_rademacher_layout(self) -> None:
        """Fail fast before fused kernels enter the hot path."""

        for selected in self.selected_params:
            param = selected.param
            if not param.is_cuda:
                raise RuntimeError(f"fused_rademacher requires CUDA tensor for selected param: {selected.name}")
            if param.ndim != 2:
                raise RuntimeError(f"fused_rademacher only supports 2D selected param: {selected.name}")
            if not param.data.is_contiguous():
                raise RuntimeError(f"fused_rademacher requires contiguous selected param: {selected.name}")
            block_rows = resolve_block_rows(
                param.data,
                self.block_rows,
                full_block_max_elements=self.full_block_max_elements,
            )
            for block_index, _row_slice, block in iter_param_blocks(param.data, block_rows):
                if not block.is_cuda or not block.is_contiguous():
                    raise RuntimeError(
                        "fused_rademacher requires CUDA contiguous row blocks for "
                        f"selected param: {selected.name} block_index={block_index}"
                    )

    @torch.no_grad()
    def _perturb_selected(
        self,
        seed: int,
        scaling_factor: float,
        selected_params: list[SelectedParameter] | None = None,
    ) -> None:
        scale = float(scaling_factor) * self.zo_eps
        selected_items = selected_params if selected_params is not None else self.selected_params
        for selected in selected_items:
            block_rows = resolve_block_rows(
                selected.param.data,
                self.block_rows,
                full_block_max_elements=self.full_block_max_elements,
            )
            for block_index, _, block in iter_param_blocks(selected.param.data, block_rows):
                if self.fast_path_backend == "fused_rademacher":
                    fused_perturb_inplace_rademacher(
                        block,
                        base_seed=seed,
                        param_hash=self._param_hashes[selected.name],
                        block_index=block_index,
                        scale=scale,
                    )
                else:
                    noise = make_zo_noise_like(
                        block,
                        seed,
                        selected.name,
                        block_index=block_index,
                        distribution=self.distribution,
                    )
                    block.add_(noise, alpha=scale)

    @contextlib.contextmanager
    def _phase(self, name: str):
        with self.phase_profiler.phase(name):
            yield

    def _call_loss(self, loss_closure: Callable[[], torch.Tensor | float | object]) -> float:
        self.model.eval()
        with torch.no_grad():
            return _as_float(loss_closure())

    def _projection_skip_reason(self, loss_plus: float, loss_minus: float, p_raw: float) -> str | None:
        values = {
            "loss_plus": float(loss_plus),
            "loss_minus": float(loss_minus),
            "p_raw": float(p_raw),
        }
        for name, value in values.items():
            if not math.isfinite(value):
                return f"{name} is non-finite"

        if self.loss_ema is not None:
            threshold = self.loss_spike_ratio * max(abs(self.loss_ema), 1e-12)
            if float(loss_plus) > threshold:
                return "loss_plus exceeded loss spike threshold"
            if float(loss_minus) > threshold:
                return "loss_minus exceeded loss spike threshold"
        return None

    def _update_loss_ema(self, loss: float) -> None:
        if self.loss_ema is None:
            self.loss_ema = float(loss)
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * float(loss)

    def _clear_qk_clip_capture(self) -> None:
        if self.qk_clip is not None:
            self.qk_clip.clear()

    @torch.no_grad()
    def _snapshot_selected_params(self) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        return [(selected.param, selected.param.detach().cpu().clone()) for selected in self.selected_params]

    @torch.no_grad()
    def _restore_snapshot(self, snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
        for param, cpu_value in snapshot:
            param.data.copy_(cpu_value.to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def _restore_after_failed_projection(
        self,
        seed: int,
        displacement: int,
        snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]] | None,
    ) -> None:
        if snapshot is not None:
            self._restore_snapshot(snapshot)
        elif displacement == 1:
            self._perturb_selected(seed, scaling_factor=-1.0, selected_params=self._pending_active_params)
        elif displacement == -1:
            self._perturb_selected(seed, scaling_factor=1.0, selected_params=self._pending_active_params)
        if self.qk_clip is not None:
            self.qk_clip.clear()


@contextlib.contextmanager
def _nullcontext():
    yield
