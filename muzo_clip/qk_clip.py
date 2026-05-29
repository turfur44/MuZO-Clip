"""Optional QK-Clip support using pre-softmax logits only.

This module does not inspect post-softmax attention probabilities.  It captures
the input to torch/F.softmax while a known attention module is executing.  If an
eager attention implementation does not expose a softmax call, QK-Clip is
reported as disabled.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class QKClipApplyStats:
    enabled: bool
    exact_logits: bool
    fallback_used: bool
    disabled_reason: str | None
    qk_smax_max: float
    qk_clip_count: int


class QKClipController:
    """Capture pre-softmax logits and apply QK-Clip to q/k projections."""

    def __init__(self, model: torch.nn.Module, tau: float = 100.0, alpha: float = 0.5):
        self.model = model
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.modules: dict[str, torch.nn.Module] = {}
        self._handles: list[Any] = []
        self._module_stack: list[str] = []
        self._captured_smax: dict[str, torch.Tensor] = {}
        self._warned_unavailable = False
        self._installed = False
        self._install_hooks()

    @property
    def installed(self) -> bool:
        return self._installed

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._installed = False

    def _install_hooks(self) -> None:
        for name, module in self.model.named_modules():
            q_proj = getattr(module, "q_proj", None)
            k_proj = getattr(module, "k_proj", None)
            if (
                q_proj is not None
                and k_proj is not None
                and hasattr(q_proj, "weight")
                and hasattr(k_proj, "weight")
                and q_proj.weight.ndim == 2
                and k_proj.weight.ndim == 2
            ):
                self.modules[name] = module
                self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
                self._handles.append(module.register_forward_hook(self._make_post_hook(name)))
        self._installed = bool(self.modules)
        if not self._installed:
            logger.warning("QK-Clip disabled because no q_proj/k_proj attention modules were found")

    def _make_pre_hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._module_stack.append(name)

        return hook

    def _make_post_hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], _outputs: Any) -> None:
            if self._module_stack and self._module_stack[-1] == name:
                self._module_stack.pop()
            elif name in self._module_stack:
                self._module_stack.remove(name)

        return hook

    @contextlib.contextmanager
    def capture(self) -> Iterator[None]:
        """Capture pre-softmax logits during model forward calls."""

        if not self._installed:
            yield
            return

        original_f_softmax = F.softmax
        original_torch_softmax = torch.softmax
        controller = self

        def patched_f_softmax(input: torch.Tensor, *args: Any, **kwargs: Any):
            controller._maybe_record_softmax_input(input, args, kwargs)
            return original_f_softmax(input, *args, **kwargs)

        def patched_torch_softmax(input: torch.Tensor, *args: Any, **kwargs: Any):
            controller._maybe_record_softmax_input(input, args, kwargs)
            return original_torch_softmax(input, *args, **kwargs)

        F.softmax = patched_f_softmax
        torch.softmax = patched_torch_softmax
        try:
            yield
        finally:
            F.softmax = original_f_softmax
            torch.softmax = original_torch_softmax
            self._module_stack.clear()

    @contextlib.contextmanager
    def force_eager_attention(self, enabled: bool = True) -> Iterator[None]:
        """Temporarily ask HuggingFace attention modules to use eager attention.

        This is best-effort. Some models choose SDPA/FlashAttention classes at
        construction time and will ignore config changes. In that case
        ``capture`` simply records no logits and ``apply`` reports disabled.
        """

        if not enabled:
            yield
            return

        configs: list[Any] = []
        for module in [self.model, *self.modules.values()]:
            config = getattr(module, "config", None)
            if config is not None and not any(config is existing for existing in configs):
                configs.append(config)

        saved: list[tuple[Any, str, Any, bool]] = []
        for config in configs:
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(config, attr):
                    saved.append((config, attr, getattr(config, attr), True))
                    try:
                        setattr(config, attr, "eager")
                    except Exception:
                        logger.debug("Could not set %s on %s", attr, type(config).__name__)
                else:
                    saved.append((config, attr, None, False))

        try:
            yield
        finally:
            for config, attr, value, existed in reversed(saved):
                try:
                    if existed:
                        setattr(config, attr, value)
                except Exception:
                    logger.debug("Could not restore %s on %s", attr, type(config).__name__)

    @torch.no_grad()
    def probe(self, forward_closure: Callable[[], Any], *, force_eager: bool = True) -> QKClipApplyStats:
        """Run one clean forward to collect QK stats and then apply QK-Clip."""

        self.clear()
        if not self._installed:
            return self.apply()
        with self.force_eager_attention(force_eager):
            with self.capture():
                forward_closure()
        return self.apply()

    def _maybe_record_softmax_input(
        self,
        input_tensor: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not self._module_stack:
            return
        if not isinstance(input_tensor, torch.Tensor) or input_tensor.ndim != 4:
            return

        dim = kwargs.get("dim", args[0] if args else None)
        if dim not in (-1, input_tensor.ndim - 1):
            return

        module_name = self._module_stack[-1]
        with torch.no_grad():
            logits = input_tensor.detach()
            if not bool(torch.isfinite(logits).all().item()):
                logger.warning("QK-Clip skipped non-finite logits for %s", module_name)
                return
            per_head = logits.float().amax(dim=(0, 2, 3)).cpu()
            previous = self._captured_smax.get(module_name)
            self._captured_smax[module_name] = per_head if previous is None else torch.maximum(previous, per_head)

    @torch.no_grad()
    def apply(self) -> QKClipApplyStats:
        """Apply QK-Clip using captured pre-softmax logits."""

        if not self._installed:
            return QKClipApplyStats(
                enabled=False,
                exact_logits=False,
                fallback_used=False,
                disabled_reason="no q_proj/k_proj attention modules found",
                qk_smax_max=0.0,
                qk_clip_count=0,
            )

        if not self._captured_smax:
            reason = "QK-Clip disabled because pre-softmax logits are unavailable. Use attn_implementation='eager'."
            if not self._warned_unavailable:
                logger.warning(reason)
                self._warned_unavailable = True
            return QKClipApplyStats(
                enabled=False,
                exact_logits=False,
                fallback_used=False,
                disabled_reason=reason,
                qk_smax_max=0.0,
                qk_clip_count=0,
            )

        fallback_used = False
        clip_count = 0
        smax_max = 0.0

        for module_name, per_head_smax in self._captured_smax.items():
            module = self.modules.get(module_name)
            if module is None:
                continue
            smax_max = max(smax_max, float(per_head_smax.max().item()))
            needs_clip = per_head_smax > self.tau
            if not bool(needs_clip.any().item()):
                continue

            gamma = torch.clamp(self.tau / per_head_smax.clamp_min(1e-12), max=1.0)
            q_proj = module.q_proj
            k_proj = module.k_proj
            clipped, used_fallback = self._scale_qk(q_proj.weight, k_proj.weight, gamma, needs_clip)
            clip_count += clipped
            fallback_used = fallback_used or used_fallback

        self._captured_smax.clear()
        return QKClipApplyStats(
            enabled=True,
            exact_logits=True,
            fallback_used=fallback_used,
            disabled_reason=None,
            qk_smax_max=smax_max,
            qk_clip_count=clip_count,
        )

    def persistent_tensors(self) -> list[torch.Tensor]:
        return list(self._captured_smax.values())

    def clear(self) -> None:
        self._captured_smax.clear()

    @torch.no_grad()
    def _scale_qk(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        gamma: torch.Tensor,
        needs_clip: torch.Tensor,
    ) -> tuple[int, bool]:
        num_q_heads = int(gamma.numel())
        if num_q_heads <= 0 or q_weight.ndim != 2 or k_weight.ndim != 2:
            return self._scale_qk_fallback(q_weight, k_weight, gamma)

        if q_weight.shape[0] % num_q_heads != 0:
            return self._scale_qk_fallback(q_weight, k_weight, gamma)

        q_head_dim = q_weight.shape[0] // num_q_heads
        if q_head_dim <= 0 or k_weight.shape[0] % q_head_dim != 0:
            return self._scale_qk_fallback(q_weight, k_weight, gamma)

        num_k_heads = k_weight.shape[0] // q_head_dim
        if num_k_heads <= 0 or num_q_heads % num_k_heads != 0:
            return self._scale_qk_fallback(q_weight, k_weight, gamma)

        gamma = gamma.to(device=q_weight.device, dtype=torch.float32)
        needs_clip = needs_clip.to(device=q_weight.device)
        q_scale = gamma.pow(self.alpha).to(dtype=q_weight.dtype)

        for head in range(num_q_heads):
            if bool(needs_clip[head].item()):
                start = head * q_head_dim
                q_weight[start : start + q_head_dim].mul_(q_scale[head])

        group = num_q_heads // num_k_heads
        k_scales = []
        k_masks = []
        for k_head in range(num_k_heads):
            q_slice = slice(k_head * group, (k_head + 1) * group)
            k_scales.append(gamma[q_slice].min())
            k_masks.append(bool(needs_clip[q_slice].any().item()))

        k_scale_tensor = torch.stack(k_scales).pow(1.0 - self.alpha).to(dtype=k_weight.dtype)
        clipped_k = 0
        for k_head, should_clip in enumerate(k_masks):
            if should_clip:
                start = k_head * q_head_dim
                k_weight[start : start + q_head_dim].mul_(k_scale_tensor[k_head])
                clipped_k += 1

        return int(needs_clip.sum().item()) + clipped_k, False

    @torch.no_grad()
    def _scale_qk_fallback(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[int, bool]:
        global_gamma = torch.clamp(gamma.min(), max=1.0).to(device=q_weight.device, dtype=torch.float32)
        q_weight.mul_(global_gamma.pow(self.alpha).to(dtype=q_weight.dtype))
        k_weight.mul_(global_gamma.pow(1.0 - self.alpha).to(device=k_weight.device, dtype=k_weight.dtype))
        return 2, True
