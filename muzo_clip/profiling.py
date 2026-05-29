"""Optional profiling helpers for MuZO-Clip experiments."""

from __future__ import annotations

import contextlib
import csv
import time
from pathlib import Path
from typing import Any, Iterator

import torch


class PhaseProfiler:
    """Collect per-phase CPU/CUDA timing without changing default behavior.

    The profiler is disabled by default. When enabled, every phase context uses
    ``torch.profiler.record_function`` and optional NVTX ranges. CUDA timings
    use events and synchronize at phase end, so this is for diagnostics, not
    normal training throughput.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        csv_path: str | Path | None = None,
        use_cuda_events: bool = True,
        use_nvtx: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.csv_path = Path(csv_path) if csv_path else None
        self.use_cuda_events = bool(use_cuda_events)
        self.use_nvtx = bool(use_nvtx)
        self._rows: list[dict[str, Any]] = []
        self._current: dict[str, float] = {}

    def reset_step(self) -> None:
        self._current = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        cuda_enabled = self.use_cuda_events and torch.cuda.is_available()
        start_event = torch.cuda.Event(enable_timing=True) if cuda_enabled else None
        end_event = torch.cuda.Event(enable_timing=True) if cuda_enabled else None
        if self.use_nvtx and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
        cpu_start = time.perf_counter()
        try:
            with torch.profiler.record_function(name):
                if start_event is not None:
                    start_event.record()
                yield
                if end_event is not None:
                    end_event.record()
        finally:
            cpu_ms = (time.perf_counter() - cpu_start) * 1000.0
            self._current[f"{name}_cpu_ms"] = self._current.get(f"{name}_cpu_ms", 0.0) + cpu_ms
            if start_event is not None and end_event is not None:
                torch.cuda.synchronize()
                cuda_ms = float(start_event.elapsed_time(end_event))
                self._current[f"{name}_cuda_ms"] = self._current.get(f"{name}_cuda_ms", 0.0) + cuda_ms
            if self.use_nvtx and torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()

    def current_summary(self) -> dict[str, float]:
        return dict(self._current)

    def write_step(self, step: int, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.csv_path is None:
            return
        row: dict[str, Any] = {"step": int(step)}
        if extra:
            row.update(extra)
        row.update(self._current)
        self._rows.append(row)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for item in self._rows for key in item})
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)


@contextlib.contextmanager
def maybe_torch_profiler(
    *,
    enabled: bool,
    trace_dir: str | Path,
    wait: int = 5,
    warmup: int = 2,
    active: int = 5,
    repeat: int = 1,
) -> Iterator[Any | None]:
    """Create a scheduled torch.profiler only when explicitly enabled."""

    if not enabled:
        yield None
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_path)),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        yield profiler


def null_phase_profiler() -> PhaseProfiler:
    return PhaseProfiler(enabled=False)

