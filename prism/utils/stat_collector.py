import os
from typing import Optional

import torch


class StatCollector:
    def __init__(
        self,
        collect_density: bool = False,
        collect_select_time: bool = False,
        method_name: str = "prism",
    ) -> None:
        self.collect_density = collect_density
        self.collect_select_time = collect_select_time
        self.method_name = method_name
        self._densities = []
        self._events = []

    @classmethod
    def from_env(cls, method_name: str = "prism") -> "StatCollector":
        collect_density = os.environ.get("COLLECT_DENSITY", "false").lower() in ("1", "true", "yes")
        collect_select_time = os.environ.get("COLLECT_SELECT_TIME", "false").lower() in ("1", "true", "yes")
        return cls(
            collect_density=collect_density,
            collect_select_time=collect_select_time,
            method_name=method_name,
        )

    def reset(self) -> None:
        self._densities = []
        self._events = []

    def reset_density(self) -> None:
        self._densities = []

    def reset_select_time(self) -> None:
        self._events = []

    def maybe_start_select_timer(self, query_states: torch.Tensor):
        if self.collect_select_time and query_states.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end
        return None, None

    def finish_select_timer(self, start, end) -> None:
        if start is None or end is None:
            return
        end.record()
        self._events.append((start, end))

    def drain_select_time_ms(self) -> float:
        if not self._events:
            return 0.0
        total_ms = 0.0
        for start, end in self._events:
            total_ms += start.elapsed_time(end)
        self._events = []
        return total_ms

    def record_block_mask(self, block_mask: torch.Tensor) -> None:
        q_block_num = block_mask.shape[2]
        kv_block_num = block_mask.shape[3]
        num_causal_blocks_per_head = q_block_num * (kv_block_num - q_block_num) + (q_block_num * (q_block_num + 1)) // 2
        total_causal_numel = block_mask.shape[0] * block_mask.shape[1] * num_causal_blocks_per_head
        if total_causal_numel <= 0:
            return
        density = block_mask.sum().item() / total_causal_numel
        self._densities.append(density)

    def summary(self) -> Optional[dict]:
        if not self._densities:
            return None
        total = len(self._densities)
        avg = sum(self._densities) / total
        return {
            "method": self.method_name,
            "total": avg,
            "count": total,
        }

    def collect(self, fn):
        """Decorator for estimate functions that return a block_mask."""
        def wrapped(*args, **kwargs):
            query_states = args[0] if args else None
            start, end = self.maybe_start_select_timer(query_states)
            block_mask = fn(*args, **kwargs)
            self.finish_select_timer(start, end)
            if self.collect_density:
                self.record_block_mask(block_mask)
            return block_mask

        return wrapped
