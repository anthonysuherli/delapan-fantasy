"""
Performance profiling module for backtest runs.

Tracks execution time, memory usage, and other performance metrics
for different operations during the backtest.
"""

import time
import psutil
import os
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    memory_start_mb: float = 0.0
    memory_end_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    num_samples: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Calculate derived metrics."""
        if self.end_time:
            self.duration_seconds = self.end_time - self.start_time
        if self.memory_end_mb is not None:
            self.memory_delta_mb = self.memory_end_mb - self.memory_start_mb
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def throughput(self) -> Optional[float]:
        """Calculate throughput (samples per second)."""
        if self.duration_seconds and self.duration_seconds > 0 and self.num_samples:
            return self.num_samples / self.duration_seconds
        return None

    def __str__(self) -> str:
        """String representation."""
        parts = [f"{self.operation_name}"]
        if self.duration_seconds:
            parts.append(f"{self.duration_seconds:.2f}s")
        if self.memory_delta_mb is not None:
            parts.append(f"Δ{self.memory_delta_mb:+.1f}MB")
        if self.num_samples and self.throughput():
            parts.append(f"{self.throughput():.1f}/s")
        return " | ".join(parts)


class PerformanceProfiler:
    """Tracks performance metrics across backtest operations."""

    def __init__(self):
        self.operations: List[OperationMetrics] = []
        self.current_process = psutil.Process(os.getpid())
        self.backtest_start_time: Optional[float] = None
        self.backtest_end_time: Optional[float] = None

    def start_backtest(self):
        """Mark backtest start."""
        self.backtest_start_time = time.perf_counter()
        self._record_memory_snapshot("backtest_start")

    def end_backtest(self):
        """Mark backtest end."""
        self.backtest_end_time = time.perf_counter()
        self._record_memory_snapshot("backtest_end")

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.current_process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return 0.0

    def _record_memory_snapshot(self, label: str):
        """Record a memory snapshot."""
        try:
            mem_mb = self.get_memory_mb()
            logger.debug(f"Memory snapshot ({label}): {mem_mb:.1f}MB")
        except Exception as e:
            logger.debug(f"Could not record memory snapshot: {e}")

    @contextmanager
    def track(
        self,
        operation_name: str,
        num_samples: Optional[int] = None,
        **metadata
    ):
        """Context manager to track an operation."""
        mem_start = self.get_memory_mb()
        time_start = time.perf_counter()

        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=time_start,
            memory_start_mb=mem_start,
            num_samples=num_samples,
            metadata=metadata
        )

        try:
            yield metrics
        finally:
            metrics.end_time = time.perf_counter()
            metrics.memory_end_mb = self.get_memory_mb()
            metrics.memory_peak_mb = metrics.memory_end_mb
            metrics.finalize()
            self.operations.append(metrics)
            logger.debug(f"Profiled: {metrics}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all operations."""
        if not self.operations:
            return {"error": "No operations recorded"}

        total_time = (self.backtest_end_time or time.perf_counter()) - (
            self.backtest_start_time or self.operations[0].start_time
        )
        total_memory_delta = (
            sum(
                op.memory_delta_mb
                for op in self.operations
                if op.memory_delta_mb is not None
            )
            or 0
        )

        # Group by operation name
        op_groups: Dict[str, List[OperationMetrics]] = {}
        for op in self.operations:
            if op.operation_name not in op_groups:
                op_groups[op.operation_name] = []
            op_groups[op.operation_name].append(op)

        # Calculate aggregates
        op_summary = {}
        for op_name, ops in op_groups.items():
            durations = [op.duration_seconds for op in ops if op.duration_seconds]
            memory_deltas = [
                op.memory_delta_mb for op in ops if op.memory_delta_mb is not None
            ]
            sample_counts = [op.num_samples for op in ops if op.num_samples]

            op_summary[op_name] = {
                "count": len(ops),
                "total_time": sum(durations) if durations else 0,
                "mean_time": sum(durations) / len(durations) if durations else 0,
                "min_time": min(durations) if durations else 0,
                "max_time": max(durations) if durations else 0,
                "total_memory_delta": sum(memory_deltas) if memory_deltas else 0,
                "mean_memory_delta": sum(memory_deltas) / len(memory_deltas)
                if memory_deltas
                else 0,
                "total_samples": sum(sample_counts) if sample_counts else 0,
                "throughput": sum(sample_counts) / sum(durations)
                if sample_counts and sum(durations) > 0
                else 0,
            }

        return {
            "backtest_start": datetime.fromtimestamp(self.backtest_start_time).isoformat()
            if self.backtest_start_time
            else None,
            "backtest_end": datetime.fromtimestamp(self.backtest_end_time).isoformat()
            if self.backtest_end_time
            else None,
            "total_time": total_time,
            "total_memory_delta": total_memory_delta,
            "peak_memory": max(
                (op.memory_peak_mb for op in self.operations if op.memory_peak_mb),
                default=0,
            ),
            "operation_summary": op_summary,
            "num_operations": len(self.operations),
        }

    def format_report(self) -> str:
        """Format a human-readable performance report."""
        summary = self.get_summary()

        if "error" in summary:
            return summary["error"]

        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE REPORT")
        lines.append("=" * 80)

        if summary["backtest_start"]:
            lines.append(f"Start time: {summary['backtest_start']}")
        if summary["backtest_end"]:
            lines.append(f"End time: {summary['backtest_end']}")

        lines.append(f"Total duration: {self._format_time(summary['total_time'])}")
        lines.append(f"Peak memory: {summary['peak_memory']:.1f}MB")
        lines.append(f"Total memory delta: {summary['total_memory_delta']:+.1f}MB")
        lines.append(f"Total operations: {summary['num_operations']}")

        lines.append("")
        lines.append("Operation Summary (by type):")
        lines.append("-" * 80)
        lines.append(
            f"{'Operation':<30} {'Count':>8} {'Total':>12} {'Mean':>12} {'Throughput':>12}"
        )
        lines.append("-" * 80)

        for op_name, stats in sorted(
            summary["operation_summary"].items(), key=lambda x: x[1]["total_time"],
            reverse=True
        ):
            count = stats["count"]
            total = self._format_time(stats["total_time"])
            mean = self._format_time(stats["mean_time"])
            throughput = (
                f"{stats['throughput']:.1f}/s"
                if stats["throughput"] > 0
                else "N/A"
            )

            lines.append(
                f"{op_name:<30} {count:>8} {total:>12} {mean:>12} {throughput:>12}"
            )

        lines.append("")
        lines.append("Memory Usage by Operation:")
        lines.append("-" * 80)
        lines.append(
            f"{'Operation':<30} {'Count':>8} {'Mean Δ':>12} {'Total Δ':>12}"
        )
        lines.append("-" * 80)

        for op_name, stats in sorted(
            summary["operation_summary"].items(),
            key=lambda x: x[1]["total_memory_delta"],
            reverse=True,
        ):
            if stats["total_memory_delta"] != 0:
                count = stats["count"]
                mean_delta = f"{stats['mean_memory_delta']:+.1f}MB"
                total_delta = f"{stats['total_memory_delta']:+.1f}MB"
                lines.append(f"{op_name:<30} {count:>8} {mean_delta:>12} {total_delta:>12}")

        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.0f}s"

    def save_report(self, output_path: str):
        """Save performance report to file."""
        report = self.format_report()
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Performance report saved to {output_path}")

    def save_json(self, output_path: str):
        """Save performance metrics as JSON."""
        import json

        summary = self.get_summary()

        # Convert to serializable format
        data = {
            "backtest_start": summary.get("backtest_start"),
            "backtest_end": summary.get("backtest_end"),
            "total_time": summary.get("total_time"),
            "total_memory_delta": summary.get("total_memory_delta"),
            "peak_memory": summary.get("peak_memory"),
            "num_operations": summary.get("num_operations"),
            "operation_summary": summary.get("operation_summary", {}),
            "operations": [op.to_dict() for op in self.operations],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Performance metrics saved to {output_path}")
