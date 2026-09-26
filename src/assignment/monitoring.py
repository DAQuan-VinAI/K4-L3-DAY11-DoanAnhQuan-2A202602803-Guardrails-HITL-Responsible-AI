"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def record(self, *, blocked: bool, rate_limited: bool = False,
               judge_checked: bool = False, judge_failed: bool = False):
        """Update counters after one request passes through the pipeline."""
        self.total_requests += 1
        self.blocked_requests += int(blocked)
        self.rate_limit_hits += int(rate_limited)
        self.judge_checks += int(judge_checked)
        self.judge_fails += int(judge_failed)

    def check_metrics(self) -> list[Alert]:
        """Compute rates and append Alert objects when thresholds are exceeded."""
        snap = self.snapshot()
        new_alerts = []
        if snap["block_rate"] > self.block_rate_threshold:
            new_alerts.append(Alert(
                metric="block_rate",
                value=snap["block_rate"],
                threshold=self.block_rate_threshold,
                message=(
                    f"Block rate {snap['block_rate']:.0%} exceeds "
                    f"{self.block_rate_threshold:.0%} — possible attack wave "
                    "or guardrails too strict."
                ),
            ))
        if self.rate_limit_hits > self.rate_limit_hit_threshold:
            new_alerts.append(Alert(
                metric="rate_limit_hits",
                value=self.rate_limit_hits,
                threshold=self.rate_limit_hit_threshold,
                message=(
                    f"{self.rate_limit_hits} rate-limit hits exceed "
                    f"{self.rate_limit_hit_threshold} — possible flooding / cost attack."
                ),
            ))
        if snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            new_alerts.append(Alert(
                metric="judge_fail_rate",
                value=snap["judge_fail_rate"],
                threshold=self.judge_fail_rate_threshold,
                message=(
                    f"Judge fail rate {snap['judge_fail_rate']:.0%} exceeds "
                    f"{self.judge_fail_rate_threshold:.0%} — model output quality degraded."
                ),
            ))
        self.alerts = new_alerts
        return new_alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
