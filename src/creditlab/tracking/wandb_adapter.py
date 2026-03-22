from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from creditlab.core.models import MetricRow
from creditlab.tracking.local import LocalTracker


class OptionalWandbTracker(LocalTracker):
    tracker_name = "wandb"

    def __init__(self, config: dict[str, Any], artifact_root: Path) -> None:
        super().__init__(artifact_root=artifact_root)
        self.config = config

    def log_metrics(self, run_id: str, rows: Sequence[MetricRow]) -> None:
        super().log_metrics(run_id, rows)
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        project = self.config.get("project", "creditlab")
        mode = self.config.get("mode", "disabled")
        with wandb.init(project=project, name=run_id, mode=mode, reinit=True):
            for row in rows:
                wandb.log({f"{row.scope}/{row.name}": row.value, **row.metadata})
