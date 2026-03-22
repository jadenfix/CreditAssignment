from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from creditlab.core.interfaces import TrackerBackend
from creditlab.core.models import MetricRow


class LocalTracker(TrackerBackend):
    tracker_name = "local"

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)

    def log_metrics(self, run_id: str, rows: Sequence[MetricRow]) -> None:
        run_dir = self.artifact_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")

    def log_artifact(self, run_id: str, name: str, payload: dict[str, Any]) -> Path:
        run_dir = self.artifact_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = run_dir / f"{name}.json"
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return artifact_path

