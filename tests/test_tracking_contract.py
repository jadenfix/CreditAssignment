from pathlib import Path

from creditlab.core.models import MetricRow
from creditlab.tracking.local import LocalTracker
from creditlab.tracking.wandb_adapter import OptionalWandbTracker


def test_local_and_optional_wandb_tracking_write_same_metric_schema(tmp_path: Path):
    rows = [
        MetricRow(run_id="run_123", scope="report", name="success", value=0.5, metadata={"rollouts": 4}),
        MetricRow(run_id="run_123", scope="report", name="avg_turns", value=2.0),
    ]
    local_root = tmp_path / "local"
    wandb_root = tmp_path / "wandb"

    LocalTracker(local_root).log_metrics("run_123", rows)
    OptionalWandbTracker(config={"project": "test"}, artifact_root=wandb_root).log_metrics("run_123", rows)

    local_lines = (local_root / "runs" / "run_123" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    wandb_lines = (wandb_root / "runs" / "run_123" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert local_lines == wandb_lines

