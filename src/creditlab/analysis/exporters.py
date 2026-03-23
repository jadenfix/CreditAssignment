from __future__ import annotations

import csv
import json
from pathlib import Path

from creditlab.analysis.reporting import aggregate_run_group
from creditlab.core.utils import ensure_directory, render_simple_bar_svg


def export_results_bundle(run_group_id: str, store, output_dir: Path) -> dict:
    aggregate = aggregate_run_group(run_group_id, store)
    output = ensure_directory(output_dir / run_group_id)
    summaries = aggregate["condition_summaries"]
    runs = aggregate["run_metrics"]
    summary_csv = output / "condition_summaries.csv"
    run_csv = output / "run_metrics.csv"
    budget_json = output / "budget_parity.json"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summaries for key in row}))
        writer.writeheader()
        writer.writerows(summaries)
    with run_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in runs for key in row}))
        writer.writeheader()
        writer.writerows(runs)
    budget_json.write_text(json.dumps(aggregate["budget_parity"], indent=2, sort_keys=True), encoding="utf-8")

    success_rows = [
        (row["condition_name"], row["mean"])
        for row in summaries
        if row["metric_name"] == "success"
    ]
    change_rows = [
        (row["condition_name"], row["mean"])
        for row in summaries
        if row["metric_name"] == "change_rate"
    ]
    success_svg = render_simple_bar_svg("Success By Condition", success_rows, output / "success_by_condition.svg")
    change_svg = render_simple_bar_svg("Matched-Prefix Change Rate", change_rows, output / "change_rate.svg")
    markdown = output / "results_summary.md"
    markdown.write_text(
        "\n".join(
            [
                f"# Run Group {run_group_id}",
                "",
                "## Files",
                f"- `{summary_csv.name}`",
                f"- `{run_csv.name}`",
                f"- `{budget_json.name}`",
                f"- `{success_svg.name}`",
                f"- `{change_svg.name}`",
                "",
                "## Budget Parity",
                "```json",
                json.dumps(aggregate["budget_parity"], indent=2, sort_keys=True),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_group_id": run_group_id,
        "output_dir": str(output),
        "files": [str(summary_csv), str(run_csv), str(budget_json), str(success_svg), str(change_svg), str(markdown)],
    }


def render_paper_assets(run_group_id: str, store, manuscript_path: Path, output_dir: Path) -> dict:
    bundle = export_results_bundle(run_group_id, store, output_dir)
    manuscript = Path(manuscript_path)
    generated = ensure_directory(output_dir / run_group_id) / "generated_results.md"
    generated.write_text(
        "\n".join(
            [
                f"# Generated Results For {run_group_id}",
                "",
                f"- Condition summary CSV: `{Path(bundle['files'][0]).name}`",
                f"- Run metrics CSV: `{Path(bundle['files'][1]).name}`",
                f"- Success chart: `{Path(bundle['files'][3]).name}`",
                f"- Change-rate chart: `{Path(bundle['files'][4]).name}`",
                "",
                f"Base manuscript: `{manuscript}`",
            ]
        ),
        encoding="utf-8",
    )
    return {"bundle": bundle, "generated_results": str(generated)}
