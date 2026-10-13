#!/usr/bin/env python3
"""Generate the source-backed stochastic seed integrity demo.

The displayed paths are executed through the repository's real manifest adapter.
No network service, model provider, or precomputed benchmark result is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from creditlab.core.config import load_experiment_config  # noqa: E402
from creditlab.core.models import ActionEvent  # noqa: E402
from creditlab.core.registry import create_env  # noqa: E402

WIDTH, HEIGHT = 1280, 720
FPS = 15
DURATION_SECONDS = 12
SEEDS = (0, 1)
SOURCE_INPUTS = (
    "src/creditlab/analysis/sweep.py",
    "src/creditlab/core/registry.py",
    "src/creditlab/envs/manifest_graph.py",
    "benchmarks/stochastic/v1/tasks.yaml",
    "scripts/generate_seed_integrity_demo.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def collect_trace(seed: int) -> list[dict[str, object]]:
    config = load_experiment_config(REPO_ROOT / "configs" / "stochastic_paper_local.yaml")
    config.env.config["seed"] = seed
    environment = create_env(config)
    task = environment.list_tasks("train")[0]
    state = environment.reset(task)
    trace: list[dict[str, object]] = []
    for turn in range(8):
        action_name = str(state.payload["oracle_action"])
        action = ActionEvent(action_id=f"seed-{seed}-{turn}", action_type="tool", content=action_name)
        next_state, observation, terminal = environment.step(task, state, action)
        trace.append(
            {
                "turn": turn + 1,
                "from": state.payload["raw_node"],
                "action": action_name,
                "to": next_state.payload["raw_node"],
                "observation": observation,
                "terminal": terminal,
            }
        )
        state = next_state
        if terminal:
            break
    return trace


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, radius: int = 22, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 1)


def gradient() -> Image.Image:
    # A low-resolution field scaled once is visually smooth and keeps reproducible
    # media generation comfortably fast on CPU-only machines.
    sample_width, sample_height = 80, 45
    image = Image.new("RGB", (sample_width, sample_height), "#07111f")
    pixels = image.load()
    for y in range(sample_height):
        for x in range(sample_width):
            glow = max(0.0, 1.0 - (((x - 66) / 44) ** 2 + ((y - 6) / 33) ** 2))
            pixels[x, y] = (7 + int(10 * glow), 17 + int(22 * glow), 31 + int(37 * glow))
    return image.resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)


BACKGROUND = gradient()


def draw_header(draw: ImageDraw.ImageDraw, kicker: str, title: str, subtitle: str) -> None:
    draw.text((74, 55), kicker.upper(), font=font(18, True), fill="#5eead4")
    title_size = 46
    title_font = font(title_size, True)
    while draw.textbbox((0, 0), title, font=title_font)[2] > WIDTH - 148 and title_size > 30:
        title_size -= 2
        title_font = font(title_size, True)
    draw.text((74, 90), title, font=title_font, fill="#f8fafc")
    draw.text((76, 151), subtitle, font=font(21), fill="#a8b7cb")


def draw_pipeline(draw: ImageDraw.ImageDraw, active: int) -> None:
    labels = ("Sweep seed", "Environment RNG", "Policy fallback", "Recorded path")
    for index, label in enumerate(labels):
        x = 92 + index * 292
        color = "#2dd4bf" if index <= active else "#1e3448"
        rounded(draw, (x, 248, x + 218, 330), "#0d1d2e", 18, color)
        draw.text((x + 24, 269), f"0{index + 1}", font=font(15, True), fill=color)
        draw.text((x + 24, 294), label, font=font(17, True), fill="#f8fafc")
        if index < len(labels) - 1:
            draw.line((x + 226, 289, x + 278, 289), fill=color, width=4)
            draw.polygon([(x + 278, 289), (x + 266, 282), (x + 266, 296)], fill=color)


def draw_trace_card(draw: ImageDraw.ImageDraw, x: int, y: int, seed: int, trace: list[dict[str, object]], visible: int, accent: str) -> None:
    rounded(draw, (x, y, x + 535, y + 350), "#0b1a2a", 24, "#20384c")
    draw.text((x + 30, y + 24), f"SEED {seed}", font=font(19, True), fill=accent)
    draw.text((x + 30, y + 56), "actual manifest transitions", font=font(16), fill="#91a6bb")
    for index, step in enumerate(trace[:visible]):
        row_y = y + 102 + index * 39
        succeeded = step["from"] != step["to"]
        dot = accent if succeeded else "#fb7185"
        draw.ellipse((x + 31, row_y + 4, x + 43, row_y + 16), fill=dot)
        text = f"{step['from']}  →  {step['to']}"
        draw.text((x + 57, row_y), text, font=font(17, True), fill="#e7eef7")
        draw.text((x + 337, row_y + 1), "advance" if succeeded else "retry", font=font(15), fill=dot)
    if visible >= len(trace):
        draw.text((x + 30, y + 316), f"{len(trace)} turns • terminal success", font=font(16, True), fill="#c7d5e3")


def make_frame(frame_index: int, traces: dict[int, list[dict[str, object]]], source_sha: str) -> Image.Image:
    seconds = frame_index / FPS
    image = BACKGROUND.copy()
    draw = ImageDraw.Draw(image)
    if seconds < 2.2:
        draw_header(draw, "CreditLab · provider-free fixture", "Seeds now reach every random branch", "One sweep seed controls both environment transitions and fallback policy choices.")
        draw_pipeline(draw, min(3, int(seconds / 0.5)))
        rounded(draw, (92, 401, 1188, 582), "#0b1a2a", 24, "#20384c")
        draw.text((124, 435), "Determinism contract", font=font(20, True), fill="#5eead4")
        draw.text((124, 480), "same seed  →  same path", font=font(31, True), fill="#f8fafc")
        draw.text((657, 480), "different seed  →  independent path", font=font(31, True), fill="#f8fafc")
    elif seconds < 8.7:
        draw_header(draw, "Executed against stochastic/v1", "Two reproducible paths, not two labels", "Each row below is produced by GraphManifestEnv.step() at the pinned source.")
        progress = (seconds - 2.2) / 6.5
        visible0 = min(len(traces[0]), max(1, int(progress * (len(traces[0]) + 1))))
        visible1 = min(len(traces[1]), max(1, int(progress * (len(traces[1]) + 1))))
        draw_trace_card(draw, 74, 222, 0, traces[0], visible0, "#38bdf8")
        draw_trace_card(draw, 671, 222, 1, traces[1], visible1, "#a78bfa")
    else:
        draw_header(draw, "Verified receipt", "Seeded paths are reproducible and independent.", "The demo uses local fixture data only—no model, provider, or paper-performance claim.")
        rounded(draw, (74, 232, 1206, 555), "#0b1a2a", 24, "#20384c")
        facts = (
            ("seed 0 replay", "MATCH", "#38bdf8"),
            ("seed 1 replay", "MATCH", "#a78bfa"),
            ("cross-seed paths", "DISTINCT", "#5eead4"),
            ("source", source_sha[:12], "#fbbf24"),
        )
        for index, (label, value, color) in enumerate(facts):
            x = 107 + (index % 2) * 545
            y = 271 + (index // 2) * 128
            draw.text((x, y), label.upper(), font=font(16, True), fill="#91a6bb")
            draw.text((x, y + 33), value, font=font(36, True), fill=color)
        draw.text((76, 638), "ACTUAL WORKFLOW  •  SOURCE-PINNED  •  OFFLINE", font=font(16, True), fill="#5eead4")
    draw.text((1108, 665), "creditlab", font=font(17, True), fill="#627b91")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "media" / "demo")
    parser.add_argument("--source-sha", default=None, help="Commit containing the captured runtime sources")
    parser.add_argument("--reuse-frames", action="store_true", help="Reuse a complete existing frame set")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = args.source_sha or git("rev-parse", "HEAD")
    git("cat-file", "-e", f"{source_sha}^{{commit}}")

    traces = {seed: collect_trace(seed) for seed in SEEDS}
    replays = {seed: collect_trace(seed) for seed in SEEDS}
    if traces != replays:
        raise RuntimeError("same-seed replay was not deterministic")
    if traces[0] == traces[1]:
        raise RuntimeError("demo seeds unexpectedly produced identical paths")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    expected_frames = DURATION_SECONDS * FPS
    existing_frames = sorted(frames_dir.glob("frame-*.png"))
    if not args.reuse_frames or len(existing_frames) != expected_frames:
        for frame in existing_frames:
            frame.unlink()
        for frame_index in range(expected_frames):
            make_frame(frame_index, traces, source_sha).save(frames_dir / f"frame-{frame_index:04d}.png")

    mp4_path = output_dir / "seed-integrity.mp4"
    gif_path = output_dir / "seed-integrity.gif"
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "frame-%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20", str(mp4_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "frame-%04d.png"), "-vf",
            "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=160[p];[s1][p]paletteuse=dither=bayer",
            "-loop", "0", str(gif_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    source_blobs = {path: git("rev-parse", f"{source_sha}:{path}") for path in SOURCE_INPUTS}
    receipt = {
        "schema_version": 1,
        "scope": "provider-free stochastic manifest fixture; no model or paper-performance claim",
        "source_commit": source_sha,
        "source_tree": git("rev-parse", f"{source_sha}^{{tree}}"),
        "source_blobs": source_blobs,
        "benchmark": "stochastic/v1",
        "task_id": "stoch_gate_train_01",
        "seeds": list(SEEDS),
        "same_seed_replay_matches": traces == replays,
        "cross_seed_paths_distinct": traces[0] != traces[1],
        "traces": {str(seed): trace for seed, trace in traces.items()},
        "artifacts": {
            mp4_path.name: {"sha256": sha256(mp4_path), "bytes": mp4_path.stat().st_size},
            gif_path.name: {"sha256": sha256(gif_path), "bytes": gif_path.stat().st_size},
        },
    }
    (output_dir / "capture.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for frame in frames_dir.glob("*.png"):
        frame.unlink()
    frames_dir.rmdir()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
