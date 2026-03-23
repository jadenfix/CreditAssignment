from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from creditlab.core.models import MachineProfile


def stable_hash(payload: dict[str, Any]) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:16]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def action_label(action: str) -> str:
    if ":" in action:
        return action
    return re.sub(r"_\d+$", "", action)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def try_command(args: list[str]) -> tuple[bool, str]:
    try:
        output = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        output = exc.output if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        return False, output.strip()
    return True, output.strip()


def collect_machine_profile() -> MachineProfile:
    docker_ok, docker_output = try_command(["docker", "info", "--format", "{{.ServerVersion}}"])
    gpu_ok, gpu_output = try_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    ctk_ok, _ = try_command(["nvidia-ctk", "--version"])
    gpu_names = [line.strip() for line in gpu_output.splitlines() if line.strip()] if gpu_ok else []
    return MachineProfile(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        docker_available=docker_ok,
        docker_runtime=docker_output if docker_ok else None,
        gpu_available=gpu_ok,
        gpu_name=", ".join(gpu_names) if gpu_names else None,
        gpu_count=len(gpu_names) if gpu_names else None,
        nvidia_ctk_available=ctk_ok,
    )


def render_simple_bar_svg(title: str, rows: list[tuple[str, float]], output_path: str | Path) -> Path:
    output = Path(output_path)
    width = 720
    height = 80 + 40 * max(len(rows), 1)
    max_value = max((value for _, value in rows), default=1.0) or 1.0
    bars: list[str] = [
        f'<text x="20" y="32" font-size="22" font-family="monospace">{title}</text>'
    ]
    for index, (label, value) in enumerate(rows):
        y = 60 + index * 34
        bar_width = int((value / max_value) * 460)
        bars.append(
            f'<text x="20" y="{y}" font-size="14" font-family="monospace">{label}</text>'
            f'<rect x="220" y="{y - 14}" width="{bar_width}" height="18" fill="#2952cc" />'
            f'<text x="{230 + bar_width}" y="{y}" font-size="13" font-family="monospace">{value:.3f}</text>'
        )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="#f8fafc" />'
        + "".join(bars)
        + "</svg>",
        encoding="utf-8",
    )
    return output

