from __future__ import annotations

import json
from pathlib import Path

from creditlab.core.utils import collect_machine_profile, try_command


def build_host_report() -> dict:
    profile = collect_machine_profile().model_dump(mode="json")
    image_ok, _ = try_command(["docker", "image", "inspect", "nvidia/cuda:12.3.2-base-ubuntu22.04"])
    if profile["docker_available"] and profile["gpu_available"] and image_ok:
        cuda_ok, cuda_output = try_command(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.3.2-base-ubuntu22.04", "nvidia-smi"]
        )
    elif profile["docker_available"] and profile["gpu_available"]:
        cuda_ok, cuda_output = False, "CUDA image not present locally; run scripts/aws/validate_cuda_container.sh on the AWS host."
    else:
        cuda_ok, cuda_output = False, "Docker or NVIDIA GPU unavailable on this host."
    return {
        "machine_profile": profile,
        "docker_cuda_probe": {
            "ok": cuda_ok,
            "output": cuda_output,
        },
        "recommended_workspace": "/workspace/creditlab",
        "recommended_data_root": "/data/creditlab",
    }


def write_host_report(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_host_report(), indent=2, sort_keys=True), encoding="utf-8")
    return target
