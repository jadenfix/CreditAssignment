from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any


def stable_hash(payload: dict[str, Any]) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:16]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def action_label(action: str) -> str:
    if ":" in action:
        return action
    return re.sub(r"_\d+$", "", action)
