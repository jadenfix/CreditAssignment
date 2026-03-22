from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FileFocus(BaseModel):
    path: str
    rationale: str | None = None


class CodeRepoState(BaseModel):
    repo_diff: dict[str, Any] = Field(default_factory=dict)
    failing_tests: list[str] = Field(default_factory=list)
    file_focus: list[FileFocus] = Field(default_factory=list)


class CodeVerifierSpec(BaseModel):
    name: str
    command: str
    produces_artifact: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

