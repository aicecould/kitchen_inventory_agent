"""Read-only Markdown user profile adapter."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    markdown: str
    allergens: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)


def load_user_profile(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"User profile not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_user_profile(markdown: str) -> UserProfile:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
        elif current and line.startswith("- "):
            value = line[2:].strip()
            if value and value not in {"待填写", "暂无"}:
                sections[current].append(value)

    return UserProfile(
        markdown=markdown,
        allergens=sections.get("过敏原", []),
        preferences=sections.get("饮食偏好", []),
        history=sections.get("历史摘要", []),
    )


def read_user_profile(path: Path) -> UserProfile:
    return parse_user_profile(load_user_profile(path))
