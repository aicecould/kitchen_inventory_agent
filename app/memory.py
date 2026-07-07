"""Markdown user profile adapter."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    markdown: str
    allergens: list[str] = Field(default_factory=list)
    allergen_intolerances: list[str] = Field(default_factory=list)
    custom_allergens: list[str] = Field(default_factory=list)
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

    broad = sections.get("广义过敏原", [])
    custom = [*sections.get("自定义过敏食材", []), *sections.get("过敏原", [])]
    return UserProfile(
        markdown=markdown,
        allergens=list(dict.fromkeys([*broad, *custom])),
        allergen_intolerances=broad,
        custom_allergens=list(dict.fromkeys(custom)),
        preferences=sections.get("饮食偏好", []),
        history=sections.get("历史摘要", []),
    )


def read_user_profile(path: Path) -> UserProfile:
    return parse_user_profile(load_user_profile(path))


def write_user_allergens(
    path: Path,
    allergen_intolerances: list[str],
    custom_allergens: list[str],
) -> UserProfile:
    markdown = load_user_profile(path)
    target_sections = {"过敏原", "广义过敏原", "自定义过敏食材"}
    retained: list[str] = []
    skipping = False
    for line in markdown.splitlines():
        if line.startswith("## "):
            skipping = line[3:].strip() in target_sections
        if not skipping:
            retained.append(line)

    cleaned = "\n".join(retained).rstrip()
    broad_lines = [f"- {value}" for value in allergen_intolerances] or ["- 暂无"]
    custom_lines = [f"- {value}" for value in custom_allergens] or ["- 暂无"]
    updated = (
        f"{cleaned}\n\n## 广义过敏原\n\n"
        + "\n".join(broad_lines)
        + "\n\n## 自定义过敏食材\n\n"
        + "\n".join(custom_lines)
        + "\n"
    )
    path.write_text(updated, encoding="utf-8")
    return parse_user_profile(updated)
