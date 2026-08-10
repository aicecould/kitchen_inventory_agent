from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.profile import (
    ProfileFact,
    ProfileFormatError,
    ProfileRepository,
    ProfileRevisionError,
    ProfileState,
    UserProfile,
    parse_user_profile,
    render_user_profile,
)


NOW = datetime(2026, 8, 10, 4, 0, tzinfo=UTC)


def _fact(**overrides: object) -> ProfileFact:
    values: dict[str, object] = {
        "fact_id": "fact_" + "1" * 32,
        "category": "preference",
        "key": "ingredient.cilantro",
        "value": "不喜欢香菜",
        "origin": "user_confirmed",
        "status": "active",
        "evidence_ids": ["hist_" + "2" * 32],
        "reason": "用户确认",
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return ProfileFact.model_validate(values)


def test_legacy_profile_becomes_virtual_confirmed_facts() -> None:
    profile = parse_user_profile(
        "# 用户画像\n\n"
        "## 广义过敏原\n\n- Peanut\n\n"
        "## 自定义过敏食材\n\n- 腰果\n\n"
        "## 饮食偏好\n\n- 少辣\n"
    )

    assert profile.state.schema_version == 1
    assert profile.allergen_intolerances == ["Peanut"]
    assert profile.custom_allergens == ["腰果"]
    assert profile.preferences == ["少辣"]
    assert {fact.origin.value for fact in profile.state.facts} == {"user_confirmed"}


def test_v2_round_trip_projects_only_active_facts() -> None:
    state = ProfileState(
        revision=3,
        facts=[
            _fact(),
            _fact(
                fact_id="fact_" + "3" * 32,
                value="喜欢辣椒",
                status="superseded",
            ),
        ],
        last_summarized_segment_id="hist_" + "4" * 32,
    )
    profile = UserProfile.from_state(state)

    markdown = render_user_profile(profile)
    reparsed = parse_user_profile(markdown)

    assert "## 厨房情况" in markdown
    assert "## 必须要求" in markdown
    assert "## 默认情景" in markdown
    assert "不喜欢香菜" in markdown
    assert "喜欢辣椒" not in markdown.split("<!-- kitchen-profile-state:v2", 1)[0]
    assert reparsed.state == state
    assert reparsed.preferences == ["不喜欢香菜"]


def test_malformed_v2_state_is_rejected() -> None:
    markdown = (
        "# 用户画像\n\n## 饮食偏好\n\n- 少辣\n\n"
        "<!-- kitchen-profile-state:v2\n{not json}\n-->\n"
    )

    with pytest.raises(ProfileFormatError, match="invalid v2 profile state"):
        parse_user_profile(markdown)


def test_repository_legacy_read_does_not_rewrite_file(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    legacy = "# 用户画像\n\n## 饮食偏好\n\n- 少辣\n"
    path.write_text(legacy, encoding="utf-8")

    profile = ProfileRepository(path).read()

    assert profile.preferences == ["少辣"]
    assert path.read_text(encoding="utf-8") == legacy


def test_repository_requires_audit_and_expected_revision(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    repository = ProfileRepository(path)
    initial = repository.read()
    changed = initial.state.model_copy(
        update={"revision": 1, "facts": [_fact()]}, deep=True
    )

    with pytest.raises(PermissionError, match="audit"):
        repository.replace(
            expected_revision=0,
            state=changed,
            audit_passed=False,
        )

    saved = repository.replace(
        expected_revision=0,
        state=changed,
        audit_passed=True,
    )
    with pytest.raises(ProfileRevisionError, match="revision"):
        repository.replace(
            expected_revision=0,
            state=changed.model_copy(update={"revision": 2}),
            audit_passed=True,
        )

    assert saved.state.revision == 1
    assert repository.read().preferences == ["不喜欢香菜"]


def test_repository_preserves_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    malformed = "<!-- kitchen-profile-state:v2\n{}"
    path.write_text(malformed, encoding="utf-8")

    with pytest.raises(ProfileFormatError):
        ProfileRepository(path).read()

    assert path.read_text(encoding="utf-8") == malformed
