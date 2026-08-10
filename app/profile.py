"""Versioned, bounded user-profile documents backed by Markdown."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import fcntl
from pydantic import BaseModel, Field, model_validator


DEFAULT_USER_PROFILE_TEMPLATE = (
    Path(__file__).resolve().parent / "templates" / "user_profile.md"
)
_STATE_START = "<!-- kitchen-profile-state:v2"
_STATE_PATTERN = re.compile(
    r"<!-- kitchen-profile-state:v2\s*\n(?P<json>.*?)\n-->\s*$",
    re.DOTALL,
)
_ID_PATTERN = re.compile(r"^(?:fact|prof|hist)_[0-9a-f]{32}$")
_CATEGORY_HEADINGS = {
    "kitchen_setup": "厨房情况",
    "preference": "饮食偏好",
    "requirement": "必须要求",
    "default_context": "默认情景",
}
_PROCESS_LOCKS: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
_LEGACY_TIME = datetime(1970, 1, 1, tzinfo=UTC)


class ProfileFormatError(ValueError):
    """The Markdown profile contains invalid machine state."""


class ProfileRevisionError(ValueError):
    """A profile mutation was built from a stale revision."""


class ProfileCategory(StrEnum):
    KITCHEN_SETUP = "kitchen_setup"
    PREFERENCE = "preference"
    REQUIREMENT = "requirement"
    DEFAULT_CONTEXT = "default_context"


class ProfileOrigin(StrEnum):
    USER_CONFIRMED = "user_confirmed"
    INFERRED = "inferred"


class ProfileFactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class ProfileProposalStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ProfileFact(BaseModel):
    fact_id: str
    category: ProfileCategory
    key: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=300)
    origin: ProfileOrigin
    status: ProfileFactStatus = ProfileFactStatus.ACTIVE
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=500)
    created_at: datetime
    updated_at: datetime
    replaced_by: str | None = None

    @model_validator(mode="after")
    def validate_fact(self) -> "ProfileFact":
        if not re.fullmatch(r"fact_[0-9a-f]{32}", self.fact_id):
            raise ValueError("invalid fact_id")
        _validate_text(self.key, "fact key")
        _validate_text(self.value, "fact value")
        _validate_evidence(self.evidence_ids)
        _validate_aware(self.created_at)
        _validate_aware(self.updated_at)
        return self


class ProfileProposal(BaseModel):
    proposal_id: str
    user_id: str = Field(min_length=1, max_length=120)
    operation: Literal["upsert", "replace", "delete"]
    category: ProfileCategory
    key: str = Field(min_length=1, max_length=120)
    value: str | None = Field(default=None, max_length=300)
    conflicting_fact_id: str | None = None
    source: Literal["explicit_user_statement", "autonomous_summary"]
    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    reason: str = Field(min_length=1, max_length=500)
    status: ProfileProposalStatus = ProfileProposalStatus.PENDING
    base_revision: int = Field(ge=0)
    resolved_revision: int | None = Field(default=None, ge=0)
    created_at: datetime
    expires_at: datetime
    resolved_at: datetime | None = None
    preview: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_proposal(self) -> "ProfileProposal":
        if not re.fullmatch(r"prof_[0-9a-f]{32}", self.proposal_id):
            raise ValueError("invalid proposal_id")
        _validate_text(self.key, "proposal key")
        if self.value is not None:
            _validate_text(self.value, "proposal value")
        _validate_evidence(self.evidence_ids)
        _validate_aware(self.created_at)
        _validate_aware(self.expires_at)
        if self.resolved_at is not None:
            _validate_aware(self.resolved_at)
        if self.operation != "delete" and self.value is None:
            raise ValueError("profile upserts require a value")
        return self


class ProfileAuditEvent(BaseModel):
    operation: str = Field(min_length=1, max_length=80)
    status: Literal["passed", "failed"]
    reason: str = Field(default="", max_length=500)
    created_at: datetime
    subject_id: str | None = None


class ProfileState(BaseModel):
    schema_version: Literal[1, 2] = 2
    revision: int = Field(default=0, ge=0)
    facts: list[ProfileFact] = Field(default_factory=list)
    proposals: list[ProfileProposal] = Field(default_factory=list)
    audit_events: list[ProfileAuditEvent] = Field(default_factory=list, max_length=200)
    last_summarized_segment_id: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "ProfileState":
        active: set[tuple[ProfileCategory, str]] = set()
        counts: defaultdict[ProfileCategory, int] = defaultdict(int)
        for fact in self.facts:
            if fact.status != ProfileFactStatus.ACTIVE:
                continue
            identity = (fact.category, fact.key.casefold())
            if identity in active:
                raise ValueError("duplicate active profile fact key")
            active.add(identity)
            counts[fact.category] += 1
            if counts[fact.category] > 20:
                raise ValueError("profile category exceeds 20 active facts")
        if self.last_summarized_segment_id is not None and not re.fullmatch(
            r"hist_[0-9a-f]{32}", self.last_summarized_segment_id
        ):
            raise ValueError("invalid summary cursor")
        return self


class UserProfile(BaseModel):
    markdown: str
    state: ProfileState
    allergens: list[str] = Field(default_factory=list)
    allergen_intolerances: list[str] = Field(default_factory=list)
    custom_allergens: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    kitchen_setup: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    default_context: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: ProfileState) -> "UserProfile":
        profile = _profile_from_state(state, markdown="")
        markdown = render_user_profile(profile)
        return profile.model_copy(update={"markdown": markdown})


class ProfileRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock_path = path.with_name(f".{path.name}.lock")

    def read(self) -> UserProfile:
        if not self.path.exists():
            template = DEFAULT_USER_PROFILE_TEMPLATE.read_text(encoding="utf-8")
            return parse_user_profile(template)
        return parse_user_profile(self.path.read_text(encoding="utf-8"))

    def replace(
        self,
        *,
        expected_revision: int,
        state: ProfileState,
        audit_passed: bool,
    ) -> UserProfile:
        if not audit_passed:
            raise PermissionError("profile write requires a passed audit")
        if state.revision != expected_revision + 1:
            raise ProfileRevisionError("new profile revision must increment by one")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        process_lock = _PROCESS_LOCKS[str(self.path.resolve())]
        with process_lock:
            self._lock_path.touch(exist_ok=True)
            with self._lock_path.open("r+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                current = self.read()
                if current.state.revision != expected_revision:
                    raise ProfileRevisionError("profile revision has changed")
                profile = UserProfile.from_state(
                    state.model_copy(update={"schema_version": 2}, deep=True)
                )
                parsed = parse_user_profile(profile.markdown)
                if parsed.state != profile.state:
                    raise ProfileFormatError("rendered profile failed round-trip validation")
                _atomic_write(self.path, profile.markdown)
                return parsed


def load_user_profile(path: Path) -> str:
    if not path.exists():
        return DEFAULT_USER_PROFILE_TEMPLATE.read_text(encoding="utf-8")
    return path.read_text(encoding="utf-8")


def read_user_profile(path: Path) -> UserProfile:
    return parse_user_profile(load_user_profile(path))


def parse_user_profile(markdown: str) -> UserProfile:
    state_match = _STATE_PATTERN.search(markdown)
    if _STATE_START in markdown and state_match is None:
        raise ProfileFormatError("invalid v2 profile state: unterminated block")
    if state_match is not None:
        try:
            raw_state = json.loads(state_match.group("json"))
            state = ProfileState.model_validate(raw_state)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProfileFormatError("invalid v2 profile state") from exc
        if state.schema_version != 2:
            raise ProfileFormatError("invalid v2 profile state: schema mismatch")
        return _profile_from_state(state, markdown=markdown)

    sections = _parse_sections(markdown)
    facts: list[ProfileFact] = []
    broad = sections.get("广义过敏原", [])
    custom = [*sections.get("自定义过敏食材", []), *sections.get("过敏原", [])]
    legacy_categories = [
        (ProfileCategory.REQUIREMENT, "allergen.broad", broad),
        (ProfileCategory.REQUIREMENT, "allergen.custom", custom),
        (ProfileCategory.PREFERENCE, "preference", sections.get("饮食偏好", [])),
        (ProfileCategory.KITCHEN_SETUP, "kitchen", sections.get("厨房情况", [])),
        (ProfileCategory.DEFAULT_CONTEXT, "default", sections.get("默认情景", [])),
    ]
    for category, prefix, values in legacy_categories:
        for index, value in enumerate(dict.fromkeys(values)):
            token = _legacy_id(f"{category}:{prefix}:{value}:{index}")
            facts.append(
                ProfileFact(
                    fact_id=f"fact_{token}",
                    category=category,
                    key=f"{prefix}.{_slug(value)}",
                    value=value,
                    origin=ProfileOrigin.USER_CONFIRMED,
                    evidence_ids=[],
                    reason="legacy profile migration",
                    created_at=_LEGACY_TIME,
                    updated_at=_LEGACY_TIME,
                )
            )
    state = ProfileState(schema_version=1, facts=facts)
    profile = _profile_from_state(state, markdown=markdown)
    return profile.model_copy(update={"history": sections.get("历史摘要", [])})


def render_user_profile(profile: UserProfile) -> str:
    state = profile.state.model_copy(update={"schema_version": 2}, deep=True)
    lines = ["# 用户画像", ""]
    for category, heading in _CATEGORY_HEADINGS.items():
        values = [
            fact.value
            for fact in state.facts
            if fact.category.value == category
            and fact.status == ProfileFactStatus.ACTIVE
        ]
        lines.extend([f"## {heading}", ""])
        lines.extend([f"- {value}" for value in values] or ["- 暂无"])
        lines.append("")
    payload = json.dumps(state.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    lines.extend([_STATE_START, payload, "-->", ""])
    return "\n".join(lines)


def profile_without_legacy_history(markdown: str) -> str:
    state_match = _STATE_PATTERN.search(markdown)
    if state_match is not None:
        markdown = markdown[: state_match.start()].rstrip()
    retained: list[str] = []
    skipping = False
    for line in markdown.splitlines():
        if line.startswith("## "):
            skipping = line[3:].strip() == "历史摘要"
        if not skipping:
            retained.append(line)
    return "\n".join(retained).rstrip()


def _profile_from_state(state: ProfileState, *, markdown: str) -> UserProfile:
    active = [fact for fact in state.facts if fact.status == ProfileFactStatus.ACTIVE]
    broad = [
        fact.value
        for fact in active
        if fact.category == ProfileCategory.REQUIREMENT
        and fact.key.startswith("allergen.broad.")
    ]
    custom = [
        fact.value
        for fact in active
        if fact.category == ProfileCategory.REQUIREMENT
        and fact.key.startswith("allergen.custom.")
    ]
    values_by_category = {
        category: [fact.value for fact in active if fact.category == category]
        for category in ProfileCategory
    }
    return UserProfile(
        markdown=markdown,
        state=state,
        allergens=list(dict.fromkeys([*broad, *custom])),
        allergen_intolerances=broad,
        custom_allergens=custom,
        preferences=values_by_category[ProfileCategory.PREFERENCE],
        kitchen_setup=values_by_category[ProfileCategory.KITCHEN_SETUP],
        requirements=values_by_category[ProfileCategory.REQUIREMENT],
        default_context=values_by_category[ProfileCategory.DEFAULT_CONTEXT],
    )


def _parse_sections(markdown: str) -> dict[str, list[str]]:
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
    return sections


def _validate_text(value: str, label: str) -> None:
    if value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be one trimmed line")


def _validate_evidence(values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError("profile evidence IDs must be unique")
    if any(not re.fullmatch(r"hist_[0-9a-f]{32}", value) for value in values):
        raise ValueError("invalid profile evidence ID")


def _validate_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("profile timestamps must be timezone-aware")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or _legacy_id(value)[:12]


def _legacy_id(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _atomic_write(path: Path, content: str) -> None:
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)
