# Long-term User Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded Markdown-backed long-term user profile with explicit confirmation, conversation evidence, autonomous conflict-aware updates, and the existing audit pipeline.

**Architecture:** A new `app/profile.py` owns the versioned Markdown document and atomic repository, while `app/profile_actions.py` applies typed mutations through reusable validators added to `app/audit.py`. The Agent can create proposals, a structured summarizer processes each ten eligible history rows, and the Pipeline injects only active facts and exposes minimal inline confirmation cards.

**Tech Stack:** Python 3.13, Pydantic 2, SQLite, LangChain/ChatOpenAI, FastAPI, browser JavaScript, pytest.

## Global Constraints

- `user_profile.md` remains the only profile-state artifact; do not create a profile SQLite table or JSON sidecar.
- Explicit durable statements require a second user click before becoming active.
- Autonomous changes may silently replace only inferred facts; mandatory requirements and conflicts with confirmed facts become proposals.
- Every profile write extends the existing `app/audit.py` and tool-history architecture.
- Maximums are 20 active facts per category, 5 evidence IDs per item, 30-day proposal expiry, and 200 audit events.
- Conversation history remains permanent and is the canonical evidence store.
- Use test-driven development: every production behavior must first be observed failing for the intended reason.
- Preserve the existing allergen API and update the existing draft PR rather than opening another PR.

---

## File Map

- Create `app/profile.py`: profile enums/models, v1/v2 parsing, projection rendering, file locking, and atomic writes.
- Create `app/profile_actions.py`: proposal lifecycle, inferred batch updates, capacity/conflict rules, and audited mutations.
- Create `app/profile_summary.py`: structured summary candidate schema and ten-record summarization orchestration.
- Modify `app/memory.py`: history ID reservation, evidence lookup, and records-after-cursor query; remove profile implementation while re-exporting compatibility names.
- Modify `app/audit.py`: profile operation allowlist, preview/result validation, and canonical profile proposal output.
- Modify `app/context.py`: current segment ID and structured profile proposal response fields.
- Modify `app/agent.py`: `propose_profile_change` tool and profile service dependency.
- Modify `app/prompts.py`: durable-memory and untrusted-history rules.
- Modify `app/pipeline.py`: service construction, reserved history IDs, summary trigger, profile reload, and confirmation recording.
- Modify `app/web.py`: profile confirm/reject endpoints and audited allergen batch writes.
- Modify `frontend/index.html`, `frontend/app.js`, and `frontend/styles.css`: one inline proposal-card container and interaction.
- Create `tests/test_profile.py`, `tests/test_profile_actions.py`, and `tests/test_profile_summary.py`.
- Modify `tests/test_memory.py`, `tests/test_audit.py`, `tests/test_agent_allergens.py`, `tests/test_pipeline.py`, `tests/test_web_limits.py`, and `tests/test_frontend_markdown.py`.

---

### Task 1: Versioned Markdown Profile Document

**Files:**
- Create: `app/profile.py`
- Modify: `app/memory.py`
- Create: `tests/test_profile.py`
- Modify: `tests/test_memory.py`

**Interfaces:**
- Produces `ProfileCategory`, `ProfileOrigin`, `ProfileFactStatus`, `ProfileProposalStatus`, `ProfileFact`, `ProfileProposal`, `ProfileAuditEvent`, `ProfileState`, and expanded `UserProfile` Pydantic models.
- Produces `parse_user_profile(markdown: str) -> UserProfile`, `render_user_profile(profile: UserProfile) -> str`, `read_user_profile(path: Path) -> UserProfile`, and `ProfileRepository(path: Path)`.
- `ProfileRepository.read() -> UserProfile` performs v1 virtual migration without rewriting an existing file.
- `ProfileRepository.replace(expected_revision: int, state: ProfileState, audit_passed: bool) -> UserProfile` rejects unaudited or stale writes and performs a locked atomic replacement.

**Concrete test contract:**

```python
def test_v2_round_trip_hides_inactive_state() -> None:
    profile = UserProfile.from_state(ProfileState(
        revision=3,
        facts=[
            ProfileFact(fact_id="fact_" + "1" * 32, category="preference",
                        key="ingredient.cilantro", value="不喜欢香菜",
                        origin="user_confirmed", status="active",
                        evidence_ids=["hist_" + "2" * 32], reason="用户确认"),
        ],
    ))
    reparsed = parse_user_profile(render_user_profile(profile))
    assert reparsed.state == profile.state
    assert reparsed.preferences == ["不喜欢香菜"]
```

- [ ] **Step 1: Write failing parsing and rendering tests**

Add literal fixtures proving that v1 allergens/preferences become virtual `user_confirmed` facts, v2 hidden JSON round-trips all metadata, readable output contains only active facts in four sections, and malformed v2 raises `ProfileFormatError` without rewriting the file.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_profile.py tests/test_memory.py -q`

Expected: collection or assertion failures because `app.profile` and v2 types do not exist.

- [ ] **Step 3: Implement the domain models and pure parser/renderer**

Use exact enum values from the design and an HTML comment beginning `<!-- kitchen-profile-state:v2`. Validate one active fact per category/key, evidence lists of at most five IDs, category limits, and the single-line/value length contracts before rendering.

- [ ] **Step 4: Verify pure behavior GREEN**

Run: `pytest tests/test_profile.py -q`

Expected: all profile parser/renderer tests pass.

- [ ] **Step 5: Add failing repository tests**

Cover audited initialization, no-write legacy reads, expected revision conflicts, refusal when `audit_passed=False`, atomic replacement, and preservation of malformed input.

- [ ] **Step 6: Verify repository RED**

Run: `pytest tests/test_profile.py -q`

Expected: repository tests fail because locked audited writes are missing.

- [ ] **Step 7: Implement `ProfileRepository` and compatibility re-exports**

Use a per-path reentrant process lock plus Linux `fcntl.flock`, a same-directory temporary file, `flush`, `os.fsync`, and `os.replace`. Re-export old profile names from `app.memory` so existing imports continue working while history remains in that module.

- [ ] **Step 8: Verify and commit**

Run: `pytest tests/test_profile.py tests/test_memory.py -q`

Expected: PASS.

Commit: `git add app/profile.py app/memory.py tests/test_profile.py tests/test_memory.py && git commit -m "feat: add versioned markdown user profiles"`

---

### Task 2: Profile Actions and Existing Audit Integration

**Files:**
- Create: `app/profile_actions.py`
- Modify: `app/audit.py`
- Create: `tests/test_profile_actions.py`
- Modify: `tests/test_audit.py`

**Interfaces:**
- Produces `ProfileChangeOperation`, `ProfileChangeSource`, `ProfileChangeRequest`, `ProfileMutationPreview`, and `ProfileActionService`.
- `ProfileActionService.propose(user_id: str, request: ProfileChangeRequest, evidence_ids: list[str], trusted_current_segment_id: str | None = None) -> ProfileProposal`.
- `confirm(proposal_id: str, user_id: str) -> ProfileProposal` and `reject(proposal_id: str, user_id: str) -> ProfileProposal` are idempotent.
- `apply_inferred_batch(user_id: str, candidates: list[ProfileChangeRequest], evidence_by_key: dict[str, list[str]], summarized_through: str) -> list[ProfileProposal]` applies accepted direct changes and cursor movement in one replacement.
- Produces `validate_profile_mutation_preview(arguments: dict[str, object], preview: dict[str, object]) -> str | None` in `app/audit.py`; both action service and `_validate_tool_record` call it.

**Concrete test contract:**

```python
def test_inference_cannot_replace_confirmed_fact(service: ProfileActionService) -> None:
    confirmed = service.seed_confirmed("u", "preference", "cuisine", "喜欢意大利菜")
    proposals = service.apply_inferred_batch(
        "u",
        [ProfileChangeRequest(operation="replace", category="preference",
                              key="cuisine", value="喜欢川菜", reason="近期选择")],
        {"cuisine": ["hist_" + "1" * 32, "hist_" + "2" * 32]},
        "hist_" + "2" * 32,
    )
    assert proposals[0].status == "pending"
    assert service.repository.read().state.active_fact("preference", "cuisine") == confirmed
```

**Concrete implementation shape:**

```python
def validate_profile_mutation_preview(
    arguments: dict[str, object], preview: dict[str, object]
) -> str | None:
    if preview.get("after_revision") != preview.get("before_revision", -1) + 1:
        return "Profile preview revision is invalid"
    return None
```

- [ ] **Step 1: Write failing action lifecycle tests**

Cover explicit pending proposals, confirm/reject idempotence, 30-day expiry, cross-user denial, stale revision 409 behavior, and one active category/key.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_profile_actions.py -q`

Expected: import failure because `ProfileActionService` does not exist.

- [ ] **Step 3: Implement minimal proposal lifecycle**

Build before/after previews from the latest repository state, require evidence IDs, append bounded audit events, call `validate_profile_mutation_preview`, and pass only successful audit results to `ProfileRepository.replace`.

- [ ] **Step 4: Verify lifecycle GREEN**

Run: `pytest tests/test_profile_actions.py -q`

Expected: lifecycle tests pass.

- [ ] **Step 5: Add failing conflict, safety, and capacity tests**

Prove inferred facts can replace inferred facts, inferred changes against confirmed facts become pending proposals, all `requirement` changes become proposals, confirmed facts are never silently evicted, evidence is capped at five, active categories at twenty, and audit events at two hundred.

- [ ] **Step 6: Verify policy RED**

Run: `pytest tests/test_profile_actions.py -q`

Expected: policy tests fail at the unimplemented branches.

- [ ] **Step 7: Implement conflict and bounded-state rules**

Apply a whole inferred batch to an in-memory state, drop invalid candidates before mutation, and perform one audited repository replacement containing facts, proposals, pruning, audit events, and summary cursor.

- [ ] **Step 8: Add failing existing-auditor tests**

Add literal structured tool records for `propose_profile_change`, `confirm_profile_change`, `reject_profile_change`, and `apply_inferred_profile_change`; prove invalid IDs, mismatched arguments/previews, unsafe inferred requirements, nonexistent evidence, and non-pending proposal results are rejected.

- [ ] **Step 9: Verify auditor RED**

Run: `pytest tests/test_audit.py -q`

Expected: new records are accepted or rendered incorrectly because the allowlist is missing.

- [ ] **Step 10: Extend `app/audit.py` and canonical responses**

Add `prof_`/`fact_` ID validation, profile operation labels, preview validation through the shared pure function, pending-result checks, and canonical pending-profile copy without altering inventory/recipe behavior.

- [ ] **Step 11: Verify and commit**

Run: `pytest tests/test_profile_actions.py tests/test_audit.py -q`

Expected: PASS.

Commit: `git add app/profile_actions.py app/audit.py tests/test_profile_actions.py tests/test_audit.py && git commit -m "feat: audit profile change actions"`

---

### Task 3: Evidence Queries and Autonomous Summarizer

**Files:**
- Modify: `app/memory.py`
- Create: `app/profile_summary.py`
- Modify: `app/prompts.py`
- Modify: `tests/test_memory.py`
- Create: `tests/test_profile_summary.py`

**Interfaces:**
- `ConversationHistoryRepository.reserve_segment_id() -> str`; the existing keyword-only `append` gains `segment_id: str | None = None` and still returns `ConversationRecord`.
- `get_evidence(user_id: str, segment_ids: list[str]) -> list[ConversationRecord]` preserves requested order and never returns another user's rows.
- `list_completed_after(user_id: str, segment_id: str | None, limit: int) -> list[ConversationRecord]` returns only relevant `completed` rows in ascending order.
- Produces `ProfileSummaryCandidate` and `ProfileSummaryBatch` Pydantic models.
- Produces `ProfileSummarizer(model: object, history: ConversationHistoryRepository, actions: ProfileActionService)` with `run_if_due(user_id: str) -> list[ProfileProposal]`.

**Concrete test contract:**

```python
def test_summary_waits_for_ten_completed_records(stack: SummaryStack) -> None:
    stack.append_completed(9)
    assert stack.summarizer.run_if_due("u") == []
    stack.append_completed(1)
    stack.model.result = ProfileSummaryBatch(candidates=[])
    assert stack.summarizer.run_if_due("u") == []
    assert stack.model.calls == 1
    assert stack.repository.read().state.last_summarized_segment_id == stack.last_id
```

**Concrete implementation shape:**

```python
def run_if_due(self, user_id: str) -> list[ProfileProposal]:
    records = self.history.list_completed_after(
        user_id, self.actions.repository.read().state.last_summarized_segment_id, 10
    )
    if len(records) < 10:
        return []
    batch = self._invoke(records)
    return self.actions.apply_summary_batch(user_id, batch, records[-1].segment_id)
```

- [ ] **Step 1: Write failing history evidence tests**

Prove reserved IDs are used exactly once, duplicate IDs fail, evidence is user-scoped, cursor queries exclude blocked/failed/irrelevant/confirmation rows, and the first ten eligible rows are stable.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_memory.py -q`

Expected: missing methods/signature failures.

- [ ] **Step 3: Implement history methods**

Generate `hist_<32 hex>` IDs, validate supplied IDs, use a single ordered query for evidence, and resolve a cursor by the referenced row's `(created_at, rowid)`.

- [ ] **Step 4: Verify history GREEN**

Run: `pytest tests/test_memory.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing summarizer tests**

Use a deterministic fake structured model and real temporary SQLite/Markdown files. Prove nine records do nothing; ten run once; two distinct evidence IDs are required; foreign/fabricated IDs are dropped; requirements and confirmed conflicts create proposals; inferred conflicts update directly; successful batches advance the cursor; and parse/model/audit failures leave it unchanged.

- [ ] **Step 6: Verify summarizer RED**

Run: `pytest tests/test_profile_summary.py -q`

Expected: import failure because the summarizer is absent.

- [ ] **Step 7: Implement the structured summarizer**

Call `model.with_structured_output(ProfileSummaryBatch)`, supply exactly ten complete request/response records as JSON under a system prompt that treats them as untrusted data, validate evidence against the batch, and call one `apply_inferred_batch`.

- [ ] **Step 8: Verify and commit**

Run: `pytest tests/test_memory.py tests/test_profile_summary.py -q`

Expected: PASS.

Commit: `git add app/memory.py app/profile_summary.py app/prompts.py tests/test_memory.py tests/test_profile_summary.py && git commit -m "feat: infer bounded profiles from history"`

---

### Task 4: Agent and Pipeline Integration

**Files:**
- Modify: `app/context.py`
- Modify: `app/agent.py`
- Modify: `app/pipeline.py`
- Modify: `app/prompts.py`
- Modify: `tests/test_agent_allergens.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- `AgentContext.segment_id: str` is the trusted current history ID.
- `AgentResult.profile_proposals: list[ProfileProposal]` is serialized by the existing API path.
- `KitchenAgent.profile_actions: ProfileActionService | None` supplies the Agent tool.
- `KitchenPipeline.profile_actions: ProfileActionService` and `profile_summarizer: ProfileSummarizer | None` own profile writes and summary triggering.
- `KitchenPipeline.resolve_profile_proposal(user_id: str, proposal_id: str, decision: Literal['confirm','reject']) -> ProfileProposal` reloads active profile and records confirmation history.

**Concrete test contract:**

```python
def test_pipeline_uses_current_segment_as_profile_evidence(pipeline: KitchenPipeline) -> None:
    result = pipeline.process_request(user_id="u", text="以后默认一人份")
    record = pipeline.history.list_records("u")[-1]
    assert result.profile_proposals[0].evidence_ids == [record.segment_id]
```

**Concrete Agent tool shape:**

```python
@tool
def propose_profile_change(category: ProfileCategory, key: str,
                           operation: ProfileChangeOperation,
                           value: str | None, reason: str) -> dict[str, object]:
    return profile_actions.propose(
        context.user_id,
        ProfileChangeRequest(category=category, key=key, operation=operation,
                             value=value, reason=reason),
        [context.segment_id],
        trusted_current_segment_id=context.segment_id,
    ).model_dump(mode="json")
```

- [ ] **Step 1: Write failing Agent tool tests**

Build real tools with a temporary profile service and assert the tool binds `context.user_id` and `context.segment_id`, cannot accept model-supplied evidence/user IDs, returns a pending structured result, and produces canonical audited content.

- [ ] **Step 2: Verify Agent RED**

Run: `pytest tests/test_agent_allergens.py -q`

Expected: the profile tool and context fields are missing.

- [ ] **Step 3: Implement Agent context and tool**

Add typed category/operation/value/key/reason arguments only, include the tool in normal Agent runs but not recipe-recovery retries, and update `SYSTEM_PROMPT` so durable statements use the proposal tool and ordinary transient requests do not.

- [ ] **Step 4: Verify Agent GREEN**

Run: `pytest tests/test_agent_allergens.py tests/test_audit.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing Pipeline tests**

Prove each request reserves one ID and persists it, created proposals appear in `AgentResult`, confirmation/rejection records `ConversationOutcome.CONFIRMATION`, confirmed changes reload the prompt profile, summaries trigger only at ten eligible rows, and summary failure leaves the normal result unblocked.

- [ ] **Step 6: Verify Pipeline RED**

Run: `pytest tests/test_pipeline.py -q`

Expected: missing service fields and behaviors.

- [ ] **Step 7: Implement Pipeline construction and orchestration**

Construct one `ProfileRepository`/`ProfileActionService`, pass it to the Agent, reserve IDs before processing, append results with that ID, collect profile proposals from tool history, run the summarizer after append, attach summary proposals before returning, and reload `self.profile` after accepted writes.

- [ ] **Step 8: Verify and commit**

Run: `pytest tests/test_agent_allergens.py tests/test_pipeline.py -q`

Expected: PASS.

Commit: `git add app/context.py app/agent.py app/pipeline.py app/prompts.py tests/test_agent_allergens.py tests/test_pipeline.py && git commit -m "feat: connect profile memory to the agent pipeline"`

---

### Task 5: Confirmation API and Allergen Compatibility

**Files:**
- Modify: `app/web.py`
- Modify: `tests/test_web_limits.py`

**Interfaces:**
- `POST /api/profile-proposals/{proposal_id}/confirm` returns a serialized `ProfileProposal` under `proposal`.
- `POST /api/profile-proposals/{proposal_id}/reject` returns a serialized `ProfileProposal` under `proposal` plus a fixed `message`.
- `/api/process` naturally serializes `profile_proposals` through `AgentResult.model_dump`.
- `PUT /api/allergens` calls `ProfileActionService.replace_confirmed_allergens(user_id: str, broad: list[str], custom: list[str]) -> UserProfile` rather than direct Markdown rewriting.

**Concrete endpoint shape:**

```python
@app.post("/api/profile-proposals/{proposal_id}/confirm")
def confirm_profile_proposal(proposal_id: str) -> dict[str, object]:
    proposal = get_pipeline().resolve_profile_proposal(
        SHOWCASE_USER_ID, proposal_id, "confirm"
    )
    return {"proposal": proposal.model_dump(mode="json")}
```

- [ ] **Step 1: Write failing endpoint tests**

Cover confirm, reject, duplicate resolution, wrong/missing/expired/stale IDs, immediate pipeline profile reload, and confirmation history/tool trace output.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_web_limits.py -q`

Expected: 404 for new endpoints or missing pipeline methods.

- [ ] **Step 3: Implement endpoints and error mapping**

Mirror inventory endpoint handling: `KeyError`→404, `PermissionError`→403, and expired/stale/state `ValueError`→409, with controlled failure history for unsuccessful clicks.

- [ ] **Step 4: Add failing allergen compatibility tests**

Prove GET reads active v2 facts, PUT remains one-click confirmed, writes profile audit/tool records, preserves unrelated categories, and rejects unsupported categories without touching the file.

- [ ] **Step 5: Verify allergen RED**

Run: `pytest tests/test_web_limits.py -q`

Expected: direct legacy writer does not satisfy audit/state assertions.

- [ ] **Step 6: Route allergens through profile actions**

Represent broad and custom allergens as confirmed `requirement` facts with stable keys, supersede removed allergen facts in the same audited batch, and reload a live pipeline.

- [ ] **Step 7: Verify and commit**

Run: `pytest tests/test_web_limits.py tests/test_profile_actions.py -q`

Expected: PASS.

Commit: `git add app/web.py tests/test_web_limits.py && git commit -m "feat: expose profile confirmation endpoints"`

---

### Task 6: Inline Profile Confirmation Card

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`
- Modify: `tests/test_frontend_markdown.py`

**Interfaces:**
- Add `#profile-proposal-list` inside the answer card after `#answer-content`.
- Add `renderProfileProposals(proposals)` and one delegated click handler.
- Card buttons call `/api/profile-proposals/{id}/confirm` or `/reject` and update only that card.

**Concrete renderer shape:**

```javascript
function renderProfileProposals(proposals) {
  profileProposalList.innerHTML = proposals.map((proposal) => `
    <article class="profile-proposal-card" data-proposal-id="${escapeHtml(proposal.proposal_id)}">
      <p>${escapeHtml(proposal.reason)}</p>
      <button data-decision="reject">拒绝</button>
      <button data-decision="confirm">确认写入</button>
    </article>`).join("");
  profileProposalList.hidden = proposals.length === 0;
}
```

- [ ] **Step 1: Write failing browser-source behavior tests**

Execute `frontend/markdown.js` and the extracted pure card renderer in Node with literal proposal payloads. Assert escaped values/reasons, before/after labels, evidence IDs, endpoint paths, and absent markup for an empty list; do not assert source-text presence alone.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_frontend_markdown.py -q`

Expected: missing exported renderer behavior.

- [ ] **Step 3: Implement the minimal inline card**

Render below the current answer only, use `escapeHtml` for every server value, disable both buttons during requests, and mark the card confirmed/rejected without adding a settings page or persistent list.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_frontend_markdown.py -q`

Expected: PASS.

Commit: `git add frontend/index.html frontend/app.js frontend/styles.css tests/test_frontend_markdown.py && git commit -m "feat: confirm profile changes inline"`

---

### Task 7: Documentation and Complete Verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example` only if a new configurable setting is actually consumed.
- Modify: `docs/superpowers/plans/2026-08-10-long-term-user-profile.md` to check completed steps.

**Interfaces:**
- README documents four profile categories, explicit confirmation, autonomous evidence, the ten-record trigger, conflict priority, profile limits, and both endpoints.

- [ ] **Step 1: Update operator documentation**

Document the v2 Markdown ownership rule, backup/corruption behavior, confirmation UI, and that summaries require the configured DeepSeek model.

- [ ] **Step 2: Run focused verification**

Run: `pytest tests/test_profile.py tests/test_profile_actions.py tests/test_profile_summary.py tests/test_memory.py tests/test_audit.py tests/test_agent_allergens.py tests/test_pipeline.py tests/test_web_limits.py tests/test_frontend_markdown.py -q`

Expected: PASS with zero failures.

- [ ] **Step 3: Run full regression verification**

Run: `pytest -q`

Expected: PASS with zero failures.

- [ ] **Step 4: Run repository checks**

Run: `git diff --check && git status --short && git log -8 --oneline`

Expected: no whitespace errors; only intended documentation changes remain unstaged after feature commits.

- [ ] **Step 5: Commit documentation**

Run: `git add README.md .env.example && git add -f docs/superpowers/specs/2026-08-10-long-term-user-profile-design.md docs/superpowers/plans/2026-08-10-long-term-user-profile.md && git commit -m "docs: document long-term user profile memory"`

- [ ] **Step 6: Apply completion skills**

Read and follow `superpowers:verification-before-completion`, `superpowers:requesting-code-review`, and `superpowers:finishing-a-development-branch`; inspect the complete diff, rerun required verification, push `codex/conversation-memory`, and update the existing draft PR.
