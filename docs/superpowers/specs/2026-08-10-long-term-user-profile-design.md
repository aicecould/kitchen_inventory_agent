# Long-term User Profile with Confirmation, Evidence, and Conflict Updates

**Date:** 2026-08-10
**Status:** Approved design
**Branch:** `codex/conversation-memory`

## 1. Goal

Extend the current Markdown user profile into a bounded, auditable long-term
memory system. The system must support:

- kitchen setup, preferences, mandatory requirements, and default context;
- explicit user statements that become permanent only after a second click;
- autonomous history summarization with verifiable conversation evidence;
- conflict replacement without silently overriding confirmed facts;
- bounded active facts, evidence, proposals, and audit history;
- immediate injection of the active profile into later Agent requests; and
- the existing deterministic audit architecture for every profile write.

The complete conversation history implemented in phase 1 remains the canonical
evidence store. Profile state stores only stable `segment_id` references, never
copies of the original conversation text.

## 2. Scope

### In scope

- A versioned Markdown profile format and legacy migration.
- Explicit profile-change proposals exposed as an Agent tool.
- One inline confirmation card with confirm and reject actions.
- Autonomous summarization after each batch of ten eligible conversations.
- Evidence validation, inferred-fact replacement, confirmed-fact protection,
  safety restrictions, expiry, pruning, and audit history.
- Compatibility with the current allergen API and profile prompt injection.
- Unit, integration, API, pipeline, audit, and frontend tests.

### Out of scope

- A separate profile management page.
- Multiple simultaneous user profiles in the showcase UI.
- Semantic/vector retrieval for profile facts.
- Moving profile state into SQLite.
- A scheduler or external job queue.

## 3. Confirmed Product Rules

1. An explicit durable statement such as “I never want cilantro again” creates
   a pending proposal. It does not become active until the user clicks confirm.
2. Autonomous summaries do not require a click unless they touch a mandatory
   requirement, including allergens or medical dietary restrictions, or
   conflict with a user-confirmed fact.
3. Autonomous updates may silently replace or delete only facts whose origin is
   `inferred`.
4. A confirmed fact always remains active while a conflicting proposal is
   pending.
5. Every profile write, including proposal creation, rejection, expiry,
   migration, inference, and pruning, uses the existing audit architecture.
6. Only one inline confirmation-card interaction is added. No settings page is
   introduced.

## 4. Markdown as the Source of Truth

`user_profile.md` remains the only profile persistence artifact. It contains a
human-readable projection followed by a hidden, versioned machine-state block.

Conceptual shape:

```markdown
# 用户画像

## 厨房情况
- 没有烤箱

## 饮食偏好
- 喜欢意大利菜

## 必须要求
- 对花生过敏

## 默认情景
- 默认一人份

<!-- kitchen-profile-state:v2
{"schema_version":2,"revision":7,"facts":[...],"proposals":[...],...}
-->
```

The machine block is authoritative. The readable sections are regenerated from
active facts after every accepted mutation. Pending proposals, superseded facts,
evidence metadata, cursors, and audit events never enter the Agent prompt.

### 4.1 Profile state

The v2 state contains:

- `schema_version`;
- a monotonically increasing `revision`;
- `facts`;
- `proposals`;
- `audit_events`;
- `last_summarized_segment_id`; and
- timestamps needed for expiry and deterministic ordering.

### 4.2 Facts

Each fact contains:

- stable `fact_id`;
- `category`: `kitchen_setup`, `preference`, `requirement`, or `default_context`;
- normalized semantic `key`;
- human-readable `value`;
- `origin`: `user_confirmed` or `inferred`;
- `status`: `active` or `superseded`;
- up to five evidence `segment_id` values;
- a short reason;
- creation and update timestamps; and
- the replacing fact or proposal identifier when applicable.

Only one fact may be active for a category/key pair.

### 4.3 Proposals

Each proposal contains:

- stable `proposal_id`;
- operation: `upsert`, `replace`, or `delete`;
- target category/key and candidate value;
- the currently active conflicting fact, if any;
- source: `explicit_user_statement` or `autonomous_summary`;
- supporting evidence IDs and reason;
- `status`: `pending`, `confirmed`, `rejected`, or `expired`;
- base and resolved revisions; and
- creation, expiry, and resolution timestamps.

Pending proposals expire after 30 days. Resolution is idempotent.

### 4.4 Audit events and limits

- At most 20 active facts per category.
- At most five evidence references per fact or proposal.
- At most 200 recent profile audit events per user profile.
- Conversation history remains permanent and is not affected by profile
  pruning.
- When a category is full, an autonomous candidate may replace the oldest
  inferred fact but may never evict a confirmed fact.
- If all 20 facts are confirmed and no same-key replacement applies, proposal
  creation fails with a capacity explanation.

## 5. Legacy Migration

Legacy Markdown remains readable without an immediate write. Existing broad
allergens, custom allergens, and preferences are represented in memory as
`user_confirmed` facts.

On the first actual profile mutation, migration and the requested mutation are
combined into one before/after preview, reviewed by the existing auditor, and
written atomically. This avoids an unaudited write during ordinary reads.

Malformed v2 state is never replaced by a blank or default profile. Reads report
the corruption and writes stop until it is repaired.

A missing profile may still be created on first use for compatibility, but its
initial v2 document is generated through `initialize_profile_v2` and the same
preview, audit, and atomic-write path rather than the old direct template write.

## 6. Write Operations and Existing Audit Architecture

Profile writes follow the same architecture as inventory writes:

1. validate typed operation arguments;
2. build a structured before/after preview;
3. audit the preview using pure rules in `app/audit.py`;
4. claim a pending action when human confirmation is required;
5. execute the mutation under a lock;
6. record structured tool history and execution trace; and
7. run the existing post-tool `review_tool_calls` checks.

The implementation extends, rather than duplicates:

- tool-name allowlists;
- `_validate_tool_record`;
- proposal-preview validation;
- canonical response generation;
- `AuditResult` failure handling; and
- `ExecutionTraceEvent` reporting.

The same pure profile-preview validator is called before persistence and from
`review_tool_calls` afterward. Therefore pre-write and post-tool review cannot
silently drift apart.

Audited profile operations are:

- `initialize_profile_v2`;
- `propose_profile_change`;
- `confirm_profile_change`;
- `reject_profile_change`;
- `apply_inferred_profile_change`;
- `expire_profile_proposals`;
- `prune_profile_state`; and
- `migrate_profile_v2`.

The auditor verifies user scope, evidence ownership, proposal state and expiry,
allowed state transitions, safety rules, confirmed-fact precedence, limits,
revision consistency, and Markdown round-trip equivalence. An audit failure
leaves the original file unchanged.

## 7. Explicit User Flow

1. The pipeline reserves the current conversation `segment_id` before the Agent
   runs. The history repository accepts this reserved ID when it appends the
   completed record.
2. The Agent receives `propose_profile_change` as its only profile mutation
   tool. The model supplies category, key, operation, value, and reason.
3. Trusted service code binds the user ID, reserved segment ID, source, and
   timestamps. The model cannot provide or override them.
4. The profile service builds and audits a pending proposal preview. Evidence
   validation accepts the reserved ID only when it exactly matches the trusted
   current-request context; arbitrary unpersisted IDs remain invalid.
5. The response uses canonical output derived from the tool record and includes
   structured `profile_proposals` for the frontend.
6. The frontend renders a compact inline card below the response, showing the
   before/after values, reason, evidence excerpt, and confirm/reject buttons.
7. A confirm click claims and revalidates the proposal against the latest
   revision, applies it, and reloads the pipeline profile. A reject click only
   changes proposal state.
8. Both outcomes are appended to conversation history with
   `ConversationOutcome.CONFIRMATION`, a confirmation result, tool history, and
   a human-checkpoint trace.

Duplicate clicks return the already-resolved result and never create duplicate
facts. A stale proposal whose preview no longer applies returns HTTP 409.

The proposal mutation and current history append use a coordinated commit while
the profile lock is held. The Markdown write is prepared and audited, the
reserved history row is inserted in an open SQLite transaction, the prepared
profile is atomically installed, and only then is the SQLite transaction
committed. Ordinary failures roll back the database transaction and restore the
previous file before the lock is released, so no proposal card is returned.
Startup consistency checks compare profile evidence and profile tool records in
both directions and surface any orphan left by an unrecoverable process or
filesystem failure rather than silently accepting it.

## 8. Autonomous Summary Flow

Eligible records have all of these properties:

- belong to the current user;
- are newer than `last_summarized_segment_id`;
- are relevant; and
- have outcome `completed`.

After every successful history append, the pipeline checks the eligible count.
At ten records it invokes the summarizer inline after the normal answer content
has been produced but before the API response is assembled. No external
scheduler or competing task is introduced. Any confirmation proposal created by
the summary can therefore be attached to that response as an inline card.

The summarizer receives complete request/response pairs plus server-issued
segment IDs as untrusted source material. It must return validated structured
candidates. A behavioral inference requires at least two distinct evidence
segments. Evidence IDs not present in the supplied batch invalidate that
candidate.

For each valid candidate:

- a non-conflicting kitchen, preference, or default fact may be added directly;
- an inferred fact with the same semantic key may be replaced or deleted;
- a mandatory requirement always becomes a proposal; and
- any conflict with a `user_confirmed` fact becomes a proposal while the old
  fact remains active.

Every accepted autonomous change is represented by a structured tool record and
passes the same audit rules as Agent-initiated changes. Valid candidates and the
new summary cursor are applied as one Markdown mutation. Invalid candidates are
dropped before that batch is built; an audit or persistence failure rejects the
whole batch. The cursor moves only after the accepted actions complete. A model,
audit, or persistence failure leaves the cursor in place for retry and does not
fail the user-facing request.

## 9. Prompt Projection

The Agent receives only the readable projection of active facts, grouped into:

- kitchen setup;
- preferences;
- mandatory requirements; and
- default context.

Confirmed and inferred facts may be labelled by provenance, but pending,
rejected, expired, superseded, audit, and cursor data are excluded. Mandatory
requirements remain higher priority than preferences and defaults.

After a successful confirm or autonomous update, the in-memory profile is
reloaded before the next Agent call.

## 10. API and Frontend

`/api/process` adds a `profile_proposals` collection for proposals created by the
current response.

Two minimal endpoints resolve the inline card:

- `POST /api/profile-proposals/{proposal_id}/confirm`
- `POST /api/profile-proposals/{proposal_id}/reject`

They mirror current action endpoint conventions for 404, 403, 409, idempotence,
failure recording, and confirmation history.

The existing allergen API reads from the active v2 projection. Saving the
existing allergen dialog is already a direct user click, so it is treated as a
confirmed batch change, but it still uses the same preview, audit, trace, and
atomic persistence path.

No profile settings page or second proposal list is added.

## 11. Concurrency and Atomicity

- Read/modify/write operations hold an OS-backed lock associated with the
  profile path.
- The service rereads the current revision under the lock before auditing and
  applying a resolution.
- Writes use a temporary file in the same directory, flush and fsync it, then
  replace the profile atomically.
- A newly rendered document is parsed before replacement, and the parsed active
  state must match the audited preview.
- Revision conflicts do not merge silently; they are re-evaluated where safe or
  returned as HTTP 409.

## 12. Error Handling

- Invalid Agent tool arguments are rejected by the current tool-call audit path.
- Missing, cross-user, expired, rejected, or stale proposals return explicit
  errors without changing state.
- Missing or foreign evidence IDs reject the affected operation.
- A summary failure is traced but does not change the normal response or cursor.
- Profile parse or atomic-write failures preserve the previous file and are
  recorded as controlled failures.
- Existing allergen and recipe safety behavior continues to use only active
  profile facts.

## 13. Test Strategy

Implementation is test-driven and covers:

1. Markdown v1 virtual migration, v2 serialization, round-trip parsing, and
   corrupted-state protection.
2. Fact/proposal lifecycle, idempotent confirm/reject, expiry, revision
   conflicts, and all capacity rules.
3. Evidence existence and ownership, two-segment inference minimum, safety
   restrictions, inferred replacement, and confirmed-fact protection.
4. Extensions to existing tool-record validation, canonical responses, and
   rejected before/after previews.
5. The ten-record summary trigger, cursor advancement, retry behavior, and
   active-profile reload.
6. Agent context projection and exclusion of hidden/pending state.
7. API status behavior, confirmation history records, and allergen API
   compatibility.
8. Inline card rendering and confirm/reject requests.
9. Atomic write and concurrent revision tests.
10. The complete existing test suite to prevent regressions.

## 14. Acceptance Criteria

The phase is complete when:

- explicit durable profile changes remain pending until a user click;
- each active or proposed fact exposes valid conversation evidence;
- autonomous summaries run at the approved ten-record boundary;
- inferred facts may update inferred facts but never silently replace confirmed
  facts or mandatory requirements;
- every profile write is accepted by extensions to the existing audit path;
- the profile and its auxiliary state remain within approved bounds;
- the next Agent request observes each accepted change;
- the inline card can confirm or reject exactly once; and
- all focused and full-suite tests pass.
