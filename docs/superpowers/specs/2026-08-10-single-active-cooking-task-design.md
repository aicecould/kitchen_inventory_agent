# Single Active Cooking Task, Recipe Confirmation, and Inventory Reconciliation

**Date:** 2026-08-10  
**Status:** Approved design  
**Branch:** `codex/single-active-task`

## 1. Goal

Add one optional active cooking task to the existing kitchen Agent. A task is a
small structured attachment containing the user's current recipe choice and
serving count. It supplements the Agent prompt but does not own, partition, or
replace conversation context.

The feature must support:

- at most one active task per user;
- confirming a recipe into the active task;
- creating a new task or switching to a paused task;
- completing a task after the user says the dish was cooked;
- reconciling used inventory through the existing reviewed
  `inventory.replace` and `inventory.remove` actions; and
- clearing the short-term context window and the active task through one user
  command while keeping the two persistence models independent.

## 2. Confirmed Product Rules

1. A task is only supplemental prompt information. Conversation history and
   the ten-segment short-term window never depend on a task ID.
2. Switching, pausing, updating, or completing a task does not clear or change
   the conversation window.
3. The clear-context operation independently resets the short-term window and
   completes the active task. It does not delete permanent conversation
   history or physically delete the task.
4. Confirming a recipe only saves the task. It does not change inventory.
5. Inventory reconciliation starts only after the user clicks **Cooked**.
6. Inventory quantities remain opaque `quantity_text`. The feature performs no
   subtraction, parsing, scaling, or unit conversion.
7. A fully consumed inventory instance produces an existing
   `inventory.remove` proposal. A partially consumed instance produces an
   existing `inventory.replace` proposal containing the complete new
   `quantity_text`.
8. If the complete replacement value cannot be established reliably, the
   Agent asks the user to provide either "used up" or the full remaining
   quantity. It must not guess.
9. Each inventory instance has its own proposal and can be confirmed or
   cancelled independently through the existing reviewer.
10. The serving count comes from an explicit value on recipe confirmation,
    then the long-term profile's default context, then a one-serving fallback.
    The first version does not scale ingredient text.

## 3. Scope

### In scope

- A small SQLite task repository and service.
- One active/paused/completed lifecycle.
- Server-verified recipe snapshots sourced from persisted `search_recipes`
  output.
- Compact task prompt injection.
- Minimal task controls in the existing chat UI.
- A context reset cursor that is independent from task storage.
- Task-linked inventory proposals using the existing action service and
  confirmation UI.
- Unit, API, pipeline, audit, migration, and frontend contract tests.

### Out of scope

- Task-specific conversation windows or `task_id` on conversation history.
- Multiple simultaneous active tasks.
- Automatic topic classification or automatic task switching.
- Nested tasks, dependencies, labels, timelines, or a task management page.
- Reopening a completed task.
- Ingredient parsing, recipe scaling, quantity arithmetic, or unit conversion.
- Batch inventory transactions.
- A separate inventory reconciliation form.

## 4. Architecture

The feature has four deliberately small parts:

1. `CookingTaskRepository` stores structured task rows in the existing SQLite
   database.
2. `CookingTaskService` validates lifecycle transitions, recipe evidence,
   optimistic versions, ownership, and idempotency.
3. The pipeline reads the optional active task and appends a compact, trusted
   summary to the Agent context.
4. The current `InventoryActionService` continues to own every inventory
   mutation proposal and confirmation.

Task writes originate from explicit UI actions. They are validated by the task
domain service and recorded in the task row; they do not enter the response
supervisor or the inventory pending-action state machine. Inventory writes
remain proposals and continue through the existing deterministic audit and
human confirmation architecture.

## 5. Persistence

### 5.1 `cooking_tasks`

```sql
CREATE TABLE cooking_tasks (
    task_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'paused', 'completed')),
    query_text TEXT NOT NULL,
    recipe_json TEXT,
    servings INTEGER NOT NULL CHECK (servings > 0),
    servings_source TEXT NOT NULL
        CHECK (servings_source IN ('user', 'profile', 'fallback')),
    completion_reason TEXT
        CHECK (completion_reason IS NULL OR completion_reason IN
               ('cooked', 'context_cleared')),
    reconciliation_open INTEGER NOT NULL DEFAULT 0 CHECK
        (reconciliation_open IN (0, 1)),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    last_request_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
```

A partial unique index on `user_id WHERE status = 'active'` enforces the single
active task invariant in the database as well as in service code.

`recipe_json` is a snapshot, not a reference to a mutable external API record.
It contains the selected recipe's ID, original title, instructions, original
ingredient strings, source, source URL, image URL, source history segment ID,
and the user's original recipe query.

`version` provides optimistic concurrency. `last_request_id` makes a repeated
button request idempotent without introducing a generic event or workflow
table. This showcase has one local user flow and accepts one retained
idempotency key per task; stale duplicates with an older version return the
already-current representation when their requested result is already true.

### 5.2 Context reset state

Context clearing requires a small store independent from tasks:

```sql
CREATE TABLE conversation_context_state (
    user_id TEXT PRIMARY KEY,
    reset_after_rowid INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
```

`list_context_window` continues to return at most ten relevant input/output
segments, but only rows whose SQLite `rowid` is greater than the user's reset
cursor. Time-range history recall, permanent history listing, and autonomous
profile evidence ignore this cursor and continue to see all eligible persisted
history.

No `task_id` or context epoch is added to `conversation_history`.

### 5.3 Inventory action source

`pending_actions` receives one nullable `source_task_id` column. Existing rows
remain valid. The source is bound by trusted reconciliation context; the model
cannot select another user's task or override the source ID. No new inventory
operation is introduced.

## 6. Task Lifecycle

### 6.1 Recipe confirmation

The recipe result UI offers **Confirm recipe** and, when an active task already
exists, **Use as new task**.

The request supplies only:

- history segment ID;
- recipe ID;
- optional positive integer serving count;
- expected task version when updating; and
- request ID.

The server loads the conversation record for the current user and finds the
exact recipe in its persisted `search_recipes` tool result. It rejects a
missing recipe, a cross-user segment, or frontend-supplied recipe content.

When no active task exists, confirmation creates one. **Confirm recipe** updates
an existing active task. **Use as new task** atomically changes the existing
task to `paused` and creates the new active task. Confirming a different recipe
in the current task is the supported task-edit operation.

### 6.2 Serving resolution

The service resolves servings in this order:

1. an explicit positive integer from the current confirmation;
2. a parseable default-context fact in the active long-term profile; or
3. one serving.

It records both the value and `user`, `profile`, or `fallback` source. Profile
text that is absent or not safely parseable is ignored. Ingredient strings are
stored exactly as returned and are not rescaled.

### 6.3 Switching

The UI displays the current active task and a compact list of paused tasks.
Activating a paused task atomically pauses the current task, if any, and marks
the chosen task active. Completed tasks are view-only in the first version.

Switching does not alter conversation history, the short-term window, or the
profile. Ordinary topic changes do not automatically create or switch tasks.

### 6.4 Completion

**Cooked** changes the active task to `completed`, sets
`completion_reason = 'cooked'`, and opens inventory reconciliation. The task
snapshot remains available to the reconciliation turn even though there is no
active task.

The clear-context operation changes an active task to `completed` with
`completion_reason = 'context_cleared'` and does not open reconciliation.
Completed task rows are retained for history and source validation.

## 7. Context Clearing

Context clearing is one user-visible command that orchestrates two independent
domain operations:

1. advance the user's context reset cursor to the latest persisted history
   row; and
2. complete the current active task with `context_cleared`, if one exists.

The task repository has no reference to the context cursor and conversation
rows have no task reference. Switching or completing a task never invokes the
context reset operation. Advancing the cursor never deletes a history row.

The endpoint is idempotent. Repeating it before another conversation leaves the
same empty window and no active task.

## 8. Prompt Injection

Normal Agent requests keep the existing global ten-segment context window.
When an active task exists, the pipeline appends a compact trusted block:

```text
Current cooking task attachment:
- task_id: ...
- user goal: ...
- confirmed recipe: ...
- servings: ...
- ingredients: ...
```

The block is data, not instructions. It excludes internal status, versions,
request IDs, and reconciliation flags. Without an active task, no task block is
added.

An inventory-reconciliation request explicitly names a completed cooked task.
Trusted pipeline code verifies ownership and completion reason, injects its
read-only recipe snapshot, and binds its `task_id` as the source for any
inventory proposals. This special request does not create a new task and does
not replace the normal conversation window.

## 9. Inventory Reconciliation

After **Cooked**, the frontend starts an inline reconciliation Agent turn for
the completed task.

The Agent reads current inventory through existing tools and compares it with
the task's original ingredient strings. It may propose only:

- `inventory.remove` when the user has established that a specific instance
  was used up; or
- `inventory.replace` when a complete new `quantity_text` for a specific
  instance is known.

If one or more outcomes are unclear, the response asks one consolidated
question listing the affected inventory instances. The user replies with
"used up" or a complete remaining quantity for each. The follow-up request
retains the completed source task ID but remains ordinary global conversation
context.

Every instance creates an independent existing pending action. The action
service builds the current/proposed preview, the deterministic auditor reviews
it, and the existing inline card handles confirmation or cancellation. A
failure or cancellation for one instance does not roll back another proposal
and does not change the completed task.

Reconciliation closes after every explicitly reported outcome has either
produced a proposal or been skipped, or when the user skips the whole update.
An unresolved completed task is returned by the task API so its chat-based
reconciliation can be resumed, but it never blocks creation of a new active
task.

## 10. API and UI

Minimal API surface:

- `GET /api/tasks` — active and paused tasks plus completed tasks whose
  reconciliation is still open;
- `GET /api/tasks/active` — active task or null;
- `POST /api/tasks/recipe` — create/update from verified recipe evidence;
- `POST /api/tasks/recipe-as-new` — pause current and create from evidence;
- `POST /api/tasks/{task_id}/activate` — switch to a paused task;
- `PATCH /api/tasks/{task_id}/servings` — change servings;
- `POST /api/tasks/{task_id}/cooked` — complete and start reconciliation;
- `POST /api/tasks/{task_id}/reconciliation/close` — explicitly skip/close;
- `POST /api/context/clear` — independently reset context and clear the active
  task attachment.

All writes validate ownership, lifecycle state, expected version, and request
ID. Stale versions return HTTP 409. Invalid evidence or state transitions
return a bounded 4xx response and leave state unchanged.

The existing chat page adds only:

- a compact active-task attachment;
- recipe confirmation buttons;
- a paused-task selector;
- a servings edit control; and
- **Cooked** and **Clear context** actions.

There is no task management page, event timeline, or inventory reconciliation
form. Ambiguous inventory outcomes are collected in chat.

## 11. Failure and Concurrency Rules

- SQLite transactions and the partial unique index prevent two active tasks.
- Repeated requests are idempotent and do not create duplicate tasks or
  inventory proposals.
- A recipe missing from the referenced persisted tool result is never saved.
- A stale active-task version cannot overwrite a newer recipe or serving edit.
- A context reset never deletes permanent history or hides it from time recall.
- The clear-context endpoint updates the context cursor and completes the
  active task in one SQLite transaction. Either both user-visible effects
  commit or neither does, while their data models remain independent.
- Inventory proposal failures do not reopen or alter a completed task.
- `source_task_id` is always server-bound and ownership-checked.

Tasks, context state, conversation history, and pending actions live in the
same SQLite database. The clear-context orchestration therefore uses one
connection and transaction while preserving separate repositories and no
cross-domain foreign key.

## 12. Verification Strategy

Implementation follows test-driven development and covers:

1. schema creation, migration, constraints, and restart persistence;
2. one-active-task enforcement and concurrent switches;
3. lifecycle transition, ownership, optimistic version, and idempotency tests;
4. exact recipe evidence validation and cross-user rejection;
5. serving priority: explicit, profile default, then one;
6. task injection without any change to the global ten-segment window;
7. switching and completion leaving context unchanged;
8. clear-context hiding prior rows only from the short-term window while time
   recall and profile summarization still see them;
9. cooked completion producing only reviewed replace/remove proposals;
10. ambiguous inventory outcomes producing questions rather than writes;
11. independent multi-instance proposal confirmation/cancellation;
12. nullable source-task migration for old pending actions;
13. endpoint limit, stale-version, ownership, and repeated-click behavior;
14. frontend source-contract tests for the minimal controls; and
15. the complete existing suite to prevent profile, history, inventory, recipe,
    audit, and supervisor regressions.

The repository's documented full-suite command is `pytest -s -q`; capture must
remain disabled in the current environment because `TMP` points to a mounted
Windows temporary directory.

## 13. Delivery

Development occurs on `codex/single-active-task`, created from the merged PR #3
state on `origin/main`. The implementation remains one focused phase and does
not alter the long-term-profile format or introduce a task-specific memory
system.
