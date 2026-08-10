# Single Active Cooking Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one optional active cooking-task attachment, verified recipe confirmation, manual task switching, independent context clearing, and cooked-task inventory reconciliation through existing reviewed replace/remove proposals.

**Architecture:** A focused `CookingTaskRepository` and `CookingTaskService` store one active/paused/completed task per user in the existing SQLite database. Conversation context remains global and gains only an independent reset cursor; the pipeline injects a compact task snapshot without assigning history to tasks. Recipe and task writes come from explicit HTTP actions, while all inventory mutations remain existing pending actions with a trusted nullable source-task ID.

**Tech Stack:** Python 3.13, Pydantic 2, SQLite, FastAPI, LangChain/ChatOpenAI, browser JavaScript, pytest, Node.js source-contract tests.

## Global Constraints

- A task is a supplemental prompt attachment and never owns or partitions conversation history.
- The short-term context remains the latest ten relevant input/output segments after the independent reset cursor.
- Permanent history, time recall, and profile summarization ignore context resets.
- At most one task is active per user.
- Recipe confirmation saves a task but never changes inventory.
- Inventory reconciliation starts only after an explicit **Cooked** click.
- Inventory quantities remain opaque `quantity_text`; perform no parsing, arithmetic, scaling, or unit conversion.
- Use only existing `inventory.replace` and `inventory.remove` pending actions, one per inventory instance.
- Ambiguous remaining quantities must produce a consolidated question, never a guessed write.
- Explicit servings override the profile default; an unavailable/unparseable profile default falls back to one.
- Do not add automatic topic detection, task-specific history, a workflow engine, a task page, or an inventory reconciliation form.
- Run project tests with `pytest -s -q` because capture mode is incompatible with the configured mounted Windows temporary directory.

---

## File Structure

- Create `app/tasks.py`: task models, SQLite repository, lifecycle service, recipe-evidence validation, serving resolution, and atomic context/task clear orchestration.
- Modify `app/memory.py`: independent context reset table/cursor and cursor-aware ten-segment window.
- Modify `app/actions.py`: nullable trusted `source_task_id` on pending actions and proposal creation.
- Modify `app/context.py`: structured optional cooking-task and reconciliation context plus recipe options in the response.
- Modify `app/agent.py`: task payload prompt injection and trusted source binding for inventory proposal tools.
- Modify `app/pipeline.py`: initialize task services, attach active/completed task context, expose recipe evidence, and close reconciliation after source-linked proposals are produced.
- Modify `app/prompts.py`: explicit opaque-quantity and cooked-task reconciliation rules.
- Modify `app/web.py`: typed task, context-clear, and reconciliation endpoints.
- Create `frontend/tasks.js`: safe task/recipe card rendering and endpoint helpers.
- Modify `frontend/index.html`, `frontend/app.js`, and `frontend/styles.css`: minimal attachment, switcher, confirmation, cooked, and clear controls.
- Create `tests/test_tasks.py`: repository, service, evidence, lifecycle, serving, concurrency, and clear tests.
- Modify `tests/test_memory.py`, `tests/test_actions.py`, `tests/test_agent_allergens.py`, `tests/test_pipeline.py`, and `tests/test_web_limits.py`: integration and regression coverage.
- Create `tests/test_frontend_tasks.py`: Node rendering and browser source-contract coverage.
- Modify `README.md`: persistence, API, prompt, and usage documentation.

---

### Task 1: Task Repository and Lifecycle Service

**Files:**
- Create: `app/tasks.py`
- Create: `tests/test_tasks.py`

**Interfaces:**
- Produces: `TaskStatus`, `CompletionReason`, `ServingsSource`, `RecipeSnapshot`, and `CookingTask` Pydantic models.
- Produces: `CookingTaskRepository(db_path: Path)` with `initialize()`, `get()`, `get_active()`, `list_available()`, and transactional mutation helpers.
- Produces: `CookingTaskService(repository, history)` with `confirm_recipe()`, `update_servings()`, `activate()`, `complete_cooked()`, `close_reconciliation()`, and `get_reconciliation_task()`; `complete_cooked()` returns `(CookingTask, bool)` where the boolean is true only for the first state-changing click.
- Consumes: `ConversationHistoryRepository.get_evidence()` and persisted `search_recipes` tool records.

- [ ] **Step 1: Write failing schema and round-trip tests**

```python
def test_repository_enforces_one_active_task_and_survives_restart(tmp_path: Path) -> None:
    repository = CookingTaskRepository(tmp_path / "inventory.db")
    repository.initialize()
    first = repository.create_active(
        user_id="u",
        query_text="番茄能做什么",
        recipe=_recipe_snapshot(),
        servings=1,
        servings_source=ServingsSource.FALLBACK,
        request_id="req-1",
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_active(
            user_id="u",
            query_text="土豆能做什么",
            recipe=_recipe_snapshot(recipe_id="recipe-2"),
            servings=1,
            servings_source=ServingsSource.FALLBACK,
            request_id="req-2",
        )
    restarted = CookingTaskRepository(tmp_path / "inventory.db")
    restarted.initialize()
    assert restarted.get_active("u") == first
```

- [ ] **Step 2: Run the repository test and verify the missing module failure**

Run: `pytest -s -q tests/test_tasks.py::test_repository_enforces_one_active_task_and_survives_restart`

Expected: FAIL because `app.tasks` does not exist.

- [ ] **Step 3: Implement typed models, schema, explicit column reads, and the unique partial index**

```python
class CookingTask(BaseModel):
    task_id: str
    user_id: str
    status: TaskStatus
    query_text: str
    recipe: RecipeSnapshot
    servings: int = Field(gt=0)
    servings_source: ServingsSource
    completion_reason: CompletionReason | None = None
    reconciliation_open: bool = False
    version: int = Field(gt=0)
    last_request_id: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
```

Use explicit SELECT column lists rather than `SELECT *`, validate IDs as
`task_[0-9a-f]{32}`, use timezone-aware UTC ISO timestamps, and create:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_task_per_user
ON cooking_tasks(user_id) WHERE status = 'active';
```

- [ ] **Step 4: Run task repository tests**

Run: `pytest -s -q tests/test_tasks.py -k 'repository or schema'`

Expected: PASS.

- [ ] **Step 5: Write failing recipe-evidence, serving, state, ownership, version, and idempotency tests**

```python
def test_confirm_recipe_uses_persisted_tool_result_not_client_content(stack) -> None:
    evidence = stack.history.append(
        user_id="u",
        request="用番茄找菜谱",
        response="找到番茄汤",
        tool_history=[{"tool": "search_recipes", "arguments": {}, "result": [_recipe_result()]}],
    )
    task = stack.service.confirm_recipe(
        user_id="u",
        segment_id=evidence.segment_id,
        recipe_id="recipe-1",
        servings=None,
        default_context=["默认两人份"],
        as_new=False,
        expected_version=None,
        request_id="confirm-1",
    )
    assert task.recipe.title == "Tomato Soup"
    assert task.servings == 2
    assert task.servings_source == ServingsSource.PROFILE
```

Also test explicit servings, one-serving fallback, cross-user evidence,
non-recipe tool records, unknown recipe IDs, active-task update, `as_new`
pause/create, paused-task activation, completed-task rejection, stale version,
duplicate request, cooked completion returning `changed=False` on retry, and
reconciliation close.

- [ ] **Step 6: Run the service tests and verify they fail at missing behavior**

Run: `pytest -s -q tests/test_tasks.py -k 'service or confirm or serving or activate or cooked or reconciliation'`

Expected: FAIL because service methods and validations are incomplete.

- [ ] **Step 7: Implement minimal service behavior and strict recipe extraction**

`confirm_recipe()` must iterate only tool records whose name is exactly
`search_recipes`, require a list result, select an exact string ID, and build
`RecipeSnapshot` from allowlisted fields. Never accept recipe content from the
caller. Serving parsing accepts safe patterns equivalent to `默认两人份`,
`2 人份`, or `servings: 2`, with Chinese numerals limited to 一 through 十;
otherwise it falls back to one.

Every mutation opens `BEGIN IMMEDIATE`, re-reads ownership and version inside
the transaction, applies only an allowlisted transition, increments `version`,
and returns the committed row. A duplicate `request_id` whose requested result
is already true returns that row without a second write.

- [ ] **Step 8: Run all task tests**

Run: `pytest -s -q tests/test_tasks.py`

Expected: PASS.

- [ ] **Step 9: Commit the task domain**

```bash
git add app/tasks.py tests/test_tasks.py
git commit -m "feat: add single active cooking tasks"
```

---

### Task 2: Independent Short-Term Context Reset

**Files:**
- Modify: `app/memory.py`
- Modify: `app/tasks.py`
- Modify: `tests/test_memory.py`
- Modify: `tests/test_tasks.py`

**Interfaces:**
- Produces: `ConversationHistoryRepository.advance_context_cursor(user_id, *, connection=None) -> int`.
- Changes: `ConversationHistoryRepository.list_context_window()` filters relevant rows after the stored reset cursor.
- Produces: `CookingTaskService.clear_context(user_id, request_id) -> CookingTask | None`, atomically advancing the cursor and completing an active task.

- [ ] **Step 1: Write failing context reset tests**

```python
def test_context_reset_hides_window_but_not_recall_or_summary_rows(repository) -> None:
    first = repository.append(user_id="u", request="旧输入", response="旧输出")
    assert len(repository.list_context_window("u")) == 1
    repository.advance_context_cursor("u")
    assert repository.list_context_window("u") == []
    assert repository.get_evidence("u", [first.segment_id])[0].request == "旧输入"
    assert repository.search_by_time("u", DAY_START, DAY_END)[0].request == "旧输入"
    assert repository.list_completed_after("u", None, limit=10)[0].request == "旧输入"
```

Add tests that future rows re-enter the window, irrelevant rows remain skipped,
repeated reset is idempotent, other users are unaffected, and the window still
caps at ten.

- [ ] **Step 2: Run memory reset tests and verify failure**

Run: `pytest -s -q tests/test_memory.py -k 'context_reset or context_window'`

Expected: FAIL because the cursor table and method do not exist.

- [ ] **Step 3: Implement `conversation_context_state` and cursor-aware window reads**

Initialize the new table in `ConversationHistoryRepository.initialize()`.
`advance_context_cursor()` stores `COALESCE(MAX(rowid), 0)` for that user using
an upsert. `list_context_window()` uses:

```sql
WHERE user_id = ?
  AND is_relevant = 1
  AND rowid > COALESCE(
      (SELECT reset_after_rowid FROM conversation_context_state WHERE user_id = ?),
      0
  )
```

Do not change time recall, evidence reads, record listing, or profile summary
cursor queries.

- [ ] **Step 4: Run memory tests**

Run: `pytest -s -q tests/test_memory.py`

Expected: PASS.

- [ ] **Step 5: Write the failing atomic clear test**

```python
def test_clear_context_atomically_resets_window_and_completes_active_task(stack) -> None:
    task = stack.confirm_recipe()
    stack.history.append(user_id="u", request="before", response="before")
    cleared = stack.service.clear_context("u", request_id="clear-1")
    assert cleared is not None
    assert cleared.status == TaskStatus.COMPLETED
    assert cleared.completion_reason == CompletionReason.CONTEXT_CLEARED
    assert stack.repository.get_active("u") is None
    assert stack.history.list_context_window("u") == []
```

Patch one transactional helper to raise and assert both the cursor and task
remain unchanged after rollback.

- [ ] **Step 6: Run the atomic clear test and verify failure**

Run: `pytest -s -q tests/test_tasks.py -k 'clear_context'`

Expected: FAIL because orchestration is missing.

- [ ] **Step 7: Implement one-connection atomic clear orchestration**

Allow the history cursor upsert and task completion helper to accept the same
`sqlite3.Connection`. `CookingTaskService.clear_context()` uses one
`BEGIN IMMEDIATE`, advances the cursor, completes the active task with
`context_cleared`, and commits both effects. It leaves no task when none is
active and remains idempotent.

- [ ] **Step 8: Run focused context and task tests**

Run: `pytest -s -q tests/test_memory.py tests/test_tasks.py`

Expected: PASS.

- [ ] **Step 9: Commit independent context clearing**

```bash
git add app/memory.py app/tasks.py tests/test_memory.py tests/test_tasks.py
git commit -m "feat: clear short context independently of tasks"
```

---

### Task 3: Bind Inventory Proposals to Cooked Tasks

**Files:**
- Modify: `app/actions.py`
- Modify: `tests/test_actions.py`

**Interfaces:**
- Changes: `PendingAction.source_task_id: str | None = None`.
- Changes: `PendingActionRepository.create(..., source_task_id: str | None = None)` persists the trusted source.
- Changes: `InventoryActionService.propose(..., source_task_id: str | None = None)` and `propose_add(..., source_task_id: str | None = None)` forward it.

- [ ] **Step 1: Write failing source migration and round-trip tests**

```python
def test_pending_action_round_trips_nullable_source_task_id(service) -> None:
    action = service.propose(
        "u",
        "inventory.remove",
        {"instance_id": service.inventory.list_instances()[0]["instance_id"]},
        source_task_id="task_0123456789abcdef0123456789abcdef",
    )
    assert service.repository.get(action.action_id).source_task_id == action.source_task_id
```

Create a legacy `pending_actions` table without the column, initialize the
repository, and assert old rows load with `None` without changing their status.
Test rejection of malformed task IDs.

- [ ] **Step 2: Run focused action tests and verify failure**

Run: `pytest -s -q tests/test_actions.py -k 'source_task or migration'`

Expected: FAIL because source-task persistence is missing.

- [ ] **Step 3: Implement the additive nullable migration and explicit reads**

Add `source_task_id TEXT` at the end of the fresh schema and migrate old tables
with `ALTER TABLE`. Replace repository `SELECT *` calls with one shared explicit
column list so fresh and migrated databases deserialize identically. Validate
non-null IDs against `task_[0-9a-f]{32}` before insert.

- [ ] **Step 4: Run all action tests**

Run: `pytest -s -q tests/test_actions.py`

Expected: PASS.

- [ ] **Step 5: Commit source binding**

```bash
git add app/actions.py tests/test_actions.py
git commit -m "feat: link inventory proposals to cooking tasks"
```

---

### Task 4: Pipeline Task Attachment and Reconciliation Mode

**Files:**
- Modify: `app/context.py`
- Modify: `app/agent.py`
- Modify: `app/pipeline.py`
- Modify: `app/prompts.py`
- Modify: `tests/test_agent_allergens.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `CookingTaskContext` with task ID, query, recipe title, servings, and ingredient strings.
- Changes: `AgentContext.cooking_task: CookingTaskContext | None` and `inventory_source_task_id: str | None`.
- Changes: `AgentResult.segment_id: str` and `recipe_options: list[dict[str, object]]`.
- Changes: `KitchenPipeline.process_request(..., reconciliation_task_id: str | None = None)` validates and binds completed cooked tasks.
- Changes: `KitchenPipeline` owns `tasks: CookingTaskService`.

- [ ] **Step 1: Write failing prompt payload and source-binding tests**

```python
def test_user_message_injects_task_separately_from_global_history() -> None:
    context = AgentContext(
        user_id="u",
        request_text="继续",
        history_segments=[HistoryExcerpt(request="global", response="history")],
        cooking_task=CookingTaskContext(
            task_id="task_0123456789abcdef0123456789abcdef",
            query_text="做汤",
            recipe_title="Tomato Soup",
            servings=2,
            ingredients=["2 tomatoes"],
        ),
    )
    payload = json.loads(KitchenAgent._user_message(context, "zh").split("\n", 1)[1])
    assert payload["recent_history"] == [{"request": "global", "response": "history"}]
    assert payload["cooking_task"]["recipe_title"] == "Tomato Soup"
```

Add a tool test proving `propose_replace_inventory_instance` and
`propose_remove_inventory_instance` receive the trusted
`inventory_source_task_id`, while ordinary requests create actions with `None`.

- [ ] **Step 2: Run Agent tests and verify failure**

Run: `pytest -s -q tests/test_agent_allergens.py -k 'cooking_task or source_task'`

Expected: FAIL because context fields and binding are missing.

- [ ] **Step 3: Implement context models, prompt payload, and proposal binding**

Add `cooking_task` as its own JSON property beside `recent_history`; do not put
it inside a history segment. In the two inventory mutation tool closures pass
`source_task_id=context.inventory_source_task_id`. Thread the same trusted value
through `_continue_request()` so a deterministic replace/remove proposal also
receives it. Add prompt rules that cooked mode permits only read tools and
independent replace/remove proposals, and requires a question whenever the
complete replacement text is unknown.

- [ ] **Step 4: Run Agent tests**

Run: `pytest -s -q tests/test_agent_allergens.py`

Expected: PASS.

- [ ] **Step 5: Write failing pipeline attachment, recipe-option, and reconciliation tests**

```python
def test_pipeline_returns_server_derived_recipe_options_and_segment_id(stack) -> None:
    result = stack.pipeline.process_request(user_id="u", text="找番茄汤")
    assert result.segment_id.startswith("hist_")
    assert result.recipe_options == [_recipe_result()]
    persisted = stack.pipeline.history.get_evidence("u", [result.segment_id])[0]
    assert persisted.tool_history[-1]["tool"] == "search_recipes"
```

Also test active task injection without changing the same ten history segments,
missing active task injection, valid cooked-task reconciliation binding,
cross-user and non-cooked task rejection, no automatic task creation during
ordinary requests, and proposal records carrying the cooked task ID.

- [ ] **Step 6: Run pipeline tests and verify failure**

Run: `pytest -s -q tests/test_pipeline.py -k 'task or recipe_options or reconciliation'`

Expected: FAIL because task wiring and response metadata are missing.

- [ ] **Step 7: Wire the service in `build_pipeline` and implement trusted modes**

Initialize task and history repositories on the same configured database. At
normal Agent context construction, map `tasks.repository.get_active(user_id)`
to `CookingTaskContext`. When `reconciliation_task_id` is present, call
`tasks.get_reconciliation_task(user_id, task_id)`, inject its snapshot, and set
`inventory_source_task_id`. Force the Agent path for reconciliation messages so
a short answer such as “番茄用完” cannot bypass cooked-task validation. After a
successful result, close reconciliation only when its tool history contains at
least one `inventory.replace` or `inventory.remove` pending action whose
`source_task_id` equals the validated task ID; otherwise keep it open for the
consolidated user answer or an explicit close request.

Before recording a result, set its reserved `segment_id`. Derive
`recipe_options` only by flattening list results from exact `search_recipes`
tool records and copying allowlisted recipe fields. Never derive them from
assistant prose.

- [ ] **Step 8: Run focused Agent and pipeline tests**

Run: `pytest -s -q tests/test_agent_allergens.py tests/test_pipeline.py`

Expected: PASS.

- [ ] **Step 9: Commit pipeline integration**

```bash
git add app/context.py app/agent.py app/pipeline.py app/prompts.py tests/test_agent_allergens.py tests/test_pipeline.py
git commit -m "feat: inject cooking tasks into agent requests"
```

---

### Task 5: Task, Cooked, and Context-Clear HTTP APIs

**Files:**
- Modify: `app/web.py`
- Modify: `tests/test_web_limits.py`

**Interfaces:**
- Produces the endpoints listed in design section 10.
- Changes: `/api/process` accepts optional `reconciliation_task_id` form data and passes it to the pipeline after trusted validation.

- [ ] **Step 1: Write failing task endpoint tests**

```python
def test_recipe_confirmation_uses_persisted_history_and_returns_active_task(web_stack) -> None:
    segment = web_stack.recipe_history()
    response = client.post(
        "/api/tasks/recipe",
        json={
            "segment_id": segment.segment_id,
            "recipe_id": "recipe-1",
            "servings": None,
            "expected_version": None,
            "request_id": "request-1",
        },
    )
    assert response.status_code == 200
    assert response.json()["task"]["status"] == "active"
```

Cover list/active, recipe-as-new, activate, servings, cooked,
reconciliation-close, and context-clear. Cover malformed bodies, missing
evidence, cross-user ownership through direct service fixtures, stale versions,
invalid transitions, repeated clicks, and absent active tasks.

- [ ] **Step 2: Run endpoint tests and verify 404 failures**

Run: `pytest -s -q tests/test_web_limits.py -k 'task or context_clear or reconciliation'`

Expected: FAIL because routes are absent.

- [ ] **Step 3: Implement bounded request models and exception mapping**

Use Pydantic request models with one-line IDs, positive servings, expected
versions greater than zero when present, and request IDs capped at 100
characters. Map `KeyError` to 404, `PermissionError` to 403, stale or invalid
transition `ValueError` to 409, and validation errors to FastAPI's bounded 422.

Return `task.model_dump(mode="json")`; never return raw recipe evidence rows or
database errors. Reload task state after each write. The cooked response also
returns `start_reconciliation`, copied from `complete_cooked()`'s changed flag,
so a duplicate click cannot launch a duplicate Agent turn.
`/api/context/clear` calls only `CookingTaskService.clear_context()`.

- [ ] **Step 4: Implement reconciliation process binding**

Add optional `reconciliation_task_id: str | None = Form(default=None)` to
`/api/process` and pass it to `KitchenPipeline.process_request`. Pipeline
validation remains authoritative; the browser-provided ID is only a lookup
request, not a trusted action source.

- [ ] **Step 5: Run web tests**

Run: `pytest -s -q tests/test_web_limits.py`

Expected: PASS.

- [ ] **Step 6: Commit the API surface**

```bash
git add app/web.py tests/test_web_limits.py
git commit -m "feat: expose cooking task controls"
```

---

### Task 6: Minimal Chat UI Attachment

**Files:**
- Create: `frontend/tasks.js`
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `frontend/styles.css`
- Create: `tests/test_frontend_tasks.py`

**Interfaces:**
- Produces: `renderTaskAttachment(task)`, `renderRecipeOptions(segmentId, recipes, activeTask)`, and encoded endpoint helpers under `globalThis.KitchenTasks` and `module.exports`.
- Consumes: task APIs and `AgentResult.segment_id`/`recipe_options`.

- [ ] **Step 1: Write failing Node renderer safety tests**

```python
def test_task_renderer_escapes_recipe_and_encodes_ids() -> None:
    completed = subprocess.run(
        ["node", "-e", TASK_RENDER_SCRIPT],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "<script>" not in completed.stdout
    assert "&lt;script&gt;" in completed.stdout
    assert "/api/tasks/task_%2Funsafe/cooked" in completed.stdout
```

Also assert recipe cards submit only segment ID, recipe ID, servings, version,
and request ID; no recipe title, instructions, ingredients, or URLs appear in
the request payload builder.

- [ ] **Step 2: Run renderer tests and verify missing module failure**

Run: `pytest -s -q tests/test_frontend_tasks.py -k 'renderer or endpoint'`

Expected: FAIL because `frontend/tasks.js` does not exist.

- [ ] **Step 3: Implement the pure rendering/helper module**

Use the existing `KitchenMarkdown.escapeHtml`; encode every path component with
`encodeURIComponent`; render buttons with data attributes rather than inline
handlers. Export the same API to the browser global and CommonJS for Node
tests.

- [ ] **Step 4: Write failing page integration source-contract tests**

Assert the HTML contains a task attachment region, paused-task selector,
servings control, cooked button, clear-context button, and `tasks.js` before
`app.js`. Assert `app.js` loads tasks on startup, refreshes after every task
write, renders recipe options from `data.recipe_options`, and submits cooked
reconciliation through `/api/process` with `reconciliation_task_id`.

- [ ] **Step 5: Run integration contract tests and verify failure**

Run: `pytest -s -q tests/test_frontend_tasks.py -k 'page or app'`

Expected: FAIL because the chat page has no task controls.

- [ ] **Step 6: Implement the minimal attachment and event handlers**

Add one compact task strip above the prompt. The selector shows paused tasks;
recipe cards appear beneath the Agent answer. **Cooked** first completes the
task, then sends the fixed reconciliation message with the returned task ID
only when `start_reconciliation` is true. An open completed reconciliation
attachment includes one **Skip inventory update** button that calls the close
endpoint. **Clear context** calls its endpoint, clears the visible answer/trace
state, and reloads tasks. Use disabled/working states and display bounded API
errors without optimistic success claims.

- [ ] **Step 7: Add responsive styles without creating a task page**

Reuse current colors, borders, typography, focus rings, and button patterns.
Keep the attachment one row on wide screens and stacked on narrow screens.
Do not add modal workflows or a reconciliation form.

- [ ] **Step 8: Run frontend and existing Markdown tests**

Run: `pytest -s -q tests/test_frontend_tasks.py tests/test_frontend_markdown.py`

Expected: PASS.

- [ ] **Step 9: Commit the UI**

```bash
git add frontend/tasks.js frontend/index.html frontend/app.js frontend/styles.css tests/test_frontend_tasks.py
git commit -m "feat: add inline cooking task controls"
```

---

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Force-add: `docs/superpowers/specs/2026-08-10-single-active-cooking-task-design.md`
- Force-add: `docs/superpowers/plans/2026-08-10-single-active-cooking-task.md`

**Interfaces:**
- Documents the exact persistence, API, prompt, lifecycle, context reset, and reviewed inventory contracts implemented by Tasks 1–6.

- [ ] **Step 1: Update README architecture and usage examples**

Document the `cooking_tasks` and `conversation_context_state` tables, one-active
constraint, recipe confirmation evidence, serving fallback, independent clear
semantics, task prompt attachment, cooked reconciliation, and nullable
`source_task_id`. State explicitly that task switching does not affect the ten
segment context and inventory uses replacement/removal proposals rather than
subtraction.

- [ ] **Step 2: Run focused feature tests**

Run:

```bash
pytest -s -q \
  tests/test_tasks.py \
  tests/test_memory.py \
  tests/test_actions.py \
  tests/test_agent_allergens.py \
  tests/test_pipeline.py \
  tests/test_web_limits.py \
  tests/test_frontend_tasks.py \
  tests/test_frontend_markdown.py
```

Expected: PASS with only the two existing dependency deprecation warnings.

- [ ] **Step 3: Run syntax and static artifact checks**

Run:

```bash
python3 -m compileall -q app tests
node --check frontend/tasks.js
node --check frontend/app.js
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Run the complete regression suite**

Run: `pytest -s -q`

Expected: all tests pass with no new warnings or failures; the baseline before
implementation is 323 passed and two existing dependency warnings.

- [ ] **Step 5: Review scope and migration safety**

Run:

```bash
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
git status --short
```

Confirm no task ID was added to conversation history, no quantity arithmetic
was added, old pending actions remain readable, permanent history remains
searchable after clear, and only intended files changed.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md
git add -f \
  docs/superpowers/specs/2026-08-10-single-active-cooking-task-design.md \
  docs/superpowers/plans/2026-08-10-single-active-cooking-task.md
git commit -m "docs: document single active cooking tasks"
```
