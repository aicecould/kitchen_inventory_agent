(function installKitchenTasks(root, factory) {
  const helpers = factory(root && root.KitchenMarkdown);
  if (typeof module !== "undefined" && module.exports) module.exports = helpers;
  if (root) root.KitchenTasks = helpers;
})(typeof globalThis === "undefined" ? undefined : globalThis, (markdown) => {
  const escapeHtml = markdown && markdown.escapeHtml
    ? markdown.escapeHtml
    : (value) => String(value).replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[character]);

  function taskEndpoint(taskId, action) {
    const encoded = encodeURIComponent(taskId);
    const paths = {
      activate: `/api/tasks/${encoded}/activate`,
      servings: `/api/tasks/${encoded}/servings`,
      cooked: `/api/tasks/${encoded}/cooked`,
      close: `/api/tasks/${encoded}/reconciliation/close`,
    };
    if (!paths[action]) throw new Error("Unsupported task action");
    return paths[action];
  }

  function buildRecipeRequest(values) {
    const rawServings = values.servings;
    return {
      segment_id: String(values.segmentId),
      recipe_id: String(values.recipeId),
      servings: rawServings === "" || rawServings === null || rawServings === undefined
        ? null
        : Number(rawServings),
      expected_version: values.expectedVersion === null || values.expectedVersion === undefined
        ? null
        : Number(values.expectedVersion),
      request_id: String(values.requestId),
    };
  }

  function reconciliationFields(taskId) {
    return {
      text: "我已完成这道菜，请核对本次使用后的库存。",
      language: "zh",
      reconciliation_task_id: String(taskId),
    };
  }

  function renderTaskPanel(tasks) {
    const values = Array.isArray(tasks) ? tasks : [];
    if (!values.length) {
      return '<p class="task-empty">尚未确认菜谱；task 挂件不会影响对话上下文。</p>';
    }
    return values.map((task) => {
      const recipe = task.recipe || {};
      const active = task.status === "active";
      const paused = task.status === "paused";
      const open = task.status === "completed" && task.reconciliation_open;
      const controls = active ? `
        <label class="task-servings">份数
          <input type="number" min="1" max="100" value="${escapeHtml(task.servings)}" />
        </label>
        <button type="button" data-task-action="servings">更新份数</button>
        <button type="button" class="task-primary" data-task-action="cooked">已烹饪</button>
      ` : paused ? `
        <button type="button" data-task-action="activate">切换到此 task</button>
      ` : open ? `
        <button type="button" data-task-action="close">跳过库存更新</button>
      ` : "";
      return `
        <article class="task-card task-${escapeHtml(task.status)}"
          data-task-id="${escapeHtml(task.task_id)}"
          data-task-version="${escapeHtml(task.version)}">
          <div class="task-copy">
            <span>${active ? "当前 task" : paused ? "暂停" : "等待库存核对"}</span>
            <strong>${escapeHtml(recipe.title || task.query_text)}</strong>
            <small>${escapeHtml(task.query_text)} · ${escapeHtml(task.servings)} 人份</small>
          </div>
          <div class="task-controls">${controls}</div>
        </article>
      `;
    }).join("");
  }

  function renderRecipeOptions(segmentId, recipes, activeTask) {
    if (!Array.isArray(recipes) || !recipes.length) return "";
    return recipes.map((recipe) => {
      const ingredients = Array.isArray(recipe.ingredients)
        ? recipe.ingredients.slice(0, 4).join(" · ")
        : "";
      return `
        <article class="recipe-option" data-segment-id="${escapeHtml(segmentId)}"
          data-recipe-id="${escapeHtml(recipe.id)}">
          <div>
            <span>RECIPE RESULT</span>
            <strong>${escapeHtml(recipe.title)}</strong>
            <small>${escapeHtml(ingredients)}</small>
          </div>
          <label>目标份数
            <input type="number" min="1" max="100" placeholder="画像默认 / 1" />
          </label>
          <div class="recipe-option-actions">
            <button type="button" data-recipe-action="confirm">确认此菜谱</button>
            ${activeTask ? '<button type="button" data-recipe-action="new">作为新 task</button>' : ""}
          </div>
        </article>
      `;
    }).join("");
  }

  return {
    taskEndpoint,
    buildRecipeRequest,
    reconciliationFields,
    renderTaskPanel,
    renderRecipeOptions,
  };
});
