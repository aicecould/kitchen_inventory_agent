const { escapeHtml, renderSafeMarkdown } = globalThis.KitchenMarkdown;
const form = document.querySelector("#agent-form");
const promptInput = document.querySelector("#prompt");
const charCount = document.querySelector("#char-count");
const imageInput = document.querySelector("#image-input");
const dropzone = document.querySelector("#dropzone");
const fileTitle = document.querySelector("#file-title");
const fileMeta = document.querySelector("#file-meta");
const clearImageButton = document.querySelector("#clear-image");
const submitButton = document.querySelector("#submit-button");
const resultSection = document.querySelector("#result-section");
const resultState = document.querySelector("#result-state");
const answerContent = document.querySelector("#answer-content");
const toolTrace = document.querySelector("#tool-trace");
const inventoryList = document.querySelector("#inventory-list");
const globalStatus = document.querySelector("#global-status");
const proactiveReminder = document.querySelector("#proactive-reminder");
const proactiveReminderMessage = proactiveReminder.querySelector(".proactive-reminder-message");
const confirmationSection = document.querySelector("#confirmation-section");
const confirmationList = document.querySelector("#confirmation-list");
const visionCheckpoint = document.querySelector("#vision-checkpoint");
const visionConfirmationForm = document.querySelector("#vision-confirmation-form");
const visionCandidateList = document.querySelector("#vision-candidate-list");
const discardVisionCandidates = document.querySelector("#discard-vision-candidates");
const visionCheckpointStatus = document.querySelector("#vision-checkpoint-status");
const allergenDialog = document.querySelector("#allergen-dialog");
const allergenForm = document.querySelector("#allergen-form");
const allergenOptions = document.querySelector("#allergen-options");
const customAllergens = document.querySelector("#custom-allergens");
const allergenSaveStatus = document.querySelector("#allergen-save-status");

const allergenLabels = {
  Dairy: "乳制品",
  Egg: "蛋类",
  Gluten: "麸质",
  Grain: "谷物",
  Peanut: "花生",
  Seafood: "海鲜",
  Sesame: "芝麻",
  Shellfish: "甲壳类",
  Soy: "大豆",
  Sulfite: "亚硫酸盐",
  "Tree Nut": "坚果",
  Wheat: "小麦",
};

const requestLimits = {
  maxTextChars: 2000,
  maxImageBytes: 3 * 1024 * 1024,
};
const maxProactiveReminderChars = 100;

promptInput.addEventListener("input", () => {
  charCount.textContent = promptInput.value.length;
  promptInput.setCustomValidity(
    promptInput.value.length > requestLimits.maxTextChars
      ? `文字输入不能超过 ${requestLimits.maxTextChars} 个字符。`
      : "",
  );
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt;
    promptInput.dispatchEvent(new Event("input"));
    promptInput.focus();
  });
});

async function loadAllergens() {
  allergenSaveStatus.textContent = "正在读取…";
  const response = await fetch("/api/allergens");
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "过敏原设置读取失败");
  const selected = new Set(data.broad || []);
  allergenOptions.innerHTML = (data.options || []).map((value) => `
    <label class="allergen-option">
      <input type="checkbox" name="broad-allergen" value="${escapeHtml(value)}" ${selected.has(value) ? "checked" : ""} />
      <span>${escapeHtml(allergenLabels[value] || value)}</span>
      <small>${escapeHtml(value)}</small>
    </label>
  `).join("");
  customAllergens.value = (data.custom || []).join("，");
  allergenSaveStatus.textContent = "";
}

document.querySelector("#open-allergen-settings").addEventListener("click", async () => {
  allergenDialog.showModal();
  try {
    await loadAllergens();
  } catch (error) {
    allergenSaveStatus.textContent = error.message;
  }
});

document.querySelector("#close-allergen-settings").addEventListener("click", () => {
  allergenDialog.close();
});

allergenForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const broad = Array.from(
    allergenOptions.querySelectorAll('input[name="broad-allergen"]:checked'),
    (input) => input.value,
  );
  const custom = Array.from(new Set(
    customAllergens.value.split(/[，,\n]/).map((value) => value.trim()).filter(Boolean),
  ));
  if (custom.length > 30) {
    allergenSaveStatus.textContent = "自定义过敏食材不能超过 30 项。";
    return;
  }
  allergenSaveStatus.textContent = "正在保存…";
  try {
    const response = await fetch("/api/allergens", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ broad, custom }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "保存失败");
    allergenSaveStatus.textContent = `已保存 ${data.broad.length + data.custom.length} 项过敏原。`;
  } catch (error) {
    allergenSaveStatus.textContent = error.message;
  }
});

function resetImageInput() {
  imageInput.value = "";
  imageInput.setCustomValidity("");
  dropzone.classList.remove("has-file");
  clearImageButton.hidden = true;
  fileTitle.textContent = "放一张食材照片";
  fileMeta.textContent = "JPG / PNG / BMP · ≤3 MiB · 15–4096px · ≤3:1";
}

function rejectImage(message) {
  imageInput.value = "";
  imageInput.setCustomValidity(message);
  dropzone.classList.remove("has-file");
  clearImageButton.hidden = true;
  fileTitle.textContent = "图片不符合要求";
  fileMeta.textContent = message;
  imageInput.reportValidity();
}

async function updateFile(file) {
  if (!file) {
    resetImageInput();
    return true;
  }
  if (!new Set(["image/jpeg", "image/png", "image/bmp"]).has(file.type)) {
    rejectImage("仅支持 JPG、PNG 或 BMP 图片。");
    return false;
  }
  if (file.size > requestLimits.maxImageBytes) {
    rejectImage(`图片不能超过 ${(requestLimits.maxImageBytes / 1024 / 1024).toFixed(0)} MiB。`);
    return false;
  }
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    rejectImage("图片无法解码，请重新选择。");
    return false;
  }
  const { width, height } = bitmap;
  bitmap.close();
  if (Math.min(width, height) < 15 || Math.max(width, height) > 4096) {
    rejectImage("图片边长必须在 15–4096px 之间。");
    return false;
  }
  if (Math.max(width, height) / Math.min(width, height) > 3) {
    rejectImage("图片长宽比不能超过 3:1。");
    return false;
  }
  imageInput.setCustomValidity("");
  dropzone.classList.add("has-file");
  clearImageButton.hidden = false;
  fileTitle.textContent = file.name;
  fileMeta.textContent = `${width}×${height}px · ${(file.size / 1024 / 1024).toFixed(2)} MiB · 点击可更换`;
  return true;
}

imageInput.addEventListener("change", async () => updateFile(imageInput.files[0]));
clearImageButton.addEventListener("click", resetImageInput);
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});
dropzone.addEventListener("drop", async (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  await updateFile(file);
});

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (data.limits) {
      requestLimits.maxTextChars = data.limits.max_text_chars;
      requestLimits.maxImageBytes = data.limits.max_image_bytes;
      promptInput.maxLength = requestLimits.maxTextChars;
      document.querySelector("#text-limit").textContent = requestLimits.maxTextChars;
      promptInput.dispatchEvent(new Event("input"));
      if (imageInput.files[0]) await updateFile(imageInput.files[0]);
    }
    globalStatus.classList.toggle("ready", data.ready);
    globalStatus.querySelector("span:last-child").textContent = data.ready ? "Agent 已就绪" : "等待 API 配置";
  } catch {
    globalStatus.querySelector("span:last-child").textContent = "服务不可达";
  }
}

async function loadInventory() {
  inventoryList.innerHTML = '<p class="empty-note">正在翻找储物柜…</p>';
  try {
    const response = await fetch("/api/inventory");
    const data = await response.json();
    if (!data.items.length) {
      inventoryList.innerHTML = '<p class="empty-note">储物柜还是空的。告诉 Agent 你买了什么吧。</p>';
      return;
    }
    inventoryList.innerHTML = data.items.map((item) => `
      <article class="inventory-item">
        <header class="inventory-item-heading">
          <strong>${escapeHtml(item.name)}</strong>
          <span class="inventory-instance-id" title="完整库存实例 ID">${escapeHtml(item.instance_id)}</span>
        </header>
        <p class="inventory-quantity">${escapeHtml(item.quantity_text)}</p>
        <dl class="inventory-details">
          <div><dt>入库</dt><dd>${formatLocalDateTime(item.stocked_at)}</dd></div>
          <div><dt>过期</dt><dd>${formatLocalDateTime(item.expires_at, "过期未知")}</dd></div>
          <div><dt>确认</dt><dd>${formatLocalDateTime(item.last_confirmed_at)}</dd></div>
        </dl>
      </article>
    `).join("");
  } catch {
    inventoryList.innerHTML = '<p class="empty-note">库存暂时打不开，请稍后再试。</p>';
  }
}

async function loadProactiveReminder() {
  proactiveReminder.hidden = true;
  proactiveReminderMessage.textContent = "";
  try {
    const response = await fetch("/api/proactive-check", { method: "POST" });
    const data = await response.json();
    const message = typeof data.message === "string" ? data.message.trim() : "";
    if (!response.ok || !message || message.length > maxProactiveReminderChars || /[\r\n]/.test(message)) return;
    proactiveReminderMessage.textContent = message;
    proactiveReminder.hidden = false;
  } catch {
    // A reminder must never block the independent pantry and status requests.
  }
}

async function loadPendingActions() {
  try {
    const response = await fetch("/api/actions");
    const data = await response.json();
    const actions = data.actions || [];
    confirmationSection.hidden = actions.length === 0;
    confirmationList.innerHTML = actions.map((action) => {
      const previewReady = action.preview
        && typeof action.preview === "object"
        && Object.keys(action.preview).length > 0;
      const confirmAttributes = previewReady ? "" : "disabled hidden aria-disabled=\"true\"";
      return `
      <article class="confirmation-card" data-action-id="${escapeHtml(action.action_id)}" data-preview-ready="${previewReady}">
        <div class="confirmation-seal">待<br>确认</div>
        <div class="confirmation-copy">
          <span>${escapeHtml(operationLabel(action.operation))}</span>
          <h3>${escapeHtml(action.summary)}</h3>
          ${renderActionPreview(action)}
          <small>${escapeHtml(action.action_id)} · 过期时间 ${formatTime(action.expires_at)}</small>
        </div>
        <div class="confirmation-actions">
          <button type="button" class="cancel-action" data-action="cancel">取消</button>
          <button type="button" class="confirm-action" data-action="confirm" ${confirmAttributes}>确认执行</button>
        </div>
      </article>
    `;
    }).join("");
  } catch {
    confirmationSection.hidden = true;
  }
}

confirmationList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const card = button.closest("[data-action-id]");
  const actionId = card.dataset.actionId;
  const action = button.dataset.action;
  card.classList.add("working");
  card.querySelectorAll("button").forEach((item) => { item.disabled = true; });
  try {
    const response = await fetch(`/api/actions/${encodeURIComponent(actionId)}/${action}`, {
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "操作失败");
    card.classList.add(action === "confirm" ? "confirmed" : "cancelled");
    card.querySelector("h3").textContent = action === "confirm"
      ? "操作已执行"
      : (data.message || "库存操作已取消，库存未发生变化。");
    await loadInventory();
    window.setTimeout(loadPendingActions, 550);
  } catch (error) {
    card.classList.remove("working");
    card.querySelector("small").textContent = error.message;
    card.querySelectorAll("button").forEach((item) => { item.disabled = false; });
  }
});

function showVisionCheckpoint(confirmation) {
  visionCheckpoint.dataset.confirmationId = confirmation.confirmation_id;
  visionCandidateList.innerHTML = (confirmation.candidates || []).map((candidate, index) => `
    <article class="vision-candidate" data-candidate-index="${index}">
      <input
        class="vision-candidate-toggle"
        type="checkbox"
        aria-label="采用候选 ${escapeHtml(candidate.name)}"
      />
      <div class="vision-candidate-copy">
        <span>识别候选<strong>${escapeHtml(candidate.name)}</strong></span>
        <label>
          校对后的名称
          <input class="vision-candidate-name" type="text" maxlength="100" value="${escapeHtml(candidate.name)}" disabled />
        </label>
      </div>
      <span class="vision-confidence">${(Number(candidate.confidence) * 100).toFixed(1)}%</span>
    </article>
  `).join("");
  visionCheckpointStatus.textContent = "";
  visionCheckpoint.hidden = false;
  visionCheckpoint.scrollIntoView({ behavior: "smooth", block: "center" });
}

visionCandidateList.addEventListener("change", (event) => {
  if (!event.target.matches(".vision-candidate-toggle")) return;
  const card = event.target.closest(".vision-candidate");
  const nameInput = card.querySelector(".vision-candidate-name");
  nameInput.disabled = !event.target.checked;
  if (event.target.checked) {
    nameInput.focus();
    nameInput.select();
  }
});

async function resolveVisionCheckpoint(selections) {
  const confirmationId = visionCheckpoint.dataset.confirmationId;
  const buttons = visionConfirmationForm.querySelectorAll("button");
  buttons.forEach((button) => { button.disabled = true; });
  visionCheckpointStatus.textContent = "已记录校对，正在继续原请求…";
  resultState.textContent = "继续处理";
  answerContent.innerHTML = document.querySelector("#loading-template").innerHTML;
  try {
    const response = await fetch(
      `/api/vision-confirmations/${encodeURIComponent(confirmationId)}/resolve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selections }),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "图片校对结果提交失败");
    visionCheckpoint.hidden = true;
    renderAgentResult(data);
    await loadInventory();
    await loadPendingActions();
  } catch (error) {
    visionCheckpointStatus.textContent = error.message;
    resultState.textContent = "等待校对";
    answerContent.textContent = "校对结果尚未提交，可以修改后重试。";
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
  }
}

visionConfirmationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selections = Array.from(
    visionCandidateList.querySelectorAll(".vision-candidate"),
  ).filter(
    (card) => card.querySelector(".vision-candidate-toggle").checked,
  ).map((card) => ({
    candidate_index: Number(card.dataset.candidateIndex),
    name: card.querySelector(".vision-candidate-name").value.trim(),
  }));
  if (selections.some((selection) => !selection.name)) {
    visionCheckpointStatus.textContent = "已选择的候选必须填写食材名称。";
    return;
  }
  await resolveVisionCheckpoint(selections);
});

discardVisionCandidates.addEventListener("click", async () => {
  await resolveVisionCheckpoint([]);
});

function renderAgentResult(data) {
  answerContent.innerHTML = renderSafeMarkdown(data.content);
  resultState.textContent = data.blocked ? "已拦截" : "完成";
  resultState.classList.toggle("blocked", data.blocked);
  renderExecutionTrace(data.execution_trace || []);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  const imageIsValid = !file || await updateFile(file);
  if (
    !promptInput.value.length
    || promptInput.value.length > requestLimits.maxTextChars
    || !imageIsValid
  ) {
    promptInput.dispatchEvent(new Event("input"));
    form.reportValidity();
    return;
  }
  submitButton.disabled = true;
  resultSection.hidden = false;
  resultState.textContent = "处理中";
  resultState.classList.remove("blocked");
  answerContent.innerHTML = document.querySelector("#loading-template").innerHTML;
  toolTrace.innerHTML = '<p class="empty-note">等待后端执行轨迹…</p>';
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });

  const payload = new FormData(form);
  if (!imageInput.files.length) payload.delete("image");
  try {
    const response = await fetch("/api/process", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "请求失败");

    renderAgentResult(data);
    if (data.awaiting_vision_confirmation && data.vision_confirmation) {
      resultState.textContent = "等待校对";
      showVisionCheckpoint(data.vision_confirmation);
      resetImageInput();
      return;
    }
    visionCheckpoint.hidden = true;
    await loadInventory();
    await loadPendingActions();
  } catch (error) {
    answerContent.textContent = error.message;
    resultState.textContent = "出错";
    resultState.classList.add("blocked");
    toolTrace.innerHTML = '<p class="empty-note">本次请求没有可展示的工具记录。</p>';
  } finally {
    submitButton.disabled = false;
  }
});

function renderExecutionTrace(trace) {
  if (!trace.length) {
    toolTrace.innerHTML = '<p class="empty-note">本次请求没有可展示的执行轨迹。</p>';
    return;
  }
  toolTrace.innerHTML = trace.map((entry, index) => `
    <div class="trace-entry trace-${escapeHtml(entry.status)}">
      <strong>${String(index + 1).padStart(2, "0")} · ${escapeHtml(entry.stage)} / ${escapeHtml(entry.name)}</strong>
      <small>${escapeHtml(entry.status)} · ${Number(entry.duration_ms || 0)} ms · ${escapeHtml(shorten(entry.detail || "", 180))}</small>
    </div>
  `).join("");
}

function shorten(value, limit) {
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function previewField(label, value, className = "") {
  const displayValue = value === null || value === undefined || value === ""
    ? "未知"
    : String(value);
  return `
    <div class="${escapeHtml(className)}">
      <dt>${escapeHtml(label)}</dt>
      <dd>${escapeHtml(displayValue)}</dd>
    </div>
  `;
}

function previewDate(value) {
  return value ? formatLocalDateTime(value, "未知") : "未知";
}

function currentInstanceFields(current, { includeConfirmed = false } = {}) {
  if (!current || typeof current !== "object") return [];
  const fields = [
    previewField("完整实例 ID", current.instance_id, "confirmation-preview-id"),
    previewField("食材", current.name),
    previewField("当前数量", current.quantity_text),
    previewField("当前过期时间", previewDate(current.expires_at)),
    previewField("日期来源", current.expiry_source),
  ];
  if (includeConfirmed) {
    fields.push(previewField("上次确认时间", previewDate(current.last_confirmed_at)));
  }
  return fields;
}

function renderActionPreview(action) {
  const preview = action.preview && typeof action.preview === "object"
    ? action.preview
    : {};
  let fields = [];
  let items = "";

  if (action.operation === "inventory.add") {
    const proposed = preview.proposed || {};
    fields = [
      previewField("食材", proposed.name),
      previewField("拟新增数量", proposed.quantity_text),
      previewField("拟定过期时间", previewDate(proposed.expires_at)),
      previewField("日期来源", proposed.expiry_source),
    ];
  } else if (action.operation === "inventory.replace") {
    const proposed = preview.proposed || {};
    fields = [
      ...currentInstanceFields(preview.current),
      previewField("拟替换数量", proposed.quantity_text),
      previewField("拟定过期时间", previewDate(proposed.expires_at)),
      previewField("拟定日期来源", proposed.expiry_source),
    ];
  } else if (action.operation === "inventory.remove") {
    fields = currentInstanceFields(preview.current);
  } else if (action.operation === "inventory.confirm") {
    fields = currentInstanceFields(preview.current, { includeConfirmed: true });
  } else if (action.operation === "inventory.remove_all") {
    fields = [
      previewField("精确食材名称", preview.name),
      previewField("预计删除数量", preview.expected_count),
    ];
    const previewItems = Array.isArray(preview.items) ? preview.items : [];
    items = `
      <ul class="confirmation-preview-items">
        ${previewItems.map((item) => `
          <li>
            <code>${escapeHtml(item.instance_id)}</code>
            <span>${escapeHtml(item.name)} · ${escapeHtml(item.quantity_text)} · 过期 ${escapeHtml(previewDate(item.expires_at))} · 日期来源 ${escapeHtml(item.expiry_source)}</span>
          </li>
        `).join("")}
      </ul>
    `;
  }

  if (!fields.length && !items) {
    return '<p class="confirmation-preview-empty">暂无结构化预览，请取消并重新创建提案。</p>';
  }
  return `
    <section class="confirmation-preview" data-preview-operation="${escapeHtml(action.operation)}" aria-label="操作预览">
      <dl>${fields.join("")}</dl>
      ${items}
    </section>
  `;
}

function operationLabel(operation) {
  return ({
    "inventory.add": "库存新增",
    "inventory.replace": "库存替换",
    "inventory.remove": "单项删除",
    "inventory.remove_all": "同名全部删除",
    "inventory.confirm": "库存校准",
  })[operation] || operation;
}

function formatTime(value) {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function formatLocalDateTime(value, fallback = "时间未知") {
  if (!value) return fallback;
  const time = new Date(value);
  if (Number.isNaN(time.getTime())) return fallback;
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(time);
}

document.querySelector("#refresh-inventory").addEventListener("click", loadInventory);
loadStatus();
loadInventory();
loadPendingActions();
loadProactiveReminder();
