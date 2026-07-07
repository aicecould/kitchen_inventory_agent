const form = document.querySelector("#agent-form");
const promptInput = document.querySelector("#prompt");
const charCount = document.querySelector("#char-count");
const imageInput = document.querySelector("#image-input");
const dropzone = document.querySelector("#dropzone");
const fileTitle = document.querySelector("#file-title");
const fileMeta = document.querySelector("#file-meta");
const submitButton = document.querySelector("#submit-button");
const resultSection = document.querySelector("#result-section");
const resultState = document.querySelector("#result-state");
const answerContent = document.querySelector("#answer-content");
const toolTrace = document.querySelector("#tool-trace");
const inventoryList = document.querySelector("#inventory-list");
const globalStatus = document.querySelector("#global-status");
const confirmationSection = document.querySelector("#confirmation-section");
const confirmationList = document.querySelector("#confirmation-list");
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

function updateFile(file) {
  if (!file) {
    dropzone.classList.remove("has-file");
    fileTitle.textContent = "放一张食材照片";
    fileMeta.textContent = "PNG / JPG / BMP · 不超过 3 MiB";
    return;
  }
  if (file.size > requestLimits.maxImageBytes) {
    imageInput.value = "";
    imageInput.setCustomValidity(
      `图片不能超过 ${(requestLimits.maxImageBytes / 1024 / 1024).toFixed(0)} MiB。`,
    );
    dropzone.classList.remove("has-file");
    fileTitle.textContent = "图片太大，请重新选择";
    fileMeta.textContent = imageInput.validationMessage;
    imageInput.reportValidity();
    return false;
  }
  imageInput.setCustomValidity("");
  dropzone.classList.add("has-file");
  fileTitle.textContent = file.name;
  fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MiB · 点击可更换`;
  return true;
}

imageInput.addEventListener("change", () => updateFile(imageInput.files[0]));
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
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  updateFile(file);
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
      document.querySelector("#file-limit").textContent =
        `${(requestLimits.maxImageBytes / 1024 / 1024).toFixed(0)} MiB`;
      promptInput.dispatchEvent(new Event("input"));
      if (imageInput.files[0]) updateFile(imageInput.files[0]);
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
      <div class="inventory-item">
        <strong>${escapeHtml(item.name)}</strong>
        <span>${item.quantity} ${escapeHtml(item.unit)}</span>
      </div>
    `).join("");
  } catch {
    inventoryList.innerHTML = '<p class="empty-note">库存暂时打不开，请稍后再试。</p>';
  }
}

async function loadPendingActions() {
  try {
    const response = await fetch("/api/actions");
    const data = await response.json();
    const actions = data.actions || [];
    confirmationSection.hidden = actions.length === 0;
    confirmationList.innerHTML = actions.map((action) => `
      <article class="confirmation-card" data-action-id="${escapeHtml(action.action_id)}">
        <div class="confirmation-seal">待<br>确认</div>
        <div class="confirmation-copy">
          <span>${escapeHtml(operationLabel(action.operation))}</span>
          <h3>${escapeHtml(action.summary)}</h3>
          <small>${escapeHtml(action.action_id)} · 过期时间 ${formatTime(action.expires_at)}</small>
        </div>
        <div class="confirmation-actions">
          <button type="button" class="cancel-action" data-action="cancel">取消</button>
          <button type="button" class="confirm-action" data-action="confirm">确认执行</button>
        </div>
      </article>
    `).join("");
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
    card.querySelector("h3").textContent = action === "confirm" ? "操作已执行" : "操作已取消";
    await loadInventory();
    window.setTimeout(loadPendingActions, 550);
  } catch (error) {
    card.classList.remove("working");
    card.querySelector("small").textContent = error.message;
    card.querySelectorAll("button").forEach((item) => { item.disabled = false; });
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (
    !promptInput.value.length
    || promptInput.value.length > requestLimits.maxTextChars
    || (file && file.size > requestLimits.maxImageBytes)
  ) {
    if (file) updateFile(file);
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

    answerContent.textContent = data.content;
    resultState.textContent = data.blocked ? "已拦截" : "完成";
    resultState.classList.toggle("blocked", data.blocked);
    renderExecutionTrace(data.execution_trace || []);
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

function displayResult(value) {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function operationLabel(operation) {
  return ({
    "inventory.add": "库存新增",
    "inventory.update": "库存修改",
    "inventory.remove": "库存删除",
  })[operation] || operation;
}

function formatTime(value) {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

document.querySelector("#refresh-inventory").addEventListener("click", loadInventory);
loadStatus();
loadInventory();
loadPendingActions();
