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
const serviceGrid = document.querySelector("#service-grid");
const confirmationSection = document.querySelector("#confirmation-section");
const confirmationList = document.querySelector("#confirmation-list");

const serviceNames = {
  deepseek: ["DeepSeek", "Agent / Tool Calling"],
  baidu_vision: ["百度识图", "物体与场景识别"],
  baidu_translate: ["百度翻译", "多语言菜谱输出"],
  spoonacular: ["Spoonacular", "菜谱路由 A"],
  themealdb: ["TheMealDB", "菜谱路由 B"],
};

promptInput.addEventListener("input", () => {
  charCount.textContent = promptInput.value.length;
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt;
    promptInput.dispatchEvent(new Event("input"));
    promptInput.focus();
  });
});

function updateFile(file) {
  if (!file) {
    dropzone.classList.remove("has-file");
    fileTitle.textContent = "放一张食材照片";
    fileMeta.textContent = "PNG / JPG / WEBP · 不超过 8 MiB";
    return;
  }
  dropzone.classList.add("has-file");
  fileTitle.textContent = file.name;
  fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MiB · 点击可更换`;
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
    globalStatus.classList.toggle("ready", data.ready);
    globalStatus.querySelector("span:last-child").textContent = data.ready ? "Agent 已就绪" : "等待 API 配置";
    serviceGrid.innerHTML = Object.entries(serviceNames).map(([key, labels], index) => `
      <article class="service-card ${data.services[key] ? "configured" : ""}">
        <span class="service-index">0${index + 1}</span>
        <h3>${labels[0]}</h3>
        <p>${labels[1]} · ${data.services[key] ? "已配置" : "待配置"}</p>
      </article>
    `).join("");
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
  submitButton.disabled = true;
  resultSection.hidden = false;
  resultState.textContent = "处理中";
  resultState.classList.remove("blocked");
  answerContent.innerHTML = document.querySelector("#loading-template").innerHTML;
  toolTrace.innerHTML = '<p class="empty-note">等待工具调用…</p>';
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
    renderToolTrace(data.tool_history || []);
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

function renderToolTrace(history) {
  if (!history.length) {
    toolTrace.innerHTML = '<p class="empty-note">Agent 本次直接作答，没有调用工具。</p>';
    return;
  }
  toolTrace.innerHTML = history.map((entry, index) => `
    <div class="trace-entry">
      <strong>${String(index + 1).padStart(2, "0")} · ${escapeHtml(entry.tool)}</strong>
      <small>${escapeHtml(shorten(displayResult(entry.result), 180))}</small>
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
