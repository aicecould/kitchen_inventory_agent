(function (root) {
  async function readQueryStream(response, onEvent) {
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "请求失败");
    }
    if (!response.body) throw new Error("浏览器不支持流式读取日志。");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result;
    let completed = false;
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) >= 0) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const type = frame.split("\n").find((line) => line.startsWith("event: "))?.slice(7);
          const data = frame.split("\n").filter((line) => line.startsWith("data: ")).map((line) => line.slice(6)).join("\n");
          if (!type || !data) continue;
          const payload = JSON.parse(data);
          if (type === "error") throw new Error(payload.detail || "请求失败");
          if (type === "result") { result = payload; completed = true; }
          else onEvent(type, payload);
        }
        if (done) break;
      }
      if (!completed) throw new Error("连接已中断，请查看已有日志和任务状态后再操作。");
      return result;
    } finally {
      await reader.cancel();
      reader.releaseLock();
    }
  }

  const kindLabels = {
    request: "请求", component: "处理步骤", event: "执行记录", llm: "模型调用", tool: "工具调用",
  };
  const stageLabels = {
    request: "请求处理", routing: "意图判断", database: "库存操作", model: "模型处理",
    agent: "主助手处理", output: "回答检查", supervisor: "回答监督", profile: "用户画像",
    external_api: "外部服务", human_checkpoint: "人工确认",
  };
  const stepLabels = {
    process_request: "处理本轮请求", resolve_vision_confirmation: "继续处理图片校对结果",
    kitchen_agent: "主助手生成回答", qwen_agent: "本地模型生成回答", deepseek_agent: "云端模型生成回答",
    deterministic_route: "匹配快捷处理规则", food_query_classifier: "识别食材请求",
    food_query_classifier_fallback: "食材识别备用处理", intent_match: "识别请求意图",
    input_validation: "校验输入", input_limit: "检查输入长度", request_cancel: "识别取消请求",
    inventory_query_regex: "匹配库存查询", inventory_write_regex: "匹配库存变更",
    list_inventory: "查看库存清单", list_inventory_instances: "查看库存清单",
    find_inventory: "查找库存食材", find_inventory_instances: "查找库存食材",
    get_inventory_instance: "读取食材详情", inventory_recipe_context: "读取菜谱所需库存",
    create_pending_action: "创建待确认操作", propose_add_inventory_instance: "提议新增库存",
    propose_replace_inventory_instance: "提议替换库存", propose_remove_inventory_instance: "提议删除一项库存",
    propose_remove_all_inventory_instances: "提议删除同名库存", propose_confirm_inventory_instance: "提议校准库存",
    inventory_confirmation: "确认库存操作", propose_profile_change: "提议更新用户画像",
    profile_confirmation: "确认画像变更", profile_summary: "整理用户画像",
    search_conversation_history_by_time: "查询历史对话", expiry_inference: "估算食材保质期",
    search_recipes: "搜索菜谱", recipe_web_search: "联网搜索菜谱", deepseek_web_search: "调用联网搜索模型",
    baidu_vision: "识别图片食材", low_confidence_image: "校对图片识别结果",
    proactive_reminder: "生成主动提醒", regex_audit: "检查回答内容", tool_call_audit: "核对工具使用",
    final_text: "检查最终回答", response_audit: "审核回答", response_repair: "修复回答",
    supervisor_classification: "判断回答修复方式", supervisor_evaluation: "评估回答质量",
    subagent_rewrite: "修复助手重写回答", subagent_reaudit: "复核修复后的回答",
    main_agent_retry: "主助手重新尝试", answer_rewrite: "重写最终回答", recipe_recovery: "补充菜谱依据",
    final_fallback: "生成兜底答复", fts5_recall: "按关键词检索菜谱", faiss_recall: "按语义检索菜谱",
    rrf_fusion: "合并菜谱检索结果", recipe_filter: "筛选合适菜谱", recipe_ranking: "排列推荐菜谱",
    local: "检索本地菜谱", spoonacular: "查询第一菜谱服务", themealdb: "查询第二菜谱服务",
    gate_cost_probe: "评估模型调用开销",
  };
  const statusLabels = {
    running: "执行中", success: "完成", failed: "失败", passed: "检查通过", blocked: "已拦截",
    completed: "完成", pending: "等待确认", resolved: "已确认", matched: "已匹配",
    skipped: "已跳过", unavailable: "暂不可用", degraded: "已使用备用方案", "fail-open": "已转为备用处理",
  };
  const failureLabels = {
    ValidationError: "返回数据校验未通过", OutputParserException: "模型返回内容解析失败",
    ValueError: "数据不符合处理要求", TypeError: "数据类型不符合要求", KeyError: "缺少所需数据",
    OpenAIAPIError: "模型服务调用异常", APIError: "模型服务调用异常",
    APIConnectionError: "无法连接模型服务", ConnectError: "无法连接外部服务",
    APITimeoutError: "模型服务响应超时", TimeoutError: "处理超时", ReadTimeout: "等待服务响应超时",
    ConnectTimeout: "连接服务超时", PoolTimeout: "等待可用连接超时",
    HTTPStatusError: "外部服务返回异常", BadRequestError: "模型服务未接受请求",
    AuthenticationError: "服务身份验证失败", PermissionDeniedError: "服务访问权限不足",
    RateLimitError: "服务请求过于频繁或额度不足", InternalServerError: "模型服务内部异常",
    NotFoundError: "服务资源不存在", ConnectionError: "服务连接异常",
    GraphRecursionError: "处理步骤达到上限", RuntimeError: "执行过程出现异常",
    task_impossible: "当前请求无法完成", retry_exhausted: "重试次数已用尽",
    invalid_supervisor_output: "回答监督结果不符合要求", recipe_tool_missing: "缺少菜谱查询依据",
    recipe_result_mismatch: "回答与菜谱结果不一致", repairable_output_violation: "回答内容未通过检查",
    user_cancelled: "用户已取消请求",
  };

  function stepPresentation(step) {
    // Event rows describe the recording operation; their business status lives in content.
    const event = step.kind === "event" && step.content && typeof step.content === "object" ? step.content : null;
    const state = step.status === "failed" ? "failed" : event?.status || step.status;
    const name = stepLabels[step.name] || stageLabels[step.parameters?.stage] || kindLabels[step.kind] || "处理步骤";
    const status = statusLabels[state] || "状态待确认";
    const failure = ["failed", "blocked", "unavailable"].includes(state);
    const reason = failure ? failureLabels[step.error_type] || failureLabels[event?.detail]
      || (state === "failed" ? "执行异常，展开查看详情" : "展开查看具体原因") : "";
    const duration = event?.duration_ms ?? step.time?.duration_ms;
    return {
      state,
      text: `${kindLabels[step.kind] || "处理步骤"} / ${name} · ${status}${reason ? `：${reason}` : ""} · ${duration == null ? "计时中" : `${Math.round(duration)} 毫秒`}`,
    };
  }

  function createLogView(container, status) {
    const entries = new Map();
    return {
      reset() {
        entries.clear();
        container.replaceChildren();
        status.textContent = "等待本轮日志写入…";
      },
      update(type, data) {
        if (type === "log_status") { status.textContent = data.message; return; }
        if (type !== "steps") return;

        for (const step of data.steps) {
          let entry = entries.get(step.id);
          if (!entry) {
            const card = document.createElement("details");
            const summary = document.createElement("summary");
            const body = document.createElement("pre");
            card.append(summary, body);
            container.append(card);
            entry = { card, summary, body, index: entries.size + 1 };
            entries.set(step.id, entry);
          }
          const presentation = stepPresentation(step);
          entry.state = presentation.state;
          const style = ["running", "success", "failed", "blocked", "passed"].includes(entry.state) ? entry.state : "unknown";
          entry.card.className = `trace-entry trace-${style}`;
          entry.summary.textContent = `${String(entry.index).padStart(2, "0")} · ${presentation.text}`;
          // Only materialize potentially large prompts/results when expanded.
          entry.step = step;
          const show = () => { entry.body.textContent = JSON.stringify(entry.step, null, 2); };
          entry.card.ontoggle = () => { if (entry.card.open) show(); };
          if (entry.card.open) show();
        }
        const failed = Array.from(entries.values()).filter((entry) => entry.state === "failed").length;
        status.textContent = `本轮${statusLabels[data.status] || "状态待确认"} · 共 ${entries.size} 步${failed ? ` · ${failed} 步失败` : ""}`;
        status.title = `本轮日志编号：${data.query_id}`;
      },
    };
  }

  const api = { readQueryStream, createLogView };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.QueryLogs = api;
})(typeof window !== "undefined" ? window : globalThis);
