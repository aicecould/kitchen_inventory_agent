(function installProfileProposalCards(root, factory) {
  const escape = root && root.KitchenMarkdown
    ? root.KitchenMarkdown.escapeHtml
    : require("./markdown.js").escapeHtml;
  const api = factory(escape);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.KitchenProfileProposals = api;
})(typeof globalThis === "undefined" ? undefined : globalThis, (escapeHtml) => {
  const categoryLabels = {
    kitchen_setup: "厨房情况",
    preference: "饮食偏好",
    requirement: "必须要求",
    default_context: "默认情景",
  };

  function renderProfileProposalCards(proposals) {
    return (proposals || []).map((proposal) => {
      const preview = proposal.preview || {};
      const current = preview.current && preview.current.value
        ? preview.current.value
        : "暂无";
      const proposed = proposal.operation === "delete"
        ? "删除该事实"
        : proposal.value || "暂无";
      const evidence = (proposal.evidence_ids || [])
        .map((value) => `<code>${escapeHtml(value)}</code>`)
        .join(" ");
      return `
        <article class="profile-proposal-card" data-proposal-id="${escapeHtml(proposal.proposal_id)}">
          <div class="profile-proposal-heading">
            <span>待确认画像变更</span>
            <strong>${escapeHtml(categoryLabels[proposal.category] || proposal.category)}</strong>
          </div>
          <dl>
            <div><dt>当前</dt><dd>${escapeHtml(current)}</dd></div>
            <div><dt>写入后</dt><dd>${escapeHtml(proposed)}</dd></div>
          </dl>
          <p>${escapeHtml(proposal.reason || "")}</p>
          <details><summary>依据记录</summary>${evidence}</details>
          <div class="profile-proposal-actions">
            <button type="button" data-decision="reject">拒绝</button>
            <button type="button" data-decision="confirm">确认写入</button>
          </div>
          <p class="profile-proposal-status" aria-live="polite"></p>
        </article>`;
    }).join("");
  }

  function profileProposalEndpoint(proposalId, decision) {
    if (decision !== "confirm" && decision !== "reject") {
      throw new Error("Unsupported profile proposal decision");
    }
    return `/api/profile-proposals/${encodeURIComponent(proposalId)}/${decision}`;
  }

  return { renderProfileProposalCards, profileProposalEndpoint };
});
