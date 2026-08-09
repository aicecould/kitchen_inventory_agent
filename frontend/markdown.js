(function installKitchenMarkdown(root, factory) {
  const renderer = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = renderer;
  if (root) root.KitchenMarkdown = renderer;
})(typeof globalThis === "undefined" ? undefined : globalThis, () => {
  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[character]);
  }

  function safeHttpUrl(value) {
    try {
      const parsed = new URL(value);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
      return parsed.href;
    } catch {
      return null;
    }
  }

  function protectSafeLinks(markdown) {
    const links = [];
    const source = String(markdown || "").replace(
      /\[([^\]\n]+)\]\(([^)\s]+)\)/g,
      (matched, label, url) => {
        const safeUrl = safeHttpUrl(url);
        if (!safeUrl) return matched;
        const index = links.push({ label, url: safeUrl }) - 1;
        return `\uE100${index}\uE101`;
      },
    );
    return { source, links };
  }

  function renderInlineMarkdown(text, links) {
    const codeSpans = [];
    let rendered = text.replace(/`([^`\n]+)`/g, (_, code) => {
      const index = codeSpans.push(code) - 1;
      return `\uE000${index}\uE001`;
    });
    rendered = rendered
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
      .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
      .replace(/\uE000(\d+)\uE001/g, (_, index) => {
        return `<code>${codeSpans[Number(index)]}</code>`;
      });
    return rendered.replace(/\uE100(\d+)\uE101/g, (_, index) => {
      const link = links[Number(index)];
      return `<a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.label)}</a>`;
    });
  }

  function renderSafeMarkdown(markdown) {
    const protectedLinks = protectSafeLinks(markdown);
    const lines = escapeHtml(protectedLinks.source).replace(/\r\n?/g, "\n").split("\n");
    const output = [];
    let paragraph = [];
    let listType = null;
    let inCodeBlock = false;
    let codeLines = [];

    const renderInline = (value) => renderInlineMarkdown(value, protectedLinks.links);
    const closeParagraph = () => {
      if (!paragraph.length) return;
      output.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
      paragraph = [];
    };
    const closeList = () => {
      if (!listType) return;
      output.push(`</${listType}>`);
      listType = null;
    };
    const closeCodeBlock = () => {
      output.push(`<pre><code>${codeLines.join("\n")}</code></pre>`);
      codeLines = [];
      inCodeBlock = false;
    };

    lines.forEach((line) => {
      if (/^\s*```/.test(line)) {
        if (inCodeBlock) {
          closeCodeBlock();
        } else {
          closeParagraph();
          closeList();
          inCodeBlock = true;
        }
        return;
      }
      if (inCodeBlock) {
        codeLines.push(line);
        return;
      }
      if (!line.trim()) {
        closeParagraph();
        closeList();
        return;
      }
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        closeParagraph();
        closeList();
        const level = heading[1].length;
        output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        return;
      }
      if (/^\s*(?:---|___|\*\*\*)\s*$/.test(line)) {
        closeParagraph();
        closeList();
        output.push("<hr>");
        return;
      }
      const unordered = line.match(/^\s*[-*]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (unordered || ordered) {
        closeParagraph();
        const nextListType = unordered ? "ul" : "ol";
        if (listType !== nextListType) {
          closeList();
          listType = nextListType;
          output.push(`<${listType}>`);
        }
        output.push(`<li>${renderInline((unordered || ordered)[1])}</li>`);
        return;
      }
      const quote = line.match(/^&gt;\s?(.+)$/);
      if (quote) {
        closeParagraph();
        closeList();
        output.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
        return;
      }
      paragraph.push(line);
    });

    if (inCodeBlock) closeCodeBlock();
    closeParagraph();
    closeList();
    return output.join("");
  }

  return { escapeHtml, renderSafeMarkdown, safeHttpUrl };
});
