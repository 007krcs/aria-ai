/**
 * MarkdownRenderer — Claude-quality markdown rendering for ARIA
 * Uses marked.js for parsing + highlight.js for syntax highlighting.
 */
import { useMemo, useEffect, useRef } from "react";
import { marked } from "marked";
import hljs from "highlight.js/lib/core";
// Register only the languages ARIA commonly outputs
import langPython     from "highlight.js/lib/languages/python";
import langJS         from "highlight.js/lib/languages/javascript";
import langTS         from "highlight.js/lib/languages/typescript";
import langBash       from "highlight.js/lib/languages/bash";
import langJSON       from "highlight.js/lib/languages/json";
import langSQL        from "highlight.js/lib/languages/sql";
import langCSS        from "highlight.js/lib/languages/css";
import langXML        from "highlight.js/lib/languages/xml";
import langMarkdown   from "highlight.js/lib/languages/markdown";
import langGo         from "highlight.js/lib/languages/go";
import langRust       from "highlight.js/lib/languages/rust";
import langJava       from "highlight.js/lib/languages/java";
import langCpp        from "highlight.js/lib/languages/cpp";
import langYaml       from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("python",     langPython);
hljs.registerLanguage("javascript", langJS);
hljs.registerLanguage("typescript", langTS);
hljs.registerLanguage("bash",       langBash);
hljs.registerLanguage("shell",      langBash);
hljs.registerLanguage("json",       langJSON);
hljs.registerLanguage("sql",        langSQL);
hljs.registerLanguage("css",        langCSS);
hljs.registerLanguage("html",       langXML);
hljs.registerLanguage("xml",        langXML);
hljs.registerLanguage("markdown",   langMarkdown);
hljs.registerLanguage("go",         langGo);
hljs.registerLanguage("rust",       langRust);
hljs.registerLanguage("java",       langJava);
hljs.registerLanguage("cpp",        langCpp);
hljs.registerLanguage("yaml",       langYaml);

// ── marked configuration ───────────────────────────────────────────────────────
const renderer = new marked.Renderer();

// Code blocks with copy button wrapper
renderer.code = (code, lang) => {
  const language = lang && hljs.getLanguage(lang) ? lang : "plaintext";
  let highlighted;
  try {
    highlighted = hljs.highlight(code, { language }).value;
  } catch {
    highlighted = code.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }
  const label = language === "plaintext" ? "" : language;
  return `
<div class="md-code-block">
  <div class="md-code-header">
    <span class="md-code-lang">${label}</span>
    <button class="md-copy-btn" onclick="
      navigator.clipboard.writeText(this.closest('.md-code-block').querySelector('code').innerText);
      this.textContent='Copied!';
      setTimeout(()=>this.textContent='Copy',1500)
    ">Copy</button>
  </div>
  <pre><code class="hljs language-${language}">${highlighted}</code></pre>
</div>`;
};

// Inline code
renderer.codespan = (code) =>
  `<code class="md-inline-code">${code}</code>`;

// Tables
renderer.table = (header, body) =>
  `<div class="md-table-wrap"><table class="md-table"><thead>${header}</thead><tbody>${body}</tbody></table></div>`;

// Blockquotes
renderer.blockquote = (quote) =>
  `<blockquote class="md-blockquote">${quote}</blockquote>`;

// Links — open in new tab safely
renderer.link = (href, title, text) =>
  `<a href="${href}" target="_blank" rel="noopener noreferrer" class="md-link" title="${title||''}">${text}</a>`;

// Checkboxes in task lists
renderer.listitem = (text, task, checked) => {
  if (task) {
    return `<li class="md-task-item"><input type="checkbox" ${checked ? "checked" : ""} disabled> ${text}</li>`;
  }
  return `<li>${text}</li>`;
};

marked.setOptions({
  renderer,
  gfm: true,
  breaks: true,
  pedantic: false,
  smartLists: true,
});

// ── Component ──────────────────────────────────────────────────────────────────
export default function MarkdownRenderer({ text, streaming = false }) {
  const ref = useRef(null);

  const html = useMemo(() => {
    if (!text) return "";
    try {
      return marked.parse(text);
    } catch {
      return `<p>${text}</p>`;
    }
  }, [text]);

  // Re-run hljs on newly rendered code blocks (for streaming updates)
  useEffect(() => {
    if (ref.current) {
      ref.current.querySelectorAll("pre code:not(.hljs)").forEach(el => {
        hljs.highlightElement(el);
      });
    }
  }, [html]);

  return (
    <div
      ref={ref}
      className="md-body"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
