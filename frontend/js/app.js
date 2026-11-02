/* ==========================================================
 * 企业智能客服 - 前端逻辑
 * 通过 fetch + ReadableStream 解析 SSE，支持流式打字机效果
 * ========================================================== */
"use strict";

// ---------------- 配置 ----------------
const API_BASE = window.__API_BASE__ || "http://localhost:8000/api/v1";

// ---------------- 状态 ----------------
const state = {
  userId: localStorage.getItem("userId") || "",
  conversationId: localStorage.getItem(`conv_${localStorage.getItem("userId") || ""}`) || null,
  conversations: [],
  sending: false,
};

// ---------------- DOM ----------------
const $ = (id) => document.getElementById(id);
const chatBox = $("chatBox");
const queryInput = $("queryInput");
const sendBtn = $("sendBtn");
const userIdInput = $("userIdInput");
const convList = $("convList");
const convTitle = $("convTitle");
const backendStatus = $("backendStatus");

// ==========================================================
//  初始化
// ==========================================================
function init() {
  userIdInput.value = state.userId;
  if (!state.userId) {
    // 无用户 ID 时生成一个，保证开箱即用
    state.userId = "guest_" + Math.random().toString(36).slice(2, 8);
    userIdInput.value = state.userId;
    persistUser();
  }
  bindEvents();
  checkHealth();
  loadConversations();
  loadDocs();
  autoResize(queryInput);
}

function persistUser() {
  localStorage.setItem("userId", state.userId);
}

function persistConv() {
  localStorage.setItem(`conv_${state.userId}`, state.conversationId || "");
}

// ---------------- 事件绑定 ----------------
function bindEvents() {
  sendBtn.addEventListener("click", send);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  $("newChatBtn").addEventListener("click", newChat);

  userIdInput.addEventListener("change", () => {
    const v = userIdInput.value.trim();
    if (!v) return;
    // 切换用户：清空当前会话视图，加载新用户的会话列表
    state.userId = v;
    state.conversationId = null;
    persistUser();
    persistConv();
    resetChatBox();
    loadConversations();
  });

  // 建议问题
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.dataset.q;
      send();
    });
  });

  // 知识库抽屉
  $("uploadToggleBtn").addEventListener("click", openDrawer);
  $("drawerCloseBtn").addEventListener("click", closeDrawer);
  $("drawerMask").addEventListener("click", closeDrawer);

  // 文件上传
  const zone = $("uploadZone");
  const fileInput = $("fileInput");
  zone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
    fileInput.value = "";
  });
  ["dragover", "dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.toggle("dragover", evt === "dragover");
      if (evt === "drop" && e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
    })
  );
}

function autoResize(el) {
  el.addEventListener("input", () => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  });
}

// ==========================================================
//  健康检查
// ==========================================================
async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`);
    const data = await resp.json();
    if (resp.ok && data.status === "healthy") {
      backendStatus.textContent = `已连接 · 记忆:${data.memory_backend.replace("ConversationMemory", "")} · 文档:${data.vector_store.total_documents}`;
      backendStatus.className = "status-badge ok";
    } else {
      backendStatus.textContent = "服务降级（Redis 未连接）";
      backendStatus.className = "status-badge err";
    }
  } catch {
    backendStatus.textContent = "无法连接后端";
    backendStatus.className = "status-badge err";
  }
}

// ==========================================================
//  对话流（SSE）
// ==========================================================
async function send() {
  const query = queryInput.value.trim();
  if (!query || state.sending) return;

  state.sending = true;
  sendBtn.disabled = true;
  queryInput.value = "";
  queryInput.style.height = "auto";
  hideWelcome();

  appendUserMsg(query);

  // 构造助手占位气泡
  const { bubble, toolCardSlot, sourcesSlot } = appendAssistantShell();
  let answerText = "";
  let toolCard = null;
  let staleSession = false;

  try {
    // 最多两次尝试：会话失效（如后端内存记忆重启丢失）时清空并以新会话自动重试
    for (let attempt = 0; attempt < 2; attempt++) {
      staleSession = false;
      const resp = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          user_id: state.userId,
          conversation_id: state.conversationId,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.message || `HTTP ${resp.status}`);
      }

      // 逐行解析 SSE
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // 最后一段可能不完整，留待下轮

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          let event;
          try {
            event = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }
          handleEvent(event);
        }
      }

      if (staleSession && attempt === 0) {
        state.conversationId = null;
        persistConv();
        answerText = "";
        toolCard = null;
        bubble.style.color = "";
        bubble.classList.add("thinking");
        bubble.textContent = "会话已失效，正在以新会话重试…";
        continue;
      }
      break;
    }

    function handleEvent(event) {
      const { type, data } = event;
      switch (type) {
        case "thinking":
          state.conversationId = data.conversation_id;
          persistConv();
          bubble.textContent = "思考中…";
          bubble.style.color = "";
          bubble.classList.add("thinking");
          convTitle.textContent = `会话 ${data.conversation_id.slice(0, 8)}`;
          break;

        case "tool_start": {
          bubble.classList.remove("thinking");
          toolCard = document.createElement("div");
          toolCard.className = "tool-card";
          toolCard.innerHTML = `<span class="tool-name">🔧 ${escapeHtml(data.tool)}</span><div class="tool-detail">调用中…</div>`;
          toolCardSlot.appendChild(toolCard);
          scrollBottom();
          break;
        }

        case "tool_end":
          if (toolCard) {
            const detail = toolCard.querySelector(".tool-detail");
            detail.textContent = data.result ? data.result.slice(0, 200) + (data.result.length > 200 ? "…" : "") : "完成";
            toolCard = null;
          }
          break;

        case "sources":
          renderSources(sourcesSlot, data.sources || []);
          break;

        case "token":
          bubble.classList.remove("thinking");
          if (bubble.textContent === "思考中…") bubble.textContent = "";
          answerText += data.text;
          bubble.textContent = answerText;
          scrollBottom();
          break;

        case "done":
          bubble.classList.remove("thinking");
          if (!answerText) bubble.textContent = "（无回复内容）";
          loadConversations(); // 刷新会话列表
          break;

        case "error":
          if (data.code === "FORBIDDEN") staleSession = true;
          bubble.classList.remove("thinking");
          bubble.textContent = `⚠️ ${data.message || "出错了"}`;
          bubble.style.color = "#dc2626";
          break;
      }
    }
  } catch (err) {
    bubble.textContent = `⚠️ 请求失败：${err.message}`;
    bubble.style.color = "#dc2626";
    bubble.classList.remove("thinking");
  } finally {
    state.sending = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

// ---------------- 消息渲染 ----------------
function hideWelcome() {
  const w = $("welcome");
  if (w) w.remove();
}

function resetChatBox() {
  chatBox.innerHTML = `
    <div class="welcome" id="welcome">
      <h2>你好，我是企业智能客服 👋</h2>
      <p>我可以回答公司制度、流程、报销标准等问题，也支持联网搜索与数值计算。</p>
    </div>`;
  convTitle.textContent = "新对话";
}

function appendUserMsg(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<div class="avatar">🙋</div><div class="bubble"></div>`;
  div.querySelector(".bubble").textContent = text;
  chatBox.appendChild(div);
  scrollBottom();
}

function appendAssistantShell() {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  msg.innerHTML = `<div class="avatar">🤖</div><div class="bubble thinking">思考中…</div>`;
  chatBox.appendChild(msg);

  const toolCardSlot = document.createElement("div");
  const sourcesSlot = document.createElement("div");
  chatBox.appendChild(toolCardSlot);
  chatBox.appendChild(sourcesSlot);
  scrollBottom();

  return { bubble: msg.querySelector(".bubble"), toolCardSlot, sourcesSlot };
}

function renderSources(slot, sources) {
  if (!sources.length) return;
  slot.className = "sources";
  slot.innerHTML =
    `<div class="sources-title">📎 引用来源</div>` +
    sources
      .map(
        (s) =>
          `<div class="source-item"><b>${escapeHtml(s.doc_name)}</b>` +
          `${s.page ? ` 第${s.page}页` : ""} · 相似度 ${s.score}` +
          `：${escapeHtml(s.content.slice(0, 120))}…</div>`
      )
      .join("");
  scrollBottom();
}

function scrollBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ==========================================================
//  会话管理
// ==========================================================
async function loadConversations() {
  try {
    const resp = await fetch(`${API_BASE}/chat/conversations?user_id=${encodeURIComponent(state.userId)}`);
    const data = await resp.json();
    state.conversations = data.conversation_ids || [];
    // 内存记忆后端重启后会话会丢失：本地残留的当前会话不在列表中则自动清理
    if (state.conversationId && !state.conversations.includes(state.conversationId)) {
      state.conversationId = null;
      persistConv();
      convTitle.textContent = "新对话";
    }
    renderConvList();
  } catch {
    /* 忽略列表加载失败 */
  }
}

function renderConvList() {
  convList.innerHTML = "";
  for (const cid of state.conversations) {
    const li = document.createElement("li");
    if (cid === state.conversationId) li.classList.add("active");
    li.innerHTML = `<span>💬 ${cid.slice(0, 10)}…</span><button class="del-btn" title="删除">🗑</button>`;
    li.querySelector("span").addEventListener("click", () => switchConv(cid));
    li.querySelector(".del-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConv(cid);
    });
    convList.appendChild(li);
  }
}

async function switchConv(cid) {
  state.conversationId = cid;
  persistConv();
  renderConvList();
  resetChatBox();
  hideWelcome();
  convTitle.textContent = `会话 ${cid.slice(0, 8)}`;

  try {
    const resp = await fetch(
      `${API_BASE}/chat/conversations/${cid}/history?user_id=${encodeURIComponent(state.userId)}`
    );
    if (!resp.ok) return;
    const data = await resp.json();

    // 重放历史消息
    for (const m of data.messages) {
      if (m.role === "user") appendUserMsg(m.content);
      else if (m.role === "assistant") {
        hideWelcome();
        const div = document.createElement("div");
        div.className = "msg assistant";
        div.innerHTML = `<div class="avatar">🤖</div><div class="bubble"></div>`;
        div.querySelector(".bubble").textContent = m.content;
        chatBox.appendChild(div);
      }
    }
    // 摘要提示
    if (data.summary) {
      const tip = document.createElement("div");
      tip.className = "source-item";
      tip.style.margin = "0 auto 16px";
      tip.style.maxWidth = "70%";
      tip.textContent = `📌 早期对话摘要：${data.summary}`;
      chatBox.insertBefore(tip, chatBox.firstChild);
    }
    scrollBottom();
  } catch {
    /* 历史加载失败不阻塞 */
  }
}

async function deleteConv(cid) {
  if (!confirm("确定删除该会话及其全部历史？")) return;
  try {
    await fetch(
      `${API_BASE}/chat/conversations/${cid}?user_id=${encodeURIComponent(state.userId)}`,
      { method: "DELETE" }
    );
    if (cid === state.conversationId) {
      state.conversationId = null;
      persistConv();
      resetChatBox();
    }
    loadConversations();
  } catch {
    /* 删除失败静默 */
  }
}

function newChat() {
  state.conversationId = null;
  persistConv();
  resetChatBox();
  renderConvList();
  queryInput.focus();
}

// ==========================================================
//  知识库管理
// ==========================================================
function openDrawer() {
  $("drawer").classList.add("open");
  $("drawerMask").classList.add("open");
  loadDocs();
}
function closeDrawer() {
  $("drawer").classList.remove("open");
  $("drawerMask").classList.remove("open");
}

async function loadDocs() {
  try {
    const resp = await fetch(`${API_BASE}/documents`);
    const data = await resp.json();
    const list = $("docList");
    const empty = $("docEmpty");
    list.innerHTML = "";
    empty.hidden = data.total > 0;

    for (const doc of data.documents || []) {
      const li = document.createElement("li");
      li.className = "doc-item";
      li.innerHTML = `
        <div>
          <div class="doc-name">${escapeHtml(doc.doc_name)}</div>
          <div class="doc-meta">${doc.chunk_count} 个片段 · ${doc.file_type.toUpperCase()} · ${new Date(doc.uploaded_at).toLocaleString()}</div>
        </div>
        <button class="doc-del" title="删除">🗑</button>`;
      li.querySelector(".doc-del").addEventListener("click", () => deleteDoc(doc.doc_id));
      list.appendChild(li);
    }
  } catch {
    /* 列表加载失败静默 */
  }
}

async function uploadFile(file) {
  const allowed = /\.(pdf|docx|txt|md|csv|xlsx|xls)$/i;
  if (!allowed.test(file.name)) {
    alert("仅支持 PDF / DOCX / TXT / MD / CSV / XLSX 文件");
    return;
  }

  const progress = $("uploadProgress");
  progress.hidden = false;
  progress.textContent = `正在解析「${file.name}」并向量化，请稍候…`;

  try {
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);

    progress.textContent = `✅ 入库成功：${data.chunk_count} 个片段`;
    setTimeout(() => (progress.hidden = true), 2500);
    loadDocs();
  } catch (err) {
    progress.textContent = `❌ 上传失败：${err.message}`;
    setTimeout(() => (progress.hidden = true), 4000);
  }
}

// ---------------- 启动 ----------------
init();
