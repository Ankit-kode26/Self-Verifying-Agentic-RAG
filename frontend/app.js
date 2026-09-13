/* ============================================================
   AGENTIC RAG — Neural Command Center
   Application Logic
   ============================================================ */

// Auto-detect environment: use localhost when testing locally, Render URL when deployed on Vercel
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const DEFAULT_RENDER_URL = 'https://agentic-rag-backend-os1d.onrender.com';

const API_BASE = isLocal
  ? 'http://localhost:8000'
  : (localStorage.getItem('RAG_API_BASE') || DEFAULT_RENDER_URL);


// ── DOM References ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  healthStatus: $('#health-status'),
  statusDot: $('#status-dot'),
  statusText: $('#status-text'),
  sidebar: $('#sidebar'),
  sidebarToggle: $('#sidebar-toggle'),
  uploadZone: $('#upload-zone'),
  uploadInput: $('#upload-input'),
  uploadProgress: $('#upload-progress'),
  progressFill: $('#progress-bar-fill'),
  progressText: $('#progress-text'),
  docList: $('#doc-list'),
  docEmpty: $('#doc-empty'),
  docCount: $('#doc-count'),
  chatContainer: $('#chat-container'),
  welcomeScreen: $('#welcome-screen'),
  pipeline: $('#pipeline'),
  queryInput: $('#query-input'),
  sendBtn: $('#send-btn'),
  toastContainer: $('#toast-container'),
  clearChatBtn: $('#clear-chat-btn'),
};

// ── State ───────────────────────────────────────────────────
const state = {
  backendOnline: false,
  documents: [],
  messages: [],
  isQuerying: false,
  isUploading: false,
};

// ── Constants ───────────────────────────────────────────────
const UPLOAD_TIMEOUT_MS = 120_000;
const QUERY_TIMEOUT_MS = 60_000;
const ALLOWED_EXTS = ['.pdf', '.txt', '.md', '.csv', '.json', '.log'];

const THINKING_MESSAGES = [
  { emoji: '🔍', text: 'Searching your documents…' },
  { emoji: '🎯', text: 'Reranking by relevance…' },
  { emoji: '⚖️', text: 'Agent is judging context…' },
  { emoji: '🔄', text: 'Refining search query…' },
  { emoji: '✨', text: 'Generating cited answer…' },
  { emoji: '📑', text: 'Attaching source citations…' },
];

// (3D carousel removed — replaced with CSS radial glow in styles.css)

// ── API Client ──────────────────────────────────────────────
const api = {
  async health() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Backend offline');
    return res.json();
  },

  async ingest(file) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);

    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${res.status})`);
      }
      return res.json();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        throw new Error('Upload timed out. Try a smaller document or check your connection.');
      }
      throw err;
    }
  },

  async documents() {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error('Failed to fetch documents');
    return res.json();
  },

  async deleteDocument(name) {
    const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete document');
    return res.json();
  },

  async query(question, history = [], { signal } = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), QUERY_TIMEOUT_MS);
    if (signal) signal.addEventListener('abort', () => controller.abort());

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Query failed (${res.status})`);
      }
      return res.json();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        throw new Error('Query timed out. The backend may be busy — please try again.');
      }
      if (err.message === 'Failed to fetch' || err.name === 'TypeError') {
        throw new Error(
          'Cannot reach backend. Make sure the backend server is running at http://localhost:8000.'
        );
      }
      throw err;
    }
  },
};

// ── Toast Notifications ─────────────────────────────────────
function showToast(type, title, message, duration = 4000) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${escapeHtml(title)}</div>
      ${message ? `<div class="toast-message">${escapeHtml(message)}</div>` : ''}
    </div>
    <button class="toast-close" aria-label="Close notification">✕</button>
  `;
  toast.querySelector('.toast-close').addEventListener('click', () => toast.remove());
  els.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('removing');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Health Check ────────────────────────────────────────────
async function checkHealth() {
  try {
    await api.health();
    const wasOffline = !state.backendOnline;
    state.backendOnline = true;
    els.statusDot.className = 'status-dot online';
    els.statusText.textContent = 'Backend online';
    if (wasOffline || state.documents.length === 0) {
      await loadDocuments();
    }
  } catch {
    state.backendOnline = false;
    els.statusDot.className = 'status-dot offline';
    els.statusText.textContent = 'Backend offline';
  }
}

// ── Document Management ─────────────────────────────────────
async function loadDocuments() {
  try {
    const data = await api.documents();
    state.documents = data.sources || [];
    renderDocuments();
  } catch {
    // Silently fail on initial load
  }
}

function renderDocuments(newDoc = null) {
  els.docCount.textContent = state.documents.length;

  if (state.documents.length === 0) {
    els.docEmpty.style.display = 'block';
    els.docList.querySelectorAll('.doc-card').forEach(c => c.remove());
    return;
  }

  els.docEmpty.style.display = 'none';
  els.docList.querySelectorAll('.doc-card').forEach(c => c.remove());

  state.documents.forEach(docName => {
    const card = document.createElement('div');
    card.className = 'doc-card';
    card.style.position = 'relative';
    const isNew = newDoc && docName === newDoc.source;

    if (isNew) {
      card.classList.add('just-added');
      setTimeout(() => card.classList.remove('just-added'), 2000);
    }

    const pagesText = isNew ? `${newDoc.pages_ingested} pages` : '';
    const chunksText = isNew ? `${newDoc.chunks_created} chunks` : '';
    const safeName = escapeHtml(docName);

    card.innerHTML = `
      <div class="doc-icon">📄</div>
      <div class="doc-info">
        <div class="doc-name" title="${safeName}">${safeName}</div>
        <div class="doc-meta">
          ${pagesText ? `<span class="doc-meta-item">📃 ${pagesText}</span>` : ''}
          ${chunksText ? `<span class="doc-meta-item">🧩 ${chunksText}</span>` : ''}
        </div>
      </div>
      <button class="doc-delete-btn" data-doc="${safeName}" title="Delete document">✕</button>
    `;

    const delBtn = card.querySelector('.doc-delete-btn');
    if (delBtn) {
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await api.deleteDocument(docName);
          showToast('info', 'Document removed', docName);
          await loadDocuments();
        } catch (err) {
          showToast('error', 'Delete failed', err.message);
        }
      });
    }

    els.docList.appendChild(card);

    if (isNew) {
      spawnCelebrationParticles(card);
    }
  });
}

// ── Celebration Particles ───────────────────────────────────
function spawnCelebrationParticles(container) {
  const celebration = document.createElement('div');
  celebration.className = 'upload-celebration';

  const colors = ['#06d6a0', '#8b5cf6', '#f59e0b', '#06d6a0'];
  const numParticles = 16;

  for (let i = 0; i < numParticles; i++) {
    const particle = document.createElement('div');
    particle.className = 'celebration-particle';
    const angle = (i / numParticles) * 360;
    const distance = 40 + Math.random() * 40;
    const dx = Math.cos(angle * Math.PI / 180) * distance;
    const dy = Math.sin(angle * Math.PI / 180) * distance;

    particle.style.cssText = `
      left: 50%; top: 50%;
      background: ${colors[i % colors.length]};
      --dx: ${dx}px; --dy: ${dy}px;
    `;
    celebration.appendChild(particle);
  }

  container.appendChild(celebration);
  setTimeout(() => celebration.remove(), 1000);
}

// ── File Upload ─────────────────────────────────────────────
function isAllowedFile(name) {
  const lower = name.toLowerCase();
  return ALLOWED_EXTS.some(ext => lower.endsWith(ext));
}

function setupUpload() {
  const zone = els.uploadZone;
  const input = els.uploadInput;

  zone.addEventListener('click', () => {
    if (!state.isUploading) input.click();
  });

  input.addEventListener('change', () => {
    if (input.files.length > 0) handleFiles(Array.from(input.files));
    input.value = '';
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files).filter(f => isAllowedFile(f.name));
    if (files.length > 0) {
      handleFiles(files);
    } else {
      showToast('error', 'Invalid file', 'Supported formats: PDF, TXT, MD, CSV.');
    }
  });
}

async function handleFiles(files) {
  if (!state.backendOnline) {
    showToast('error', 'Backend offline', 'Please start the backend server first.');
    return;
  }
  if (state.isUploading) return;

  state.isUploading = true;

  const totalFiles = files.length;
  let completed = 0;

  els.uploadProgress.classList.add('active');
  els.uploadZone.classList.add('processing');

  for (const file of files) {
    let retries = 0;
    const maxRetries = 1;

    while (retries <= maxRetries) {
      try {
        els.progressText.textContent = `Processing ${escapeHtml(file.name)}${retries > 0 ? ' (retrying…)' : ''}…`;
        els.progressFill.style.width = `${(completed / totalFiles) * 100}%`;

        const progressInterval = setInterval(() => {
          const current = parseFloat(els.progressFill.style.width) || 0;
          const target = ((completed + 0.9) / totalFiles) * 100;
          if (current < target) {
            els.progressFill.style.width = `${Math.min(current + 1.5, target)}%`;
          }
        }, 300);

        const result = await api.ingest(file);
        clearInterval(progressInterval);

        completed++;
        els.progressFill.style.width = `${(completed / totalFiles) * 100}%`;

        await loadDocuments();
        renderDocuments(result);

        showToast('success', 'Document ingested',
          `${result.source} — ${result.pages_ingested} pages, ${result.chunks_created} chunks`
        );
        break;

      } catch (err) {
        if (retries < maxRetries && !err.message.includes('too large')) {
          retries++;
          showToast('info', 'Retrying upload', `${file.name} — attempt ${retries + 1}`);
          continue;
        }
        completed++;
        showToast('error', `Failed: ${file.name}`, err.message);
        break;
      }
    }
  }

  els.progressText.textContent = 'Done!';
  setTimeout(() => {
    els.uploadProgress.classList.remove('active');
    els.uploadZone.classList.remove('processing');
    els.progressFill.style.width = '0%';
  }, 1500);

  state.isUploading = false;
}

// ── Pipeline Visualizer ─────────────────────────────────────
const pipelineSteps = ['search', 'rerank', 'judge', 'generate'];
let pipelineTimer = null;

function startPipeline() {
  els.pipeline.classList.add('active');
  resetPipeline();

  let currentStep = 0;
  const stepDurations = [1200, 1500, 2000, 2500];

  function advanceStep() {
    if (currentStep >= pipelineSteps.length) return;

    const stepName = pipelineSteps[currentStep];
    const stepEl = $(`#pipe-${stepName}`);

    for (let i = 0; i < currentStep; i++) {
      const prevName = pipelineSteps[i];
      $(`#pipe-${prevName}`).classList.remove('active');
      $(`#pipe-${prevName}`).classList.add('done');
      const conn = $(`#pipe-conn-${i + 1}`);
      if (conn) conn.classList.add('active');
    }

    stepEl.classList.add('active');

    currentStep++;
    if (currentStep < pipelineSteps.length) {
      pipelineTimer = setTimeout(advanceStep, stepDurations[currentStep]);
    }
  }

  advanceStep();
}

function stopPipeline() {
  if (pipelineTimer) clearTimeout(pipelineTimer);

  pipelineSteps.forEach(name => {
    $(`#pipe-${name}`).classList.remove('active');
    $(`#pipe-${name}`).classList.add('done');
  });

  setTimeout(() => {
    els.pipeline.classList.remove('active');
    resetPipeline();
  }, 500);
}

function resetPipeline() {
  pipelineSteps.forEach((name, i) => {
    $(`#pipe-${name}`).classList.remove('active', 'done');
    const conn = $(`#pipe-conn-${i + 1}`);
    if (conn) conn.classList.remove('active');
  });
}

// ── Chat Rendering ──────────────────────────────────────────
function hideWelcome() {
  if (els.welcomeScreen) {
    els.welcomeScreen.style.display = 'none';
  }
}

function addUserMessage(text) {
  hideWelcome();

  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.innerHTML = `
    <div class="message-avatar">👤</div>
    <div class="message-body">
      <div class="message-content">${escapeHtml(text)}</div>
    </div>
  `;
  els.chatContainer.appendChild(msg);
  scrollToBottom();
}

// ── Enhanced Thinking Message ───────────────────────────────
let thinkingInterval = null;
let thinkingMsgIndex = 0;

function addThinkingMessage() {
  hideWelcome();

  const msg = document.createElement('div');
  msg.className = 'message assistant';
  msg.id = 'thinking-msg';

  const waveBars = Array.from({ length: 12 }, () =>
    '<div class="wave-bar"></div>'
  ).join('');

  msg.innerHTML = `
    <div class="message-avatar">🧠</div>
    <div class="message-body">
      <div class="message-content">
        <div class="thinking-container">
          <div class="neural-wave">${waveBars}</div>
          <div class="thinking-status">
            <span class="status-emoji">🔍</span>
            <span class="status-text">Searching your documents…</span>
          </div>
          <div class="skeleton-loader">
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
          </div>
        </div>
      </div>
    </div>
  `;
  els.chatContainer.appendChild(msg);
  scrollToBottom();

  thinkingMsgIndex = 0;
  thinkingInterval = setInterval(() => {
    thinkingMsgIndex = (thinkingMsgIndex + 1) % THINKING_MESSAGES.length;
    const current = THINKING_MESSAGES[thinkingMsgIndex];
    const statusEl = msg.querySelector('.thinking-status');
    if (statusEl) {
      statusEl.innerHTML = `
        <span class="status-emoji">${current.emoji}</span>
        <span class="status-text">${current.text}</span>
      `;
    }
  }, 2500);
}

function removeThinkingMessage() {
  if (thinkingInterval) {
    clearInterval(thinkingInterval);
    thinkingInterval = null;
  }
  const thinking = $('#thinking-msg');
  if (thinking) thinking.remove();
}

// ── Structured Answer Parser ────────────────────────────────
/**
 * Parses the raw LLM answer text into structured HTML.
 * Handles:
 *   **Bold Headers** -> section headings
 *   "1. ", "2. " -> styled numbered list items
 *   (Source: X, Page: Y) -> citation badges
 *   Paragraphs -> proper paragraph elements
 * No dash processing needed here — backend already removes them.
 */
function parseAnswerToHtml(rawText) {
  if (!rawText) return '';

  // Replace citation patterns first before splitting into lines
  // so we can style them before escaping
  const lines = rawText.split('\n');
  const segments = [];
  let currentList = [];

  function flushList() {
    if (currentList.length > 0) {
      const items = currentList.map((item, idx) => `
        <li>
          <span class="list-num">${idx + 1}</span>
          <span class="list-text">${formatInline(item)}</span>
        </li>
      `).join('');
      segments.push(`<ol class="answer-numbered-list">${items}</ol>`);
      currentList = [];
    }
  }

  for (let line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      flushList();
      continue;
    }

    // Numbered list item: "1. text", "2. text" etc.
    const numMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (numMatch) {
      currentList.push(numMatch[2]);
      continue;
    }

    flushList();

    // Section header: **Bold text** on its own line
    if (trimmed.startsWith('**') && trimmed.endsWith('**') && trimmed.length > 4) {
      const headerText = trimmed.slice(2, -2);
      segments.push(`<div class="answer-section-header">${escapeHtml(headerText)}</div>`);
      continue;
    }

    // Regular paragraph
    segments.push(`<p class="answer-paragraph">${formatInline(trimmed)}</p>`);
  }

  flushList();

  return `<div class="answer-body">${segments.join('')}</div>`;
}

/**
 * Formats inline text: handles **bold**, citations, and escaping.
 */
function formatInline(text) {
  // Escape HTML first
  let result = escapeHtml(text);

  // Bold text: **text** -> <strong>text</strong>
  result = result.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Citation badges: (Source: X, Page: Y)
  result = result.replace(
    /\(Source:\s*([^,]+),\s*Page:\s*(\d+)\)/gi,
    (match, source, page) => {
      const safeSource = source.trim();
      return `<span class="citation-badge" title="${safeSource}, Page ${page}">
        <span class="cite-icon">📎</span>${safeSource}, p.${page}
      </span>`;
    }
  );

  return result;
}

function addAssistantMessage(data) {
  removeThinkingMessage();

  const msg = document.createElement('div');
  msg.className = 'message assistant';

  const parsedAnswer = parseAnswerToHtml(data.answer);
  const sourcesHtml = renderSourcesPanel(data.sources_used);
  const traceHtml = renderTracePanel(data.retrieval_attempts);

  msg.innerHTML = `
    <div class="message-avatar">🧠</div>
    <div class="message-body">
      <div class="message-content">${parsedAnswer}</div>
      <div class="message-extras">
        ${sourcesHtml}
        ${traceHtml}
      </div>
    </div>
  `;

  els.chatContainer.appendChild(msg);
  scrollToBottom();

  // Attach toggle listeners for sources/trace panels
  msg.querySelectorAll('.extras-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const panelId = btn.dataset.target;
      const panel = msg.querySelector(`#${panelId}`);
      btn.classList.toggle('open');
      panel.classList.toggle('open');
    });
  });

  // Attach chunk preview toggle listeners
  msg.querySelectorAll('.chunk-preview-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const previewId = btn.dataset.preview;
      const preview = msg.querySelector(`#${previewId}`);
      if (!preview) return;
      preview.classList.toggle('open');
      btn.textContent = preview.classList.contains('open')
        ? '🙈 Hide chunk text'
        : '👁 View chunk text';
    });
  });
}

function addErrorMessage(errorText) {
  removeThinkingMessage();

  const msg = document.createElement('div');
  msg.className = 'message assistant';
  msg.innerHTML = `
    <div class="message-avatar">🧠</div>
    <div class="message-body">
      <div class="message-content" style="border-color: rgba(244,63,94,0.2);">
        ⚠️ ${escapeHtml(errorText)}
      </div>
    </div>
  `;
  els.chatContainer.appendChild(msg);
  scrollToBottom();
}

// ── Sources Panel with Chunk Preview ───────────────────────
function renderSourcesPanel(sources) {
  if (!sources || sources.length === 0) return '';

  const panelId = `sources-${Date.now()}`;
  const rows = sources.map((s, idx) => {
    const score = s.relevance_score;
    const normalizedScore = Math.min(Math.max((score + 5) / 15, 0), 1);
    const pct = (normalizedScore * 100).toFixed(0);
    const level = normalizedScore > 0.6 ? 'high' : normalizedScore > 0.3 ? 'medium' : 'low';
    const previewId = `chunk-preview-${panelId}-${idx}`;
    // Safely show chunk text with HTML escaping
    const chunkText = s.chunk_text ? escapeHtml(s.chunk_text) : '(no chunk text available)';

    return `
      <tr>
        <td>
          <span style="font-weight:600;color:var(--text-primary)">${escapeHtml(s.source)}</span>
          <br>
          <button class="chunk-preview-btn" data-preview="${previewId}">👁 View chunk text</button>
          <div class="chunk-text-preview" id="${previewId}">${chunkText}</div>
        </td>
        <td style="text-align:center;font-size:0.85rem;color:var(--accent-violet);font-weight:600">
          ${s.page_number}
        </td>
        <td>
          <span class="relevance-bar relevance-${level}">
            <span class="relevance-bar-fill" style="width: ${pct}%"></span>
          </span>
          <span style="font-size:0.68rem;color:var(--text-muted)">${score.toFixed(2)}</span>
        </td>
      </tr>
    `;
  }).join('');

  return `
    <button class="extras-toggle" data-target="${panelId}">
      <span class="toggle-arrow">▼</span>
      📊 Sources Used (${sources.length}) — Click to view chunks
    </button>
    <div class="extras-panel" id="${panelId}">
      <table class="source-table">
        <thead>
          <tr>
            <th>Source &amp; Chunk</th>
            <th>Page</th>
            <th>Relevance</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

// ── Retrieval Trace Panel ───────────────────────────────────
function renderTracePanel(attempts) {
  if (!attempts || attempts.length === 0) return '';

  const panelId = `trace-${Date.now()}`;
  const steps = attempts.map((a, i) => {
    const isSufficient = a.sufficient === true;
    const isNoChunks = a.verdict === 'no_chunks_found';
    const dotClass = isNoChunks ? 'no-chunks' : (isSufficient ? 'sufficient' : 'insufficient');
    const icon = isNoChunks ? '✕' : (isSufficient ? '✓' : '↻');

    return `
      <div class="trace-step">
        <div class="trace-dot ${dotClass}">${icon}</div>
        <div class="trace-info">
          <div class="trace-query">"${escapeHtml(a.query)}"</div>
          <div class="trace-verdict">
            Attempt ${a.attempt}${a.reason ? ` — ${escapeHtml(a.reason)}` : ''}
          </div>
        </div>
      </div>
    `;
  }).join('');

  const label = attempts.length === 1
    ? '🔄 Agent Trace (1 attempt)'
    : `🔄 Agent Trace (${attempts.length} attempts — query rewritten)`;

  return `
    <button class="extras-toggle" data-target="${panelId}">
      <span class="toggle-arrow">▼</span>
      ${label}
    </button>
    <div class="extras-panel" id="${panelId}">
      <div class="trace-timeline">${steps}</div>
    </div>
  `;
}

// ── Query Handler ───────────────────────────────────────────
async function handleQuery() {
  const question = els.queryInput.value.trim();
  if (!question || state.isQuerying) return;

  if (!state.backendOnline) {
    showToast('error', 'Backend offline', 'Please make sure the backend server is running on port 8000.');
    return;
  }

  if (state.documents.length === 0) {
    await loadDocuments();
  }
  if (state.documents.length === 0) {
    showToast('info', 'No documents', 'Please upload a document (.pdf, .txt, .md) first.');
    return;
  }

  state.isQuerying = true;
  els.queryInput.value = '';
  els.sendBtn.disabled = true;
  els.queryInput.disabled = true;

  addUserMessage(question);
  addThinkingMessage();
  startPipeline();

  try {
    // Send previous dialogue turns so follow-ups have full conversational context
    const historyPayload = state.messages
      .slice(-6)
      .map(m => ({ role: m.role, content: m.content }));

    const result = await api.query(question, historyPayload);
    stopPipeline();
    addAssistantMessage(result);
    state.messages.push({
      role: 'user', content: question
    }, {
      role: 'assistant',
      content: result.answer,
      sources: result.sources_used,
      attempts: result.retrieval_attempts,
    });
  } catch (err) {
    stopPipeline();
    addErrorMessage(err.message || 'Something went wrong. Please try again.');
    showToast('error', 'Query failed', err.message);
  } finally {
    state.isQuerying = false;
    els.sendBtn.disabled = false;
    els.queryInput.disabled = false;
    els.queryInput.focus();
    updateSendBtn();
  }
}

// ── Clear Chat Handler ──────────────────────────────────────
function handleClearChat() {
  state.messages = [];
  const messages = els.chatContainer.querySelectorAll('.message');
  messages.forEach(m => m.remove());
  if (els.welcomeScreen) {
    els.welcomeScreen.style.display = 'block';
  }
  showToast('info', 'Chat reset', 'Conversation history has been cleared.');
}

// ── Utilities ───────────────────────────────────────────────
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
  });
}

function updateSendBtn() {
  els.sendBtn.disabled = !els.queryInput.value.trim() || state.isQuerying;
}

// ── Event Listeners ─────────────────────────────────────────
function setupEventListeners() {
  els.sendBtn.addEventListener('click', handleQuery);

  els.queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  });

  els.queryInput.addEventListener('input', updateSendBtn);

  els.sidebarToggle.addEventListener('click', () => {
    els.sidebar.classList.toggle('mobile-open');
  });

  if (els.clearChatBtn) {
    els.clearChatBtn.addEventListener('click', handleClearChat);
  }
}

// ── Initialization ──────────────────────────────────────────
async function init() {
  setupUpload();
  setupEventListeners();

  await checkHealth();

  if (state.backendOnline) {
    await loadDocuments();
  }

  // Periodic health check every 5s if offline, 10s if online
  setInterval(async () => {
    await checkHealth();
  }, state.backendOnline ? 10000 : 4000);

  els.queryInput.focus();
}

document.addEventListener('DOMContentLoaded', init);
