const sessionId = crypto.randomUUID();
const selectedStorageKey = "paper-rag-selected-documents";
const selectedSnapshotStorageKey = "paper-rag-selected-document-snapshots";
const scopeStorageKey = "paper-rag-scope";
const uploadStorageKey = "paper-rag-upload-jobs";
const conversationStorageKey = "paper-rag-current-conversation";

function restoreUploadItems() {
  try {
    const items = JSON.parse(localStorage.getItem(uploadStorageKey) || "[]");
    return Array.isArray(items) ? items.map((item) => ({ ...item, file: null })) : [];
  } catch {
    return [];
  }
}

function restoreSelectedDocumentSnapshots() {
  try {
    const items = JSON.parse(sessionStorage.getItem(selectedSnapshotStorageKey) || "[]");
    if (!Array.isArray(items)) return new Map();
    return new Map(items
      .filter((item) => item && typeof item.document_id === "string")
      .map((item) => [item.document_id, item]));
  } catch {
    return new Map();
  }
}

const state = {
  documents: [],
  documentCache: new Map(),
  selectedDocumentSnapshots: restoreSelectedDocumentSnapshots(),
  selectedIds: new Set(JSON.parse(sessionStorage.getItem(selectedStorageKey) || "[]")),
  scope: sessionStorage.getItem(scopeStorageKey) || "all",
  nextCursor: null,
  libraryQuery: "",
  libraryRequestController: null,
  libraryRequestGeneration: 0,
  sessionHasKey: false,
  asking: false,
  abortController: null,
  deletionCheck: null,
  uploadItems: restoreUploadItems(),
  uploadRunning: false,
  uploadController: null,
  uploadPollTimer: null,
  conversations: [],
  conversationId: localStorage.getItem(conversationStorageKey),
};
state.selectedDocumentSnapshots.forEach((item, id) => state.documentCache.set(id, item));

const elements = {
  documentList: document.querySelector("#document-list"),
  documentCount: document.querySelector("#document-count"),
  documentCountLabel: document.querySelector("#document-count-label"),
  selectedCount: document.querySelector("#selected-count"),
  scopeSummary: document.querySelector("#scope-summary"),
  loadMore: document.querySelector("#load-more"),
  refreshLibrary: document.querySelector("#refresh-library"),
  filenameSearch: document.querySelector("#filename-search"),
  librarySearchStatus: document.querySelector("#library-search-status"),
  serviceState: document.querySelector("#service-state"),
  fileInput: document.querySelector("#pdf-file"),
  fileLabel: document.querySelector("#file-label"),
  uploadForm: document.querySelector("#upload-form"),
  uploadSubmit: document.querySelector("#upload-submit"),
  uploadQueue: document.querySelector("#upload-queue"),
  uploadSummary: document.querySelector("#upload-summary"),
  uploadItems: document.querySelector("#upload-items"),
  cancelUpload: document.querySelector("#cancel-upload"),
  clearUpload: document.querySelector("#clear-upload"),
  ingestionWrap: document.querySelector("#ingestion-wrap"),
  ingestionProgress: document.querySelector("#ingestion-progress"),
  ingestionPercent: document.querySelector("#ingestion-percent"),
  ingestionStatus: document.querySelector("#ingestion-status"),
  keyForm: document.querySelector("#key-form"),
  keyInput: document.querySelector("#deepseek-key"),
  keyStatus: document.querySelector("#key-status"),
  chatForm: document.querySelector("#chat-form"),
  question: document.querySelector("#question"),
  sendButton: document.querySelector("#send-button"),
  cancelButton: document.querySelector("#cancel-button"),
  conversation: document.querySelector("#conversation"),
  emptyState: document.querySelector("#empty-state"),
  conversationSelect: document.querySelector("#conversation-select"),
  newConversation: document.querySelector("#new-conversation"),
  renameConversation: document.querySelector("#rename-conversation"),
  resetConversation: document.querySelector("#reset-conversation"),
  deleteConversation: document.querySelector("#delete-conversation"),
  deleteDialog: document.querySelector("#delete-dialog"),
  deleteCheckContent: document.querySelector("#delete-check-content"),
  deleteForm: document.querySelector("#delete-form"),
  deleteConfirmFilename: document.querySelector("#delete-confirm-filename"),
  deleteSubmit: document.querySelector("#delete-submit"),
  deleteCancel: document.querySelector("#delete-cancel"),
};

const stageProgress = { queued: 5, classifying: 12, parsing: 30, chunking: 52, embedding: 72, indexing: 90, completed: 100, review_required: 100, quarantined: 100, failed: 100 };
const stageNames = { queued: "等待处理", classifying: "识别 PDF 类型", parsing: "解析正文与版面", chunking: "构建知识切片", embedding: "生成语义向量", indexing: "写入本地索引", completed: "入库完成", review_required: "领域相关性待人工确认", quarantined: "疑似非超表面文档，已隔离", failed: "入库失败" };
const domainNames = { unclassified: "未判定", accepted: "领域匹配", review_required: "待确认", quarantined: "已隔离", manual_approved: "人工确认" };
const admissionDecisionNames = {
  positive_evidence_quorum: "正向证据充分",
  insufficient_parse_evidence: "可用正文不足",
  insufficient_independent_regions: "缺少跨区域一致证据",
  missing_domain_relationship: "缺少领域对象与电磁作用关系",
  inconsistent_positive_evidence: "正文中的领域证据不一致",
  reference_only_evidence: "领域信息仅出现在参考文献",
  gate_dependency_unavailable: "系统暂时无法完成判断",
  gate_internal_error: "系统暂时无法完成判断",
  gate_safe_mode: "准入服务处于安全复核模式",
};
const admissionRequirementNames = {
  parse_quality: "正文解析质量",
  domain_identity: "明确的超表面领域对象",
  domain_relationship: "领域对象与电磁作用关系",
  independent_regions: "跨区域一致证据",
  embedding_provider: "本地语义模型可用性",
  page_contract: "页面解析合同",
  safe_mode: "准入服务正常运行",
};
const genreNames = { unclassified: "类型待判定", research_paper: "研究论文", review_paper: "综述", thesis: "学位论文", conference_paper: "会议论文" };
const profileNames = { building: "结构分析中", ready: "论文地图就绪", failed: "论文地图失败", stale: "论文地图待更新" };
const terminalJobStates = new Set(["completed", "review_required", "quarantined", "failed"]);
const terminalUploadStates = new Set(["upload_failed", "cancelled"]);
const uploadStateNames = { pending: "等待上传", uploading: "上传中", accepted: "已排队", upload_failed: "上传失败", cancelled: "已取消" };

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

const MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML";

function parseSafeMathMl(value) {
  if (!value || typeof value !== "string") return null;
  const parsed = new DOMParser().parseFromString(value, "application/xml");
  if (parsed.querySelector("parsererror")) return null;
  const root = parsed.documentElement;
  if (!root || root.localName !== "math" || root.namespaceURI !== MATHML_NAMESPACE) return null;
  const elements = [root, ...root.querySelectorAll("*")];
  const unsafe = elements.some((element) => element.namespaceURI !== MATHML_NAMESPACE
    || [...element.attributes].some((attribute) => attribute.name.toLowerCase().startsWith("on")
      || attribute.name.toLowerCase().endsWith("href")));
  return unsafe ? null : document.importNode(root, true);
}

function persistSelection() {
  sessionStorage.setItem(selectedStorageKey, JSON.stringify([...state.selectedIds]));
  sessionStorage.setItem(scopeStorageKey, state.scope);
  const snapshots = [...state.selectedIds]
    .map((id) => state.documentCache.get(id) || state.selectedDocumentSnapshots.get(id))
    .filter(Boolean);
  state.selectedDocumentSnapshots = new Map(
    snapshots.map((item) => [item.document_id, item]),
  );
  sessionStorage.setItem(selectedSnapshotStorageKey, JSON.stringify(snapshots));
}

function renderScope() {
  document.querySelectorAll(".scope-option").forEach((button) => button.classList.toggle("active", button.dataset.scope === state.scope));
  elements.selectedCount.textContent = state.selectedIds.size;
  elements.scopeSummary.textContent = state.scope === "all"
    ? `将从全部 ${state.documents.filter((item) => item.status === "completed").length} 篇已入库论文中检索证据`
    : `将从已选择的 ${state.selectedIds.size} 篇论文中检索证据`;
}

function renderConversationSelect() {
  elements.conversationSelect.innerHTML = state.conversations
    .map((item) => `<option value="${item.conversation_id}" ${item.conversation_id === state.conversationId ? "selected" : ""}>${escapeHtml(item.title)}</option>`)
    .join("");
}

async function createConversation(title = "新研究会话") {
  const response = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, scope: state.scope, document_ids: [...state.selectedIds] }),
  });
  if (!response.ok) throw new Error("无法创建研究会话");
  const conversation = await response.json();
  state.conversationId = conversation.conversation_id;
  localStorage.setItem(conversationStorageKey, state.conversationId);
  await loadConversations({ restore: true });
}

async function loadConversations({ restore = true } = {}) {
  const response = await fetch("/api/conversations");
  if (!response.ok) throw new Error("无法加载研究会话");
  state.conversations = (await response.json()).items;
  if (!state.conversations.length) {
    await createConversation();
    return;
  }
  if (!state.conversations.some((item) => item.conversation_id === state.conversationId)) {
    state.conversationId = state.conversations[0].conversation_id;
    localStorage.setItem(conversationStorageKey, state.conversationId);
  }
  renderConversationSelect();
  if (restore) await loadConversation(state.conversationId);
}

function renderEmptyConversation() {
  elements.conversation.innerHTML = `<div class="empty-state" id="empty-state"><i data-lucide="library-big"></i><h2>知识库已就绪</h2><p id="scope-summary">${escapeHtml(elements.scopeSummary?.textContent || "输入问题开始研究")}</p></div>`;
  elements.emptyState = document.querySelector("#empty-state");
  elements.scopeSummary = document.querySelector("#scope-summary");
  refreshIcons();
}

function renderTranscript(messages) {
  elements.conversation.innerHTML = "";
  const byTurn = new Map();
  messages.forEach((message) => {
    const entry = byTurn.get(message.turn_index) || {};
    entry[message.role] = message;
    byTurn.set(message.turn_index, entry);
  });
  [...byTurn.entries()].sort((left, right) => left[0] - right[0]).forEach(([turnIndex, entry]) => {
    if (!entry.user) return;
    const view = createQuestionView(entry.user.content, `restored-${turnIndex}`);
    if (entry.assistant?.response) renderAnswer(view.answer, entry.assistant.response);
    else if (entry.assistant) view.answer.innerHTML = `<div class="answer-body ${entry.assistant.status === "failed" ? "refusal" : ""}">${escapeHtml(entry.assistant.content)}</div>`;
  });
  if (!messages.length) renderEmptyConversation();
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

async function loadConversation(conversationId) {
  const response = await fetch(`/api/conversations/${conversationId}`);
  if (!response.ok) throw new Error("无法恢复研究会话");
  const conversation = await response.json();
  state.conversationId = conversation.conversation_id;
  localStorage.setItem(conversationStorageKey, state.conversationId);
  state.scope = conversation.scope;
  state.selectedIds = new Set(conversation.document_ids);
  renderConversationSelect();
  renderLibrary();
  renderTranscript(conversation.messages || []);
}

function renderLibrary() {
  const { matchedDocuments, selectedDocuments } = deriveLibraryView();
  elements.documentCount.textContent = state.documents.length;
  elements.documentCountLabel.textContent = state.libraryQuery ? "篇匹配" : "篇已收录";
  const matchedHtml = matchedDocuments.map(renderDocumentItem).join("");
  const emptyHtml = state.documents.length
    ? ""
    : `<p class="document-meta">${state.libraryQuery ? "未找到匹配的论文文件" : "论文库为空"}</p>`;
  const selectedHtml = state.libraryQuery && selectedDocuments.length
    ? `<div class="selected-papers-divider" data-selected-papers-anchor>已选论文 · ${selectedDocuments.length}</div>${selectedDocuments.map(renderDocumentItem).join("")}`
    : "";
  elements.documentList.innerHTML = `${matchedHtml}${emptyHtml}${selectedHtml}`;
  elements.loadMore.hidden = !state.nextCursor;
  persistSelection();
  renderScope();
  refreshIcons();
}

function renderDocumentItem(item) {
  const selected = state.selectedIds.has(item.document_id);
  const ready = item.status === "completed";
  const retryable = ["gate_dependency_unavailable", "gate_internal_error", "gate_safe_mode"].includes(item.domain_decision_code);
  const reviewable = item.domain_status === "review_required" && item.domain_assessment_id && !retryable;
  const reviewMarkup = renderAdmissionReview(item);
  return `<div class="document-item ${selected ? "selected" : ""}">
      <input type="checkbox" data-document-id="${item.document_id}" ${selected ? "checked" : ""} ${ready ? "" : "disabled"} aria-label="选择 ${escapeHtml(item.original_filename)}">
      <span><p class="document-name">${escapeHtml(item.original_filename)}</p>
      <span class="document-meta"><span class="document-status domain-${escapeHtml(item.domain_status)}">${escapeHtml(domainNames[item.domain_status] || item.domain_status)}</span><span>${escapeHtml(genreNames[item.document_genre] || item.document_genre)}</span><span>${escapeHtml(profileNames[item.profile_status] || "论文地图待生成")}</span><span>${item.page_count ?? "-"} 页</span><span>${item.chunk_count} chunks</span></span>
      ${reviewMarkup}
      <span class="document-actions">${reviewable ? `<button class="approve-document" type="button" data-approve-id="${item.document_id}" data-assessment-id="${item.domain_assessment_id}">确认入库</button>` : ""}${item.domain_status === "review_required" && retryable ? `<button class="approve-document" type="button" data-reindex-id="${item.document_id}">重新判断</button>` : ""}<button class="delete-document icon-button" type="button" data-delete-id="${item.document_id}" title="删除论文" aria-label="删除 ${escapeHtml(item.original_filename)}"><i data-lucide="trash-2"></i></button></span></span>
    </div>`;
}

function renderAdmissionReview(item) {
  if (item.domain_status !== "review_required") return "";
  const retryable = ["gate_dependency_unavailable", "gate_internal_error", "gate_safe_mode"].includes(item.domain_decision_code);
  const title = retryable ? "系统暂时无法完成判断" : "正向证据不足，尚未进入知识索引";
  const decision = admissionDecisionNames[item.domain_decision_code] || item.domain_decision_code || "等待复核";
  const failed = (item.domain_failed_requirements || []).map((requirement) => admissionRequirementNames[requirement] || requirement);
  const evidence = (item.domain_evidence || []).slice(0, 2).map((entry) => {
    const pages = (entry.page_numbers || []).join("、");
    return `<li>${pages ? `第 ${escapeHtml(pages)} 页 · ` : ""}${escapeHtml(entry.excerpt || "")}</li>`;
  }).join("");
  return `<div class="admission-review ${retryable ? "admission-system" : ""}">
    <p class="admission-review-title">${escapeHtml(title)}</p>
    <p class="admission-review-reason">${escapeHtml(decision)}${failed.length ? ` · ${escapeHtml(failed.join("、"))}` : ""}</p>
    ${evidence ? `<details><summary>查看证据位置</summary><ul>${evidence}</ul></details>` : ""}
  </div>`;
}

function renderAdmissionRequirements(check) {
  const passed = (check.passed_requirements || []).map((item) => admissionRequirementNames[item] || item);
  const failed = (check.failed_requirements || []).map((item) => admissionRequirementNames[item] || item);
  const evidence = (check.evidence || []).map((entry) => {
    const pages = (entry.page_numbers || []).join("、");
    return `<li>${pages ? `第 ${escapeHtml(pages)} 页 · ` : ""}${escapeHtml(entry.excerpt || "")}</li>`;
  }).join("");
  return `<dl class="admission-requirements"><div><dt>已满足</dt><dd>${escapeHtml(passed.join("、") || "无")}</dd></div><div><dt>未满足</dt><dd>${escapeHtml(failed.join("、") || "无")}</dd></div></dl>${evidence ? `<details><summary>查看判定证据</summary><ul>${evidence}</ul></details>` : ""}`;
}

function cacheDocuments(items) {
  items.forEach((item) => state.documentCache.set(item.document_id, item));
}

function deriveLibraryView() {
  if (!state.libraryQuery) {
    return { matchedDocuments: state.documents, selectedDocuments: [] };
  }
  return {
    matchedDocuments: state.documents.filter((item) => !state.selectedIds.has(item.document_id)),
    selectedDocuments: [...state.selectedIds]
      .map((id) => state.documentCache.get(id))
      .filter(Boolean),
  };
}

function scrollSelectedPapersIntoView() {
  if (!state.libraryQuery || !elements.documentList.querySelector("[data-selected-papers-anchor]")) return;
  window.requestAnimationFrame(() => {
    elements.documentList.scrollTo({
      top: elements.documentList.scrollHeight,
      behavior: "smooth",
    });
  });
}

async function loadDocuments({ append = false, revealSelected = false } = {}) {
  state.libraryRequestController?.abort();
  const controller = new AbortController();
  state.libraryRequestController = controller;
  const generation = ++state.libraryRequestGeneration;
  const url = new URL("/api/documents", location.origin);
  url.searchParams.set("limit", "50");
  if (state.libraryQuery) url.searchParams.set("filename", state.libraryQuery);
  if (append && state.nextCursor) url.searchParams.set("cursor", state.nextCursor);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error("无法加载论文库");
    const payload = await response.json();
    if (generation !== state.libraryRequestGeneration) return;
    cacheDocuments(payload.items);
    state.documents = append ? [...state.documents, ...payload.items] : payload.items;
    state.nextCursor = payload.next_cursor;
    elements.librarySearchStatus.textContent = "";
    elements.librarySearchStatus.hidden = true;
    elements.librarySearchStatus.classList.remove("error");
    renderLibrary();
    if (revealSelected) scrollSelectedPapersIntoView();
  } catch (error) {
    if (error.name === "AbortError") return;
    if (generation !== state.libraryRequestGeneration) return;
    elements.librarySearchStatus.textContent = "文件名检索失败，请重试";
    elements.librarySearchStatus.hidden = false;
    elements.librarySearchStatus.classList.add("error");
  } finally {
    if (generation === state.libraryRequestGeneration) state.libraryRequestController = null;
  }
}

async function checkReadiness() {
  try {
    const response = await fetch("/ready");
    const report = await response.json();
    elements.serviceState.className = `service-state ${report.ready ? "ready" : "error"}`;
    elements.serviceState.lastElementChild.textContent = report.ready ? "服务就绪" : "服务未就绪";
  } catch {
    elements.serviceState.className = "service-state error";
    elements.serviceState.lastElementChild.textContent = "连接失败";
  }
}

async function waitForJob(jobId, documentId) {
  elements.ingestionWrap.hidden = false;
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    const progress = stageProgress[job.state] ?? 0;
    elements.ingestionProgress.value = progress;
    elements.ingestionPercent.textContent = `${progress}%`;
    elements.ingestionStatus.textContent = stageNames[job.state] || job.state;
    if (job.state === "completed") {
      state.selectedIds.add(documentId);
      state.scope = "selected";
      await loadDocuments();
      return;
    }
    if (["review_required", "quarantined"].includes(job.state)) {
      await loadDocuments();
      return;
    }
    if (job.state === "failed") {
      elements.ingestionStatus.textContent = `入库失败：${job.error_message || job.error_code || "未知错误"}`;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "大小未知";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** index)).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function persistUploadItems() {
  const accepted = state.uploadItems
    .filter((item) => item.jobId)
    .slice(-100)
    .map(({ file, ...item }) => item);
  localStorage.setItem(uploadStorageKey, JSON.stringify(accepted));
}

function uploadItemProgress(item) {
  if (terminalUploadStates.has(item.uploadState)) return 100;
  if (item.uploadState === "pending") return 0;
  if (item.uploadState === "uploading") return 3;
  return stageProgress[item.jobState] ?? 5;
}

function uploadItemStatus(item) {
  if (item.uploadState === "upload_failed") return item.error || uploadStateNames.upload_failed;
  if (item.uploadState === "cancelled") return uploadStateNames.cancelled;
  if (item.uploadState !== "accepted") return uploadStateNames[item.uploadState] || item.uploadState;
  if (item.duplicate && item.jobState === "completed") return "已收录（重复）";
  return stageNames[item.jobState] || uploadStateNames.accepted;
}

function renderUploadQueue() {
  const items = state.uploadItems;
  elements.uploadQueue.hidden = items.length === 0;
  elements.fileLabel.textContent = items.length
    ? `队列 ${items.length} 篇 PDF`
    : "选择 PDF";
  const completed = items.filter((item) => terminalJobStates.has(item.jobState)).length;
  const failed = items.filter((item) => item.uploadState === "upload_failed" || item.jobState === "failed").length;
  const active = items.filter((item) => item.uploadState === "uploading" || (item.jobId && !terminalJobStates.has(item.jobState))).length;
  elements.uploadSummary.textContent = `${items.length} 篇 · ${completed} 已结束 · ${active} 处理中${failed ? ` · ${failed} 失败` : ""}`;
  elements.uploadItems.innerHTML = items.map((item) => {
    const progress = uploadItemProgress(item);
    const statusClass = item.uploadState === "upload_failed" || item.jobState === "failed"
      ? "failed"
      : terminalJobStates.has(item.jobState) ? "completed" : "";
    return `<div class="upload-item" data-upload-id="${item.id}">
      <div class="upload-item-name"><strong title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</strong><span>${formatBytes(item.size)}${item.duplicate ? " · 重复文件" : ""}</span></div>
      <progress max="100" value="${progress}" aria-label="${escapeHtml(item.filename)} 进度 ${progress}%"></progress>
      <span class="upload-item-status ${statusClass}" title="${escapeHtml(item.error || "")}">${escapeHtml(uploadItemStatus(item))}</span>
    </div>`;
  }).join("");
  elements.uploadSubmit.disabled = state.uploadRunning || !items.some((item) => item.file && item.uploadState === "pending");
  elements.cancelUpload.hidden = !state.uploadRunning;
  elements.clearUpload.disabled = !items.some((item) => terminalUploadStates.has(item.uploadState) || terminalJobStates.has(item.jobState));
  refreshIcons();
}

function buildUploadBatch(files) {
  const acceptedJobs = state.uploadItems.filter((item) => item.jobId && !terminalJobStates.has(item.jobState));
  const capacity = Math.max(0, 100 - acceptedJobs.length);
  const selected = [...files].slice(0, capacity);
  state.uploadItems = [
    ...acceptedJobs,
    ...selected.map((file) => ({
      id: crypto.randomUUID(),
      file,
      filename: file.name,
      size: file.size,
      uploadState: "pending",
      jobState: null,
      documentId: null,
      jobId: null,
      duplicate: false,
      error: null,
    })),
  ];
  if (files.length > capacity) {
    elements.ingestionWrap.hidden = false;
    elements.ingestionStatus.textContent = `队列最多 100 篇，已保留前 ${capacity} 篇`;
  }
  renderUploadQueue();
}

async function uploadFilesSequentially() {
  const pendingItems = state.uploadItems.filter((item) => item.file && item.uploadState === "pending");
  if (!pendingItems.length || state.uploadRunning) return;
  state.uploadRunning = true;
  state.uploadController = new AbortController();
  renderUploadQueue();
  for (const item of pendingItems) {
    if (state.uploadController.signal.aborted) break;
    item.uploadState = "uploading";
    item.error = null;
    renderUploadQueue();
    const body = new FormData();
    body.append("file", item.file);
    try {
      const response = await fetch("/api/documents", { method: "POST", body, signal: state.uploadController.signal });
      if (!response.ok) throw new Error(await response.text());
      const accepted = await response.json();
      item.uploadState = "accepted";
      item.jobState = accepted.status;
      item.documentId = accepted.document_id;
      item.jobId = accepted.job_id;
      item.duplicate = accepted.duplicate;
      item.file = null;
      persistUploadItems();
      scheduleBatchPolling(0);
    } catch (error) {
      if (error.name === "AbortError") {
        item.uploadState = "cancelled";
        break;
      }
      item.uploadState = "upload_failed";
      item.error = error.message || "上传失败";
    }
    renderUploadQueue();
  }
  if (state.uploadController.signal.aborted) {
    state.uploadItems.forEach((item) => {
      if (item.uploadState === "pending") item.uploadState = "cancelled";
    });
  }
  state.uploadRunning = false;
  state.uploadController = null;
  elements.fileInput.value = "";
  persistUploadItems();
  renderUploadQueue();
  scheduleBatchPolling(0);
}

function scheduleBatchPolling(delay = 1200) {
  if (state.uploadPollTimer) return;
  const activeJobs = state.uploadItems.some((item) => item.jobId && !terminalJobStates.has(item.jobState));
  if (!activeJobs) return;
  state.uploadPollTimer = setTimeout(() => {
    state.uploadPollTimer = null;
    pollBatchJobs();
  }, delay);
}

async function pollBatchJobs() {
  const activeItems = state.uploadItems.filter((item) => item.jobId && !terminalJobStates.has(item.jobState));
  if (!activeItems.length) return;
  try {
    const response = await fetch("/api/jobs/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_ids: activeItems.map((item) => item.jobId) }),
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    const jobs = new Map(payload.jobs.map((job) => [job.job_id, job]));
    activeItems.forEach((item) => {
      const job = jobs.get(item.jobId);
      if (job) {
        item.jobState = job.state;
        item.documentId = job.document_id;
        item.error = job.error_message || job.error_code;
      } else if ((payload.missing_job_ids || []).includes(item.jobId)) {
        item.jobState = "failed";
        item.error = "服务端任务不存在";
      }
    });
    persistUploadItems();
    renderUploadQueue();
    if (payload.jobs.some((job) => terminalJobStates.has(job.state))) await loadDocuments();
  } catch {
    elements.uploadSummary.textContent = "任务状态暂时不可用，正在重试";
  }
  scheduleBatchPolling();
}

function createQuestionView(question, clientTurnId) {
  elements.emptyState?.remove();
  const turn = document.createElement("article");
  turn.className = "conversation-turn";
  turn.dataset.clientTurnId = clientTurnId;
  turn.innerHTML = `<div class="question-block"><p class="question-text">${escapeHtml(question)}</p></div><div class="execution-log"></div><div class="answer-panel"></div>`;
  elements.conversation.appendChild(turn);
  return { turn, log: turn.querySelector(".execution-log"), answer: turn.querySelector(".answer-panel") };
}

function conversationIsNearBottom() {
  return elements.conversation.scrollHeight - elements.conversation.scrollTop - elements.conversation.clientHeight < 96;
}

function appendExecutionStep(log, message, event) {
  const shouldFollow = conversationIsNearBottom();
  log.querySelectorAll(".execution-step.active").forEach((step) => { step.classList.remove("active"); step.classList.add("done"); });
  const step = document.createElement("div");
  step.className = `execution-step ${event === "complete" || event === "refused" ? "done" : "active"}`;
  step.textContent = message;
  log.appendChild(step);
  if (shouldFollow) elements.conversation.scrollTo({ top: elements.conversation.scrollHeight, behavior: "smooth" });
}

function renderAnswer(container, payload) {
  const shouldFollow = conversationIsNearBottom();
  const citations = payload.citations || [];
  const formulaAssets = payload.formula_assets || [];
  const headings = { answer: "回答", partial: "部分回答", clarify: "需要澄清", refuse: "无法回答", error: "服务异常" };
  const action = payload.action || (payload.refused ? "refuse" : "answer");
  const errorHeadings = {
    formula_not_extracted: "公式无法可靠还原",
    formula_text_corrupted: "公式无法可靠还原",
    model_schema_failure: "模型结构校验失败",
    missing_formula_claim: "公式解释结构不完整",
    unknown_citation: "引用校验失败",
    provider_failure: "模型服务不可用",
    strong_reference_not_found: "未找到指定对象",
    strong_reference_ambiguous: "需要明确指定对象",
    strong_reference_stale: "对象索引待修复",
    strong_reference_invalid: "对象标识无效",
    reference_index_inconsistent: "对象证据索引异常",
  };
  const heading = errorHeadings[payload.audit_result] || headings[action] || "回答";
  const formulaState = formulaAssets.length
    ? (formulaAssets.every((asset) => asset.fidelity_status === "source_exact")
      ? { label: "公式已可靠提取", className: "formula-reliable" }
      : { label: "公式已定位，文本待复核", className: "formula-review" })
    : (["formula_not_extracted", "formula_text_corrupted"].includes(payload.audit_result)
      ? { label: "公式无法可靠还原", className: "formula-unavailable" }
      : null);
  const debugPayload = { ...payload };
  delete debugPayload.reasoning_content;
  delete debugPayload.reasoning;
  const epistemicNames = { source_fact: "原文事实", evidence_synthesis: "证据综合", deterministic_derivation: "确定性推导", evidence_bounded_hypothesis: "待验证假设" };
  const epistemicLabel = epistemicNames[payload.epistemic_level] || "证据回答";
  const hypothesisNotice = payload.epistemic_level === "evidence_bounded_hypothesis"
    ? "<div class='answer-body refusal'>以下内容是基于库内证据的条件性推测，不是论文已经验证的结论。</div>"
    : "";
  const citationHtml = citations.map((citation) => `<article class="citation"><div class="citation-title"><span>${escapeHtml(citation.paper_title)}</span><span>第 ${citation.page_start}${citation.page_end !== citation.page_start ? `–${citation.page_end}` : ""} 页</span></div><blockquote>${escapeHtml(citation.quoted_snippet)}</blockquote></article>`).join("");
  const formulaHtml = formulaAssets.map((asset, index) => `<figure class="formula-source"><figcaption>公式 ${escapeHtml(asset.formula_number ? `(${asset.formula_number})` : "")} · 第 ${escapeHtml(asset.page_number)} 页</figcaption><div class="formula-mathml" data-formula-index="${index}"></div><details class="formula-crop-audit"><summary>查看原文裁剪</summary><img class="formula-source-image" src="${escapeHtml(asset.image_url)}" alt="原文公式 ${escapeHtml(asset.formula_number || "")}" loading="eager"></details></figure>`).join("");
  const formulaQualityHtml = formulaState
    ? `<div class="formula-quality ${formulaState.className}">${escapeHtml(formulaState.label)}</div>`
    : "";
  container.innerHTML = `<h2>${escapeHtml(heading)}</h2>
    <div class="answer-meta"><span class="passed">${escapeHtml(epistemicLabel)}</span><span>${escapeHtml(payload.answer_mode || "synthesize")}</span></div>
    ${formulaQualityHtml}
    ${hypothesisNotice}
    <div class="answer-body ${payload.refused ? "refusal" : ""}">${escapeHtml(payload.answer)}</div>
    ${formulaHtml ? `<section class="formula-sources">${formulaHtml}</section>` : ""}
    ${(payload.unsupported_parts || []).length ? `<div class="answer-body refusal">未支持部分：${escapeHtml(payload.unsupported_parts.join("；"))}</div>` : ""}
    <div class="answer-meta"><span class="${payload.audit_result === "passed" ? "passed" : ""}">审计：${escapeHtml(payload.audit_result)}</span><span>风险：${escapeHtml(payload.hallucination_risk)}</span></div>
    ${citationHtml ? `<section class="citations">${citationHtml}</section>` : ""}
    <details class="debug"><summary>调试信息</summary><pre>${escapeHtml(JSON.stringify(debugPayload, null, 2))}</pre></details>`;
  formulaAssets.forEach((asset, index) => {
    const host = container.querySelector(`[data-formula-index="${index}"]`);
    const math = parseSafeMathMl(asset.rendered_mathml);
    if (host && math) host.replaceChildren(math);
    else if (host) host.remove();
  });
  if (shouldFollow) elements.conversation.scrollTo({ top: elements.conversation.scrollHeight, behavior: "smooth" });
}

async function consumeSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const event = block.split("\n").find((line) => line.startsWith("event:"))?.slice(6).trim();
      const data = block.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).join("\n");
      if (event && data) onEvent(event, JSON.parse(data));
    }
    if (done) break;
  }
}

function referenceResolutionMessage(payload) {
  const kindNames = {
    formula: "公式",
    figure: "图",
    table: "表",
    section: "章节",
    page: "页码",
    document: "论文",
  };
  const labels = (payload.resolutions || []).map((item) => {
    const kind = kindNames[item.entity_type] || item.entity_type || "对象";
    const key = item.entity_type === "formula" ? `(${item.canonical})` : item.canonical;
    return `${kind} ${key} [${item.resolution_status}]`;
  });
  return labels.length ? `已识别强标识：${labels.join("；")}` : (payload.message || "强标识解析完成");
}

elements.documentList.addEventListener("change", (event) => {
  const id = event.target.dataset.documentId;
  if (!id) return;
  if (event.target.checked) {
    state.selectedIds.add(id);
    state.scope = "selected";
  } else {
    state.selectedIds.delete(id);
  }
  renderLibrary();
  if (event.target.checked && state.libraryQuery) scrollSelectedPapersIntoView();
});

elements.documentList.addEventListener("click", async (event) => {
  const reindexButton = event.target.closest("[data-reindex-id]");
  if (reindexButton) {
    reindexButton.disabled = true;
    reindexButton.textContent = "处理中";
    const response = await fetch(`/api/documents/${reindexButton.dataset.reindexId}/reindex`, { method: "POST" });
    if (!response.ok) {
      reindexButton.disabled = false;
      reindexButton.textContent = "重新判断";
      return;
    }
    const accepted = await response.json();
    await waitForJob(accepted.job_id, accepted.document_id);
    return;
  }
  const approveButton = event.target.closest("[data-approve-id]");
  if (approveButton) {
    approveButton.disabled = true;
    approveButton.textContent = "处理中";
    const response = await fetch(`/api/documents/${approveButton.dataset.approveId}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assessment_id: approveButton.dataset.assessmentId }),
    });
    if (!response.ok) {
      approveButton.disabled = false;
      approveButton.textContent = "确认入库";
      return;
    }
    const accepted = await response.json();
    await waitForJob(accepted.job_id, accepted.document_id);
    return;
  }
  const deleteButton = event.target.closest("[data-delete-id]");
  if (deleteButton) await openDeletionReview(deleteButton.dataset.deleteId);
});

async function openDeletionReview(documentId) {
  state.deletionCheck = null;
  elements.deleteForm.hidden = true;
  elements.deleteCheckContent.innerHTML = "<p class='document-meta'>正在重新计算领域相关性与删除影响...</p>";
  elements.deleteDialog.showModal();
  refreshIcons();
  const response = await fetch(`/api/documents/${documentId}/deletion-check`, { method: "POST" });
  if (!response.ok) {
    elements.deleteCheckContent.innerHTML = `<p class="deletion-warning">预检失败：${escapeHtml(await response.text())}</p>`;
    return;
  }
  const check = await response.json();
  state.deletionCheck = check;
  elements.deleteCheckContent.innerHTML = `${check.warning ? `<p class="deletion-warning">${escapeHtml(check.warning)}</p>` : ""}
    <p class="delete-filename">${escapeHtml(check.original_filename)}</p>
    <dl class="impact-grid"><div><dt>复检结论</dt><dd>${escapeHtml(domainNames[check.fresh_domain_status] || check.fresh_domain_status)}</dd></div><div><dt>判定依据</dt><dd>${escapeHtml(admissionDecisionNames[check.fresh_decision_code] || check.fresh_decision_code)}</dd></div><div><dt>页数</dt><dd>${check.page_count}</dd></div><div><dt>知识切片</dt><dd>${check.chunk_count}</dd></div><div><dt>向量</dt><dd>${check.vector_count}</dd></div><div><dt>历史问答</dt><dd>${check.answer_audit_count}</dd></div></dl>
    ${renderAdmissionRequirements(check)}`;
  elements.deleteConfirmFilename.value = "";
  elements.deleteSubmit.disabled = true;
  elements.deleteForm.hidden = false;
}

elements.deleteConfirmFilename.addEventListener("input", () => {
  elements.deleteSubmit.disabled = !state.deletionCheck || elements.deleteConfirmFilename.value !== state.deletionCheck.original_filename;
});
elements.deleteCancel.addEventListener("click", () => elements.deleteDialog.close());
elements.deleteForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.deletionCheck || elements.deleteSubmit.disabled) return;
  elements.deleteSubmit.disabled = true;
  const check = state.deletionCheck;
  const response = await fetch(`/api/documents/${check.document_id}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmation_token: check.confirmation_token, confirm_filename: elements.deleteConfirmFilename.value }),
  });
  if (!response.ok) {
    elements.deleteCheckContent.insertAdjacentHTML("afterbegin", `<p class="deletion-warning">删除失败：${escapeHtml(await response.text())}</p>`);
    return;
  }
  state.selectedIds.delete(check.document_id);
  state.documentCache.delete(check.document_id);
  state.selectedDocumentSnapshots.delete(check.document_id);
  persistSelection();
  elements.deleteDialog.close();
  await loadDocuments();
});

document.querySelectorAll(".scope-option").forEach((button) => button.addEventListener("click", () => { state.scope = button.dataset.scope; persistSelection(); renderScope(); }));
elements.conversationSelect.addEventListener("change", () => loadConversation(elements.conversationSelect.value));
elements.newConversation.addEventListener("click", () => createConversation());
elements.renameConversation.addEventListener("click", async () => {
  const current = state.conversations.find((item) => item.conversation_id === state.conversationId);
  const title = window.prompt("会话名称", current?.title || "研究会话")?.trim();
  if (!title) return;
  const response = await fetch(`/api/conversations/${state.conversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (response.ok) await loadConversations({ restore: false });
});
elements.resetConversation.addEventListener("click", async () => {
  if (!window.confirm("清空当前会话的全部问答记录？论文库不会受影响。")) return;
  const response = await fetch(`/api/conversations/${state.conversationId}/reset`, { method: "POST" });
  if (response.ok) renderTranscript((await response.json()).messages || []);
});
elements.deleteConversation.addEventListener("click", async () => {
  if (!window.confirm("永久删除当前研究会话？论文库不会受影响。")) return;
  const response = await fetch(`/api/conversations/${state.conversationId}`, { method: "DELETE" });
  if (!response.ok) return;
  localStorage.removeItem(conversationStorageKey);
  state.conversationId = null;
  await loadConversations({ restore: true });
});
let librarySearchTimer = null;
elements.filenameSearch.addEventListener("input", () => {
  window.clearTimeout(librarySearchTimer);
  state.libraryRequestController?.abort();
  state.libraryRequestGeneration += 1;
  elements.librarySearchStatus.hidden = true;
  elements.librarySearchStatus.classList.remove("error");
  const query = elements.filenameSearch.value.trim();
  if (!query) {
    state.libraryQuery = "";
    loadDocuments();
    return;
  }
  librarySearchTimer = window.setTimeout(() => {
    state.libraryQuery = query;
    loadDocuments({ revealSelected: true });
  }, 250);
});
elements.filenameSearch.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  window.clearTimeout(librarySearchTimer);
  state.libraryRequestController?.abort();
  state.libraryRequestGeneration += 1;
  state.libraryQuery = elements.filenameSearch.value.trim();
  loadDocuments({ revealSelected: Boolean(state.libraryQuery) });
});
elements.refreshLibrary.addEventListener("click", () => loadDocuments());
elements.loadMore.addEventListener("click", () => loadDocuments({ append: true }));
elements.fileInput.addEventListener("change", () => buildUploadBatch(elements.fileInput.files));

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await uploadFilesSequentially();
});

elements.cancelUpload.addEventListener("click", () => state.uploadController?.abort());
elements.clearUpload.addEventListener("click", () => {
  state.uploadItems = state.uploadItems.filter((item) => !terminalUploadStates.has(item.uploadState) && !terminalJobStates.has(item.jobState));
  persistUploadItems();
  renderUploadQueue();
});

elements.keyForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/api/session/deepseek-key", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId, api_key: elements.keyInput.value }) });
  state.sessionHasKey = response.ok;
  elements.keyInput.value = "";
  elements.keyStatus.textContent = response.ok ? "已连接" : "连接失败";
});

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.asking) return;
  if (!state.sessionHasKey) { elements.keyStatus.textContent = "请先连接模型"; elements.keyInput.focus(); return; }
  if (state.scope === "selected" && state.selectedIds.size === 0) { elements.scopeSummary.textContent = "请至少选择一篇论文"; return; }
  const question = elements.question.value.trim();
  if (!question) return;
  state.asking = true;
  state.abortController = new AbortController();
  elements.sendButton.disabled = true;
  elements.cancelButton.hidden = false;
  const clientTurnId = crypto.randomUUID();
  const view = createQuestionView(question, clientTurnId);
  const payload = { session_id: sessionId, conversation_id: state.conversationId, client_turn_id: clientTurnId, question, top_n: 8, scope: state.scope, document_ids: state.scope === "selected" ? [...state.selectedIds] : [] };
  elements.question.value = "";
  try {
    const response = await fetch("/api/chat/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: state.abortController.signal });
    await consumeSse(response, (eventName, data) => {
      if (eventName === "result") renderAnswer(view.answer, data);
      else if (eventName === "error") appendExecutionStep(view.log, data.detail || "问答失败", "refused");
      else if (eventName === "reference_resolution") appendExecutionStep(view.log, referenceResolutionMessage(data), eventName);
      else appendExecutionStep(view.log, data.message || eventName, eventName);
    });
  } catch (error) {
    appendExecutionStep(view.log, error.name === "AbortError" ? "已取消问答" : "连接中断，请稍后重试", "refused");
  } finally {
    state.asking = false;
    state.abortController = null;
    elements.sendButton.disabled = false;
    elements.cancelButton.hidden = true;
    await loadConversations({ restore: false }).catch(() => {});
  }
});

elements.cancelButton.addEventListener("click", () => state.abortController?.abort());

renderUploadQueue();
scheduleBatchPolling(0);
Promise.all([loadDocuments(), checkReadiness()])
  .then(() => loadConversations({ restore: true }))
  .finally(() => { renderScope(); refreshIcons(); });

window.addEventListener("pagehide", () => {
  if (!state.sessionHasKey) return;
  fetch(`/api/session/deepseek-key/${sessionId}`, {
    method: "DELETE",
    keepalive: true,
  });
});
