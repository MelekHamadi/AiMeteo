// Will be set after DOM switches to chat mode
let chat = null;
let selectedProject = null;
let projectsList = [];
let conversationContext = { lastProject: null };
let hasStartedChat = false;

/* =====================================================================
   SESSION STORAGE
   ===================================================================== */
const SESSIONS_KEY = "pmo_sessions_v2";
const RECENTS_HIDDEN_KEY = "pmo_recents_hidden";

let currentSessionId = null;
let viewingSessionId = null;

function getSessions() {
    try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]"); }
    catch { return []; }
}
function saveSessions(sessions) {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}

function createSession(project) {
    const sessions = getSessions();
    const id = "s_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7);
    const now = new Date().toISOString();
    sessions.unshift({ id, title: null, project: project || null, messages: [], createdAt: now, updatedAt: now });
    saveSessions(sessions);
    return id;
}

function appendMessageToSession(sessionId, role, content) {
    const sessions = getSessions();
    const idx = sessions.findIndex(s => s.id === sessionId);
    if (idx === -1) return;
    const msg = { role, content, timestamp: new Date().toISOString() };
    sessions[idx].messages.push(msg);
    sessions[idx].updatedAt = msg.timestamp;
    if (!sessions[idx].title && role === "user") {
        sessions[idx].title = content.trim().substring(0, 60) + (content.length > 60 ? "…" : "");
    }
    saveSessions(sessions);
    renderRecentsList();
}

function deleteSession(sessionId) {
    let sessions = getSessions();
    sessions = sessions.filter(s => s.id !== sessionId);
    saveSessions(sessions);
    if (currentSessionId === sessionId) currentSessionId = null;
    if (viewingSessionId === sessionId) closeSessionViewer();
    renderRecentsList();
}

function getDateGroup(isoDate) {
    const d = new Date(isoDate);
    const now = new Date();
    const diffDays = Math.floor((now - d) / 86400000);
    if (diffDays === 0) return "Aujourd'hui";
    if (diffDays === 1) return "Hier";
    if (diffDays < 7) return "Cette semaine";
    if (diffDays < 30) return "Ce mois-ci";
    return "Plus ancien";
}

const GROUP_ORDER = ["Aujourd'hui", "Hier", "Cette semaine", "Ce mois-ci", "Plus ancien"];

function renderRecentsList() {
    const list = document.getElementById("recentsList");
    if (!list) return;
    const sessions = getSessions();

    if (sessions.length === 0) {
        list.innerHTML = `
            <div class="recents-empty">
                <i class="fa-regular fa-message"></i>
                <span>Aucune conversation</span>
            </div>`;
        return;
    }

    const groups = {};
    sessions.forEach(s => {
        const g = getDateGroup(s.updatedAt || s.createdAt);
        if (!groups[g]) groups[g] = [];
        groups[g].push(s);
    });

    let html = "";
    GROUP_ORDER.forEach(groupName => {
        if (!groups[groupName]) return;
        html += `<div class="recents-date-group">${groupName}</div>`;
        groups[groupName].forEach(s => {
            const isActive = s.id === currentSessionId ? "active-session" : "";
            const projectTag = s.project ? ` — ${s.project}` : "";
            const title = escapeHtml(s.title || "Nouvelle conversation") + escapeHtml(projectTag);
            html += `
                <div class="recent-item ${isActive}" data-id="${s.id}">
                    <button class="recent-item-btn" onclick="openSessionViewer('${s.id}')" title="${title}">
                        ${title}
                    </button>
                    <div class="recent-item-actions">
                        <button class="recent-del-btn" onclick="confirmDeleteSession(event,'${s.id}')" title="Supprimer">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                </div>`;
        });
    });

    list.innerHTML = html;
}

function confirmDeleteSession(event, sessionId) {
    event.stopPropagation();
    deleteSession(sessionId);
    showToast("Conversation supprimée");
}

function toggleRecents() {
    const section = document.getElementById("recentsSection");
    const hidden = section.classList.toggle("collapsed-recents");
    localStorage.setItem(RECENTS_HIDDEN_KEY, hidden ? "1" : "0");
}

/* ---------- SESSION VIEWER ---------- */
function openSessionViewer(sessionId) {
    const sessions = getSessions();
    const session = sessions.find(s => s.id === sessionId);
    if (!session) return;

    viewingSessionId = sessionId;
    document.getElementById("sessionViewerTitle").textContent = session.title || "Nouvelle conversation";
    document.getElementById("sessionViewerDate").textContent =
        formatFullDate(session.createdAt) + (session.project ? ` · ${session.project}` : "");

    const body = document.getElementById("sessionViewerBody");
    if (!session.messages || session.messages.length === 0) {
        body.innerHTML = `<div style="color:var(--text-muted);font-size:13px;text-align:center;padding:40px 0;">Aucun message dans cette conversation.</div>`;
    } else {
        body.innerHTML = session.messages.map(msg => {
            if (msg.role === "system") return "";
            const isUser = msg.role === "user";
            const avatarClass = isUser ? "sv-user" : "sv-bot";
            const bubbleClass = isUser ? "sv-user" : "sv-bot";
            const icon = isUser ? "fa-user" : "fa-robot";
            const contentHtml = isUser ? escapeHtml(msg.content) : renderMarkdown(msg.content);
            return `
                <div class="sv-msg ${isUser ? "sv-user" : ""}">
                    <div class="sv-avatar ${avatarClass}"><i class="fa-solid ${icon}"></i></div>
                    <div class="sv-bubble ${bubbleClass}">${contentHtml}</div>
                </div>`;
        }).join("");
    }

    document.getElementById("sessionViewer").classList.add("open");
    document.getElementById("sessionOverlay").classList.add("active");
    setTimeout(() => { body.scrollTop = body.scrollHeight; }, 60);
}

function closeSessionViewer() {
    document.getElementById("sessionViewer").classList.remove("open");
    document.getElementById("sessionOverlay").classList.remove("active");
    viewingSessionId = null;
}

function restoreSession() {
    const sessions = getSessions();
    const session = sessions.find(s => s.id === viewingSessionId);
    if (!session) return;

    closeSessionViewer();
    hasStartedChat = false;
    const chatBodyR = document.getElementById("chatBody");
    const welcomeR = document.getElementById("welcomeScreen");
    if (welcomeR) { welcomeR.style.display = "none"; welcomeR.style.opacity = ""; }
    if (chatBodyR) chatBodyR.style.display = "flex";
    chat = document.getElementById("chat");
    hasStartedChat = true;

    const taChatR = document.getElementById("questionChat");
    if (taChatR && !taChatR._wired) {
        taChatR._wired = true;
        taChatR.addEventListener("input", () => { taChatR.style.height = "auto"; taChatR.style.height = taChatR.scrollHeight + "px"; });
        taChatR.addEventListener("keydown", function(e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
    }

    chat.innerHTML = "";
    conversationContext = { lastProject: null };

    if (session.project) {
        selectedProject = session.project;
        const disp = document.getElementById("selected-project-display");
        if (disp) disp.textContent = session.project;
        conversationContext.lastProject = session.project;
    } else {
        selectedProject = null;
        const disp = document.getElementById("selected-project-display");
        if (disp) disp.textContent = "Tous les projets";
    }

    session.messages.forEach(msg => {
        if (msg.role === "system") return;
        if (msg.role === "user") {
            createMessage(msg.content, "user", null);
        } else if (msg.role === "bot") {
            const msgId = Date.now() + Math.random();
            const bubble = createMessage("", "bot", msgId);
            bubble.innerHTML = renderMarkdown(msg.content);
        }
    });

    currentSessionId = session.id;
    showToast("Conversation restaurée");
    renderChips();
    showChips();
}

function deleteCurrentSession() {
    if (!viewingSessionId) return;
    deleteSession(viewingSessionId);
    closeSessionViewer();
    showToast("Conversation supprimée");
}

function formatFullDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return d.toLocaleDateString("fr-FR", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

/* =====================================================================
   CONTEXTUAL CHIPS
   ===================================================================== */
const CHIP_SUGGESTIONS = {
    default: [
        { label: "Bilan des risques", query: "Bilan des risques sur tous les projets" },
        { label: "Phase actuelle", query: "Quelle est la phase actuelle de chaque projet ?" },
        { label: "Projet le plus critique", query: "Quel est le projet le plus à risque ?" },
    ],
    project: (name) => [
        { label: "Synthèse des risques", query: `Fais une synthèse des risques du projet ${name}` },
        { label: "Santé du projet", query: `Quelle est la santé du projet ${name} ?` },
        { label: "Que faire ?", query: `Que faire pour le projet ${name} ?` },
        { label: "Prédiction", query: `Anticipe les problèmes du projet ${name}` },
        { label: "Budget", query: `Quel est le budget du projet ${name} ?` },
        { label: "Chef de projet", query: `Qui est le chef de projet du projet ${name} ?` },
    ]
};

function renderChips() {
    const chips = selectedProject
        ? CHIP_SUGGESTIONS.project(selectedProject)
        : CHIP_SUGGESTIONS.default;
    const html = chips.map(chip => `
        <button class="chip-btn" onclick="useChip('${escapeAttr(chip.query)}')">
            ${escapeHtml(chip.label)}
        </button>
    `).join("");
    const cWelcome = document.getElementById("chipsContainerWelcome");
    const cChat = document.getElementById("chipsContainer");
    if (cWelcome) cWelcome.innerHTML = html;
    if (cChat) cChat.innerHTML = html;
}

function useChip(query) {
    const input = document.getElementById("question");
    input.value = query;
    input.focus();
    setTimeout(() => sendMessage(), 100);
}

/* =====================================================================
   MARKDOWN RENDERER
   ===================================================================== */
function renderMarkdown(text) {
    if (!text) return "";

    text = text
        .replace(/^["']|["']$/gm, "")
        .replace(/^- "(.+)"$/gm, "- $1")
        .replace(/\*\*Action:\s*/gi, "")
        .replace(/\*\*Impact:\s*/gi, "Impact : ")
        .replace(/\*\*Effort:\s*/gi, "Effort : ");

    const lines = text.split("\n");
    let html = "";
    let inList = false, inOrderedList = false, inTable = false;
    let tableRows = [];

    const flushTable = () => {
        if (!tableRows.length) return;
        let th = '<div class="resp-table-wrap"><table class="resp-table">';
        tableRows.forEach((row, i) => {
            const cells = row.split("|").map(c => c.trim()).filter(Boolean);
            if (i === 0) {
                th += "<thead><tr>" + cells.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
            } else if (i === 1 && cells.every(c => /^[-:]+$/.test(c))) {
                // skip separator
            } else {
                th += "<tr>" + cells.map(c => `<td>${inlineMarkdown(c)}</td>`).join("") + "</tr>";
            }
        });
        th += "</tbody></table></div>";
        html += th; tableRows = []; inTable = false;
    };
    const flushList = () => {
        if (inList) { html += "</ul>"; inList = false; }
        if (inOrderedList) { html += "</ol>"; inOrderedList = false; }
    };

    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i];
        const line = raw.trim();

        if (line.startsWith("|")) { flushList(); inTable = true; tableRows.push(line); continue; }
        else if (inTable) { flushTable(); }
        if (!line) { flushList(); continue; }

        const h4 = line.match(/^####\s+(.+)/);
        const h3 = line.match(/^###\s+(.+)/);
        const h2 = line.match(/^##\s+(.+)/);
        const h1 = line.match(/^#\s+(.+)/);
        if (h1) { flushList(); html += `<h1 class="resp-h1">${inlineMarkdown(h1[1])}</h1>`; continue; }
        if (h2) { flushList(); html += `<h2 class="resp-h2">${inlineMarkdown(h2[1])}</h2>`; continue; }
        if (h3) { flushList(); html += `<h3 class="resp-h3 ${detectSectionClass(h3[1])}">${inlineMarkdown(h3[1])}</h3>`; continue; }
        if (h4) { flushList(); html += `<h4 class="resp-h4 ${detectSectionClass(h4[1])}">${inlineMarkdown(h4[1])}</h4>`; continue; }

        const bullet = line.match(/^[-*]\s+(.+)/);
        if (bullet) {
            if (inOrderedList) { html += "</ol>"; inOrderedList = false; }
            if (!inList) { html += "<ul class='resp-list'>"; inList = true; }
            html += `<li>${inlineMarkdown(bullet[1])}</li>`; continue;
        }
        const ordered = line.match(/^\d+\.\s+(.+)/);
        if (ordered) {
            if (inList) { html += "</ul>"; inList = false; }
            if (!inOrderedList) { html += "<ol class='resp-list resp-olist'>"; inOrderedList = true; }
            html += `<li>${inlineMarkdown(ordered[1])}</li>`; continue;
        }

        const actionLine = line.match(/Action\s*:\s*(.+?)\s*\|\s*Impact\s*:\s*(.+?)\s*\|\s*Effort\s*:\s*(.+)/i);
        if (actionLine) {
            flushList();
            const impact = actionLine[2].trim().toLowerCase();
            const effort = actionLine[3].trim().toLowerCase();
            const impactClass = impact.includes("fort") ? "tag-high" : impact.includes("moyen") ? "tag-med" : "tag-low";
            const effortClass = effort.includes("fort") ? "tag-high" : effort.includes("moyen") ? "tag-med" : "tag-low";
            html += `<div class="action-card"><div class="action-title">${inlineMarkdown(actionLine[1])}</div><div class="action-tags"><span class="action-tag ${impactClass}">Impact : ${actionLine[2].trim()}</span><span class="action-tag ${effortClass}">Effort : ${actionLine[3].trim()}</span></div></div>`;
            continue;
        }

        const scoreMatch = line.match(/Score de santé.+?:\s*(\d+(?:\.\d+)?)\s*\/\s*100/i);
        if (scoreMatch) {
            flushList();
            const score = parseFloat(scoreMatch[1]);
            const color = score >= 80 ? "#3B6D11" : score >= 60 ? "#639922" : score >= 40 ? "#854F0B" : "#A32D2D";
            html += `<div class="health-bar-wrap"><div class="health-bar-label">${inlineMarkdown(line)}</div><div class="health-bar-bg"><div class="health-bar-fill" style="width:${Math.min(score,100)}%;background:${color}"></div></div><div class="health-bar-score" style="color:${color}">${score}/100</div></div>`;
            continue;
        }

        const kvLine = line.match(/^\*\*(.+?)\*\*\s*[:\-]\s*(.+)/);
        if (kvLine) {
            flushList();
            html += `<div class="resp-kv"><span class="resp-kv-key">${kvLine[1]}</span><span class="resp-kv-val">${inlineMarkdown(kvLine[2])}</span></div>`;
            continue;
        }

        flushList();
        if (line.length > 0) html += `<p class="resp-p">${inlineMarkdown(line)}</p>`;
    }
    flushList(); flushTable();
    return html;
}

function detectSectionClass(title) {
    const t = title.toLowerCase();
    if (t.includes("risque") || t.includes("critique") || t.includes("alerte")) return "section-risk";
    if (t.includes("action") || t.includes("recommand") || t.includes("priorit")) return "section-action";
    if (t.includes("santé") || t.includes("score") || t.includes("health")) return "section-health";
    if (t.includes("prédiction") || t.includes("futur") || t.includes("anticip")) return "section-pred";
    if (t.includes("résumé") || t.includes("exécutif") || t.includes("conclusion") || t.includes("verdict")) return "section-summary";
    if (t.includes("budget") || t.includes("financ")) return "section-budget";
    return "";
}

function inlineMarkdown(text) {
    if (!text) return "";
    return text
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`(.+?)`/g, "<code>$1</code>")
        .replace(/~~(.+?)~~/g, "<del>$1</del>")
        .replace(/\b(À redresser)\b/g, '<span class="status-badge badge-critical">$1</span>')
        .replace(/\b(À surveiller)\b/g, '<span class="status-badge badge-warning">$1</span>')
        .replace(/\b(En contrôle)\b/g, '<span class="status-badge badge-ok">$1</span>')
        .replace(/\b(Amélioration)\b/g, '<span class="trend-badge trend-up">↑ $1</span>')
        .replace(/\b(Détérioration)\b/g, '<span class="trend-badge trend-down">↓ $1</span>')
        .replace(/\b(Stand by)\b/g, '<span class="trend-badge trend-neutral">— $1</span>')
        .replace(/\b(CRITIQUE|critique)\b/g, '<span class="lvl-badge lvl-critical">$1</span>')
        .replace(/\b(ÉLEVÉ|élevé|ELEVE|eleve)\b/g, '<span class="lvl-badge lvl-high">$1</span>')
        .replace(/\b(MODÉRÉ|modéré|MODERE|modere)\b/g, '<span class="lvl-badge lvl-med">$1</span>')
        .replace(/\b(FAIBLE|faible)\b/g, '<span class="lvl-badge lvl-low">$1</span>');
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
function escapeAttr(str) {
    return str.replace(/'/g, "\\'").replace(/"/g, "&quot;").replace(/\n/g, " ");
}

/* =====================================================================
   MESSAGE UI
   ===================================================================== */
let lastUserMessage = "";
let feedbackState = {};

function createMessage(text, type, msgId) {
    const row = document.createElement("div");
    row.className = "msg-row " + type;

    const wrapper = document.createElement("div");
    wrapper.className = "msg-wrapper";

    const bubble = document.createElement("div");
    bubble.className = `message ${type}`;
    if (type === "bot") {
        bubble.innerHTML = text;
    } else {
        bubble.textContent = text;
    }

    wrapper.appendChild(bubble);

    // Action bar for bot messages (copy, thumbs, retry)
    if (type === "bot" && msgId) {
        wrapper.appendChild(createBotActionBar(msgId, text, bubble));
    }

    // Action bar for user messages (copy, edit, retry)
    if (type === "user") {
        wrapper.appendChild(createUserActionBar(text, bubble, row));
    }

    row.appendChild(wrapper);

    const chatEl = chat || document.getElementById("chat");
    if (chatEl) { chatEl.appendChild(row); chatEl.scrollTop = chatEl.scrollHeight; }
    return bubble;
}

/* Action bar for BOT messages: copy · thumbs up · thumbs down · retry */
function createBotActionBar(msgId, rawText, bubble) {
    const bar = document.createElement("div");
    bar.className = "msg-action-bar";

    // Copy
    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn";
    copyBtn.title = "Copier la réponse";
    copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
    copyBtn.onclick = () => {
        const plain = bubble.innerText || bubble.textContent;
        navigator.clipboard.writeText(plain).then(() => {
            copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
            copyBtn.classList.add("copied");
            setTimeout(() => { copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>'; copyBtn.classList.remove("copied"); }, 1800);
        });
    };

    // Thumbs up
    const thumbUpBtn = document.createElement("button");
    thumbUpBtn.className = "msg-action-btn";
    thumbUpBtn.title = "Bonne réponse";
    thumbUpBtn.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';

    // Thumbs down
    const thumbDownBtn = document.createElement("button");
    thumbDownBtn.className = "msg-action-btn";
    thumbDownBtn.title = "Réponse incorrecte";
    thumbDownBtn.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';

    thumbUpBtn.onclick = () => {
        if (feedbackState[msgId] === "up") {
            feedbackState[msgId] = null; thumbUpBtn.classList.remove("active-feedback");
            thumbUpBtn.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';
        } else {
            feedbackState[msgId] = "up"; thumbUpBtn.classList.add("active-feedback");
            thumbUpBtn.innerHTML = '<i class="fa-solid fa-thumbs-up"></i>';
            thumbDownBtn.classList.remove("active-feedback");
            thumbDownBtn.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';
            showToast("Merci pour votre retour !");
        }
    };
    thumbDownBtn.onclick = () => {
        if (feedbackState[msgId] === "down") {
            feedbackState[msgId] = null; thumbDownBtn.classList.remove("active-feedback");
            thumbDownBtn.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';
        } else {
            feedbackState[msgId] = "down"; thumbDownBtn.classList.add("active-feedback");
            thumbDownBtn.innerHTML = '<i class="fa-solid fa-thumbs-down"></i>';
            thumbUpBtn.classList.remove("active-feedback");
            thumbUpBtn.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';
            showToast("Retour enregistré — nous améliorerons.");
        }
    };

    // Retry (regenerate)
    const retryBtn = document.createElement("button");
    retryBtn.className = "msg-action-btn";
    retryBtn.title = "Régénérer la réponse";
    retryBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i>';
    retryBtn.onclick = () => {
        if (lastUserMessage) {
            const row = bar.closest(".msg-row");
            if (row) row.remove();
            askQuestion(lastUserMessage);
        }
    };

    bar.appendChild(copyBtn);
    bar.appendChild(thumbUpBtn);
    bar.appendChild(thumbDownBtn);
    bar.appendChild(retryBtn);
    return bar;
}

/* Action bar for USER messages: copy · edit · resend */
function createUserActionBar(text, bubble, row) {
    const bar = document.createElement("div");
    bar.className = "msg-action-bar";

    // Copy
    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn";
    copyBtn.title = "Copier le message";
    copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
    copyBtn.onclick = () => {
        navigator.clipboard.writeText(text).then(() => {
            copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
            copyBtn.classList.add("copied");
            setTimeout(() => { copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>'; copyBtn.classList.remove("copied"); }, 1800);
        });
    };

    // Edit — puts the message text back in the input for editing
    const editBtn = document.createElement("button");
    editBtn.className = "msg-action-btn";
    editBtn.title = "Modifier le message";
    editBtn.innerHTML = '<i class="fa-regular fa-pen-to-square"></i>';
    editBtn.onclick = () => {
        const input = getActiveInput();
        if (!input) return;
        input.value = text;
        input.style.height = "auto";
        input.style.height = input.scrollHeight + "px";
        input.focus();
        // Move cursor to end
        input.selectionStart = input.selectionEnd = input.value.length;
        showToast("Message copié dans le champ de saisie");
    };

    // Resend — re-ask the exact same question
    const resendBtn = document.createElement("button");
    resendBtn.className = "msg-action-btn";
    resendBtn.title = "Renvoyer ce message";
    resendBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i>';
    resendBtn.onclick = () => {
        askQuestion(text);
    };

    bar.appendChild(copyBtn);
    bar.appendChild(editBtn);
    bar.appendChild(resendBtn);
    return bar;
}

function showToast(message) {
    let toast = document.getElementById("pmo-toast");
    if (!toast) { toast = document.createElement("div"); toast.id = "pmo-toast"; document.body.appendChild(toast); }
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2500);
}

function addSystemMessage(text) {
    const chatEl = chat || document.getElementById("chat");
    if (!chatEl) return;
    const row = document.createElement("div");
    row.className = "msg-row system";
    const bubble = document.createElement("div");
    bubble.className = "message system";
    bubble.innerHTML = `<i class="fa-solid fa-info-circle"></i> ${text}`;
    row.appendChild(bubble);
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;
}

async function streamRender(element, html) {
    element.innerHTML = "";
    const temp = document.createElement("div");
    temp.innerHTML = html;
    const nodes = Array.from(temp.childNodes);
    const chatEl = chat || document.getElementById("chat");
    for (const node of nodes) {
        element.appendChild(node.cloneNode(true));
        if (chatEl) chatEl.scrollTop = chatEl.scrollHeight;
        await new Promise(r => setTimeout(r, 18));
    }
}

/* =====================================================================
   SEND MESSAGE
   ===================================================================== */
function getActiveInput() {
    if (!hasStartedChat) return document.getElementById("question");
    return document.getElementById("questionChat");
}

function switchToChatMode() {
    if (hasStartedChat) return;
    hasStartedChat = true;

    const welcome = document.getElementById("welcomeScreen");
    const chatBody = document.getElementById("chatBody");

    welcome.style.transition = "opacity 0.25s ease, transform 0.25s ease";
    welcome.style.opacity = "0";
    welcome.style.transform = "translateY(-12px)";

    setTimeout(() => {
        welcome.style.display = "none";
        chatBody.style.display = "flex";

        chat = document.getElementById("chat");

        const ta = document.getElementById("questionChat");
        if (ta) {
            ta.addEventListener("input", () => {
                ta.style.height = "auto";
                ta.style.height = ta.scrollHeight + "px";
            });
            ta.addEventListener("keydown", function(e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            ta.focus();
        }
        renderChips();
        showChips();
    }, 240);
}

async function sendMessage() {
    const input = getActiveInput();
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    input.style.height = "auto";

    if (!hasStartedChat) {
        switchToChatMode();
        setTimeout(() => {
            hideChips();
            askQuestion(message);
        }, 260);
    } else {
        hideChips();
        await askQuestion(message);
    }
}

async function askQuestion(message) {
    lastUserMessage = message;

    if (!chat) chat = document.getElementById("chat");

    if (!currentSessionId) {
        currentSessionId = createSession(selectedProject);
    }

    appendMessageToSession(currentSessionId, "user", message);
    createMessage(message, "user", null);

    // Loading bubble
    const loadingRow = document.createElement("div");
    loadingRow.className = "msg-row bot";
    const lb = document.createElement("div");
    lb.className = "msg-wrapper";
    const lbInner = document.createElement("div");
    lbInner.className = "message bot loading-bubble";
    lbInner.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
    lb.appendChild(lbInner);
    loadingRow.appendChild(lb);
    chat.appendChild(loadingRow);
    chat.scrollTop = chat.scrollHeight;

    try {
        const payload = { message };
        if (selectedProject) payload.project = selectedProject;
        if (conversationContext.lastProject) payload.context_project = conversationContext.lastProject;

        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        chat.removeChild(loadingRow);

        const answer = data.answer || "Information non disponible.";
        if (data.detected_project) {
            conversationContext.lastProject = data.detected_project;
        }

        const msgId = Date.now();
        const renderedHtml = renderMarkdown(answer);
        const botBubble = createMessage("", "bot", msgId);
        await streamRender(botBubble, renderedHtml);

        appendMessageToSession(currentSessionId, "bot", answer);
        showChips();

    } catch (e) {
        chat.removeChild(loadingRow);
        createMessage("Erreur serveur.", "bot", null);
        console.error("Erreur:", e);
    }
}

/* =====================================================================
   CHIPS
   ===================================================================== */
function showChips() {
    const zone = document.getElementById("chipsZone");
    if (zone) zone.style.display = "flex";
}
function hideChips() {
    const zone = document.getElementById("chipsZone");
    if (zone) zone.style.display = "none";
}

/* =====================================================================
   PROJECT SELECTOR
   ===================================================================== */
function toggleProjectSelector() {
    const dropdown = document.getElementById("projectDropdown");
    const arrow = document.getElementById("selector-arrow");
    dropdown.classList.toggle("show");
    arrow.style.transform = dropdown.classList.contains("show") ? "rotate(180deg)" : "rotate(0deg)";
    if (dropdown.classList.contains("show")) loadProjects();
}

document.addEventListener("click", function(event) {
    const selector = document.querySelector(".project-selector-container");
    const dropdown = document.getElementById("projectDropdown");
    const arrow = document.getElementById("selector-arrow");
    if (selector && !selector.contains(event.target) && dropdown && dropdown.classList.contains("show")) {
        dropdown.classList.remove("show");
        if (arrow) arrow.style.transform = "rotate(0deg)";
    }
    const overlay = document.getElementById("sessionOverlay");
    if (overlay && event.target === overlay) closeSessionViewer();
});

async function loadProjects() {
    try {
        const response = await fetch("/projects");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        projectsList = data && data.projects && Array.isArray(data.projects) ? data.projects : Array.isArray(data) ? data : [];
        const statusResponse = await fetch("/project_status");
        const statusData = await statusResponse.json();
        displayProjects(projectsList, statusData);
        updateSidebarAlertBadge(statusData);
    } catch (error) {
        console.error("Erreur chargement projets:", error);
        const listDiv = document.getElementById("projectList");
        if (listDiv) listDiv.innerHTML = `<div class="project-item" style="color:red;"><i class="fa-solid fa-exclamation-triangle"></i><span>Erreur de chargement</span></div>`;
    }
}

function updateSidebarAlertBadge(statusData) {
    if (!statusData) return;
    const criticalProjects = Object.entries(statusData).filter(([, v]) => v.risk === "critique").map(([name]) => name);
    const count = criticalProjects.length;

    let badge = document.getElementById("sidebar-alert-badge");
    if (!badge) {
        badge = document.createElement("span");
        badge.id = "sidebar-alert-badge";
        badge.style.cssText = `display:inline-flex;align-items:center;justify-content:center;min-width:18px;height:18px;padding:0 5px;background:#A32D2D;color:#fff;border-radius:99px;font-size:11px;font-weight:700;margin-left:auto;flex-shrink:0;`;
        if (!document.getElementById("badge-anim-style")) {
            const style = document.createElement("style");
            style.id = "badge-anim-style";
            style.textContent = `@keyframes pulse-badge{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.75;transform:scale(1.15)}}.project-item .alert-dot{width:7px;height:7px;background:#A32D2D;border-radius:50%;flex-shrink:0;box-shadow:0 0 5px #A32D2D88;animation:pulse-badge 2s infinite}`;
            document.head.appendChild(style);
        }
        const header = document.querySelector(".project-selector-header");
        if (header) header.appendChild(badge);
    }
    if (count > 0) {
        badge.textContent = count;
        badge.title = `${count} projet${count > 1 ? "s" : ""} critique${count > 1 ? "s" : ""} : ${criticalProjects.join(", ")}`;
        badge.style.display = "inline-flex";
    } else {
        badge.style.display = "none";
    }
}

function displayProjects(projects, statusData) {
    const listDiv = document.getElementById("projectList");
    if (!listDiv) return;
    const searchTerm = document.getElementById("projectSearch") ? document.getElementById("projectSearch").value.toLowerCase() : "";

    let html = `
        <div class="project-item all-projects ${!selectedProject ? "selected" : ""}" onclick="selectProject(null,'Tous les projets',event)">
            <i class="fa-solid fa-globe"></i>
            <span class="project-name">Tous les projets</span>
            <span class="project-status unknown"></span>
        </div>`;

    if (projects && Array.isArray(projects) && projects.length > 0) {
        projects.filter(proj => proj && proj.toLowerCase().includes(searchTerm)).forEach(proj => {
            const status = statusData && statusData[proj] ? statusData[proj].risk : "inconnu";
            const statusClass = status === "faible" ? "healthy" : status === "critique" ? "critical" : "unknown";
            const selectedClass = (selectedProject === proj) ? "selected" : "";
            const isCritical = status === "critique";
            const alertDot = isCritical ? `<span class="alert-dot" title="Projet critique"></span>` : "";
            html += `
                <div class="project-item ${selectedClass}" onclick="selectProject('${proj.replace(/'/g,"\\'")}','${proj.replace(/'/g,"\\'")}',event)" title="${isCritical ? '⚠ Projet critique' : proj}">
                    <i class="fa-solid fa-folder${isCritical ? '-open' : ''}" style="${isCritical ? 'color:#A32D2D' : ''}"></i>
                    <span class="project-name" style="${isCritical ? 'font-weight:600' : ''}">${proj}</span>
                    ${alertDot}
                    <span class="project-status ${statusClass}"></span>
                </div>`;
        });
    } else {
        html += `<div class="project-item"><i class="fa-solid fa-info-circle"></i><span>Aucun projet disponible</span></div>`;
    }
    listDiv.innerHTML = html;
}

function filterProjects() { loadProjects(); }

function selectProject(projectName, displayName, event) {
    if (event) event.stopPropagation();
    selectedProject = projectName;
    const displayElement = document.getElementById("selected-project-display");
    if (displayElement) displayElement.textContent = displayName;
    const dropdown = document.getElementById("projectDropdown");
    const arrow = document.getElementById("selector-arrow");
    if (dropdown) dropdown.classList.remove("show");
    if (arrow) arrow.style.transform = "rotate(0deg)";
    document.querySelectorAll(".project-item").forEach(item => item.classList.remove("selected"));
    if (event && event.currentTarget) event.currentTarget.classList.add("selected");
    else loadProjects();
    addSystemMessage(projectName ? `Projet "${displayName}" sélectionné` : `Mode tous projets activé`);
    renderChips();
    showChips();
}

/* =====================================================================
   NOUVEAU CHAT
   ===================================================================== */
function newChat() {
    currentSessionId = null;
    hasStartedChat = false;

    selectedProject = null;
    const displayElement = document.getElementById("selected-project-display");
    if (displayElement) displayElement.textContent = "Tous les projets";
    document.querySelectorAll(".project-item").forEach(item => item.classList.remove("selected"));
    const allProjectsItem = document.querySelector(".all-projects");
    if (allProjectsItem) allProjectsItem.classList.add("selected");

    const chatEl = document.getElementById("chat");
    if (chatEl) chatEl.innerHTML = "";
    chat = null;

    conversationContext = { lastProject: null };

    const welcome = document.getElementById("welcomeScreen");
    const chatBody = document.getElementById("chatBody");
    if (chatBody) chatBody.style.display = "none";
    if (welcome) {
        welcome.style.display = "";
        welcome.style.opacity = "";
        welcome.style.transform = "";
        welcome.style.transition = "";
    }

    const textarea = document.getElementById("question");
    if (textarea) { textarea.style.height = "auto"; textarea.value = ""; textarea.focus(); }

    renderChips();
    renderRecentsList();
}

/* =====================================================================
   AUTRES FONCTIONS
   ===================================================================== */
function toggleSidebar() {
    document.getElementById("sidebar").classList.toggle("collapsed");
}

const chatSection = document.getElementById("chatSection");
const dashboardSection = document.getElementById("dashboardSection");

function showDashboard() { chatSection.classList.remove("active"); dashboardSection.classList.add("active"); }
function showChat() { dashboardSection.classList.remove("active"); chatSection.classList.add("active"); }

const modal = document.getElementById("uploadModal");
function openUploadModal() { modal.classList.add("active"); }
modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("active"); });

async function uploadProject() {
    const file = document.getElementById("projectFile").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    await fetch("/upload_project", { method: "POST", body: form });
    modal.classList.remove("active");
    loadProjects();
    addSystemMessage("Projet uploadé avec succès");
}

function toggleTheme() {
    document.body.classList.toggle("dark");
    localStorage.setItem("pmo_theme", document.body.classList.contains("dark") ? "dark" : "light");
    const btn = document.getElementById("themeBtn");
    if (btn) btn.innerHTML = document.body.classList.contains("dark")
        ? '<i class="fa-solid fa-sun"></i> <span>Thème</span>'
        : '<i class="fa-solid fa-moon"></i> <span>Thème</span>';
}

/* =====================================================================
   AUTO RESIZE TEXTAREA
   ===================================================================== */
const textarea = document.getElementById("question");
if (textarea) {
    textarea.addEventListener("input", () => {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    });
    textarea.addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

/* =====================================================================
   INIT
   ===================================================================== */
window.addEventListener("load", () => {
    if (localStorage.getItem("pmo_theme") === "dark") {
        document.body.classList.add("dark");
        const btn = document.getElementById("themeBtn");
        if (btn) btn.innerHTML = '<i class="fa-solid fa-sun"></i> <span>Thème</span>';
    }

    const recentsHidden = localStorage.getItem(RECENTS_HIDDEN_KEY) === "1";
    if (recentsHidden) {
        const section = document.getElementById("recentsSection");
        if (section) section.classList.add("collapsed-recents");
    }

    loadProjects();
    renderChips();
    renderRecentsList();
    showChips();

    setInterval(async () => {
        try {
            const res = await fetch("/project_status");
            const statusData = await res.json();
            updateSidebarAlertBadge(statusData);
        } catch (e) { /* silencieux */ }
    }, 60000);
});