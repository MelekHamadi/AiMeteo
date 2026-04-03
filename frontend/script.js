const chat = document.getElementById("chat");
let selectedProject = null;
let projectsList = [];

/* ================= HISTORY MANAGEMENT ================= */
const HISTORY_KEY = "pmo_chat_history";

function getHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "{}"); }
    catch { return {}; }
}

function saveMessage(projectKey, role, content) {
    const history = getHistory();
    const key = projectKey || "__global__";
    if (!history[key]) history[key] = [];
    history[key].push({
        role,
        content,
        timestamp: new Date().toISOString(),
        id: Date.now() + Math.random()
    });
    // Keep max 50 messages per project
    if (history[key].length > 50) history[key] = history[key].slice(-50);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistoryPanel();
}

function clearHistory(projectKey) {
    const history = getHistory();
    const key = projectKey || "__global__";
    delete history[key];
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistoryPanel();
}

function formatTimestamp(iso) {
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return "à l'instant";
    if (diff < 3600000) return `il y a ${Math.floor(diff / 60000)} min`;
    if (diff < 86400000) return `il y a ${Math.floor(diff / 3600000)} h`;
    return d.toLocaleDateString("fr-FR", { day: "2-digit", month: "short" });
}

function renderHistoryPanel() {
    const panel = document.getElementById("historyPanel");
    if (!panel) return;
    const history = getHistory();
    const keys = Object.keys(history);

    if (keys.length === 0) {
        panel.innerHTML = `<div class="history-empty"><i class="fa-regular fa-clock"></i><span>Aucun historique</span></div>`;
        return;
    }

    let html = "";
    keys.reverse().forEach(key => {
        const msgs = history[key];
        const displayName = key === "__global__" ? "Tous les projets" : key;
        const lastMsg = msgs[msgs.length - 1];
        const preview = lastMsg.content.replace(/<[^>]+>/g, "").substring(0, 60) + "…";
        const userMsgs = msgs.filter(m => m.role === "user");
        const count = userMsgs.length;

        html += `
        <div class="history-group">
            <div class="history-group-header">
                <div class="history-group-meta">
                    <span class="history-project-name">${displayName}</span>
                    <span class="history-count">${count} échange${count > 1 ? "s" : ""}</span>
                </div>
                <button class="history-clear-btn" onclick="clearHistory('${key === '__global__' ? '' : key}')" title="Supprimer">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            </div>
            <div class="history-messages">`;

        msgs.slice(-4).reverse().forEach(msg => {
            if (msg.role === "user") {
                html += `
                <div class="history-item" onclick="replayMessage('${escapeAttr(msg.content)}')">
                    <div class="history-item-role user-role"><i class="fa-solid fa-user"></i></div>
                    <div class="history-item-body">
                        <span class="history-item-text">${escapeHtml(msg.content.substring(0, 70))}${msg.content.length > 70 ? "…" : ""}</span>
                        <span class="history-item-time">${formatTimestamp(msg.timestamp)}</span>
                    </div>
                </div>`;
            }
        });

        html += `</div></div>`;
    });

    panel.innerHTML = html;
}

function replayMessage(content) {
    const input = document.getElementById("question");
    input.value = content;
    input.focus();
    showHistory(false);
}

function escapeAttr(str) {
    return str.replace(/'/g, "\\'").replace(/"/g, "&quot;").replace(/\n/g, " ");
}

/* ================= CONTEXTUAL CHIPS ================= */
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
    const container = document.getElementById("chipsContainer");
    if (!container) return;

    const chips = selectedProject
        ? CHIP_SUGGESTIONS.project(selectedProject)
        : CHIP_SUGGESTIONS.default;

    container.innerHTML = chips.map(chip => `
        <button class="chip-btn" onclick="useChip('${escapeAttr(chip.query)}')">
            ${escapeHtml(chip.label)}
        </button>
    `).join("");
}

function useChip(query) {
    const input = document.getElementById("question");
    input.value = query;
    input.focus();
    // Auto-send after small delay
    setTimeout(() => sendMessage(), 100);
}

/* ================= HISTORY PANEL TOGGLE ================= */
let historyVisible = false;

function showHistory(show) {
    historyVisible = show;
    const panel = document.getElementById("historyPanel");
    const overlay = document.getElementById("historyOverlay");
    if (!panel) return;
    if (show) {
        renderHistoryPanel();
        panel.classList.add("open");
        if (overlay) overlay.classList.add("active");
    } else {
        panel.classList.remove("open");
        if (overlay) overlay.classList.remove("active");
    }
}

function toggleHistory() {
    showHistory(!historyVisible);
}

/* ================= MARKDOWN RENDERER ================= */
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
    let inList = false;
    let inOrderedList = false;
    let inTable = false;
    let tableRows = [];

    const flushTable = () => {
        if (tableRows.length === 0) return;
        let tableHtml = '<div class="resp-table-wrap"><table class="resp-table">';
        tableRows.forEach((row, i) => {
            const cells = row.split("|").map(c => c.trim()).filter(Boolean);
            if (i === 0) {
                tableHtml += "<thead><tr>" + cells.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
            } else if (i === 1 && cells.every(c => /^[-:]+$/.test(c))) {
                // skip separator
            } else {
                tableHtml += "<tr>" + cells.map(c => `<td>${inlineMarkdown(c)}</td>`).join("") + "</tr>";
            }
        });
        tableHtml += "</tbody></table></div>";
        html += tableHtml;
        tableRows = [];
        inTable = false;
    };

    const flushList = () => {
        if (inList) { html += "</ul>"; inList = false; }
        if (inOrderedList) { html += "</ol>"; inOrderedList = false; }
    };

    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i];
        const line = raw.trim();

        if (line.startsWith("|")) {
            flushList();
            inTable = true;
            tableRows.push(line);
            continue;
        } else if (inTable) {
            flushTable();
        }

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
            html += `<li>${inlineMarkdown(bullet[1])}</li>`;
            continue;
        }

        const ordered = line.match(/^\d+\.\s+(.+)/);
        if (ordered) {
            if (inList) { html += "</ul>"; inList = false; }
            if (!inOrderedList) { html += "<ol class='resp-list resp-olist'>"; inOrderedList = true; }
            html += `<li>${inlineMarkdown(ordered[1])}</li>`;
            continue;
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

    flushList();
    flushTable();
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

/* ================= MESSAGE UI ================= */
let lastUserMessage = "";
let feedbackState = {}; // msgId -> "up" | "down" | null

function createMessage(text, type, msgId) {
    const row = document.createElement("div");
    row.className = "msg-row " + type;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${type}`;
    avatar.innerHTML = type === "user"
        ? '<i class="fa-solid fa-user"></i>'
        : '<i class="fa-solid fa-robot"></i>';

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

    // Add action bar for bot messages
    if (type === "bot" && msgId) {
        const actionBar = createActionBar(msgId, text, bubble);
        wrapper.appendChild(actionBar);
    }

    row.appendChild(avatar);
    row.appendChild(wrapper);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    return bubble;
}

function createActionBar(msgId, rawText, bubble) {
    const bar = document.createElement("div");
    bar.className = "msg-action-bar";

    // Copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn";
    copyBtn.title = "Copier la réponse";
    copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
    copyBtn.onclick = () => {
        const plain = bubble.innerText || bubble.textContent;
        navigator.clipboard.writeText(plain).then(() => {
            copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
            copyBtn.classList.add("copied");
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
                copyBtn.classList.remove("copied");
            }, 1800);
        });
    };

    // Thumbs up
    const thumbUpBtn = document.createElement("button");
    thumbUpBtn.className = "msg-action-btn";
    thumbUpBtn.title = "Bonne réponse";
    thumbUpBtn.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';
    thumbUpBtn.onclick = () => {
        if (feedbackState[msgId] === "up") {
            feedbackState[msgId] = null;
            thumbUpBtn.classList.remove("active-feedback");
            thumbUpBtn.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';
        } else {
            feedbackState[msgId] = "up";
            thumbUpBtn.classList.add("active-feedback");
            thumbUpBtn.innerHTML = '<i class="fa-solid fa-thumbs-up"></i>';
            thumbDownBtn.classList.remove("active-feedback");
            thumbDownBtn.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';
            showToast("Merci pour votre retour !");
        }
    };

    // Thumbs down
    const thumbDownBtn = document.createElement("button");
    thumbDownBtn.className = "msg-action-btn";
    thumbDownBtn.title = "Réponse incorrecte";
    thumbDownBtn.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';
    thumbDownBtn.onclick = () => {
        if (feedbackState[msgId] === "down") {
            feedbackState[msgId] = null;
            thumbDownBtn.classList.remove("active-feedback");
            thumbDownBtn.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';
        } else {
            feedbackState[msgId] = "down";
            thumbDownBtn.classList.add("active-feedback");
            thumbDownBtn.innerHTML = '<i class="fa-solid fa-thumbs-down"></i>';
            thumbUpBtn.classList.remove("active-feedback");
            thumbUpBtn.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';
            showToast("Retour enregistré — nous améliorerons.");
        }
    };

    // Retry button
    const retryBtn = document.createElement("button");
    retryBtn.className = "msg-action-btn";
    retryBtn.title = "Réessayer";
    retryBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i>';
    retryBtn.onclick = () => {
        if (lastUserMessage) {
            // Remove current bot row and re-ask
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

function showToast(message) {
    let toast = document.getElementById("pmo-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "pmo-toast";
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2500);
}

function addSystemMessage(text) {
    const row = document.createElement("div");
    row.className = "msg-row system";
    const bubble = document.createElement("div");
    bubble.className = "message system";
    bubble.innerHTML = `<i class="fa-solid fa-info-circle"></i> ${text}`;
    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
}

async function streamRender(element, html) {
    element.innerHTML = "";
    const temp = document.createElement("div");
    temp.innerHTML = html;
    const nodes = Array.from(temp.childNodes);
    for (const node of nodes) {
        element.appendChild(node.cloneNode(true));
        chat.scrollTop = chat.scrollHeight;
        await new Promise(r => setTimeout(r, 18));
    }
}

/* ================= SEND MESSAGE ================= */
async function sendMessage() {
    const input = document.getElementById("question");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    input.style.height = "auto";
    hideChips();
    await askQuestion(message);
}

async function askQuestion(message) {
    lastUserMessage = message;

    // Save user message
    saveMessage(selectedProject, "user", message);

    createMessage(message, "user", null);

    // Loading indicator
    const loadingRow = document.createElement("div");
    loadingRow.className = "msg-row bot";
    const loadingAvatar = document.createElement("div");
    loadingAvatar.className = "avatar bot";
    loadingAvatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
    const loadingBubble = document.createElement("div");
    loadingBubble.className = "message bot loading-bubble";
    loadingBubble.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
    loadingRow.appendChild(loadingAvatar);
    loadingRow.appendChild(loadingBubble);
    chat.appendChild(loadingRow);
    chat.scrollTop = chat.scrollHeight;

    try {
        const payload = { message };
        if (selectedProject) payload.project = selectedProject;

        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        chat.removeChild(loadingRow);

        const answer = data.answer || "Information non disponible.";
        const msgId = Date.now();
        const renderedHtml = renderMarkdown(answer);
        const botBubble = createMessage("", "bot", msgId);
        await streamRender(botBubble, renderedHtml);

        // Save bot response
        saveMessage(selectedProject, "bot", answer);

        // Show chips again after response
        showChips();

    } catch (e) {
        chat.removeChild(loadingRow);
        createMessage("Erreur serveur.", "bot", null);
        console.error("Erreur:", e);
    }
}

/* ================= CHIPS SHOW/HIDE ================= */
function showChips() {
    const zone = document.getElementById("chipsZone");
    if (zone) zone.style.display = "flex";
}

function hideChips() {
    const zone = document.getElementById("chipsZone");
    if (zone) zone.style.display = "none";
}

/* ================= PROJECT SELECTOR ================= */
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
    // Close history panel on overlay click
    const overlay = document.getElementById("historyOverlay");
    if (overlay && event.target === overlay) showHistory(false);
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
    } catch (error) {
        console.error("Erreur chargement projets:", error);
        const listDiv = document.getElementById("projectList");
        if (listDiv) listDiv.innerHTML = `<div class="project-item" style="color:red;"><i class="fa-solid fa-exclamation-triangle"></i><span>Erreur de chargement</span></div>`;
    }
}

function displayProjects(projects, statusData) {
    const listDiv = document.getElementById("projectList");
    if (!listDiv) return;
    const searchTerm = document.getElementById("projectSearch") ? document.getElementById("projectSearch").value.toLowerCase() : "";
    let html = `
        <div class="project-item all-projects ${!selectedProject ? "selected" : ""}"
             onclick="selectProject(null, 'Tous les projets', event)">
            <i class="fa-solid fa-globe"></i>
            <span class="project-name">Tous les projets</span>
            <span class="project-status unknown"></span>
        </div>`;
    if (projects && Array.isArray(projects) && projects.length > 0) {
        projects.filter(proj => proj && proj.toLowerCase().includes(searchTerm)).forEach(proj => {
            const status = statusData && statusData[proj] ? statusData[proj].risk : "inconnu";
            const statusClass = status === "faible" ? "healthy" : status === "critique" ? "critical" : "unknown";
            const selectedClass = (selectedProject === proj) ? "selected" : "";
            html += `
                <div class="project-item ${selectedClass}"
                     onclick="selectProject('${proj.replace(/'/g, "\\'")}', '${proj.replace(/'/g, "\\'")}', event)">
                    <i class="fa-solid fa-folder"></i>
                    <span class="project-name">${proj}</span>
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

/* ================= NOUVEAU CHAT ================= */
function newChat() {
    selectedProject = null;
    const displayElement = document.getElementById("selected-project-display");
    if (displayElement) displayElement.textContent = "Tous les projets";
    document.querySelectorAll(".project-item").forEach(item => item.classList.remove("selected"));
    const allProjectsItem = document.querySelector(".all-projects");
    if (allProjectsItem) allProjectsItem.classList.add("selected");
    chat.innerHTML = "";
    const textarea = document.getElementById("question");
    if (textarea) { textarea.style.height = "auto"; textarea.value = ""; }
    renderChips();
    showChips();
    setTimeout(() => {
        addSystemMessage("Nouvelle conversation — tout réinitialisé");
        setTimeout(() => addSystemMessage("Mode actif : Tous les projets"), 600);
    }, 100);
}

/* ================= AUTRES FONCTIONS ================= */
function toggleSidebar() {
    document.getElementById("sidebar").classList.toggle("collapsed");
}

const chatSection = document.getElementById("chatSection");
const dashboardSection = document.getElementById("dashboardSection");

function showDashboard() {
    chatSection.classList.remove("active");
    dashboardSection.classList.add("active");
}

function showChat() {
    dashboardSection.classList.remove("active");
    chatSection.classList.add("active");
}

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
    const btn = document.getElementById("themeBtn");
    if (btn) btn.innerHTML = document.body.classList.contains("dark")
        ? '<i class="fa-solid fa-sun"></i>'
        : '<i class="fa-solid fa-moon"></i>';
}

/* ================= AUTO RESIZE ================= */
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

/* ================= INIT ================= */
window.addEventListener("load", () => {
    loadProjects();
    renderChips();
    renderHistoryPanel();
    showChips();
    setTimeout(() => {
        addSystemMessage("Bienvenue sur PMO AI Copilot");
        setTimeout(() => addSystemMessage("Sélectionnez un projet dans la sidebar ou posez une question globale"), 800);
    }, 400);
});