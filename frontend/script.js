const chat = document.getElementById("chat");
let selectedProject = null;
let projectsList = [];

/* ================= MARKDOWN RENDERER ================= */
function renderMarkdown(text) {
    if (!text) return "";

    // Nettoyer les guillemets parasites et artefacts LLM
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
    let inSection = false;
    let currentSectionClass = "";

    const flushTable = () => {
        if (tableRows.length === 0) return;
        let tableHtml = '<div class="resp-table-wrap"><table class="resp-table">';
        tableRows.forEach((row, i) => {
            const cells = row.split("|").map(c => c.trim()).filter(Boolean);
            if (i === 0) {
                tableHtml += "<thead><tr>" + cells.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
            } else if (i === 1 && cells.every(c => /^[-:]+$/.test(c))) {
                // ligne séparatrice — skip
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

        // Tables
        if (line.startsWith("|")) {
            flushList();
            inTable = true;
            tableRows.push(line);
            continue;
        } else if (inTable) {
            flushTable();
        }

        // Lignes vides
        if (!line) {
            flushList();
            html += "";
            continue;
        }

        // Détection des sections (#### ou ###)
        const h4 = line.match(/^####\s+(.+)/);
        const h3 = line.match(/^###\s+(.+)/);
        const h2 = line.match(/^##\s+(.+)/);
        const h1 = line.match(/^#\s+(.+)/);

        if (h1) {
            flushList();
            html += `<h1 class="resp-h1">${inlineMarkdown(h1[1])}</h1>`;
            continue;
        }
        if (h2) {
            flushList();
            html += `<h2 class="resp-h2">${inlineMarkdown(h2[1])}</h2>`;
            continue;
        }
        if (h3) {
            flushList();
            const sectionClass = detectSectionClass(h3[1]);
            html += `<h3 class="resp-h3 ${sectionClass}">${inlineMarkdown(h3[1])}</h3>`;
            continue;
        }
        if (h4) {
            flushList();
            const sectionClass = detectSectionClass(h4[1]);
            html += `<h4 class="resp-h4 ${sectionClass}">${inlineMarkdown(h4[1])}</h4>`;
            continue;
        }

        // Listes à puces
        const bullet = line.match(/^[-*]\s+(.+)/);
        if (bullet) {
            if (inOrderedList) { html += "</ol>"; inOrderedList = false; }
            if (!inList) { html += "<ul class='resp-list'>"; inList = true; }
            html += `<li>${inlineMarkdown(bullet[1])}</li>`;
            continue;
        }

        // Listes numérotées
        const ordered = line.match(/^\d+\.\s+(.+)/);
        if (ordered) {
            if (inList) { html += "</ul>"; inList = false; }
            if (!inOrderedList) { html += "<ol class='resp-list resp-olist'>"; inOrderedList = true; }
            html += `<li>${inlineMarkdown(ordered[1])}</li>`;
            continue;
        }

        // Lignes "Action | Impact | Effort"
        const actionLine = line.match(/Action\s*:\s*(.+?)\s*\|\s*Impact\s*:\s*(.+?)\s*\|\s*Effort\s*:\s*(.+)/i);
        if (actionLine) {
            flushList();
            const impact = actionLine[2].trim().toLowerCase();
            const effort = actionLine[3].trim().toLowerCase();
            const impactClass = impact.includes("fort") ? "tag-high" : impact.includes("moyen") ? "tag-med" : "tag-low";
            const effortClass = effort.includes("fort") ? "tag-high" : effort.includes("moyen") ? "tag-med" : "tag-low";
            html += `
            <div class="action-card">
                <div class="action-title">${inlineMarkdown(actionLine[1])}</div>
                <div class="action-tags">
                    <span class="action-tag ${impactClass}">Impact : ${actionLine[2].trim()}</span>
                    <span class="action-tag ${effortClass}">Effort : ${actionLine[3].trim()}</span>
                </div>
            </div>`;
            continue;
        }

        // Score de santé "Score de santé du projet X : 82/100"
        const scoreMatch = line.match(/Score de santé.+?:\s*(\d+(?:\.\d+)?)\s*\/\s*100/i);
        if (scoreMatch) {
            flushList();
            const score = parseFloat(scoreMatch[1]);
            const color = score >= 80 ? "#3B6D11" : score >= 60 ? "#639922" : score >= 40 ? "#854F0B" : "#A32D2D";
            const pct = Math.min(score, 100);
            html += `
            <div class="health-bar-wrap">
                <div class="health-bar-label">${inlineMarkdown(line)}</div>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" style="width:${pct}%;background:${color}"></div>
                </div>
                <div class="health-bar-score" style="color:${color}">${score}/100</div>
            </div>`;
            continue;
        }

        // Lignes "Semaine | Niveau | ..." (tableau semaines)
        // (déjà géré par le bloc table ci-dessus)

        // Texte de niveau diagnostic (lignes avec ":")
        const kvLine = line.match(/^\*\*(.+?)\*\*\s*[:\-]\s*(.+)/);
        if (kvLine) {
            flushList();
            html += `<div class="resp-kv"><span class="resp-kv-key">${kvLine[1]}</span><span class="resp-kv-val">${inlineMarkdown(kvLine[2])}</span></div>`;
            continue;
        }

        // Paragraphe normal
        flushList();
        if (line.length > 0) {
            html += `<p class="resp-p">${inlineMarkdown(line)}</p>`;
        }
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
        // Badges statut
        .replace(/\b(À redresser)\b/g, '<span class="status-badge badge-critical">$1</span>')
        .replace(/\b(À surveiller)\b/g, '<span class="status-badge badge-warning">$1</span>')
        .replace(/\b(En contrôle)\b/g, '<span class="status-badge badge-ok">$1</span>')
        // Badges tendance
        .replace(/\b(Amélioration)\b/g, '<span class="trend-badge trend-up">↑ $1</span>')
        .replace(/\b(Détérioration)\b/g, '<span class="trend-badge trend-down">↓ $1</span>')
        .replace(/\b(Stand by)\b/g, '<span class="trend-badge trend-neutral">— $1</span>')
        // Niveaux risque
        .replace(/\b(CRITIQUE|critique)\b/g, '<span class="lvl-badge lvl-critical">$1</span>')
        .replace(/\b(ÉLEVÉ|élevé|ELEVE|eleve)\b/g, '<span class="lvl-badge lvl-high">$1</span>')
        .replace(/\b(MODÉRÉ|modéré|MODERE|modere)\b/g, '<span class="lvl-badge lvl-med">$1</span>')
        .replace(/\b(FAIBLE|faible)\b/g, '<span class="lvl-badge lvl-low">$1</span>');
}

/* ================= MESSAGE UI ================= */
function createMessage(text, type) {
    const row = document.createElement("div");
    row.className = "msg-row " + type;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${type}`;
    avatar.innerHTML = type === "user"
        ? '<i class="fa-solid fa-user"></i>'
        : '<i class="fa-solid fa-robot"></i>';

    const bubble = document.createElement("div");
    bubble.className = `message ${type}`;

    if (type === "bot") {
        bubble.innerHTML = text;
    } else {
        bubble.textContent = text;
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    return bubble;
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

    createMessage(message, "user");
    input.value = "";
    input.style.height = "auto";

    // Indicateur de chargement
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

        const renderedHtml = renderMarkdown(data.answer || "Information non disponible.");
        const botBubble = createMessage("", "bot");
        await streamRender(botBubble, renderedHtml);

    } catch (e) {
        chat.removeChild(loadingRow);
        createMessage("Erreur serveur.", "bot");
        console.error("Erreur:", e);
    }
}

/* ================= PROJECT SELECTOR ================= */
function toggleProjectSelector() {
    const dropdown = document.getElementById("projectDropdown");
    const arrow = document.getElementById("selector-arrow");
    dropdown.classList.toggle("show");
    arrow.style.transform = dropdown.classList.contains("show") ? "rotate(180deg)" : "rotate(0deg)";
    if (dropdown.classList.contains("show")) loadProjects();
}

document.addEventListener('click', function(event) {
    const selector = document.querySelector('.project-selector-container');
    const dropdown = document.getElementById("projectDropdown");
    const arrow = document.getElementById("selector-arrow");
    if (selector && !selector.contains(event.target) && dropdown && dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        if (arrow) arrow.style.transform = "rotate(0deg)";
    }
});

async function loadProjects() {
    try {
        const response = await fetch('/projects');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        projectsList = data && data.projects && Array.isArray(data.projects) ? data.projects : Array.isArray(data) ? data : [];
        const statusResponse = await fetch('/project_status');
        const statusData = await statusResponse.json();
        displayProjects(projectsList, statusData);
    } catch (error) {
        console.error("Erreur chargement projets:", error);
        const listDiv = document.getElementById('projectList');
        if (listDiv) listDiv.innerHTML = `<div class="project-item" style="color:red;"><i class="fa-solid fa-exclamation-triangle"></i><span>Erreur de chargement</span></div>`;
    }
}

function displayProjects(projects, statusData) {
    const listDiv = document.getElementById('projectList');
    if (!listDiv) return;
    const searchTerm = document.getElementById('projectSearch') ? document.getElementById('projectSearch').value.toLowerCase() : '';
    let html = `
        <div class="project-item all-projects ${!selectedProject ? 'selected' : ''}"
             onclick="selectProject(null, 'Tous les projets', event)">
            <i class="fa-solid fa-globe"></i>
            <span class="project-name">Tous les projets</span>
            <span class="project-status unknown"></span>
        </div>`;
    if (projects && Array.isArray(projects) && projects.length > 0) {
        projects.filter(proj => proj && proj.toLowerCase().includes(searchTerm)).forEach(proj => {
            const status = statusData && statusData[proj] ? statusData[proj].risk : 'inconnu';
            const statusClass = status === 'faible' ? 'healthy' : status === 'critique' ? 'critical' : 'unknown';
            const selectedClass = (selectedProject === proj) ? 'selected' : '';
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
    const displayElement = document.getElementById('selected-project-display');
    if (displayElement) displayElement.textContent = displayName;
    const dropdown = document.getElementById('projectDropdown');
    const arrow = document.getElementById('selector-arrow');
    if (dropdown) dropdown.classList.remove('show');
    if (arrow) arrow.style.transform = "rotate(0deg)";
    document.querySelectorAll('.project-item').forEach(item => item.classList.remove('selected'));
    if (event && event.currentTarget) event.currentTarget.classList.add('selected');
    else loadProjects();
    addSystemMessage(projectName ? `Projet "${displayName}" sélectionné` : `Mode tous projets activé`);
}

/* ================= NOUVEAU CHAT ================= */
function newChat() {
    selectedProject = null;
    const displayElement = document.getElementById('selected-project-display');
    if (displayElement) displayElement.textContent = "Tous les projets";
    document.querySelectorAll('.project-item').forEach(item => item.classList.remove('selected'));
    const allProjectsItem = document.querySelector('.all-projects');
    if (allProjectsItem) allProjectsItem.classList.add('selected');
    chat.innerHTML = "";
    const textarea = document.getElementById("question");
    if (textarea) { textarea.style.height = "auto"; textarea.value = ""; }
    setTimeout(() => {
        addSystemMessage("Nouvelle conversation — tout réinitialisé");
        setTimeout(() => {
            addSystemMessage("Mode actif : Tous les projets");
        }, 600);
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

/* ================= THEME TOGGLE ================= */
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
    setTimeout(() => {
        addSystemMessage("Bienvenue sur PMO AI Copilot");
        setTimeout(() => {
            addSystemMessage("Sélectionnez un projet dans la sidebar ou posez une question globale");
        }, 800);
    }, 400);
});