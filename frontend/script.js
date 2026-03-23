const chat = document.getElementById("chat");
let selectedProject = null;
let projectsList = [];

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
    bubble.textContent = text;

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

async function typeText(element, text) {
    element.textContent = "";
    for (let i = 0; i < text.length; i++) {
        element.textContent += text[i];
        await new Promise(r => setTimeout(r, 12));
        chat.scrollTop = chat.scrollHeight;
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

    const botBubble = createMessage("PMO AI réfléchit...", "bot");

    try {
        const payload = { message };
        if (selectedProject) {
            payload.project = selectedProject;
        }
        
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        await typeText(botBubble, data.answer || "Information non disponible.");
    } catch (e) {
        botBubble.textContent = "Erreur serveur.";
        console.error("Erreur:", e);
    }
}

/* ================= PROJECT SELECTOR ================= */
function toggleProjectSelector() {
    const dropdown = document.getElementById("projectDropdown");
    const arrow = document.getElementById("selector-arrow");
    dropdown.classList.toggle("show");
    arrow.style.transform = dropdown.classList.contains("show") ? "rotate(180deg)" : "rotate(0deg)";
    
    if (dropdown.classList.contains("show")) {
        loadProjects();
    }
}

// Fermer le dropdown si on clique ailleurs
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
    console.log("Chargement des projets...");
    
    try {
        const response = await fetch('/projects');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        if (data && data.projects && Array.isArray(data.projects)) {
            projectsList = data.projects;
        } else if (Array.isArray(data)) {
            projectsList = data;
        } else {
            console.warn("Format de données inattendu:", data);
            projectsList = [];
        }
        
        console.log("Projets extraits:", projectsList);
        
        const statusResponse = await fetch('/project_status');
        const statusData = await statusResponse.json();
        console.log("Statuts reçus:", statusData);
        
        displayProjects(projectsList, statusData);
        
    } catch (error) {
        console.error("Erreur chargement projets:", error);
        const listDiv = document.getElementById('projectList');
        if (listDiv) {
            listDiv.innerHTML = `
                <div class="project-item" style="color: red;">
                    <i class="fa-solid fa-exclamation-triangle"></i>
                    <span>Erreur de chargement</span>
                </div>
            `;
        }
    }
}

function displayProjects(projects, statusData) {
    const listDiv = document.getElementById('projectList');
    if (!listDiv) return;
    
    const searchTerm = document.getElementById('projectSearch') ? 
        document.getElementById('projectSearch').value.toLowerCase() : '';
    
    let html = `
        <div class="project-item all-projects ${!selectedProject ? 'selected' : ''}" 
             onclick="selectProject(null, 'Tous les projets', event)">
            <i class="fa-solid fa-globe"></i>
            <span class="project-name">Tous les projets</span>
            <span class="project-status unknown"></span>
        </div>
    `;
    
    if (projects && Array.isArray(projects) && projects.length > 0) {
        const filteredProjects = projects.filter(proj => 
            proj && proj.toLowerCase().includes(searchTerm)
        );
        
        filteredProjects.forEach(proj => {
            const status = statusData && statusData[proj] ? statusData[proj].risk : 'inconnu';
            const statusClass = status === 'faible' ? 'healthy' : 
                               status === 'critique' ? 'critical' : 'unknown';
            const selectedClass = (selectedProject === proj) ? 'selected' : '';
            
            html += `
                <div class="project-item ${selectedClass}" 
                     onclick="selectProject('${proj.replace(/'/g, "\\'")}', '${proj.replace(/'/g, "\\'")}', event)">
                    <i class="fa-solid fa-folder"></i>
                    <span class="project-name">${proj}</span>
                    <span class="project-status ${statusClass}"></span>
                </div>
            `;
        });
    } else {
        html += `
            <div class="project-item">
                <i class="fa-solid fa-info-circle"></i>
                <span>Aucun projet disponible</span>
            </div>
        `;
    }
    
    listDiv.innerHTML = html;
}

function filterProjects() {
    loadProjects();
}

/* ================= FONCTION SELECTION CORRIGÉE ================= */
function selectProject(projectName, displayName, event) {
    if (event) {
        event.stopPropagation();
    }
    
    console.log("Sélection:", projectName, displayName);
    
    // Mettre à jour la variable globale
    selectedProject = projectName;
    
    // Mettre à jour l'affichage du sélecteur
    const displayElement = document.getElementById('selected-project-display');
    if (displayElement) {
        displayElement.textContent = displayName;
    }
    
    // Fermer le dropdown
    const dropdown = document.getElementById('projectDropdown');
    const arrow = document.getElementById('selector-arrow');
    if (dropdown) dropdown.classList.remove('show');
    if (arrow) arrow.style.transform = "rotate(0deg)";
    
    // Mettre à jour visuellement la classe "selected" sur tous les éléments
    document.querySelectorAll('.project-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Ajouter la classe "selected" à l'élément cliqué
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('selected');
    } else {
        // Fallback: recharger la liste pour que la classe selected s'applique
        loadProjects();
    }
    
    // Message de confirmation
    if (projectName) {
        addSystemMessage(`✅ Projet "${displayName}" sélectionné`);
    } else {
        addSystemMessage(`🌐 Mode tous projets activé`);
    }
}

/* ================= NOUVEAU CHAT ================= */
function newChat() {
    // Réinitialiser la sélection du projet
    selectedProject = null;
    
    // Mettre à jour l'affichage du sélecteur
    const displayElement = document.getElementById('selected-project-display');
    if (displayElement) {
        displayElement.textContent = "Tous les projets";
    }
    
    // Mettre à jour les classes "selected" dans la liste
    document.querySelectorAll('.project-item').forEach(item => {
        item.classList.remove('selected');
    });
    const allProjectsItem = document.querySelector('.all-projects');
    if (allProjectsItem) {
        allProjectsItem.classList.add('selected');
    }
    
    // Vider le chat
    chat.innerHTML = "";
    
    // Réinitialiser le textarea
    const textarea = document.getElementById("question");
    if (textarea) {
        textarea.style.height = "auto";
        textarea.value = "";
    }
    
    // Messages de bienvenue
    setTimeout(() => {
        addSystemMessage("🔄 Nouvelle conversation - Tout a été réinitialisé");
        addSystemMessage("🌐 Mode actif : Tous les projets");
        
        setTimeout(() => {
            addSystemMessage("💡 Exemples : 'Quel est le budget du projet Data Fraud Detection ?' ou 'Quels projets sont en phase MEP ?'");
        }, 1000);
    }, 100);
    
    console.log("Nouveau chat - tout réinitialisé");
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

function openUploadModal() {
    modal.classList.add("active");
}

modal.addEventListener("click", (e) => {
    if (e.target === modal) {
        modal.classList.remove("active");
    }
});

async function uploadProject() {
    const file = document.getElementById("projectFile").files[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    await fetch("/upload_project", { method: "POST", body: form });
    modal.classList.remove("active");
    loadProjects();
    addSystemMessage("✅ Projet uploadé avec succès");
}

// Auto resize textarea
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

// Initialisation
window.addEventListener("load", () => {
    console.log("Page chargée, initialisation...");
    loadProjects();
    
    setTimeout(() => {
        addSystemMessage("👋 Bienvenue sur PMO AI Copilot");
        setTimeout(() => {
            addSystemMessage("🌐 Mode actif : Tous les projets - Sélectionnez un projet dans la sidebar si besoin");
        }, 1000);
    }, 500);
});