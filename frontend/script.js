const chat = document.getElementById("chat");

/* ================= MESSAGE UI ================= */

function createMessage(text,type){

    const row=document.createElement("div");
    row.className="msg-row";

    const avatar=document.createElement("div");
    avatar.className=`avatar ${type}`;
    avatar.innerHTML = type==="user"
        ? '<i class="fa-solid fa-user"></i>'
        : '<i class="fa-solid fa-robot"></i>';

    const bubble=document.createElement("div");
    bubble.className=`message ${type}`;
    bubble.textContent=text;

    row.appendChild(avatar);
    row.appendChild(bubble);

    chat.appendChild(row);
    chat.scrollTop=chat.scrollHeight;

    return bubble;
}

/* ================= STREAMING EFFECT ================= */

async function typeText(element,text){

    element.textContent="";
    for(let i=0;i<text.length;i++){
        element.textContent+=text[i];
        await new Promise(r=>setTimeout(r,12)); // vitesse typing
        chat.scrollTop=chat.scrollHeight;
    }
}

/* ================= SEND MESSAGE ================= */

async function sendMessage(){

    const input=document.getElementById("question");
    const message=input.value.trim();
    if(!message) return;

    createMessage(message,"user");
    input.value="";

    // typing placeholder
    const botBubble=createMessage("PMO AI réfléchit...","bot");

    try{

        const res=await fetch("/chat",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({message})
        });

        const data=await res.json();

        await typeText(botBubble,data.answer || "Information non disponible.");

    }catch(e){
        botBubble.textContent="Erreur serveur.";
    }
}

/* ================= SIDEBAR ================= */

function toggleSidebar(){
    document.getElementById("sidebar").classList.toggle("collapsed");
}

/* ================= NAVIGATION ================= */

function showDashboard(){
    chatSection.classList.remove("active");
    dashboardSection.classList.add("active");
}

function showChat(){
    dashboardSection.classList.remove("active");
    chatSection.classList.add("active");
}

function newChat(){
    chat.innerHTML="";
}

/* ================= MODAL ================= */

const modal=document.getElementById("uploadModal");

function openUploadModal(){
    modal.classList.add("active");
}

modal.addEventListener("click",(e)=>{
    if(e.target===modal){
        modal.classList.remove("active");
    }
});

async function uploadProject(){

    const file=document.getElementById("projectFile").files[0];
    if(!file) return;

    const form=new FormData();
    form.append("file",file);

    await fetch("/upload_project",{method:"POST",body:form});
    modal.classList.remove("active");
}
function setActive(btn){
document.querySelectorAll(".sidebar-btn")
.forEach(b=>b.classList.remove("active"));
btn.classList.add("active");
}

/* ================= THEME TOGGLE ================= */

function toggleTheme(){

    const body = document.body;
    const icon = document.getElementById("themeIcon");

    body.classList.toggle("dark");

    if(body.classList.contains("dark")){
        icon.classList.remove("fa-moon");
        icon.classList.add("fa-sun");
        localStorage.setItem("theme","dark");
    }else{
        icon.classList.remove("fa-sun");
        icon.classList.add("fa-moon");
        localStorage.setItem("theme","light");
    }
}


/* LOAD SAVED THEME */

window.addEventListener("load", () => {

    const savedTheme = localStorage.getItem("theme");
    const icon = document.getElementById("themeIcon");

    if(savedTheme === "dark"){
        document.body.classList.add("dark");
        icon.classList.remove("fa-moon");
        icon.classList.add("fa-sun");
    }

});
const textarea = document.getElementById("question");

textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
});
textarea.addEventListener("keydown", function(e){

    if(e.key === "Enter" && !e.shiftKey){
        e.preventDefault();
        sendMessage();
    }

});
function showTyping(){

    const chat = document.getElementById("chat");

    const typing = document.createElement("div");
    typing.className = "msg-row typing";
    typing.id = "typingIndicator";

    typing.innerHTML = `
        <div class="avatar bot">AI</div>
        <div class="message bot">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
    `;

    chat.appendChild(typing);
    chat.scrollTop = chat.scrollHeight;
}
