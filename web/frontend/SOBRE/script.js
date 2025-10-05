// --- LÓGICA DE BARRA LATERAL ---

const sidebarWidth = "350px"; 

function openNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    const openBtn = document.querySelector('.openbtn');
    
    if (sidebar && mainContent && openBtn) {
        sidebar.style.width = sidebarWidth;
        mainContent.style.marginLeft = sidebarWidth;
        sidebar.classList.add('active'); 
        
    // HIDE BUTTON WHEN OPENING THE MENU
        openBtn.style.opacity = '0';
        openBtn.style.pointerEvents = 'none';
    }
}

function closeNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    const openBtn = document.querySelector('.openbtn');

    if (sidebar && mainContent && openBtn) {
        sidebar.style.width = "0";
        mainContent.style.marginLeft = "0";
        sidebar.classList.remove('active'); 
        
    // SHOW BUTTON WHEN CLOSING THE MENU
        openBtn.style.opacity = '1';
        openBtn.style.pointerEvents = 'auto';
    }
}

window.openNav = openNav;
window.closeNav = closeNav;


// --- SOBRE.HTML SPECIFIC LOGIC (Placeholder) ---

document.addEventListener('DOMContentLoaded', () => {
    // No hay funcionalidad dinámica o eventos complejos en esta página.
    console.log("'About Us' page loaded successfully.");
});