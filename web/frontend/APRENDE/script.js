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
        
        // OCULTAR BOTÓN AL ABRIR EL MENÚ
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
        
        // MOSTRAR BOTÓN AL CERRAR EL MENÚ
        openBtn.style.opacity = '1';
        openBtn.style.pointerEvents = 'auto';
    }
}

window.openNav = openNav;
window.closeNav = closeNav;


// --- LÓGICA ESPECÍFICA DE APRENDE.HTML (Placeholder) ---

document.addEventListener('DOMContentLoaded', ()  => {
    // No hay funcionalidad dinámica o eventos complejos en esta página.
    console.log("Página 'Aprende Más' cargada correctamente.");
});

window.addEventListener("scroll", function() {
  const header = document.querySelector(".page-header");
  header.classList.toggle("shrink", window.scrollY > 50);
});
