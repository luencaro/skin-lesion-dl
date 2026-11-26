// Callbacks en el navegador (sin ida y vuelta al servidor).
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  ui: {
    // Abre o cierra el menú con el botón; lo cierra al cambiar de página.
    toggleNav: function (nClicks, pathname, isOpen) {
      const trig = window.dash_clientside.callback_context.triggered.map(t => t.prop_id)[0] || "";
      if (trig.startsWith("url.")) { return false; }
      return !isOpen;
    }
  }
});

// Desplaza al inicio al navegar entre páginas (útil en móvil).
document.addEventListener("click", function (ev) {
  const a = ev.target.closest && ev.target.closest("a.nav-link");
  if (a) { window.scrollTo({ top: 0, behavior: "smooth" }); }
});
