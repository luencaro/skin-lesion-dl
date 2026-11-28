// Callbacks de cliente de Dash (sin ida y vuelta al servidor). Es el único JavaScript propio.
(function () {
  function trig() {
    const t = window.dash_clientside.callback_context.triggered || [];
    return t.length ? t[0].prop_id : "";
  }
  function trigId() {
    const p = trig();
    const raw = p.slice(0, p.lastIndexOf("."));
    try { return JSON.parse(raw); } catch (e) { return raw; }
  }
  function icon(name) {
    return { namespace: "dash_html_components", type: "Span",
             props: { className: "ic ic-" + name, style: { width: 20, height: 20 }, "aria-hidden": "true" } };
  }
  const NAV = ["/", "/demo", "/modelos", "/datos"];

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    ui: {
      // Al cargar: si no hay preferencia guardada, toma la del sistema; el botón la alterna.
      toggleTheme: function (n, path, current) {
        const sys = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        if (trig().indexOf("theme-toggle") !== 0) { return current ? window.dash_clientside.no_update : sys; }
        return (current || sys) === "dark" ? "light" : "dark";
      },
      applyTheme: function (t) {
        const mode = t || (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        return ["cr " + mode, icon(mode === "dark" ? "sun" : "moon")];
      },
      navCurrent: function (path) {
        const p = (path || "/").replace(/\/+$/, "") || "/";
        const alias = p === "/eda" ? "/datos" : p;
        return NAV.map(function (h) { return h === alias ? "cur" : ""; });
      },
      toggleMenu: function (n, path, cls) {
        const open = trig().indexOf("menu-toggle") === 0 && (cls || "").indexOf("open") < 0;
        return [open ? "site-header open" : "site-header", open ? "true" : "false"];
      }
    },
    demo: {
      // Ejemplo o carga -> fuente de la demo, miniatura activa y vista previa del estado "analizando".
      pickSource: function (clicks, contents, samples, ids) {
        const id = trigId();
        const nc = window.dash_clientside.no_update;
        if (id === "demo-upload") {
          if (!contents) { return [nc, nc, nc]; }
          return [{ kind: "upload", t: Date.now() }, ids.map(function () { return "thumb"; }), contents];
        }
        if (id && id.type === "sample") {
          const s = (samples || {})[id.id] || {};
          return [{ kind: "sample", id: id.id, t: Date.now() },
                  ids.map(function (x) { return x.id === id.id ? "thumb on" : "thumb"; }), s.src || ""];
        }
        return [nc, nc, nc];
      },
      pickModel: function (clicks, ids, names) {
        const id = trigId();
        const nc = window.dash_clientside.no_update;
        if (!id || id.type !== "model-pill") { return [nc, nc, nc, nc]; }
        const step = id.id === "cci" ? "Corrigiendo color con CCI" : "Preparando la imagen para " + ((names || {})[id.id] || id.id);
        return [id.id, ids.map(function (x) { return x.id === id.id ? "pill on" : "pill"; }), step,
                ids.map(function (x) { return x.id === id.id ? "true" : "false"; })];
      },
      // Opacidad del mapa de calor: responde sin latencia.
      opacity: function (v) {
        const val = (v === undefined || v === null || v === "") ? 65 : Number(v);
        return [{ opacity: val / 100 }, val + " %", val];
      }
    },
    models: {
      sortPills: function (clicks, ids) {
        const id = trigId();
        return ids.map(function (x) { return x.id === (id && id.id) ? "pill on" : "pill"; });
      }
    }
  });
})();
