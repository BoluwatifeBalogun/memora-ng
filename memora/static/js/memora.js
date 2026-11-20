/* Memora NG interface behaviour. No dependencies. */
(() => {
  "use strict";
  const doc = document.documentElement;
  doc.classList.add("js");
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = matchMedia("(hover: hover) and (pointer: fine)").matches;

  /* ---- reveal on scroll, number tickers, rings ---- */
  const tick = (el) => {
    const end = parseFloat(el.dataset.count);
    if (!isFinite(end) || calm) return;
    const t0 = performance.now(), dur = 1100;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 4);
      el.textContent = Math.round(end * e).toLocaleString();
      if (k < 1) requestAnimationFrame(step);
    };
    el.textContent = "0";
    requestAnimationFrame(step);
  };
  $$(".ring").forEach((r) => r.setAttribute("data-armed", ""));
  const arrive = (el) => {
    el.classList.add("is-in");
    $$("[data-count]", el).forEach(tick);
    if (el.matches("[data-count]")) tick(el);
    $$(".ring[data-armed]", el).concat(el.matches(".ring[data-armed]") ? [el] : []).forEach((r) => requestAnimationFrame(() => r.removeAttribute("data-armed")));
  };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { arrive(e.target); io.unobserve(e.target); } }), { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    $$(".reveal").forEach((el) => io.observe(el));
    $$(".ring[data-armed]").filter((r) => !r.closest(".reveal")).forEach((r) => io.observe(r));
  } else {
    $$(".reveal").forEach(arrive);
    $$(".ring").forEach((r) => r.removeAttribute("data-armed"));
  }

  /* ---- pointer spotlight on cards ---- */
  if (fine) {
    document.addEventListener("pointermove", (e) => {
      const card = e.target.closest && e.target.closest(".spot");
      if (!card) return;
      const b = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - b.left}px`);
      card.style.setProperty("--my", `${e.clientY - b.top}px`);
    }, { passive: true });
  }

  /* ---- landing: floating nav, hero parallax, magnetic buttons ---- */
  const nav = $("[data-nav]");
  if (nav) {
    const onScroll = () => nav.classList.toggle("is-stuck", scrollY > 24);
    addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }
  const tilt = $("[data-tilt]");
  if (tilt && fine && !calm) {
    addEventListener("pointermove", (e) => {
      const x = (e.clientX / innerWidth - 0.5) * -18, y = (e.clientY / innerHeight - 0.5) * -14;
      tilt.style.setProperty("--tx", x.toFixed(2));
      tilt.style.setProperty("--ty", y.toFixed(2));
    }, { passive: true });
  }
  if (fine && !calm) {
    $$("[data-magnet]").forEach((b) => {
      b.addEventListener("pointermove", (e) => {
        const r = b.getBoundingClientRect();
        b.style.transform = `translate(${((e.clientX - r.left) / r.width - 0.5) * 12}px, ${((e.clientY - r.top) / r.height - 0.5) * 10}px)`;
      });
      b.addEventListener("pointerleave", () => { b.style.transform = ""; });
    });
  }

  /* ---- dock magnification ---- */
  const dock = $("[data-dock]");
  if (dock && fine && !calm) {
    const items = $$(".dock-item", dock);
    dock.addEventListener("pointermove", (e) => {
      items.forEach((it) => {
        const r = it.getBoundingClientRect();
        const d = Math.abs(e.clientX - (r.left + r.width / 2));
        it.style.setProperty("--s", (1 + Math.max(0, 1 - d / 130) * 0.22).toFixed(3));
      });
    }, { passive: true });
    dock.addEventListener("pointerleave", () => items.forEach((it) => it.style.setProperty("--s", 1)));
  }

  /* ---- account menu ---- */
  const menu = $("[data-menu]");
  if (menu) {
    const btn = $("button", menu), pop = $(".menu-pop", menu);
    const set = (open) => { pop.hidden = !open; btn.setAttribute("aria-expanded", String(open)); };
    btn.addEventListener("click", (e) => { e.stopPropagation(); set(pop.hidden); });
    document.addEventListener("click", (e) => { if (!menu.contains(e.target)) set(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !pop.hidden) { set(false); btn.focus(); } });
  }
  document.addEventListener("click", (e) => $$("details.rowmenu[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; }));

  /* ---- toasts ---- */
  const dismiss = (t) => { t.classList.add("is-out"); setTimeout(() => t.remove(), 300); };
  $$(".toast").forEach((t) => {
    $(".toast-x", t).addEventListener("click", () => dismiss(t));
    if (t.hasAttribute("data-autoclose")) setTimeout(() => dismiss(t), 6000);
  });
  const toast = (msg) => {
    let box = $(".toasts");
    if (!box) { box = document.createElement("div"); box.className = "toasts"; box.setAttribute("role", "alert"); document.body.appendChild(box); }
    const t = document.createElement("div");
    t.className = "toast toast-error";
    const p = document.createElement("p"); p.textContent = msg;
    const x = document.createElement("button"); x.type = "button"; x.className = "toast-x"; x.setAttribute("aria-label", "Dismiss message"); x.textContent = "×";
    x.addEventListener("click", () => dismiss(t));
    t.append(p, x); box.appendChild(t);
  };

  /* ---- forms: busy state, password toggle, error focus, confirm dialog, print ---- */
  $$("form[data-busy]").forEach((f) => f.addEventListener("submit", (e) => {
    const b = e.submitter || $("button[type=submit]", f);
    if (b) setTimeout(() => b.classList.add("is-busy"), 0);
  }));
  $$("[data-pw-toggle]").forEach((b) => b.addEventListener("click", () => {
    const input = b.parentElement.querySelector("input"), show = input.type === "password";
    input.type = show ? "text" : "password";
    b.setAttribute("aria-pressed", String(show));
    b.setAttribute("aria-label", show ? "Hide password" : "Show password");
  }));
  const summary = $("[data-focus]");
  if (summary) summary.focus();
  const dlg = $("#confirm");
  $$("form[data-confirm]").forEach((f) => f.addEventListener("submit", (e) => {
    if (f.dataset.ok === "1" || !dlg || !dlg.showModal) return;
    e.preventDefault();
    $("[data-confirm-title]", dlg).textContent = f.dataset.confirm;
    $("[data-confirm-text]", dlg).textContent = f.dataset.confirmText || "";
    dlg.returnValue = "";
    dlg.showModal();
    dlg.addEventListener("close", () => { if (dlg.returnValue === "ok") { f.dataset.ok = "1"; f.submit(); } }, { once: true });
  }));
  $$("[data-print]").forEach((b) => b.addEventListener("click", () => print()));

  /* ---- analyse: drop zone, previews, staged progress ---- */
  const form = $("[data-analyse]");
  if (form) {
    const input = $("#slices"), drop = $("[data-drop]"), thumbs = $("[data-thumbs]"), label = $("[data-count-label]"), ferr = $("[data-file-error]");
    const runner = $("[data-runner]"), steps = $$("[data-step]", runner), IMG = /\.(png|jpe?g|bmp|tiff?)$/i, VOL = /\.nii(\.gz)?$/i;
    const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
    let urls = [];
    const fail = (m) => { ferr.textContent = m; ferr.hidden = !m; };
    const render = () => {
      urls.forEach(URL.revokeObjectURL); urls = [];
      thumbs.textContent = "";
      const files = Array.from(input.files).sort((a, b) => natural.compare(a.name, b.name));
      fail("");
      if (!files.length) { thumbs.hidden = true; label.textContent = "No files chosen"; return false; }
      const vols = files.filter((f) => VOL.test(f.name)), bad = files.filter((f) => !VOL.test(f.name) && !IMG.test(f.name));
      if (bad.length) { fail(`${bad[0].name} is not a supported file. Use PNG, JPG, BMP or TIFF slices, or one NIfTI volume.`); }
      else if (vols.length && files.length > 1) { fail("Upload either one NIfTI volume or a set of slice images, not both."); }
      else if (files.length > 64) { fail(`Upload at most 64 slices per study. You chose ${files.length}.`); }
      thumbs.hidden = false;
      if (vols.length === 1 && files.length === 1) {
        const fig = document.createElement("figure"); fig.className = "vol";
        fig.textContent = `${files[0].name} · ${(files[0].size / 1048576).toFixed(1)} MB NIfTI volume`;
        thumbs.appendChild(fig); label.textContent = "1 volume";
      } else {
        files.slice(0, 64).forEach((f, i) => {
          const fig = document.createElement("figure"); fig.style.setProperty("--n", i);
          if (IMG.test(f.name) && !/\.tiff?$/i.test(f.name)) { const im = new Image(); const u = URL.createObjectURL(f); urls.push(u); im.src = u; im.alt = ""; fig.appendChild(im); }
          const cap = document.createElement("figcaption"); cap.textContent = String(i + 1); fig.appendChild(cap);
          thumbs.appendChild(fig);
        });
        label.textContent = `${files.length} ${files.length === 1 ? "slice" : "slices"}`;
      }
      return ferr.hidden;
    };
    input.addEventListener("change", render);
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
    drop.addEventListener("drop", (e) => { if (e.dataTransfer && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; render(); } });

    const stage = (n) => steps.forEach((s, i) => { s.classList.toggle("is-done", i < n); s.classList.toggle("is-now", i === n); });
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!input.files.length) { fail("Choose at least one MRI slice or a NIfTI volume."); drop.focus(); return; }
      if (!render()) return;
      if (!form.patient.value) { toast("Choose the patient this scan belongs to."); form.patient.focus(); return; }
      runner.hidden = false; stage(0);
      const xhr = new XMLHttpRequest();
      let timer = null, at = 0;
      xhr.open("POST", form.getAttribute("action") || location.href);
      xhr.setRequestHeader("X-Requested-With", "fetch");
      xhr.responseType = "json";
      xhr.upload.addEventListener("load", () => { at = 1; stage(1); timer = setInterval(() => { if (at < 4) stage(++at); }, 650); });
      const stop = (msg) => { clearInterval(timer); runner.hidden = true; toast(msg); };
      xhr.addEventListener("load", () => {
        const r = xhr.response;
        if (xhr.status === 200 && r && r.ok) { clearInterval(timer); stage(5); setTimeout(() => { location.href = r.redirect; }, 450); }
        else stop((r && r.error) || "The analysis could not be completed. Try again.");
      });
      xhr.addEventListener("error", () => stop("The connection was lost during the upload. Check the network and try again."));
      xhr.send(new FormData(form));
    });
  }

  /* ---- study: slice viewer ---- */
  const viewer = $("[data-viewer]");
  if (viewer) {
    const main = $("[data-main]", viewer), box = $(".viewer", viewer), toggle = $("[data-overlay-toggle]", viewer);
    const lab = $("[data-slice-label]", viewer), wl = $("[data-weight-label]", viewer), items = $$("[data-slice]", viewer);
    const top = viewer.dataset.top.split(",").filter(Boolean).map(Number), base = viewer.dataset.base;
    let cur = top[0], overlay = true;
    const pad = (n) => String(n).padStart(2, "0");
    const show = () => {
      const has = top.includes(cur), map = has && overlay;
      main.classList.add("is-swap");
      const next = `${base}${map ? "overlay" : "slice"}_${pad(cur)}.png`;
      const pre = new Image();
      pre.onload = () => { main.src = next; main.classList.remove("is-swap"); };
      pre.src = next;
      main.alt = `MRI slice ${cur + 1}${map ? " with the attention map" : ""}`;
      box.classList.toggle("no-map", !map);
      toggle.disabled = !has;
      toggle.setAttribute("aria-pressed", String(map));
      lab.textContent = `Slice ${cur + 1} of ${items.length}`;
      const it = items[cur];
      wl.textContent = `weight ${it.dataset.weight}%`;
      items.forEach((x) => { const on = x === it; x.classList.toggle("is-on", on); x.setAttribute("aria-selected", String(on)); });
    };
    items.forEach((it) => it.addEventListener("click", () => { cur = Number(it.dataset.slice); show(); }));
    toggle.addEventListener("click", () => { overlay = !overlay; show(); });
    viewer.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      if (!e.target.closest(".strip")) return;
      e.preventDefault();
      cur = Math.max(0, Math.min(items.length - 1, cur + (e.key === "ArrowRight" ? 1 : -1)));
      show(); items[cur].focus();
    });
    wl.textContent = `weight ${items[cur].dataset.weight}%`;
  }
})();
