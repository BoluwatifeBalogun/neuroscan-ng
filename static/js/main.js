/* NeuroScan NG interactions */
(function () {
  "use strict";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Reveal on scroll (one orchestrated entrance per element) */
  const revealEls = document.querySelectorAll(".reveal");
  if (reduced || !("IntersectionObserver" in window)) {
    revealEls.forEach((el) => el.classList.add("in"));
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.15 });
    revealEls.forEach((el) => io.observe(el));
  }

  /* Spotlight cards: cursor-tracked radial highlight */
  document.querySelectorAll("[data-spotlight-group]").forEach((group) => {
    group.addEventListener("pointermove", (ev) => {
      group.querySelectorAll(".spotlight").forEach((card) => {
        const r = card.getBoundingClientRect();
        card.style.setProperty("--mx", (ev.clientX - r.left) + "px");
        card.style.setProperty("--my", (ev.clientY - r.top) + "px");
      });
    });
  });

  /* Count-up stats */
  const counters = document.querySelectorAll("[data-count]");
  const runCount = (el) => {
    const target = parseInt(el.dataset.count, 10);
    if (reduced) { el.textContent = target.toLocaleString(); return; }
    const dur = 1400;
    const start = performance.now();
    const tick = (t) => {
      const p = Math.min((t - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  if ("IntersectionObserver" in window && counters.length) {
    const cio = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { runCount(e.target); cio.unobserve(e.target); }
      });
    }, { threshold: 0.5 });
    counters.forEach((el) => cio.observe(el));
  } else {
    counters.forEach(runCount);
  }

  /* Mobile nav */
  const toggle = document.querySelector("[data-nav-toggle]");
  const links = document.querySelector("[data-nav-links]");
  if (toggle && links) {
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  /* Dropzone */
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("mriInput");
  if (dz && input) {
    const idle = document.getElementById("dzIdle");
    const preview = document.getElementById("dzPreview");
    const img = document.getElementById("previewImg");
    const nameEl = document.getElementById("previewName");
    const btn = document.getElementById("analyseBtn");
    const form = document.getElementById("uploadForm");

    const show = (file) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        img.src = e.target.result;
        nameEl.textContent = file.name;
        idle.hidden = true;
        preview.hidden = false;
        btn.disabled = false;
      };
      reader.readAsDataURL(file);
    };

    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => show(input.files[0]));

    ["dragenter", "dragover"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
    dz.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        show(e.dataTransfer.files[0]);
      }
    });

    form.addEventListener("submit", () => {
      btn.disabled = true;
      btn.querySelector(".btn-label").hidden = true;
      btn.querySelector(".btn-loading").hidden = false;
    });
  }
})();
