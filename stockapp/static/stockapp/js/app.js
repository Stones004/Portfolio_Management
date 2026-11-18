/* StockVision Capital — UI behaviour (no dependencies). */
(function () {
    "use strict";
    const root = document.documentElement;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function store(key, value) {
        try {
            if (value === undefined) return localStorage.getItem(key);
            localStorage.setItem(key, value);
        } catch (e) { return null; }
    }

    /* ---------- theme toggle ---------- */
    document.querySelectorAll("[data-theme-toggle]").forEach(btn => btn.addEventListener("click", () => {
        const next = root.dataset.theme === "light" ? "dark" : "light";
        root.dataset.theme = next;
        store("sv-theme", next);
    }));

    /* ---------- mobile navigation ---------- */
    const menuBtn = document.querySelector(".menu-btn");
    const scrim = document.querySelector(".scrim");
    if (menuBtn) menuBtn.addEventListener("click", () => document.body.classList.toggle("nav-open"));
    if (scrim) scrim.addEventListener("click", () => document.body.classList.remove("nav-open"));
    document.addEventListener("keydown", e => { if (e.key === "Escape") document.body.classList.remove("nav-open"); });

    /* ---------- keyboard shortcuts: 1/2/3 jump between pages ---------- */
    const shortcuts = {};
    document.querySelectorAll(".nav a[data-key]").forEach(a => { shortcuts[a.dataset.key] = a.href; });
    document.addEventListener("keydown", e => {
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        const t = e.target;
        if (t && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))) return;
        if (shortcuts[e.key]) leaveTo(shortcuts[e.key]);
    });

    /* ---------- page transitions ---------- */
    function leaveTo(href) {
        if (reduceMotion) { window.location.href = href; return; }
        document.body.classList.add("leaving");
        setTimeout(() => { window.location.href = href; }, 200);
    }
    document.querySelectorAll("a[data-transition]").forEach(a => a.addEventListener("click", e => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
        e.preventDefault();
        leaveTo(a.href);
    }));

    /* ---------- count-up numbers ---------- */
    function countUp(el) {
        const raw = el.textContent.trim();
        const clean = raw.replace(/,/g, "");
        const target = parseFloat(clean);
        if (!isFinite(target) || reduceMotion) return;
        const decimals = (clean.split(".")[1] || "").replace(/\D.*$/, "").length;
        const commas = raw.includes(",");
        const plus = raw.startsWith("+");
        const suffix = (raw.match(/[^\d.]+$/) || [""])[0];
        const fmt = v => {
            let s = commas
                ? v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
                : v.toFixed(decimals);
            if (plus && v >= 0) s = "+" + s;
            return s + suffix;
        };
        const duration = 1300;
        const start = performance.now();
        el.textContent = fmt(0);
        (function tick(now) {
            const p = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - p, 4);
            el.textContent = p < 1 ? fmt(target * eased) : raw;
            if (p < 1) requestAnimationFrame(tick);
        })(start);
    }

    /* ---------- scroll reveal ---------- */
    const revealTargets = document.querySelectorAll("main .page-head, main .panel, main .kpi, main .alert:not(.warning), main [data-reveal]");
    let group = null, idx = 0;
    revealTargets.forEach(el => {
        // stagger siblings that share a parent
        if (el.parentElement !== group) { group = el.parentElement; idx = 0; }
        el.style.setProperty("--i", idx++);
        el.classList.add("reveal");
    });
    function show(el) {
        el.classList.add("in");
        el.querySelectorAll("[data-count]").forEach(n => {
            if (!n.dataset.counted) { n.dataset.counted = "1"; countUp(n); }
        });
    }
    if ("IntersectionObserver" in window && !reduceMotion) {
        const io = new IntersectionObserver(entries => entries.forEach(entry => {
            if (entry.isIntersecting) { show(entry.target); io.unobserve(entry.target); }
        }), { threshold: 0.08, rootMargin: "0px 0px -30px 0px" });
        revealTargets.forEach(el => io.observe(el));
    } else {
        revealTargets.forEach(show);
    }

    /* ---------- loading overlay for <form data-loading> ---------- */
    const overlay = document.getElementById("loading");
    const loadingText = document.getElementById("loading-text");
    const loadingStep = document.getElementById("loading-step");
    let stepTimer = null;
    const defaultSteps = ["Connecting to market data", "Downloading price history", "Cleaning and aligning series", "Crunching the numbers", "Rendering the results"];
    document.querySelectorAll("form[data-loading]").forEach(form => {
        form.addEventListener("submit", e => {
            if (!overlay || (e.submitter && e.submitter.hasAttribute("data-no-loading"))) return;
            loadingText.textContent = form.dataset.loading || "Working…";
            const steps = form.dataset.steps ? form.dataset.steps.split("|") : defaultSteps;
            let i = 0;
            loadingStep.textContent = steps[0];
            clearInterval(stepTimer);
            stepTimer = setInterval(() => {
                i = Math.min(i + 1, steps.length - 1);
                loadingStep.style.opacity = 0;
                setTimeout(() => { loadingStep.textContent = steps[i]; loadingStep.style.opacity = 1; }, 250);
            }, 2600);
            overlay.classList.add("show");
            setTimeout(() => document.querySelectorAll('button[type="submit"]').forEach(b => b.disabled = true), 0);
        });
    });
    window.addEventListener("pageshow", () => {
        document.body.classList.remove("leaving");
        if (overlay) overlay.classList.remove("show");
        clearInterval(stepTimer);
        document.querySelectorAll('button[type="submit"]').forEach(b => b.disabled = false);
    });

    /* ---------- file dropzones: show chosen file name ---------- */
    document.querySelectorAll(".dropzone").forEach(zone => {
        const input = zone.querySelector("input[type=file]");
        const title = zone.querySelector(".dz-title");
        if (!input || !title) return;
        const initial = title.textContent;
        input.addEventListener("change", () => {
            title.textContent = input.files.length ? input.files[0].name : initial;
            zone.classList.toggle("has-file", !!input.files.length);
        });
        ["dragenter", "dragover"].forEach(ev => zone.addEventListener(ev, () => zone.classList.add("drag")));
        ["dragleave", "drop"].forEach(ev => zone.addEventListener(ev, () => zone.classList.remove("drag")));
    });

    /* ---------- market session clocks ---------- */
    const sessions = document.querySelectorAll(".session[data-tz]");
    function partsIn(tz) {
        const f = new Intl.DateTimeFormat("en-GB", { timeZone: tz, weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false });
        const o = {};
        f.formatToParts(new Date()).forEach(p => { o[p.type] = p.value; });
        return o;
    }
    function updateSessions() {
        sessions.forEach(s => {
            const p = partsIn(s.dataset.tz);
            const mins = (parseInt(p.hour, 10) % 24) * 60 + parseInt(p.minute, 10);
            const [oh, om] = s.dataset.open.split(":").map(Number);
            const [ch, cm] = s.dataset.close.split(":").map(Number);
            const weekday = !["Sat", "Sun"].includes(p.weekday);
            const open = weekday && mins >= oh * 60 + om && mins < ch * 60 + cm;
            s.querySelector(".pulse").classList.toggle("open", open);
            s.querySelector(".time").textContent = `${p.hour}:${p.minute}`;
            s.querySelector(".state").textContent = open ? "Open" : "Closed";
        });
    }
    if (sessions.length) { updateSessions(); setInterval(updateSessions, 20000); }
})();
