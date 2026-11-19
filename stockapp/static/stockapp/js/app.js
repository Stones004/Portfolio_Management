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

    /* ---------- sidebar: icon rail on desktop, slide-over drawer on mobile ---------- */
    const menuBtn = document.querySelector(".menu-btn");
    const scrim = document.querySelector(".scrim");
    const mobile = window.matchMedia("(max-width: 900px)");
    function syncMenu() {
        if (!menuBtn) return;
        const open = mobile.matches ? document.body.classList.contains("nav-open") : !root.classList.contains("nav-collapsed");
        menuBtn.setAttribute("aria-expanded", String(open));
        menuBtn.setAttribute("aria-label", open ? "Collapse sidebar" : "Expand sidebar");
    }
    function toggleNav() {
        if (mobile.matches) {
            document.body.classList.toggle("nav-open");
        } else {
            const collapsed = root.classList.toggle("nav-collapsed");
            store("sv-nav", collapsed ? "collapsed" : "open");
            hideTip();
        }
        syncMenu();
    }
    if (menuBtn) menuBtn.addEventListener("click", toggleNav);
    if (scrim) scrim.addEventListener("click", () => { document.body.classList.remove("nav-open"); syncMenu(); });
    document.addEventListener("keydown", e => {
        if (e.key === "Escape") { document.body.classList.remove("nav-open"); syncMenu(); }
    });
    mobile.addEventListener("change", () => { document.body.classList.remove("nav-open"); syncMenu(); });
    syncMenu();

    // Labels are hidden on the collapsed rail, so show them as a floating tooltip on hover/focus.
    const tip = document.createElement("div");
    tip.className = "nav-tip";
    tip.setAttribute("aria-hidden", "true");
    document.body.appendChild(tip);
    function hideTip() { tip.classList.remove("show"); }
    document.querySelectorAll(".sidebar [data-tip]").forEach(el => {
        const showTip = () => {
            if (mobile.matches || !root.classList.contains("nav-collapsed")) return;
            const r = el.getBoundingClientRect();
            tip.textContent = el.dataset.tip;
            tip.style.left = (r.right + 12) + "px";
            tip.style.top = (r.top + r.height / 2) + "px";
            tip.classList.add("show");
        };
        el.addEventListener("mouseenter", showTip);
        el.addEventListener("focus", showTip);
        el.addEventListener("mouseleave", hideTip);
        el.addEventListener("blur", hideTip);
    });

    /* ---------- keyboard shortcuts: 1/2/3 jump between pages ---------- */
    const shortcuts = {};
    document.querySelectorAll(".nav a[data-key]").forEach(a => { shortcuts[a.dataset.key] = a.href; });
    document.addEventListener("keydown", e => {
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        const t = e.target;
        if (t && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))) return;
        if (shortcuts[e.key]) leaveTo(shortcuts[e.key]);
        else if (e.key === "[" && menuBtn) toggleNav();
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

    /* ---------- live ticker strip ---------- */
    const tape = document.querySelector(".tape");
    if (tape) initTape(tape);

    function initTape(tape) {
        const track = tape.querySelector(".tape-track");
        const glider = tape.querySelector(".tape-glider");
        const buttons = tape.querySelectorAll("[data-market]");
        const liveDot = tape.querySelector(".tape-live .pulse");
        const liveTime = tape.querySelector(".tape-time");
        const HOURS = { IN: ["Asia/Kolkata", "09:15", "15:30"], US: ["America/New_York", "09:30", "16:00"] };
        const SPEED = 45;          // pixels per second
        const POLL_MS = 60000;
        let market = store("sv-market");
        if (!HOURS[market]) {
            const tz = (Intl.DateTimeFormat().resolvedOptions().timeZone || "");
            market = /Kolkata|Calcutta/.test(tz) ? "IN" : "US";
        }
        let shownMarket = null, keys = "", nodes = {}, timer = null, request = 0;

        function isOpen(m) {
            const [tz, o, c] = HOURS[m];
            const p = partsIn(tz);
            const mins = (parseInt(p.hour, 10) % 24) * 60 + parseInt(p.minute, 10);
            const toMin = t => { const [h, mm] = t.split(":").map(Number); return h * 60 + mm; };
            return !["Sat", "Sun"].includes(p.weekday) && mins >= toMin(o) && mins < toMin(c);
        }
        function moveGlider() {
            const on = tape.querySelector(`[data-market="${market}"]`);
            if (!on || !glider) return;
            glider.style.width = on.offsetWidth + "px";
            glider.style.transform = `translateX(${on.offsetLeft}px)`;
            buttons.forEach(b => b.setAttribute("aria-pressed", String(b === on)));
        }
        const fmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        const pct = v => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
        function escapeHtml(t) { return String(t).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

        function tickHtml(q) {
            const dir = q.change_pct == null ? "" : (q.change_pct >= 0 ? "up" : "down");
            const chg = q.change_pct == null ? "" : `<span class="chg ${dir}">${pct(q.change_pct)}</span>`;
            const inner = `<b>${escapeHtml(q.label)}</b><span class="px">${fmt.format(q.price)}</span>${chg}`;
            if (q.index) return `<span class="tick idx" data-sym="${escapeHtml(q.symbol)}">${inner}</span>`;
            const href = `${tape.dataset.dcfUrl}?ticker=${encodeURIComponent(q.symbol)}&run=1`;
            return `<a class="tick" data-sym="${escapeHtml(q.symbol)}" href="${href}" title="Run a DCF on ${escapeHtml(q.label)}">${inner}</a>`;
        }
        function message(text) {
            track.classList.remove("run");
            track.innerHTML = `<span class="tape-msg">${escapeHtml(text)}</span>`;
            keys = ""; nodes = {};
        }
        function build(items) {
            // Two identical copies: the animation scrolls by exactly one copy, so the loop is seamless.
            const once = items.map(tickHtml).join("");
            track.classList.remove("run");
            track.innerHTML = once + once.replace(/<(a|span) class="tick/g, '<$1 aria-hidden="true" tabindex="-1" class="tick');
            nodes = {};
            track.querySelectorAll(".tick").forEach(el => (nodes[el.dataset.sym] = nodes[el.dataset.sym] || []).push(el));
            requestAnimationFrame(() => {
                const width = track.scrollWidth / 2;
                track.style.setProperty("--tape-dur", Math.max(20, width / SPEED) + "s");
                if (!reduceMotion) track.classList.add("run");
            });
        }
        function update(items) {
            // Same symbols as before: update prices in place so the strip keeps scrolling smoothly.
            items.forEach(q => (nodes[q.symbol] || []).forEach(el => {
                const px = el.querySelector(".px");
                const old = parseFloat(px.dataset.v || "NaN");
                px.textContent = fmt.format(q.price);
                if (isFinite(old) && q.price !== old) {
                    px.classList.remove("flash-up", "flash-down");
                    void px.offsetWidth;   // restart the animation
                    px.classList.add(q.price > old ? "flash-up" : "flash-down");
                }
                px.dataset.v = q.price;
                const chg = el.querySelector(".chg");
                if (chg && q.change_pct != null) {
                    chg.textContent = pct(q.change_pct);
                    chg.className = "chg " + (q.change_pct >= 0 ? "up" : "down");
                }
            }));
        }
        function stamp() {
            const open = isOpen(market);
            if (liveDot) liveDot.classList.toggle("open", open);
            if (liveTime) liveTime.textContent = (open ? "Live " : "Closed · ")
                + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }

        async function load() {
            const id = ++request, m = market;
            try {
                const res = await fetch(`${tape.dataset.quotesUrl}?market=${m}`, { headers: { "Accept": "application/json" }, credentials: "same-origin" });
                const data = await res.json();
                if (id !== request) return;               // a newer request (market switch) won
                if (!data.ok || !data.items.length) { message(data.error || "Live prices are unavailable right now."); return; }
                const newKeys = data.items.map(q => q.symbol).join("|");
                const swapping = shownMarket !== m;
                if (newKeys !== keys) {
                    if (swapping && shownMarket !== null && !reduceMotion) {
                        track.classList.add("swap");
                        await new Promise(r => setTimeout(r, 300));
                        if (id !== request) return;
                    }
                    build(data.items);
                    keys = newKeys;
                    track.classList.remove("swap");
                }
                update(data.items);
                shownMarket = m;
                stamp();
            } catch (e) {
                if (id === request && !keys) message("Live prices are unavailable right now.");
            }
        }
        function schedule() {
            clearInterval(timer);
            timer = setInterval(() => { if (!document.hidden) load(); }, POLL_MS);
        }

        buttons.forEach(b => b.addEventListener("click", () => {
            if (b.dataset.market === market) return;
            market = b.dataset.market;
            store("sv-market", market);
            moveGlider();
            load();
            schedule();
        }));
        document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
        window.addEventListener("resize", moveGlider);
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(moveGlider);
        moveGlider();
        load();
        schedule();
    }
})();
