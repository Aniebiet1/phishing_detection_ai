const API = {
    health: "/health",
    predict: "/predict",
    predictRaw: "/predict-raw",
    login: "/auth/login",
    register: "/auth/register",
    me: "/auth/me",
    history: "/users/me/history",
    users: "/users/summary",
    report: "/training-report",
};

const MODEL_DETAILS = {
    linear_svc: {
        title: "Linear Support Vector Classifier",
        math: "Optimizes a maximum-margin hyperplane. The decision rule is based on sign(w.x + b), which separates phishing and legitimate text in TF-IDF feature space.",
        why: "Works well in high-dimensional sparse text classification because the separating boundary is robust even when features are mostly zeros.",
    },
    logistic_regression: {
        title: "Logistic Regression",
        math: "Uses the sigmoid function sigma(z) = 1 / (1 + e^-z) to estimate the probability that a sample is phishing.",
        why: "Strong baseline for text classification because linear weights over TF-IDF vectors remain interpretable and stable.",
    },
    passive_aggressive: {
        title: "Passive Aggressive Classifier",
        math: "Updates weights only when the current prediction is wrong or inside the margin, keeping the model passive on correct confident samples and aggressive on mistakes.",
        why: "Efficient for large-scale online learning and sparse features.",
    },
    sgd_classifier: {
        title: "Stochastic Gradient Descent Classifier",
        math: "Learns a linear decision function by iteratively minimizing a loss with gradient steps over mini updates.",
        why: "Scales well with large corpora and supports large-dimensional text representations.",
    },
    multinomial_nb: {
        title: "Multinomial Naive Bayes",
        math: "Applies Bayes' theorem under conditional independence assumptions and multiplies per-token likelihoods for each class.",
        why: "Fast and often surprisingly competitive for bag-of-words style text problems.",
    },
};

const RESEARCH_LINKS = [
    {
        title: "LIBLINEAR: A Library for Large Linear Classification",
        url: "https://www.csie.ntu.edu.tw/~cjlin/liblinear/",
        note: "Core reference for linear SVM and logistic regression optimization used widely in text classification.",
    },
    {
        title: "Scikit-learn User Guide: Linear Models",
        url: "https://scikit-learn.org/stable/modules/linear_model.html",
        note: "Practical documentation for logistic regression, SGD-based classifiers, and linear decision functions.",
    },
    {
        title: "Passive-Aggressive Algorithms",
        url: "https://jmlr.org/papers/volume7/crammer06a/crammer06a.pdf",
        note: "Seminal paper explaining passive-aggressive online learning updates.",
    },
    {
        title: "A Comparison of Event Models for Naive Bayes Text Classification",
        url: "https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-007.pdf",
        note: "Classic text classification reference for multinomial Naive Bayes.",
    },
    {
        title: "Phishing Websites Features and Machine Learning Review",
        url: "https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=phishing%20website%20machine%20learning",
        note: "Starting point for phishing-specific ML research papers and comparative studies.",
    },
];

const state = {
    theme: localStorage.getItem("micheal-ai-theme") || "light",
    source: "url",
    auth: JSON.parse(localStorage.getItem("micheal-ai-auth") || "null"),
    report: null,
};

document.addEventListener("DOMContentLoaded", () => {
    applyTheme(state.theme);
    bindThemeToggle();
    initMobileNavigation();
    initMotion();
    hydrateCommonData();

    const page = document.body.dataset.page;
    if (page === "home") {
        initHomePage();
    }
    if (page === "insights") {
        initInsightsPage();
    }
    if (page === "dashboard") {
        initDashboardPage();
    }
});

function initMobileNavigation() {
    const header = document.querySelector(".site-header");
    const toggle = document.getElementById("mobileNavToggle");
    const nav = document.getElementById("topNavMenu");
    if (!header || !toggle || !nav) {
        return;
    }

    const closeMenu = () => {
        header.classList.remove("nav-open");
        toggle.setAttribute("aria-expanded", "false");
    };

    toggle.addEventListener("click", () => {
        const isOpen = header.classList.toggle("nav-open");
        toggle.setAttribute("aria-expanded", String(isOpen));
    });

    nav.querySelectorAll("a, button").forEach((item) => {
        item.addEventListener("click", () => {
            if (window.matchMedia("(max-width: 980px)").matches) {
                closeMenu();
            }
        });
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeMenu();
        }
    });

    window.addEventListener("resize", () => {
        if (window.innerWidth > 980) {
            closeMenu();
        }
    });
}

function initMotion() {
    requestAnimationFrame(() => {
        document.body.classList.add("is-ready");
    });

    const animated = document.querySelectorAll(".reveal-up, .stagger-item, .reveal-on-scroll");
    if (!animated.length) {
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("in-view");
                    observer.unobserve(entry.target);
                }
            });
        },
        {
            threshold: 0.16,
            rootMargin: "0px 0px -40px 0px",
        },
    );

    animated.forEach((element) => observer.observe(element));
}

function bindThemeToggle() {
    const toggle = document.getElementById("themeToggle");
    if (!toggle) {
        return;
    }
    toggle.addEventListener("click", () => {
        state.theme = state.theme === "dark" ? "light" : "dark";
        localStorage.setItem("micheal-ai-theme", state.theme);
        applyTheme(state.theme);
    });
}

function applyTheme(theme) {
    document.body.dataset.theme = theme;
}

async function hydrateCommonData() {
    await Promise.allSettled([loadHealth(), loadReport()]);
}

async function loadHealth() {
    const serviceStatus = document.getElementById("serviceStatus");
    const serviceStatusText = document.getElementById("serviceStatusText");
    if (!serviceStatus || !serviceStatusText) {
        return;
    }

    try {
        const response = await fetch(API.health);
        const data = await response.json();
        serviceStatus.textContent = data.model_loaded ? "Service available and model loaded" : "Service available";
        serviceStatusText.textContent = data.model_loaded
            ? "The classifier is available and ready to evaluate submitted content."
            : "The service is responding, but the model still needs to be trained or loaded.";
    } catch (error) {
        serviceStatus.textContent = "Service unavailable";
        serviceStatusText.textContent = "The interface could not reach the FastAPI backend.";
    }
}

async function loadReport() {
    try {
        const response = await fetch(API.report);
        if (!response.ok) {
            throw new Error("Report unavailable");
        }
        state.report = await response.json();
        syncReportHighlights();
    } catch (error) {
        state.report = null;
    }
}

function syncReportHighlights() {
    if (!state.report) {
        return;
    }

    const best = state.report.models?.[0];
    setText("bestModelName", prettifyModelName(state.report.selection?.best_model || "Unavailable"));
    setText("bestModelScore", best ? formatPercent(best.phishing_f1) : "-");
    setText("rowsUsed", formatNumber(state.report.data?.rows_used));
    setText("selectionMetricPill", `Metric: ${state.report.selection?.metric || "n/a"}`);
    setText("reportBestModel", prettifyModelName(state.report.selection?.best_model || "-"));
    setText("reportRows", formatNumber(state.report.data?.rows_used));
    setText("reportSplit", `${formatNumber(state.report.data?.train_rows)} / ${formatNumber(state.report.data?.test_rows)}`);
    setText("reportScore", best ? formatPercent(best.phishing_f1) : "-");
    setText("heroBestModel", prettifyModelName(state.report.selection?.best_model || "-"));
    setText("heroBestMetric", best ? `${formatPercent(best.phishing_f1)} phishing F1` : "Validation score unavailable");
    setText("heroSplit", `${formatNumber(state.report.data?.train_rows)} / ${formatNumber(state.report.data?.test_rows)}`);
    setText("heroSelectionMetric", String(state.report.selection?.metric || "n/a").toUpperCase());

    const reportSummary = document.getElementById("reportSummary");
    if (reportSummary) {
        reportSummary.textContent = `${prettifyModelName(state.report.selection?.best_model || "The selected model")} was selected using ${state.report.selection?.metric || "the chosen metric"} after training on ${formatNumber(state.report.data?.rows_used)} combined records drawn from the email and URL datasets.`;
    }

    renderModelRanking();
    renderAlgorithmCards();
    renderResearchLinks();
}

function initHomePage() {
    drawScoreChart(0, "neutral");
    updateAuthPill();

    document.querySelectorAll(".switch-option").forEach((button) => {
        button.addEventListener("click", () => {
            state.source = button.dataset.source;
            document.querySelectorAll(".switch-option").forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
        });
    });

    const predictionForm = document.getElementById("predictionForm");
    const rawButton = document.getElementById("useRawMode");

    predictionForm?.addEventListener("submit", (event) => {
        event.preventDefault();
        submitPrediction(false);
    });

    rawButton?.addEventListener("click", () => submitPrediction(true));
}

async function submitPrediction(useRawMode) {
    const text = document.getElementById("predictionText")?.value?.trim() || "";
    if (!text) {
        renderPredictionError("Provide a URL or email body before submitting.");
        return;
    }

    const resultCard = document.querySelector(".result-card");
    if (resultCard) {
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    setText("resultHeadline", "Running classification...");
    setText("resultCopy", "Micheal Ai is evaluating the submitted content.");
    setText("resultMode", useRawMode ? "Raw text request" : "Structured request");

    try {
        const headers = { Accept: "application/json" };
        if (state.auth?.token) {
            headers.Authorization = `Bearer ${state.auth.token}`;
        }

        let response;
        if (useRawMode) {
            headers["Content-Type"] = "text/plain";
            response = await fetch(API.predictRaw, {
                method: "POST",
                headers,
                body: text,
            });
        } else {
            headers["Content-Type"] = "application/json";
            response = await fetch(API.predict, {
                method: "POST",
                headers,
                body: JSON.stringify({ text, source: state.source }),
            });
        }

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }
        if (state.auth?.token && data.details?.user_type !== "authenticated") {
            state.auth = null;
            localStorage.removeItem("micheal-ai-auth");
            updateAuthPill();
        }
        renderPredictionResult(data);
    } catch (error) {
        renderPredictionError(error.message || "Prediction failed.");
    }
}

function renderPredictionResult(data) {
    const label = data.label === "phishing" ? "Phishing" : "Legitimate";
    const tone = data.label === "phishing" ? "danger" : "success";
    setText("resultHeadline", `This is ${label}.`);
    setText("resultCopy", `The classifier assessed this submission as ${label.toLowerCase()} and returned the confidence shown below.`);
    setText("resultLabel", label);
    setText("resultScore", formatPercent(data.score));
    setText("scorePercent", formatPercent(data.score));
    setText("resultMode", data.details?.mode || "prediction");
    setText("resultUser", data.details?.user || "No signed-in account");
    drawScoreChart(Number(data.score), tone);

    const details = document.getElementById("resultDetails");
    if (details) {
        details.innerHTML = "";
        const responseEmail = data.details?.user || "";
        const displayEmail = responseEmail ? shortenEmail(responseEmail, 10) : null;
        const fragments = [
            `Source: ${data.details?.source || state.source}`,
            `Input length: ${data.details?.input_length || 0}`,
            displayEmail ? `Saved for: ${displayEmail}` : "Guest submission",
        ];
        fragments.forEach((value) => {
            const chip = document.createElement("span");
            chip.textContent = value;
            if (displayEmail && value.startsWith("Saved for:")) {
                chip.title = `Saved for: ${responseEmail}`;
            }
            details.appendChild(chip);
        });
    }
}

function renderPredictionError(message) {
    setText("resultHeadline", "Classification failed");
    setText("resultCopy", message);
    setText("resultLabel", "Error");
    setText("resultScore", "0.00%");
    setText("resultUser", state.auth?.user?.email || "No signed-in account");
    setText("scorePercent", "0%");
    setText("resultMode", "Request issue");
    drawScoreChart(0, "danger");
}

function drawScoreChart(score, tone) {
    const canvas = document.getElementById("scoreChart");
    if (!canvas) {
        return;
    }
    const ctx = canvas.getContext("2d");
    const percent = Math.max(0, Math.min(score || 0, 1));
    const styles = getComputedStyle(document.body);
    const bgRing = rgba(styles.getPropertyValue("--primary").trim(), 0.12);
    const active = tone === "danger"
        ? "#ef4444"
        : tone === "success"
            ? "#0ea573"
            : styles.getPropertyValue("--primary").trim();

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 18;
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.strokeStyle = bgRing;
    ctx.arc(120, 120, 78, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = active;
    ctx.arc(120, 120, 78, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * percent);
    ctx.stroke();

    ctx.beginPath();
    ctx.fillStyle = rgba(active, 0.1);
    ctx.arc(120, 120, 52, 0, Math.PI * 2);
    ctx.fill();
}

function initInsightsPage() {
    renderAlgorithmCards();
    renderResearchLinks();
    renderModelRanking();
}

function renderModelRanking() {
    const container = document.getElementById("modelRanking");
    if (!container || !state.report?.models) {
        return;
    }
    container.innerHTML = "";
    state.report.models.forEach((model, index) => {
        const row = document.createElement("article");
        row.className = "ranking-item";
        row.innerHTML = `
            <div class="section-heading">
                <div>
                    <span class="eyebrow">#${index + 1}</span>
                    <strong>${prettifyModelName(model.name)}</strong>
                </div>
                <span class="pill">${formatPercent(model.phishing_f1)}</span>
            </div>
            <div class="ranking-bar"><span style="width:${Math.max(6, model.phishing_f1 * 100)}%"></span></div>
            <div class="ranking-meta">Accuracy ${formatPercent(model.accuracy)} | Precision ${formatPercent(model.precision_phishing)} | Recall ${formatPercent(model.recall_phishing)} | Train ${model.train_seconds.toFixed(1)}s | Predict ${model.predict_seconds.toFixed(1)}s</div>
        `;
        container.appendChild(row);
    });
}

function renderAlgorithmCards() {
    const container = document.getElementById("algorithmCards");
    if (!container) {
        return;
    }
    container.innerHTML = "";
    const orderedNames = state.report?.ranking || Object.keys(MODEL_DETAILS);
    orderedNames.forEach((name) => {
        const details = MODEL_DETAILS[name];
        if (!details) {
            return;
        }
        const card = document.createElement("article");
        card.className = "algorithm-card";
        card.innerHTML = `
            <strong>${details.title}</strong>
            <p>${details.why}</p>
            <p>${details.math}</p>
        `;
        container.appendChild(card);
    });
}

function renderResearchLinks() {
    const container = document.getElementById("researchLinks");
    if (!container) {
        return;
    }
    container.innerHTML = "";
    RESEARCH_LINKS.forEach((item) => {
        const link = document.createElement("a");
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.innerHTML = `<strong>${item.title}</strong><p>${item.note}</p>`;
        container.appendChild(link);
    });
}

function initDashboardPage() {
    const authForm = document.getElementById("authForm");
    const registerButton = document.getElementById("registerButton");
    authForm?.addEventListener("submit", (event) => {
        event.preventDefault();
        submitAuth("login");
    });
    registerButton?.addEventListener("click", () => submitAuth("register"));
    refreshDashboard();
}

async function submitAuth(mode) {
    const email = document.getElementById("authEmail")?.value?.trim() || "";
    const password = document.getElementById("authPassword")?.value || "";
    const authMessage = document.getElementById("authMessage");

    if (!email || !password) {
        if (authMessage) {
            authMessage.textContent = "Provide both email and password.";
        }
        return;
    }

    try {
        const response = await fetch(mode === "register" ? API.register : API.login, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({ email, password }),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Authentication failed.");
        }
        state.auth = data;
        localStorage.setItem("micheal-ai-auth", JSON.stringify(state.auth));
        if (authMessage) {
            authMessage.textContent = mode === "register"
            ? "Account created and signed in successfully."
            : "Signed in successfully.";
        }
        updateAuthPill();
        refreshDashboard();
    } catch (error) {
        if (authMessage) {
            authMessage.textContent = error.message || "Authentication failed.";
        }
    }
}

async function refreshDashboard() {
    updateAuthPill();
    await loadCommunityUsers();
    if (!state.auth?.token) {
        renderProfile(null);
        renderHistory(null);
        return;
    }

    try {
        const [meResponse, historyResponse] = await Promise.all([
            fetch(API.me, { headers: authHeaders() }),
            fetch(API.history, { headers: authHeaders() }),
        ]);
        const me = await meResponse.json();
        const history = await historyResponse.json();
        if (!meResponse.ok || !historyResponse.ok) {
            throw new Error(me.detail || history.detail || "Could not load user data.");
        }
        state.auth.user = me.user;
        localStorage.setItem("micheal-ai-auth", JSON.stringify(state.auth));
        renderProfile(me.user);
        renderHistory(history);
    } catch (error) {
        renderProfile(null, error.message || "Could not load account profile.");
        renderHistory(null, error.message || "Could not load history.");
    }
}

async function loadCommunityUsers() {
    const container = document.getElementById("communityUsers");
    if (!container) {
        return;
    }
    container.innerHTML = '<div class="empty-state">Loading community activity...</div>';
    try {
        const response = await fetch(API.users);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Could not load users.");
        }
        container.innerHTML = "";
        if (!data.users.length) {
            container.innerHTML = '<div class="empty-state">No user activity is available yet.</div>';
            return;
        }
        data.users.forEach((user, index) => {
            const row = document.createElement("article");
            row.className = "community-row";
            const shortEmail = shortenEmail(user.email, 10);
            row.innerHTML = `
                <strong title="${escapeHtml(user.email)}">#${index + 1} ${shortEmail}</strong>
                <span class="community-meta">Predictions: ${user.prediction_count}</span>
                <span class="community-meta">Last activity: ${formatDate(user.last_prediction_at || user.last_login)}</span>
            `;
            container.appendChild(row);
        });
    } catch (error) {
        container.innerHTML = `<div class="empty-state">${error.message || "Could not load user activity."}</div>`;
    }
}

function renderProfile(user, message) {
    const profileCard = document.getElementById("profileCard");
    const authState = document.getElementById("dashboardAuthState");
    if (!profileCard || !authState) {
        return;
    }

    if (!user) {
        authState.textContent = "Guest";
        profileCard.className = "profile-card empty-state";
        profileCard.textContent = message || "Sign in to review account activity, classification totals, and recent scan timestamps.";
        return;
    }

    authState.textContent = "Signed in";
    profileCard.className = "profile-card";
    const shortEmail = shortenEmail(user.email, 10);
    profileCard.innerHTML = `
        <strong title="${escapeHtml(user.email)}">${shortEmail}</strong>
        <div class="profile-metrics">
            <article class="profile-metric">
                <span>Joined</span>
                <strong>${formatDate(user.joined_at)}</strong>
            </article>
            <article class="profile-metric">
                <span>Predictions saved</span>
                <strong>${user.prediction_count}</strong>
            </article>
            <article class="profile-metric">
                <span>Last login</span>
                <strong>${formatDate(user.last_login)}</strong>
            </article>
            <article class="profile-metric">
                <span>Last prediction</span>
                <strong>${formatDate(user.last_prediction_at)}</strong>
            </article>
        </div>
    `;
}

function renderHistory(payload, message) {
    const summary = document.getElementById("historySummary");
    const table = document.getElementById("historyTable");
    if (!summary || !table) {
        return;
    }

    if (!payload) {
        summary.className = "history-summary empty-state";
        summary.textContent = message || "No signed-in user yet.";
        table.innerHTML = "";
        return;
    }

    summary.className = "history-summary";
    summary.innerHTML = `
        <strong>${payload.summary.total_predictions}</strong> predictions saved. Phishing: ${payload.summary.phishing_predictions}. Legitimate: ${payload.summary.legitimate_predictions}.
    `;

    table.innerHTML = "";
    const header = document.createElement("div");
    header.className = "table-row header";
    header.innerHTML = "<span>Preview</span><span>Label</span><span>Score</span><span>Saved at</span>";
    table.appendChild(header);

    if (!payload.history.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No classifications have been saved yet. Submit content from the home page while signed in.";
        table.appendChild(empty);
        return;
    }

    payload.history.forEach((entry) => {
        const row = document.createElement("div");
        row.className = "table-row";
        row.innerHTML = `
            <span>${escapeHtml(entry.preview || "")}</span>
            <span><strong class="${entry.label === "phishing" ? "tag-danger" : "tag-success"}">${entry.label}</strong></span>
            <span>${formatPercent(entry.score)}</span>
            <span>${formatDate(entry.timestamp)}</span>
        `;
        table.appendChild(row);
    });
}

function updateAuthPill() {
    const pill = document.getElementById("authStatePill");
    if (pill) {
        if (state.auth?.user?.email) {
            const shortEmail = shortenEmail(state.auth.user.email, 10);
            pill.textContent = `Signed in as ${shortEmail}`;
            pill.title = `Signed in as ${state.auth.user.email}`;
        } else {
            pill.textContent = "Guest session";
            pill.removeAttribute("title");
        }
    }
}

function shortenEmail(email, maxLength) {
    const value = String(email || "").trim();
    if (!value || value.length <= maxLength) {
        return value;
    }

    const keep = Math.max(1, maxLength - 3);
    return `${value.slice(0, keep)}...`;
}

function authHeaders() {
    return {
        Authorization: `Bearer ${state.auth.token}`,
        Accept: "application/json",
    };
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function prettifyModelName(name) {
    return String(name || "")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return Number(value).toLocaleString();
}

function formatDate(value) {
    if (!value) {
        return "-";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }
    return date.toLocaleString();
}

function rgba(color, alpha) {
    if (color.startsWith("#")) {
        const hex = color.slice(1);
        const normalized = hex.length === 3
            ? hex.split("").map((char) => char + char).join("")
            : hex;
        const red = parseInt(normalized.slice(0, 2), 16);
        const green = parseInt(normalized.slice(2, 4), 16);
        const blue = parseInt(normalized.slice(4, 6), 16);
        return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
    }
    return color;
}

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value;
    return div.innerHTML;
}