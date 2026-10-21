const $ = (id) => document.getElementById(id);

const messageInput = $("message");
const scanButton = $("scan-button");

function formatPercent(value) {
    return value == null
        ? "—"
        : `${(value * 100).toFixed(1)}%`;
}

function showError(message) {
    $("error").textContent = message;
    $("error").hidden = false;
}

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, (character) => {
        const entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
        };

        return entities[character];
    });
}

async function loadHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();

        $("status-text").textContent = data.model_loaded
            ? "Model operational"
            : "Model not trained";

        $("status-dot").classList.toggle(
            "online",
            data.model_loaded
        );
    } catch {
        $("status-text").textContent = "API unavailable";
    }
}

async function loadMetrics() {
    try {
        const response = await fetch("/api/metrics");

        if (!response.ok) {
            return;
        }

        const report = await response.json();
        const metrics = report.metrics;

        $("accuracy").textContent =
            formatPercent(metrics.accuracy);

        $("precision").textContent =
            formatPercent(metrics.precision);

        $("recall").textContent =
            formatPercent(metrics.recall);

        $("f1").textContent =
            formatPercent(metrics.f1);

        $("auc").textContent =
            formatPercent(metrics.roc_auc);

        $("samples").textContent =
            report.test_samples.toLocaleString();

        $("demo-warning").hidden =
            !report.demo_dataset_warning;

        $("matrix").replaceChildren();

        for (const row of metrics.confusion_matrix) {
            for (const value of row) {
                const cell = document.createElement("div");
                cell.textContent = value;
                $("matrix").appendChild(cell);
            }
        }
    } catch (error) {
        console.error("Unable to load model metrics:", error);
    }
}

async function scanMessage() {
    const text = messageInput.value.trim();

    $("error").hidden = true;

    if (text.length < 3) {
        showError("Enter at least three characters.");
        return;
    }

    scanButton.disabled = true;
    scanButton.textContent = "Analyzing message...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction failed."
            );
        }

        const isSpam = data.label === "spam";

        $("empty-result").hidden = true;
        $("result").hidden = false;

        $("result-banner").classList.toggle(
            "legitimate",
            !isSpam
        );

        $("result-label").textContent = isSpam
            ? "SPAM DETECTED"
            : "LIKELY LEGITIMATE";

        $("result-description").textContent = isSpam
            ? "Review this message carefully."
            : "The model did not classify this as spam.";

        $("probability").textContent =
            formatPercent(data.spam_probability);

        $("probability-bar").style.width =
            `${data.spam_probability * 100}%`;

        const signals = $("signals");
        signals.replaceChildren();

        for (const signal of data.signals) {
            const row = document.createElement("div");
            row.className = "signal-row";

            const term = document.createElement("strong");
            term.textContent = signal.term;

            const track = document.createElement("div");
            track.className = "signal-track";

            const fill = document.createElement("div");
            fill.className = `signal-fill ${signal.direction}`;

            const width = Math.min(
                100,
                Math.max(8, Math.abs(signal.weight) * 150)
            );

            fill.style.width = `${width}%`;
            track.appendChild(fill);

            const direction = document.createElement("span");
            direction.className = "signal-direction";
            direction.textContent = signal.direction === "spam"
                ? "↑ Spam"
                : "↓ Legitimate";

            row.append(term, track, direction);
            signals.appendChild(row);
        }

        if (data.signals.length === 0) {
            signals.textContent = "No strong feature signals found.";
        }
    } catch (error) {
        showError(error.message);
    } finally {
        scanButton.disabled = false;
        scanButton.textContent = "✳ Analyze message";
    }
}

document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
        messageInput.value = button.dataset.sample;
    });
});

scanButton.addEventListener("click", scanMessage);

loadHealth();
loadMetrics();