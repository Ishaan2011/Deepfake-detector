const htmlRoot = document.documentElement;
const bodyEl = document.body;
const themeToggle = document.getElementById("theme-toggle");
const logsToggle = document.getElementById("logs-toggle");
const form = document.getElementById("analyze-form");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const overallVerdictEl = document.getElementById("overall-verdict");
const mesoScoreEl = document.getElementById("meso-score");
const d3ScoreEl = document.getElementById("d3-score");
const inputEl = document.getElementById("youtube-url");
const jobIdEl = document.getElementById("job-id");
const stageTextEl = document.getElementById("stage-text");
const modelTextEl = document.getElementById("model-text");
const downloadTextEl = document.getElementById("download-text");
const downloadBarEl = document.getElementById("download-bar");
const liveLogsEl = document.getElementById("live-logs");

let darkMode = false;
let logsPanelOpen = false;
let pollTimer = null;
let activeJobId = null;
let renderedLogCount = 0;

themeToggle.addEventListener("click", () => {
  darkMode = !darkMode;
  htmlRoot.setAttribute("data-theme", darkMode ? "dark" : "light");
  themeToggle.textContent = darkMode ? "Switch To Light" : "Switch To Dark";
});

logsToggle.addEventListener("click", () => {
  logsPanelOpen = !logsPanelOpen;
  bodyEl.classList.toggle("logs-open", logsPanelOpen);
  logsToggle.textContent = logsPanelOpen ? "Hide Logs Panel" : "Show Logs Panel";
});

const setStatus = (text, type = "") => {
  statusEl.textContent = text;
  statusEl.className = `status ${type}`.trim();
};

const stopPolling = () => {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
};

const resetLiveView = () => {
  renderedLogCount = 0;
  liveLogsEl.innerHTML = "";
  stageTextEl.textContent = "Idle";
  modelTextEl.textContent = "Waiting";
  downloadTextEl.textContent = "0.0%";
  downloadBarEl.style.width = "0%";
  jobIdEl.textContent = "Job: --";
  overallVerdictEl.textContent = "Not done yet";
  mesoScoreEl.textContent = "Not done yet";
  d3ScoreEl.textContent = "Not done yet";
};

const formatScore = (value) => {
  if (value === null || value === undefined) return "Not available";
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return "Not available";
  return numeric.toFixed(4);
};

const getModelState = (status, phase) => {
  if (status === "running_mesonet") return "MesoNet running";
  if (status === "running_temp_d3") return "Temp-D3 running";
  if (status === "running_models") return "Model pipeline running";
  if ((phase || "").includes("[MesoNet]")) return "MesoNet running";
  if ((phase || "").includes("[Temp-D3]")) return "Temp-D3 running";
  if (status === "completed") return "All models complete";
  if (status === "failed") return "Pipeline failed";
  return "Waiting";
};

const appendLogs = (logs) => {
  if (!Array.isArray(logs) || logs.length === 0) return;
  if (renderedLogCount > logs.length) {
    renderedLogCount = 0;
    liveLogsEl.innerHTML = "";
  }

  const pendingLogs = logs.slice(renderedLogCount);
  if (pendingLogs.length === 0) return;

  const fragment = document.createDocumentFragment();
  pendingLogs.forEach((entry) => {
    const line = document.createElement("div");
    const level = String(entry.level || "INFO").toLowerCase();
    line.className = `log-line ${level}`;
    line.textContent = `[${entry.ts || "--"}] [${entry.level || "INFO"}] ${entry.message || ""}`;
    fragment.appendChild(line);
  });

  liveLogsEl.appendChild(fragment);
  renderedLogCount = logs.length;
  liveLogsEl.scrollTop = liveLogsEl.scrollHeight;
};

const updateLivePanel = (snapshot) => {
  const status = snapshot.status || "unknown";
  const phase = snapshot.phase || "No phase";
  const percent = Number(snapshot.download_percent || 0);

  stageTextEl.textContent = phase;
  modelTextEl.textContent = getModelState(status, phase);
  downloadTextEl.textContent = `${percent.toFixed(1)}%`;
  downloadBarEl.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  jobIdEl.textContent = `Job: ${snapshot.job_id || "--"}`;
  appendLogs(snapshot.logs || []);
};

const handleCompletion = (snapshot) => {
  const result = snapshot.result || {};
  const overallVerdict = String(result.overall_verdict || "--").toUpperCase();
  overallVerdictEl.textContent = overallVerdict;
  mesoScoreEl.textContent = formatScore(result.mesonet_score);
  d3ScoreEl.textContent = formatScore(result.temp_d3_score);
  resultsEl.classList.remove("hidden");
  const runtime = result.processing_seconds ? ` in ${result.processing_seconds}s` : "";
  setStatus(`Analysis complete${runtime}. Verdict: ${overallVerdict}.`, "ok");
  submitBtn.disabled = false;
};

const handleFailure = (snapshot) => {
  const errorText = snapshot.error || "Analysis failed.";
  setStatus(errorText, "error");
  submitBtn.disabled = false;
};

const pollStatus = async () => {
  if (!activeJobId) return;
  try {
    const response = await fetch(`/api/analyze/${activeJobId}`);
    const snapshot = await response.json();
    if (!response.ok) {
      throw new Error(snapshot.detail || "Failed to fetch job status.");
    }

    updateLivePanel(snapshot);

    if (snapshot.status === "completed") {
      stopPolling();
      handleCompletion(snapshot);
    } else if (snapshot.status === "failed") {
      stopPolling();
      handleFailure(snapshot);
    }
  } catch (err) {
    stopPolling();
    submitBtn.disabled = false;
    setStatus(err.message || "Polling failed.", "error");
  }
};

const startPolling = (jobId) => {
  activeJobId = jobId;
  stopPolling();
  pollStatus();
  pollTimer = setInterval(pollStatus, 1000);
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const youtubeUrl = inputEl.value.trim();

  if (!youtubeUrl) {
    setStatus("Please enter a YouTube URL.", "error");
    return;
  }

  stopPolling();
  activeJobId = null;
  submitBtn.disabled = true;
  resetLiveView();
  setStatus("Starting analysis job...", "ok");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ youtube_url: youtubeUrl }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to start analysis.");
    }

    const jobId = data.job_id;
    if (!jobId) {
      throw new Error("Backend did not return a job id.");
    }

    setStatus("Job accepted. Streaming live progress...", "ok");
    startPolling(jobId);
  } catch (err) {
    submitBtn.disabled = false;
    setStatus(err.message || "Failed to start analysis.", "error");
  }
});
