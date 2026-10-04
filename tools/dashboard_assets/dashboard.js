"use strict";

const $ = (id) => document.getElementById(id);
let eventSource = null;
let lastState = null;

function showMessage(message, isError = true) {
  $("setup-message").textContent = message || "";
  $("setup-message").style.color = isError ? "#b3261e" : "#087f78";
}

function mode() { return document.querySelector('input[name="mode"]:checked').value; }

function updateMode() {
  const pending = mode() === "pending";
  $("pending-settings").classList.toggle("hidden", !pending);
  $("model-settings").classList.toggle("hidden", pending);
  $("fast-window").disabled = pending;
  $("allow-unvalidated").disabled = pending;
  if (pending) { $("fast-window").checked = false; $("allow-unvalidated").checked = false; }
}

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function refreshDevices() {
  const transport = $("transport").value;
  $("scan").disabled = true;
  showMessage(transport === "ble" ? "Scanning for PPG-LOGGER devices…" : "Refreshing serial ports…", false);
  try {
    const payload = await jsonRequest(`/api/devices?transport=${encodeURIComponent(transport)}`);
    const select = $("device"); select.replaceChildren();
    payload.devices.forEach((device) => {
      const option = document.createElement("option");
      option.value = transport === "ble" ? (device.address || device.name) : device.address;
      option.textContent = device.name + (device.address && device.address !== device.name ? ` [${device.address}]` : "");
      select.append(option);
    });
    showMessage(payload.devices.length ? `Found ${payload.devices.length} device(s).` : "No matching devices found.", !payload.devices.length);
  } catch (error) { showMessage(error.message); }
  finally { $("scan").disabled = false; }
}

async function refreshModels() {
  const participant = $("participant").value.trim();
  $("find-models").disabled = true; showMessage("Searching local model packages…", false);
  try {
    const payload = await jsonRequest(`/api/models?participant_id=${encodeURIComponent(participant)}`);
    const select = $("model"); select.replaceChildren();
    payload.models.forEach((model) => {
      const option = document.createElement("option"); option.value = model.path;
      const gate = model.viewer_eligible ? "eligible" : "not validated";
      option.textContent = `${model.run_id} · ${gate} · ${model.calibration_sbp}/${model.calibration_dbp}`;
      select.append(option);
    });
    if (payload.models.length) $("model-path").value = payload.models[0].path;
    showMessage(payload.models.length ? `Found ${payload.models.length} model package(s).` : "No model found for this participant.", !payload.models.length);
  } catch (error) { showMessage(error.message); }
  finally { $("find-models").disabled = false; }
}

function connectionControls(active) {
  $("connect").disabled = active;
  $("stop").disabled = !active;
}

async function connect() {
  const selectedModel = $("model").value || "";
  const payload = {
    transport: $("transport").value,
    device: $("device").value || "",
    participant_id: $("participant").value.trim(),
    mode: mode(),
    model_dir: $("model-path").value.trim() || selectedModel,
    calibration_sbp: $("calibration-sbp").value,
    calibration_dbp: $("calibration-dbp").value,
    fast_window: $("fast-window").checked,
    allow_unvalidated: $("allow-unvalidated").checked,
  };
  showMessage("Connecting…", false); connectionControls(true);
  try {
    await jsonRequest("/api/connect", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    showMessage("Connection started.", false); startEvents();
  } catch (error) { showMessage(error.message); connectionControls(false); }
}

async function stop() {
  $("stop").disabled = true; showMessage("Stopping and releasing the device…", false);
  try { await jsonRequest("/api/disconnect", {method: "POST"}); }
  catch (error) { showMessage(error.message); }
}

function formatAge(value) {
  if (value == null) return "--";
  const seconds = Math.max(0, Math.floor(value));
  return seconds < 60 ? `${seconds} s ago` : `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, "0")} s ago`;
}
function formatNumber(value, digits = 1, suffix = "") { return value == null ? "--" : `${Number(value).toFixed(digits)}${suffix}`; }
function addDefinition(list, name, value) { const dt = document.createElement("dt"); dt.textContent = name; const dd = document.createElement("dd"); dd.textContent = value ?? "--"; list.append(dt, dd); }

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1; const width = Math.max(200, canvas.clientWidth);
  const height = Number(canvas.getAttribute("height")) || 220;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) { canvas.width = width * ratio; canvas.height = height * ratio; }
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); return {ctx, width, height};
}

function chartFrame(ctx, width, height, minY, maxY) {
  ctx.clearRect(0, 0, width, height); ctx.strokeStyle = "#dce5e1"; ctx.lineWidth = 1; ctx.font = "11px Segoe UI"; ctx.fillStyle = "#607080";
  for (let i = 0; i <= 4; i++) { const y = 15 + i * (height - 40) / 4; ctx.beginPath(); ctx.moveTo(42, y); ctx.lineTo(width - 10, y); ctx.stroke(); const value = maxY - i * (maxY - minY) / 4; ctx.fillText(value.toFixed(0), 3, y + 4); }
}

function drawWaveform(snapshot) {
  const canvas = $("waveform"), {ctx, width, height} = resizeCanvas(canvas);
  const xs = snapshot?.waveform_x_s || [], ys = snapshot?.waveform_ir || [];
  $("waveform-empty").classList.toggle("hidden", ys.length > 1);
  if (ys.length < 2) { ctx.clearRect(0, 0, width, height); return; }
  let minY = Math.min(...ys), maxY = Math.max(...ys); if (minY === maxY) { minY -= 1; maxY += 1; }
  chartFrame(ctx, width, height, minY, maxY); ctx.strokeStyle = "#087f78"; ctx.lineWidth = 1.6; ctx.beginPath();
  ys.forEach((value, index) => { const x = 42 + ((xs[index] + 15) / 15) * (width - 52); const y = 15 + (maxY - value) / (maxY - minY) * (height - 40); index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
}

function drawHistory(history) {
  const canvas = $("history"), {ctx, width, height} = resizeCanvas(canvas); $("history-empty").classList.toggle("hidden", history.length > 0);
  const points = $("history-points"); points.replaceChildren();
  history.slice(-5).reverse().forEach((point) => { const li = document.createElement("li"); const acceptedAt = new Date(point.received_at).toLocaleTimeString(); const value = document.createElement("strong"); value.textContent = `${Math.round(point.sbp)}/${Math.round(point.dbp)} mmHg`; li.append(value, ` · ${acceptedAt}`); points.append(li); });
  if (!history.length) { ctx.clearRect(0, 0, width, height); return; }
  const values = history.flatMap((point) => [point.sbp, point.dbp]); let minY = Math.floor(Math.min(...values) / 10) * 10 - 5; let maxY = Math.ceil(Math.max(...values) / 10) * 10 + 5;
  chartFrame(ctx, width, height, minY, maxY);
  const drawPoints = (key, color) => { ctx.fillStyle = color; history.forEach((point, index) => { const x = history.length === 1 ? width / 2 : 42 + index / (history.length - 1) * (width - 52); const y = 15 + (maxY - point[key]) / (maxY - minY) * (height - 40); ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); }); };
  drawPoints("sbp", "#1e5ba8"); drawPoints("dbp", "#087f78");
}

function renderTechnical(state, snapshot) {
  const session = $("session-details"); session.replaceChildren();
  addDefinition(session, "Participant", state.session.participant_id); addDefinition(session, "Transport", state.session.transport); addDefinition(session, "Device", snapshot?.device || state.session.device); addDefinition(session, "Model", snapshot?.model_eligibility); addDefinition(session, "Calibration", state.session.calibration_sbp ? `${state.session.calibration_sbp}/${state.session.calibration_dbp} mmHg` : "--"); addDefinition(session, "Policy", state.session.fast_window ? "Experimental 30 s" : "Standard 85 s");
  const health = $("sensor-health"); health.replaceChildren(); addDefinition(health, "PPG rate", formatNumber(snapshot?.ppg_rate_hz, 1, " Hz")); addDefinition(health, "IMU rate", formatNumber(snapshot?.imu_rate_hz, 1, " Hz")); addDefinition(health, "PPG I²C / FIFO", `${snapshot?.ppg_i2c_errors ?? "--"} / ${snapshot?.ppg_fifo_overflows ?? "--"}`); addDefinition(health, "IMU I²C / FIFO", `${snapshot?.imu_i2c_errors ?? "--"} / ${snapshot?.imu_fifo_overflows ?? "--"}`);
  const ble = $("ble-health"); ble.replaceChildren(); addDefinition(ble, "Connected", snapshot?.ble_connected); addDefinition(ble, "Subscribed", snapshot?.ble_subscribed); addDefinition(ble, "MTU", snapshot?.ble_mtu); addDefinition(ble, "Dropped records", snapshot?.ble_dropped_records); addDefinition(ble, "Notify errors", snapshot?.ble_notify_errors);
  const warnings = $("warnings"); warnings.replaceChildren(); const items = snapshot?.warnings?.length ? snapshot.warnings : ["None"]; items.forEach((warning) => { const li = document.createElement("li"); li.textContent = warning; warnings.append(li); });
}

function render(state) {
  lastState = state; const snapshot = state.snapshot; const connection = state.connection_state;
  $("connection-dot").className = `status-dot ${connection}`; $("connection-title").textContent = connection === "connected" ? "Receiving sensor data" : connection === "connecting" ? "Connecting" : connection === "reconnecting" ? "Wearable disconnected" : connection === "error" ? "Connection error" : "Not connected";
  $("connection-detail").textContent = state.error || snapshot?.connection || state.session.device || "Choose a device below"; connectionControls(["connecting", "connected", "reconnecting"].includes(connection));
  if (state.error) showMessage(state.error);

  const card = $("bp-card"); card.className = `panel bp-card ${snapshot?.mode || "unavailable"}`;
  const numeric = snapshot && snapshot.mode !== "unavailable" && snapshot.sbp != null && snapshot.dbp != null;
  $("sbp").textContent = numeric ? Math.round(snapshot.sbp) : "--"; $("dbp").textContent = numeric ? Math.round(snapshot.dbp) : "--";
  const modeLabels = {current: "Current estimate", held: "Last validated", unavailable: "Unavailable"}; $("bp-mode").textContent = modeLabels[snapshot?.mode] || "Unavailable";
  $("bp-age").textContent = numeric ? `Estimate age: ${formatAge(snapshot.age_s)}` : "No validated estimate available";
  $("bp-status").textContent = snapshot?.status || (connection === "error" ? "Connection error" : "Waiting for connection");
  $("bp-reason").textContent = snapshot ? (snapshot.mode === "held" ? snapshot.current_reason : snapshot.reason) : (state.error || "Connect the wearable to begin.");
  const warning = $("experimental-warning"); warning.textContent = snapshot?.experimental_warning || ""; warning.classList.toggle("hidden", !warning.textContent);

  const target = snapshot?.target_s || 85, buffer = snapshot?.buffer_s || 0; $("buffer-label").textContent = `${buffer.toFixed(1)} / ${target.toFixed(0)} s`; $("buffer-progress").style.width = `${Math.min(100, buffer / target * 100)}%`;
  $("quality-state").textContent = snapshot?.current_status || "Waiting"; $("accepted-windows").textContent = snapshot ? `${snapshot.accepted_windows} / ${snapshot.total_windows}` : "--"; $("clean-coverage").textContent = snapshot ? `${snapshot.clean_coverage_s.toFixed(1)} s` : "--"; $("pulse-rate").textContent = snapshot?.pulse_rate_bpm != null ? `${snapshot.pulse_rate_bpm.toFixed(1)} BPM` : "-- BPM";

  const intensity = (snapshot?.intensity || "Unknown").toLowerCase(); const moving = snapshot?.motion?.toLowerCase() === "moving" || !["unknown", "still", "low"].includes(intensity); const severe = ["high", "severe"].includes(intensity); $("motion-state").textContent = snapshot?.motion || "Waiting"; $("motion-intensity").textContent = `${snapshot?.intensity || "Unknown"} intensity`; $("motion-value").textContent = snapshot?.intensity_g != null ? `Activity ${snapshot.intensity_g.toFixed(3)} g` : "Activity unavailable"; $("motion-icon").textContent = moving ? "≈" : "○"; $("motion-guidance").textContent = moving ? "No new BP estimate is accepted during movement." : "Remain still while a clean measurement window is collected."; document.querySelector(".motion-card").className = `panel motion-card ${severe ? "severe" : moving ? "moving" : "still"}`;
  drawWaveform(snapshot); drawHistory(state.history || []); renderTechnical(state, snapshot);
}

function startEvents() {
  if (eventSource) eventSource.close(); eventSource = new EventSource("/api/events");
  eventSource.addEventListener("state", (event) => render(JSON.parse(event.data)));
  eventSource.onerror = () => { $("connection-detail").textContent = "Dashboard event stream reconnecting…"; };
}

$("transport").addEventListener("change", refreshDevices); $("scan").addEventListener("click", refreshDevices); $("find-models").addEventListener("click", refreshModels); $("model").addEventListener("change", () => { $("model-path").value = $("model").value; }); document.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", updateMode)); $("connect").addEventListener("click", connect); $("stop").addEventListener("click", stop); $("reset-history").addEventListener("click", async () => { try { await jsonRequest("/api/history/reset", {method: "POST"}); } catch (error) { showMessage(error.message); } }); window.addEventListener("resize", () => { if (lastState) render(lastState); });

updateMode(); refreshDevices(); refreshModels(); startEvents(); fetch("/api/state").then((response) => response.json()).then(render).catch(() => {});
