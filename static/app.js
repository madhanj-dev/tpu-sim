/**
 * tpu-sim Web Client Application Logic
 */

let modelsData = {};
let hardwareData = {};
let currentSimulationResult = null;
let rooflineChart = null;
let activeChartType = "tradeoff"; // "tradeoff" (Throughput/s/chip vs Throughput/s/user) or "latency"

document.addEventListener("DOMContentLoaded", async () => {
  await loadInitialData();
  setupEventListeners();
  runSimulation();
});

async function loadInitialData() {
  try {
    const [modelsRes, hwRes] = await Promise.all([
      fetch("/api/models").then(r => r.json()),
      fetch("/api/hardware").then(r => r.json())
    ]);

    modelsData = modelsRes;
    hardwareData = hwRes;

    populateModelSelect();
    populateHardwareSelect();
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

function populateModelSelect() {
  const select = document.getElementById("modelSelect");
  select.innerHTML = "";
  for (const [key, model] of Object.entries(modelsData)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = model.display_name;
    if (key === "qwen3-397b") opt.selected = true;
    select.appendChild(opt);
  }
  updateModelDetails();
}

function populateHardwareSelect() {
  const select = document.getElementById("hardwareSelect");
  select.innerHTML = "";
  for (const [key, hw] of Object.entries(hardwareData)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = hw.display_name;
    if (key === "tpu-trillium") opt.selected = true;
    select.appendChild(opt);
  }
  updateHardwareDetails();
}

function updateModelDetails() {
  const key = document.getElementById("modelSelect").value;
  const m = modelsData[key];
  if (!m) return;
  const container = document.getElementById("modelDetails");
  container.innerHTML = `
    <strong>Active Params:</strong> ${m.active_params_b}B | 
    <strong>Total:</strong> ${m.total_params_b}B | 
    <strong>Type:</strong> ${m.is_moe ? 'MoE' : 'Dense'}<br>
    <em>${m.description}</em>
  `;
}

function updateHardwareDetails() {
  const key = document.getElementById("hardwareSelect").value;
  const hw = hardwareData[key];
  if (!hw) return;
  const container = document.getElementById("hardwareDetails");
  container.innerHTML = `
    <strong>Compute:</strong> ${hw.tflops} TFLOPS | 
    <strong>Memory Bandwidth:</strong> ${hw.memory_bw_tbs} TB/s | 
    <strong>HBM:</strong> ${hw.hbm_gb} GB<br>
    <em>${hw.description}</em>
  `;
}

function setupEventListeners() {
  // Model/Hardware selection changes trigger instant readout simulation
  document.getElementById("modelSelect").addEventListener("change", () => {
    updateModelDetails();
    runSimulation();
  });
  document.getElementById("hardwareSelect").addEventListener("change", () => {
    updateHardwareDetails();
    runSimulation();
  });

  // TPU Slider & Input sync
  const numTpusInput = document.getElementById("numTpus");
  const numTpusSlider = document.getElementById("numTpusSlider");

  numTpusSlider.addEventListener("input", (e) => {
    numTpusInput.value = e.target.value;
    autoSplitTpus();
    runSimulation();
  });
  numTpusInput.addEventListener("input", (e) => {
    numTpusSlider.value = e.target.value;
    autoSplitTpus();
    runSimulation();
  });

  document.getElementById("autoSplitBtn").addEventListener("click", () => {
    autoSplitTpus();
    runSimulation();
  });

  // Strategy change handler
  document.getElementById("strategySelect").addEventListener("change", (e) => {
    const customBox = document.getElementById("customFormulaBox");
    if (e.target.value === "custom") {
      customBox.classList.remove("hidden");
    } else {
      customBox.classList.add("hidden");
    }
    runSimulation();
  });

  // Run Simulation Button
  document.getElementById("runSimBtn").addEventListener("click", runSimulation);

  // Chart view toggle
  document.getElementById("viewLatencyChartBtn").addEventListener("click", () => {
    activeChartType = activeChartType === "tradeoff" ? "latency" : "tradeoff";
    document.getElementById("viewLatencyChartBtn").textContent = 
      activeChartType === "tradeoff" ? "View Latency Curve" : "View Tradeoff Curve";
    renderChart();
  });

  // Export CSV
  document.getElementById("exportCsvBtn").addEventListener("click", exportCsv);
}

function autoSplitTpus() {
  const total = parseInt(document.getElementById("numTpus").value) || 16;
  const prefill = Math.max(1, Math.floor(total / 2));
  const decode = Math.max(1, total - prefill);
  document.getElementById("prefillTpus").value = prefill;
  document.getElementById("decodeTpus").value = decode;
}

async function runSimulation() {
  const modelKey = document.getElementById("modelSelect").value;
  const hwKey = document.getElementById("hardwareSelect").value;
  const numTpus = parseInt(document.getElementById("numTpus").value) || 16;
  const prefillTpus = parseInt(document.getElementById("prefillTpus").value) || 8;
  const decodeTpus = parseInt(document.getElementById("decodeTpus").value) || 8;
  const inputLen = parseInt(document.getElementById("inputLen").value) || 2048;
  const outputLen = parseInt(document.getElementById("outputLen").value) || 512;
  const bytesPerParam = parseFloat(document.getElementById("precisionSelect").value) || 2.0;
  const strategyName = document.getElementById("strategySelect").value;
  const includeKvCache = document.getElementById("includeKvCache").checked;

  const requestBody = {
    model: modelKey,
    hardware: hwKey,
    num_tpus: numTpus,
    prefill_tpus: prefillTpus,
    decode_tpus: decodeTpus,
    input_len: inputLen,
    output_len: outputLen,
    bytes_per_param: bytesPerParam,
    strategy_name: strategyName,
    include_kv_cache: includeKvCache,
  };

  if (strategyName === "custom") {
    requestBody.custom_prefill_formula = document.getElementById("customPrefillFormula").value.trim() || null;
    requestBody.custom_decode_formula = document.getElementById("customDecodeFormula").value.trim() || null;
  }

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!res.ok) {
      const err = await res.json();
      alert("Simulation error: " + (err.error || "Unknown error"));
      return;
    }

    currentSimulationResult = await res.json();
    updateUI();
  } catch (err) {
    console.error("Simulation request failed:", err);
  }
}

function updateUI() {
  if (!currentSimulationResult) return;

  const res = currentSimulationResult;

  // Cache indicator badge
  const cacheBadge = document.getElementById("cacheBadge");
  if (res.is_precomputed) {
    cacheBadge.textContent = "⚡ Instant Precomputed Readout";
    cacheBadge.className = "badge badge-cache";
  } else {
    cacheBadge.textContent = "🔄 Live Simulated";
    cacheBadge.className = "badge badge-api";
  }

  // Update Summary Metrics
  const maxPerChip = res.max_throughput_per_chip_tps || (res.max_throughput_tps / res.config.num_tpus);
  document.getElementById("metricMaxThroughputPerChip").textContent = maxPerChip.toLocaleString(undefined, { maximumFractionDigits: 1 });
  document.getElementById("metricMaxInteractivity").textContent = res.max_interactivity_tps.toFixed(1);
  document.getElementById("metricMaxThroughput").textContent = res.max_throughput_tps.toLocaleString(undefined, { maximumFractionDigits: 0 });
  
  if (res.knee_point) {
    document.getElementById("metricKneeBatch").textContent = `B=${res.knee_point.batch_size}`;
  } else {
    document.getElementById("metricKneeBatch").textContent = "Compute Bound";
  }

  const batch1 = res.datapoints.find(d => d.batch_size === 1) || res.datapoints[0];
  if (batch1) {
    document.getElementById("metricTtftTpot").textContent = `${batch1.ttft_ms.toFixed(1)} / ${batch1.tpot_ms.toFixed(2)}`;
  }

  // Populate Data Table
  populateDataTable(res.datapoints, res.knee_point);

  // Render Chart
  renderChart();
}

function populateDataTable(datapoints, kneePoint) {
  const tbody = document.getElementById("dataTableBody");
  tbody.innerHTML = "";

  datapoints.forEach(dp => {
    const tr = document.createElement("tr");
    if (kneePoint && dp.batch_size === kneePoint.batch_size) {
      tr.classList.add("knee-row");
    }

    const stateTag = dp.decode_is_compute_bound 
      ? '<span class="state-tag state-compute">Compute Bound</span>'
      : '<span class="state-tag state-mem">Memory BW Bound</span>';

    tr.innerHTML = `
      <td><strong>${dp.batch_size}</strong> ${kneePoint && dp.batch_size === kneePoint.batch_size ? '⚡ (Knee)' : ''}</td>
      <td><strong>${dp.throughput_per_chip_tps.toLocaleString(undefined, { maximumFractionDigits: 1 })}</strong></td>
      <td>${dp.interactivity_tps.toFixed(1)}</td>
      <td>${dp.output_throughput_tps.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
      <td>${dp.ttft_ms.toFixed(1)}</td>
      <td>${dp.tpot_ms.toFixed(2)}</td>
      <td>${dp.qps.toFixed(2)}</td>
      <td>${stateTag}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderChart() {
  if (!currentSimulationResult) return;

  const ctx = document.getElementById("rooflineChart").getContext("2d");
  if (rooflineChart) {
    rooflineChart.destroy();
  }

  const datapoints = currentSimulationResult.datapoints;

  if (activeChartType === "tradeoff") {
    // Exact requested format:
    // Y-Axis: Throughput / s / chip (tokens/sec/chip)
    // X-Axis: Throughput / s / user (tokens/sec/user)
    // Curve traced out by varying Concurrencies (batch sizes)

    const chartPoints = datapoints.map(d => ({
      x: d.interactivity_tps,              // X-axis: Throughput / s / user
      y: d.throughput_per_chip_tps,        // Y-axis: Throughput / s / chip
      concurrency: d.batch_size,
      ttft: d.ttft_ms,
      tpot: d.tpot_ms,
      total_throughput: d.output_throughput_tps,
      is_compute: d.decode_is_compute_bound
    }));

    rooflineChart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Throughput / s / chip vs Throughput / s / user',
          data: chartPoints,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.15)',
          pointBackgroundColor: datapoints.map(d => d.decode_is_compute_bound ? '#3fb950' : '#d29922'),
          pointRadius: 6,
          pointHoverRadius: 9,
          borderWidth: 3,
          tension: 0.2,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Throughput / s / user (tokens/sec/user)', color: '#bc8cff', font: { size: 13, weight: 'bold' } },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          },
          y: {
            type: 'linear',
            title: { display: true, text: 'Throughput / s / chip (tokens/sec/chip)', color: '#58a6ff', font: { size: 13, weight: 'bold' } },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          }
        },
        plugins: {
          legend: { labels: { color: '#c9d1d9' } },
          tooltip: {
            callbacks: {
              title: (items) => `Concurrency (Batch Size): ${items[0].raw.concurrency}`,
              label: (item) => [
                `Throughput / s / chip: ${item.raw.y.toLocaleString(undefined, { maximumFractionDigits: 1 })} tok/s/chip`,
                `Throughput / s / user: ${item.raw.x.toFixed(1)} tok/s/user`,
                `Total Output Throughput: ${item.raw.total_throughput.toLocaleString(undefined, { maximumFractionDigits: 0 })} tok/s`,
                `TPOT: ${item.raw.tpot.toFixed(2)} ms`,
                `TTFT: ${item.raw.ttft.toFixed(1)} ms`,
                `State: ${item.raw.is_compute ? 'Compute Bound' : 'Memory Bandwidth Bound'}`
              ]
            }
          }
        }
      }
    });

  } else {
    // Latency (TTFT and TPOT) vs Concurrency
    const labels = datapoints.map(d => d.batch_size);

    rooflineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'TTFT - Time To First Token (ms)',
            data: datapoints.map(d => d.ttft_ms),
            borderColor: '#bc8cff',
            backgroundColor: 'rgba(188, 140, 255, 0.2)',
            borderWidth: 2,
            yAxisID: 'y'
          },
          {
            label: 'TPOT - Time Per Output Token (ms)',
            data: datapoints.map(d => d.tpot_ms),
            borderColor: '#3fb950',
            backgroundColor: 'rgba(63, 185, 80, 0.2)',
            borderWidth: 2,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Concurrency (Batch Size)', color: '#8b949e', font: { size: 13, weight: 'bold' } },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'TTFT (ms)', color: '#bc8cff', font: { size: 13, weight: 'bold' } },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: { display: true, text: 'TPOT (ms)', color: '#3fb950', font: { size: 13, weight: 'bold' } },
            grid: { drawOnChartArea: false },
            ticks: { color: '#c9d1d9' }
          }
        },
        plugins: {
          legend: { labels: { color: '#c9d1d9' } }
        }
      }
    });
  }
}

function exportCsv() {
  if (!currentSimulationResult) return;
  const datapoints = currentSimulationResult.datapoints;
  let csv = "Concurrency (Batch),Throughput / s / chip (tok/s/chip),Throughput / s / user (tok/s/user),Total System Throughput (tok/s),TTFT (ms),TPOT (ms),QPS,State\n";
  datapoints.forEach(dp => {
    const state = dp.decode_is_compute_bound ? "Compute Bound" : "Memory Bound";
    csv += `${dp.batch_size},${dp.throughput_per_chip_tps},${dp.interactivity_tps},${dp.output_throughput_tps},${dp.ttft_ms},${dp.tpot_ms},${dp.qps},${state}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `tpu_roofline_simulation_${currentSimulationResult.config.model_name}.csv`;
  a.click();
}
