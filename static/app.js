/**
 * tpu-sim Web Client Application Logic
 */

let modelsData = {};
let hardwareData = {};
let currentSimulationResult = null;
let rooflineChart = null;
let activeChartType = "tradeoff"; // "tradeoff" or "latency"
let isSwappedAxes = false;

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
  document.getElementById("modelSelect").addEventListener("change", updateModelDetails);
  document.getElementById("hardwareSelect").addEventListener("change", updateHardwareDetails);

  // TPU Slider & Input sync
  const numTpusInput = document.getElementById("numTpus");
  const numTpusSlider = document.getElementById("numTpusSlider");

  numTpusSlider.addEventListener("input", (e) => {
    numTpusInput.value = e.target.value;
    autoSplitTpus();
  });
  numTpusInput.addEventListener("input", (e) => {
    numTpusSlider.value = e.target.value;
    autoSplitTpus();
  });

  document.getElementById("autoSplitBtn").addEventListener("click", autoSplitTpus);

  // Strategy change handler
  document.getElementById("strategySelect").addEventListener("change", (e) => {
    const customBox = document.getElementById("customFormulaBox");
    if (e.target.value === "custom") {
      customBox.classList.remove("hidden");
    } else {
      customBox.classList.add("hidden");
    }
  });

  // Run Simulation Button
  document.getElementById("runSimBtn").addEventListener("click", runSimulation);

  // Chart view toggles
  document.getElementById("toggleYAxisBtn").addEventListener("click", () => {
    isSwappedAxes = !isSwappedAxes;
    renderChart();
  });

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
    alert("Failed to connect to simulation server.");
  }
}

function updateUI() {
  if (!currentSimulationResult) return;

  const res = currentSimulationResult;

  // Update Summary Metrics
  document.getElementById("metricMaxInteractivity").textContent = res.max_interactivity_tps.toFixed(1);
  document.getElementById("metricMaxThroughput").textContent = res.max_throughput_tps.toLocaleString(undefined, { maximumFractionDigits: 0 });
  
  if (res.knee_point) {
    document.getElementById("metricKneeBatch").textContent = `B=${res.knee_point.batch_size}`;
  } else {
    document.getElementById("metricKneeBatch").textContent = "Compute Bound";
  }

  const batch1 = res.datapoints.find(d => d.batch_size === 1) || res.datapoints[0];
  if (batch1) {
    document.getElementById("metricTtft").textContent = batch1.ttft_ms.toFixed(1);
    document.getElementById("metricTpot").textContent = batch1.tpot_ms.toFixed(2);
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
      <td>${dp.interactivity_tps.toFixed(1)}</td>
      <td>${dp.output_throughput_tps.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
      <td>${dp.ttft_ms.toFixed(1)}</td>
      <td>${dp.tpot_ms.toFixed(2)}</td>
      <td>${dp.qps.toFixed(2)}</td>
      <td>${(dp.e2e_latency_ms / 1000.0).toFixed(2)}s</td>
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
    // Interactivity vs Throughput Tradeoff Curve
    const labels = datapoints.map(d => `B=${d.batch_size}`);
    
    let xData, yData, xTitle, yTitle;

    if (!isSwappedAxes) {
      xData = datapoints.map(d => d.interactivity_tps);
      yData = datapoints.map(d => d.output_throughput_tps);
      xTitle = "Interactivity (tokens/sec per user)";
      yTitle = "Output Throughput (tokens/sec total)";
    } else {
      xData = datapoints.map(d => d.output_throughput_tps);
      yData = datapoints.map(d => d.interactivity_tps);
      xTitle = "Output Throughput (tokens/sec total)";
      yTitle = "Interactivity (tokens/sec per user)";
    }

    const chartPoints = datapoints.map((d, i) => ({
      x: xData[i],
      y: yData[i],
      batch: d.batch_size,
      ttft: d.ttft_ms,
      tpot: d.tpot_ms,
      is_compute: d.decode_is_compute_bound
    }));

    rooflineChart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Throughput vs Interactivity Curve',
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
            title: { display: true, text: xTitle, color: '#8b949e', font: { size: 13, weight: 'bold' } },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          },
          y: {
            title: { display: true, text: yTitle, color: '#8b949e', font: { size: 13, weight: 'bold' } },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          }
        },
        plugins: {
          legend: { labels: { color: '#c9d1d9' } },
          tooltip: {
            callbacks: {
              title: (items) => `Batch Size: ${items[0].raw.batch}`,
              label: (item) => [
                `Interactivity: ${item.raw.x.toFixed(1)} tok/s/user`,
                `Throughput: ${item.raw.y.toLocaleString()} tok/s`,
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
    // Latency (TTFT and TPOT) Curve
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
            title: { display: true, text: 'Batch Size', color: '#8b949e' },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'TTFT (ms)', color: '#bc8cff' },
            grid: { color: '#30363d' },
            ticks: { color: '#c9d1d9' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: { display: true, text: 'TPOT (ms)', color: '#3fb950' },
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
  let csv = "Batch Size,Interactivity (tok/s/user),Output Throughput (tok/s),TTFT (ms),TPOT (ms),QPS,E2E Latency (s),State\n";
  datapoints.forEach(dp => {
    const state = dp.decode_is_compute_bound ? "Compute Bound" : "Memory Bound";
    csv += `${dp.batch_size},${dp.interactivity_tps},${dp.output_throughput_tps},${dp.ttft_ms},${dp.tpot_ms},${dp.qps},${(dp.e2e_latency_ms/1000).toFixed(2)},${state}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `tpu_roofline_simulation_${currentSimulationResult.config.model_name}.csv`;
  a.click();
}
