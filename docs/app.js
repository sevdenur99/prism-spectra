// Aetheris v2.0 - Complete Three-Stage Spectral Pipeline Client
// Models: 1. PRISM-Theta -> 2. MODULATOR-Aging -> 3. MODULATOR-Ambient

const WAVELENGTHS = Array.from({ length: 401 }, (_, i) => 380 + i);

// CIE Standart Eğrileri (CIE S 026 s_mel ve CIE 1931 V_lambda)
const V_LAMBDA = WAVELENGTHS.map(l => Math.exp(-0.5 * Math.pow((l - 555) / 44, 2)));
const S_MEL = WAVELENGTHS.map(l => Math.exp(-0.5 * Math.pow((l - 490) / 38, 2)));
const EDI_WEIGHTS = S_MEL.map(s => s * (683.002 / 0.90564) * (1.0 / 401));

// Popüler Oyun Presets
const GAME_PRESETS = {
  "cyberpunk": { name: "Cyberpunk 2077 (Neon Gece)", r: 15, g: 140, b: 255, x: 0.5, y: 0.5, theta: 15, hours: 6000, prof: "S3", scen: "E2", bright: 0.8, ambLux: 20, ambCct: 2800, color: "#06b6d4" },
  "valorant": { name: "Valorant (Icebox A-Site)", r: 185, g: 225, b: 255, x: 0.5, y: 0.5, theta: 0, hours: 3500, prof: "S2", scen: "E1", bright: 0.7, ambLux: 80, ambCct: 4000, color: "#38bdf8" },
  "cs2": { name: "Counter-Strike 2 (Mirage)", r: 235, g: 195, b: 115, x: 0.5, y: 0.5, theta: 20, hours: 8000, prof: "S2", scen: "E1", bright: 0.65, ambLux: 60, ambCct: 3200, color: "#f59e0b" },
  "lol": { name: "League of Legends (Nehir)", r: 35, g: 195, b: 170, x: 0.5, y: 0.5, theta: 10, hours: 4500, prof: "S2", scen: "E1", bright: 0.6, ambLux: 40, ambCct: 3500, color: "#10b981" },
  "gtav": { name: "GTA V (Gün Batımı)", r: 255, g: 95, b: 30, x: 0.5, y: 0.5, theta: 25, hours: 2500, prof: "S1", scen: "E1", bright: 0.5, ambLux: 15, ambCct: 2400, color: "#f97316" },
  "eldenring": { name: "Elden Ring (Loş Zindan)", r: 50, g: 35, b: 20, x: 0.5, y: 0.5, theta: 30, hours: 12000, prof: "S3", scen: "E3", bright: 0.4, ambLux: 5, ambCct: 2200, color: "#78716c" }
};

// State Yönetimi
const state = {
  // Model 1: PRISM Theta
  r: 15, g: 140, b: 255, x: 0.5, y: 0.5, theta: 15,
  // Model 2: Yaşlanma
  calendarHours: 5000, profile: "S2", scenario: "E1", brightness: 0.6,
  // Model 3: Ortam & Biyoloji
  ambientLux: 30, ambientCct: 3000, distanceCm: 50, observerAge: 25, durationHours: 3.0,
  // Görünürlük Çipleri
  showSpd0: true,
  showSpdAging: true,
  showSpdCornea: true,
  // Hugging Face Endpoints
  endpoints: {
    theta: localStorage.getItem("hf_endpoint_theta") || "",
    aging: localStorage.getItem("hf_endpoint_aging") || "",
    ambient: localStorage.getItem("hf_endpoint_ambient") || "",
    unified: localStorage.getItem("hf_endpoint_unified") || "https://sevden123-prism-spectra.hf.space"
  },
  // Son Hesaplama Verileri
  results: null
};

// ─── 1. MATEMATİKSEL İSTEMCİ MOTORU (FALLBACK / OFFLINE SİMÜLATÖR) ──────────

function getLensTransmittance(age) {
  const ageClamped = Math.max(10, Math.min(80, age));
  return WAVELENGTHS.map(l => {
    let od1 = 0;
    let od2 = 0.015;
    if (l <= 500) {
      od1 = 1.4 * Math.exp(-Math.pow((l - 400) / 30, 2)) + 0.12 * Math.exp(-Math.pow((l - 470) / 40, 2));
      od2 = 0.28 * Math.exp(-Math.pow((l - 400) / 45, 2)) + 0.015;
    } else if (l <= 550) {
      od1 = 0.01 * Math.exp(-Math.pow((l - 500) / 25, 2));
    }
    const ageScale = ageClamped <= 30 
      ? 1.0 + 0.02 * (ageClamped - 32)
      : 1.0 + 0.02 * (ageClamped - 32) + 0.0006 * Math.pow(ageClamped - 32, 2);
    const totalOd = od1 * Math.max(0.1, ageScale) + od2;
    return Math.max(0.01, Math.min(1.0, Math.pow(10, -totalOd)));
  });
}

function generateAmbientSpd(lux, cct) {
  if (lux <= 0) return new Array(401).fill(0);
  const c1 = 3.7418e-16;
  const c2 = 1.4388e-2;
  const rawPlanck = WAVELENGTHS.map(l => {
    const lamM = l * 1e-9;
    return (c1 / Math.pow(lamM, 5)) / (Math.exp(c2 / (lamM * cct)) - 1.0);
  });
  let vSum = 0;
  for (let i = 0; i < 401; i++) vSum += rawPlanck[i] * V_LAMBDA[i] * 683.002;
  const norm = vSum > 0 ? lux / vSum : 0;
  return rawPlanck.map(val => val * norm);
}

function computeClientPipeline() {
  const t0 = performance.now();

  // ADIM 1: PRISM-Theta
  const thetaRad = (state.theta * Math.PI) / 180;
  const angularScale = Math.cos(thetaRad);
  const angleShift = Math.sin(thetaRad) * 4.5;

  const spd0 = WAVELENGTHS.map(l => {
    const shiftedL = l - angleShift;
    const bp = (state.b / 255) * 0.95 * Math.exp(-0.5 * Math.pow((shiftedL - 452) / 13, 2));
    const gp = (state.g / 255) * 0.75 * Math.exp(-0.5 * Math.pow((shiftedL - 530) / 28, 2));
    const rp = (state.r / 255) * 0.85 * Math.exp(-0.5 * Math.pow((shiftedL - 625) / 18, 2));
    const base = 0.015 + bp + gp + rp;
    return Math.max(0, base * angularScale);
  });

  // ADIM 2: Yaşlanma Modülatörü (ATLAS Model)
  const tauMap = { "S1": 60000, "S2": 30000, "S3": 15000 };
  const betaMap = { "S1": 0.70, "S2": 0.60, "S3": 0.50 };
  const kEnvMap = { "E1": 1.0, "E2": 1.3, "E3": 1.7 };

  const tau = tauMap[state.profile] || 30000;
  const beta = betaMap[state.profile] || 0.60;
  const kEnv = kEnvMap[state.scenario] || 1.0;

  const bScale = Math.pow(state.brightness / 0.5, 1.5);
  const tEffTot = kEnv * state.calendarHours * bScale;
  const gT = Math.exp(-Math.pow(tEffTot / tau, beta));

  const spdAging = spd0.map((val, i) => {
    const l = WAVELENGTHS[i];
    const blueWeight = Math.exp(-0.5 * Math.pow((l - 450) / 35, 2));
    const specGt = gT * (1.0 - 0.25 * (1.0 - gT) * blueWeight);
    return Math.max(0, val * specGt);
  });

  // ADIM 3: Ortam Aydınlatması ve Sirkadiyen Biyoloji
  const distScale = Math.pow(50 / state.distanceCm, 2);
  const screenCorneal = spdAging.map(v => v * distScale * 0.01);
  const ambientSpd = generateAmbientSpd(state.ambientLux, state.ambientCct);
  const lensTrans = getLensTransmittance(state.observerAge);

  const cornealSpd = [];
  let photopicLux = 0;
  let melanopicEdi = 0;

  for (let i = 0; i < 401; i++) {
    const totalCorneal = screenCorneal[i] + ambientSpd[i];
    cornealSpd.push(totalCorneal);
    photopicLux += totalCorneal * V_LAMBDA[i] * 683.002;
    melanopicEdi += totalCorneal * EDI_WEIGHTS[i];
  }

  const mpRatio = photopicLux > 0.001 ? melanopicEdi / photopicLux : 0;
  const cla = 1.548 * melanopicEdi;
  const cs = cla > 0 ? 0.7 * (1.0 - 1.0 / (1.0 + Math.pow(cla / 355.7, 1.1026))) : 0;

  const i50 = 30.0 + 70.0 / Math.pow(state.durationHours, 0.5);
  const suppression = (100.0 / (1.0 + Math.pow(i50 / (melanopicEdi + 1e-6), 1.0)));
  const phaseShift = melanopicEdi > 0 
    ? 3.0 * (Math.pow(melanopicEdi, 0.5) / (Math.pow(melanopicEdi, 0.5) + Math.pow(60.0, 0.5)))
    : 0;

  const t1 = performance.now();

  return {
    spd0,
    spdAging,
    cornealSpd,
    gT,
    angularScale,
    photopicLux: Math.round(photopicLux * 10) / 10,
    melanopicEdi: Math.round(melanopicEdi * 10) / 10,
    mpRatio: Math.round(mpRatio * 1000) / 1000,
    cs: Math.round(cs * 1000) / 1000,
    suppression: Math.round(suppression * 10) / 10,
    phaseShift: Math.round(phaseShift * 100) / 100,
    latencyMs: Math.round((t1 - t0) * 100) / 100,
    isHfActive: false
  };
}

// ─── 2. HUGGING FACE ENDPOINT ÇAĞRILARI (ASENKRON & GRADIO QUEUE PROTOKOLÜ) ──

async function callGradioPipeline(baseUrl, params) {
  let host = (baseUrl || "").trim();
  const hfMatch = host.match(/huggingface\.co\/spaces\/([^\/]+)\/([^\/\?#]+)/i);
  if (hfMatch) {
    const user = hfMatch[1].toLowerCase().replace(/_/g, "-");
    const space = hfMatch[2].toLowerCase().replace(/_/g, "-");
    host = `https://${user}-${space}.hf.space`;
  }
  host = host.replace(/\/+$/, "").replace(/\/predict(\/pipeline)?$/, "").replace(/\/call(\/predict_pipeline)?$/, "");

  // Gradio 4 /call/predict_pipeline kuyruk çağrısı
  const callRes = await fetch(`${host}/call/predict_pipeline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: [
        Number(params.r), Number(params.g), Number(params.b),
        Number(params.x), Number(params.y), Number(params.theta),
        Number(params.calendarHours), String(params.profile), String(params.scenario),
        Number(params.brightness), Number(params.ambientLux), Number(params.ambientCct),
        Number(params.distanceCm), Number(params.observerAge), Number(params.durationHours)
      ]
    })
  });

  if (!callRes.ok) throw new Error(`Gradio HTTP Call ${callRes.status}`);
  const { event_id } = await callRes.json();
  if (!event_id) throw new Error("Gradio event_id alınamadı");

  // SSE event sonucunu al
  const eventRes = await fetch(`${host}/call/predict_pipeline/${event_id}`);
  if (!eventRes.ok) throw new Error(`Gradio HTTP Event ${eventRes.status}`);
  const eventText = await eventRes.text();

  const dataLine = eventText.split("\n").find(l => l.startsWith("data:"));
  if (!dataLine) throw new Error("Gradio SSE verisi çözümlenemedi");

  const rawArr = JSON.parse(dataLine.slice(5).trim());
  const parsed = typeof rawArr[2] === "string" ? JSON.parse(rawArr[2]) : rawArr[2];
  return parsed;
}

async function runPipeline() {
  const unifiedHost = state.endpoints.unified ? state.endpoints.unified.trim() : "";

  if (unifiedHost !== "") {
    try {
      const data = await callGradioPipeline(unifiedHost, {
        r: state.r, g: state.g, b: state.b, x: state.x, y: state.y, theta: state.theta,
        calendarHours: state.calendarHours, profile: state.profile, scenario: state.scenario, brightness: state.brightness,
        ambientLux: state.ambientLux, ambientCct: state.ambientCct, distanceCm: state.distanceCm,
        observerAge: state.observerAge, durationHours: state.durationHours
      });

      const photopic = Number(data.photopic_lux || 0);
      const medi = Number(data.melanopic_edi || 0);
      const mp = photopic > 0 ? (medi / photopic) : 0;

      updateUIWithData({
        spd0: data.spd_0,
        spdAging: data.spd_aging,
        cornealSpd: data.corneal_spd,
        gT: Number(data.g_t || 1.0),
        angularScale: Number(data.angular_scale || 1.0),
        photopicLux: photopic,
        melanopicEdi: medi,
        mpRatio: mp,
        cs: Number(data.circadian_stimulus_cs || 0),
        suppression: Number(data.melatonin_suppression_pct || 0),
        phaseShift: Number(data.phase_shift_hours || 0),
        latencyMs: Number(data.latency_ms || 1.0),
        isHfActive: true
      });
      return;
    } catch (err) {
      console.warn("Hugging Face Canlı API çağrısı başarısız, yerel simülatöre geçiliyor:", err);
    }
  }

  // Fallback Yerel İstemci Simülatörü
  const localData = computeClientPipeline();
  updateUIWithData(localData);
}

// ─── 3. UI VE GRAFİK GÜNCELLEME ─────────────────────────────────────────────

function updateUIWithData(data) {
  state.results = data;

  // Sayısal Değerler
  document.getElementById("valMedi").textContent = data.melanopicEdi.toFixed(1);
  document.getElementById("valCs").textContent = data.cs.toFixed(3);
  document.getElementById("valPhotopic").textContent = data.photopicLux.toFixed(1);
  document.getElementById("valMp").textContent = data.mpRatio.toFixed(3);
  document.getElementById("valSuppression").textContent = `%${data.suppression.toFixed(1)}`;
  document.getElementById("valPhaseShift").textContent = `+${data.phaseShift.toFixed(1)} sa`;
  document.getElementById("valLatency").textContent = `${data.latencyMs} ms`;
  document.getElementById("valGt").textContent = `%${(data.gT * 100).toFixed(0)}`;
  document.getElementById("valCosTheta").textContent = `×${data.angularScale.toFixed(2)}`;

  // Bağlantı Durumu Rozeti
  const statusEl = document.getElementById("connectionStatus");
  if (data.isHfActive) {
    statusEl.innerHTML = `<span class="status-dot" style="background:#10b981;box-shadow:0 0 8px #10b981;"></span> Hugging Face Canlı API (sevden123)`;
    statusEl.style.color = "#34d399";
    statusEl.style.borderColor = "rgba(16, 185, 129, 0.4)";
  } else {
    statusEl.innerHTML = `<span class="status-dot" style="background:#38bdf8;box-shadow:0 0 8px #38bdf8;"></span> Yerel Simülatör (Offline/PWA)`;
    statusEl.style.color = "#38bdf8";
    statusEl.style.borderColor = "rgba(56, 189, 248, 0.3)";
  }

  // Brown et al. (2022) Sirkadiyen Risk Değerlendirmesi
  const banner = document.getElementById("riskBanner");
  const tag = document.getElementById("riskTag");
  const title = document.getElementById("riskTitle");
  const desc = document.getElementById("riskDesc");

  if (data.melanopicEdi < 10.0) {
    banner.style.background = "rgba(34, 197, 94, 0.12)";
    banner.style.borderColor = "#22c55e";
    tag.style.background = "#22c55e";
    tag.style.color = "#000";
    tag.textContent = "GÜVENLİ (DÜŞÜK)";
    title.textContent = "Biyolojik Gece Dostu";
    desc.textContent = "Melatonin salgılanması korunur; gece oturumları için sirkadiyen açıdan tamamen güvenlidir.";
  } else if (data.melanopicEdi < 50.0) {
    banner.style.background = "rgba(234, 179, 8, 0.12)";
    banner.style.borderColor = "#eab308";
    tag.style.background = "#eab308";
    tag.style.color = "#000";
    tag.textContent = "ORTA (AKŞAM EŞİĞİ)";
    title.textContent = "Hafif Melatonin Gecikmesi";
    desc.textContent = "Akşam saatlerinde biyolojik saati 30-45 dk öteleyebilir. Ekran parlaklığını kısmanız önerilir.";
  } else if (data.melanopicEdi < 150.0) {
    banner.style.background = "rgba(249, 115, 22, 0.12)";
    banner.style.borderColor = "#f97316";
    tag.style.background = "#f97316";
    tag.style.color = "#000";
    tag.textContent = "YÜKSEK (BASKILAMA)";
    title.textContent = "Güçlü Sirkadiyen Uyarım";
    desc.textContent = "Mavi pik dalga boyları uykuyu 1-2 saat geciktirir. Uyku öncesi mavi ışık filtresi devreye alınmalıdır.";
  } else {
    banner.style.background = "rgba(239, 68, 68, 0.15)";
    banner.style.borderColor = "#ef4444";
    tag.style.background = "#ef4444";
    tag.style.color = "#fff";
    tag.textContent = "KRİTİK (FAZ KAYMASI)";
    title.textContent = "ipRGC Maksimal Doyumu";
    desc.textContent = "Melatonin %80+ oranında baskılanmıştır. Biyolojik saat 2-3 saat geri kayar.";
  }

  // Plotly Çizimi
  renderPlot(data);
}

function renderPlot(data) {
  const traces = [];

  // 1. Ham SPD0 (PRISM Theta)
  if (state.showSpd0) {
    traces.push({
      x: WAVELENGTHS,
      y: data.spd0,
      mode: "lines",
      line: { color: "#38bdf8", width: 1.8, dash: "dot" },
      name: `1. Ham SPD₀ (θ=${state.theta}°)`
    });
  }

  // 2. Yaşlanmış SPD (MODULATOR Aging)
  if (state.showSpdAging) {
    traces.push({
      x: WAVELENGTHS,
      y: data.spdAging,
      mode: "lines",
      line: { color: "#f59e0b", width: 2.2 },
      name: `2. Yaşlanmış SPD (${state.calendarHours} h)`
    });
  }

  // 3. Korneal SPD (MODULATOR Ambient)
  if (state.showSpdCornea) {
    traces.push({
      x: WAVELENGTHS,
      y: data.cornealSpd,
      mode: "lines",
      line: { color: "#10b981", width: 3.0 },
      fill: "tozeroy",
      fillcolor: "rgba(16, 185, 129, 0.15)",
      name: `3. Kornea SPD (Ortam + Ekran)`
    });
  }

  // ipRGC 490 nm Referans Eğrisi
  const maxY = Math.max(0.08, ...data.cornealSpd, ...data.spdAging);
  traces.push({
    x: WAVELENGTHS,
    y: S_MEL.map(s => s * (maxY * 0.75)),
    mode: "lines",
    line: { color: "#06b6d4", width: 1.2, dash: "dash" },
    name: "ipRGC s_mel(λ) [490 nm]"
  });

  const layout = {
    template: "plotly_dark",
    plot_bgcolor: "#090d16",
    paper_bgcolor: "#090d16",
    xaxis: {
      title: "Dalga Boyu λ (nm) — CIE Spektral Bölge",
      range: [380, 780],
      gridcolor: "rgba(255, 255, 255, 0.05)"
    },
    yaxis: {
      title: "Işınım Yoğunluğu [W/m²/nm / a.u.]",
      range: [0, maxY * 1.25],
      gridcolor: "rgba(255, 255, 255, 0.05)"
    },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
    margin: { l: 45, r: 20, t: 30, b: 40 }
  };

  Plotly.react("spectrumPlot", traces, layout, { responsive: true, displayModeBar: false });
}

// ─── 4. ETKİLEŞİM VE DİNLENEN OLAYLAR ────────────────────────────────────────

function setupListeners() {
  // Model 1 Sliders
  bindSlider("sliderR", "valR", v => { state.r = Number(v); runPipeline(); });
  bindSlider("sliderG", "valG", v => { state.g = Number(v); runPipeline(); });
  bindSlider("sliderB", "valB", v => { state.b = Number(v); runPipeline(); });
  bindSlider("sliderX", "valX", v => { state.x = Number(v); runPipeline(); });
  bindSlider("sliderY", "valY", v => { state.y = Number(v); runPipeline(); });
  bindSlider("sliderTheta", "valTheta", v => { state.theta = Number(v); runPipeline(); });

  // Model 2 Sliders
  bindSlider("sliderHours", "valHours", v => { state.calendarHours = Number(v); runPipeline(); });
  bindSlider("sliderBright", "valBright", v => { state.brightness = Number(v); runPipeline(); });
  
  document.getElementById("selectProfile").addEventListener("change", e => {
    state.profile = e.target.value;
    runPipeline();
  });
  document.getElementById("selectScenario").addEventListener("change", e => {
    state.scenario = e.target.value;
    runPipeline();
  });

  // Model 3 Sliders
  bindSlider("sliderAmbLux", "valAmbLux", v => { state.ambientLux = Number(v); runPipeline(); });
  bindSlider("sliderAmbCct", "valAmbCct", v => { state.ambientCct = Number(v); runPipeline(); });
  bindSlider("sliderDist", "valDist", v => { state.distanceCm = Number(v); runPipeline(); });
  bindSlider("sliderAge", "valAge", v => { state.observerAge = Number(v); runPipeline(); });
  bindSlider("sliderDuration", "valDuration", v => { state.durationHours = Number(v); runPipeline(); });

  // Grafik Aç/Kapa Çipleri
  document.getElementById("chipSpd0").addEventListener("click", function() {
    state.showSpd0 = !state.showSpd0;
    this.classList.toggle("active-blue", state.showSpd0);
    renderPlot(state.results);
  });
  document.getElementById("chipSpdAging").addEventListener("click", function() {
    state.showSpdAging = !state.showSpdAging;
    this.classList.toggle("active-amber", state.showSpdAging);
    renderPlot(state.results);
  });
  document.getElementById("chipSpdCornea").addEventListener("click", function() {
    state.showSpdCornea = !state.showSpdCornea;
    this.classList.toggle("active-green", state.showSpdCornea);
    renderPlot(state.results);
  });

  // Presets Tıklama
  document.querySelectorAll(".preset-card").forEach(card => {
    card.addEventListener("click", () => {
      const pKey = card.getAttribute("data-preset");
      const p = GAME_PRESETS[pKey];
      if (!p) return;
      
      state.r = p.r; state.g = p.g; state.b = p.b;
      state.x = p.x; state.y = p.y; state.theta = p.theta;
      state.calendarHours = p.hours; state.profile = p.prof; state.scenario = p.scen; state.brightness = p.bright;
      state.ambientLux = p.ambLux; state.ambientCct = p.ambCct;

      syncSliders();
      runPipeline();
    });
  });

  // Modal İşlemleri
  const modal = document.getElementById("endpointModal");
  document.getElementById("btnOpenSettings").addEventListener("click", () => {
    document.getElementById("inputEndpointTheta").value = state.endpoints.theta;
    document.getElementById("inputEndpointAging").value = state.endpoints.aging;
    document.getElementById("inputEndpointAmbient").value = state.endpoints.ambient;
    document.getElementById("inputEndpointUnified").value = state.endpoints.unified;
    modal.style.display = "flex";
  });
  document.getElementById("btnCloseModal").addEventListener("click", () => modal.style.display = "none");
  document.getElementById("btnSaveEndpoints").addEventListener("click", () => {
    state.endpoints.theta = document.getElementById("inputEndpointTheta").value.trim();
    state.endpoints.aging = document.getElementById("inputEndpointAging").value.trim();
    const rawUnified = document.getElementById("inputEndpointUnified").value.trim();
    state.endpoints.unified = rawUnified || "https://sevden123-prism-spectra.hf.space";
    document.getElementById("inputEndpointUnified").value = state.endpoints.unified;

    localStorage.setItem("hf_endpoint_theta", state.endpoints.theta);
    localStorage.setItem("hf_endpoint_aging", state.endpoints.aging);
    localStorage.setItem("hf_endpoint_ambient", state.endpoints.ambient);
    localStorage.setItem("hf_endpoint_unified", state.endpoints.unified);

    modal.style.display = "none";
    runPipeline();
  });
}

function bindSlider(sliderId, valId, onChange) {
  const slider = document.getElementById(sliderId);
  const valEl = document.getElementById(valId);
  slider.addEventListener("input", e => {
    valEl.textContent = e.target.value;
    onChange(e.target.value);
  });
}

function syncSliders() {
  document.getElementById("sliderR").value = state.r;
  document.getElementById("valR").textContent = state.r;
  document.getElementById("sliderG").value = state.g;
  document.getElementById("valG").textContent = state.g;
  document.getElementById("sliderB").value = state.b;
  document.getElementById("valB").textContent = state.b;
  document.getElementById("sliderX").value = state.x;
  document.getElementById("valX").textContent = state.x;
  document.getElementById("sliderY").value = state.y;
  document.getElementById("valY").textContent = state.y;
  document.getElementById("sliderTheta").value = state.theta;
  document.getElementById("valTheta").textContent = `${state.theta}°`;

  document.getElementById("sliderHours").value = state.calendarHours;
  document.getElementById("valHours").textContent = `${state.calendarHours} h`;
  document.getElementById("sliderBright").value = state.brightness;
  document.getElementById("valBright").textContent = state.brightness;
  document.getElementById("selectProfile").value = state.profile;
  document.getElementById("selectScenario").value = state.scenario;

  document.getElementById("sliderAmbLux").value = state.ambientLux;
  document.getElementById("valAmbLux").textContent = `${state.ambientLux} lx`;
  document.getElementById("sliderAmbCct").value = state.ambientCct;
  document.getElementById("valAmbCct").textContent = `${state.ambientCct} K`;
}

// Başlangıç
document.addEventListener("DOMContentLoaded", () => {
  setupListeners();
  runPipeline();
});
