# ZeroGPU Uyumluluğu: 'spaces' kütüphanesi KESİNLİKLE en başta (Gradio'dan önce) import edilmelidir!
try:
    import spaces
except Exception:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60, **kwargs):
            if callable(func):
                return func
            def decorator(f):
                return f
            return decorator

import os
import json
import time
import math
import inspect
import numpy as np

# Hugging Face Hub HfFolder Uyumluluk Yaması (Gradio 4.x için)
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        try:
            from huggingface_hub._login import HfFolder
            huggingface_hub.HfFolder = HfFolder
        except Exception:
            class HfFolder:
                @classmethod
                def get_token(cls):
                    return os.environ.get("HF_TOKEN", None)
                @classmethod
                def save_token(cls, token):
                    pass
                @classmethod
                def delete_token(cls):
                    pass
            huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

import onnxruntime as ort
import plotly.graph_objects as go
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Gradio 4.44.0 Client & Route Koruma Yaması
try:
    import gradio_client.utils as client_utils
    _orig_json_schema = getattr(client_utils, "_json_schema_to_python_type", None)
    if _orig_json_schema:
        def _safe_json_schema(schema, defs=None):
            if not isinstance(schema, dict):
                return "Any"
            if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], dict):
                schema = {k: v for k, v in schema.items() if k != "additionalProperties"}
            return _orig_json_schema(schema, defs)
        client_utils._json_schema_to_python_type = _safe_json_schema

    _orig_get_type = getattr(client_utils, "get_type", None)
    if _orig_get_type:
        def _safe_get_type(schema):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_get_type(schema)
        client_utils.get_type = _safe_get_type
except Exception:
    pass

# Starlette Jinja2Templates.TemplateResponse Uyumluluk Yaması (Gradio 4.x için)
try:
    from starlette.templating import Jinja2Templates
    _orig_template_response = Jinja2Templates.TemplateResponse

    def _safe_template_response(self, *args, **kwargs):
        sig = inspect.signature(_orig_template_response)
        params = list(sig.parameters.keys())
        if len(params) > 1 and params[1] == "request":
            if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
                name = args[0]
                context = args[1]
                req = context.get("request")
                remaining = args[2:]
                return _orig_template_response(self, req, name, context, *remaining, **kwargs)
            elif len(args) == 1 and isinstance(args[0], str) and "context" in kwargs:
                name = args[0]
                context = kwargs.pop("context")
                req = context.get("request") if isinstance(context, dict) else kwargs.pop("request", None)
                return _orig_template_response(self, req, name, context, **kwargs)
        return _orig_template_response(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = _safe_template_response
except Exception:
    pass

# 1. Model ve Veri Yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights", "prism.onnx")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "prism.onnx")

CIE_PATH = os.path.join(BASE_DIR, "model_weights", "cie_curves.json")
if not os.path.exists(CIE_PATH):
    CIE_PATH = os.path.join(BASE_DIR, "cie_curves.json")

wavelengths = np.arange(380, 781, 1, dtype=np.float64)

# Model 1 ONNX Yükleme
session = None
input_name = None
output_name = None
if os.path.exists(MODEL_PATH):
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"PRISM ONNX Modeli başarıyla yüklendi: {MODEL_PATH}")
    except Exception as e:
        print(f"ONNX Model hatası: {e}")
else:
    print("Bilgi: prism.onnx bulunamadı, dahili Gauss fiziksel simülatörü devrede.")

# CIE Eğrileri Yükleme
v_lambda = 1.019 * np.exp(-0.5 * ((wavelengths - 559) / 42) ** 2)
s_mel_curve = np.exp(-0.5 * ((wavelengths - 490) / 28) ** 2)
edi_weights = s_mel_curve * (683.002 / 72.9)

if os.path.exists(CIE_PATH):
    try:
        with open(CIE_PATH, "r", encoding="utf-8") as f:
            cie_data = json.load(f)
        v_lambda = np.array(cie_data.get("v_lambda", v_lambda.tolist()), dtype=np.float64)
        s_mel_raw = cie_data.get("s_mel_curve") or cie_data.get("s_mel") or s_mel_curve.tolist()
        s_mel_curve = np.array(s_mel_raw, dtype=np.float64)
        if "melanopic_edi_weights" in cie_data:
            edi_weights = np.array(cie_data["melanopic_edi_weights"], dtype=np.float64)
        else:
            edi_weights = s_mel_curve * (683.002 / 72.9)
        print("CIE Eğrileri başarıyla yüklendi.")
    except Exception as e:
        print(f"CIE dosyası okunurken hata: {e}")

# Oyuncu Profilleri ve Ortam Senaryoları (NotebookLM.md ve C0-video2spd ile uyumlu)
PROFILES = {
    "S1": {"name": "Casual Gamer (S1)", "tau": 60000.0, "beta": 0.70, "desc": "Günde 1-2 saat, düşük-orta parlaklık"},
    "S2": {"name": "Core Gamer (S2)", "tau": 30000.0, "beta": 0.60, "desc": "Günde 3-5 saat, standart parlaklık"},
    "S3": {"name": "Heavy / Streamer (S3)", "tau": 15000.0, "beta": 0.50, "desc": "Günde 8-12+ saat, yüksek parlaklık, statik HUD"}
}

SCENARIOS = {
    "E1": {"name": "Serin Oda (22°C, %40 RH)", "k_env": 1.0, "desc": "Referans ev/ofis ortamı"},
    "E2": {"name": "Sıcak Oda (32°C, %50 RH)", "k_env": 1.3, "desc": "%30 daha hızlı panel aşınması"},
    "E3": {"name": "İnternet Kafe (40°C, %70 RH)", "k_env": 1.7, "desc": "%70 daha hızlı termal ve nem aşınması"}
}

# 6 Popüler Oyun Ön Ayarı (Presets)
GAME_PRESETS = {
    "Cyberpunk 2077 (Neon Gece Şehri)": {"r": 255, "g": 20, "b": 147, "x": 0.5, "y": 0.5, "theta": 25.0, "hours": 8000, "prof": "S3", "scen": "E2", "bright": 0.7, "lux": 15, "cct": 2700, "dist": 45, "age": 22, "dur": 4.0},
    "Counter-Strike 2 (Mirage A Bölgesi)": {"r": 210, "g": 180, "b": 140, "x": 0.5, "y": 0.5, "theta": 15.0, "hours": 4000, "prof": "S2", "scen": "E1", "bright": 0.6, "lux": 80, "cct": 4500, "dist": 50, "age": 24, "dur": 3.0},
    "League of Legends (Sihirdar Vadisi)": {"r": 45, "g": 160, "b": 65, "x": 0.5, "y": 0.5, "theta": 20.0, "hours": 6000, "prof": "S2", "scen": "E1", "bright": 0.5, "lux": 50, "cct": 4000, "dist": 55, "age": 20, "dur": 3.5},
    "Minecraft (Gündüz & Çimenlik)": {"r": 100, "g": 200, "b": 80, "x": 0.5, "y": 0.5, "theta": 10.0, "hours": 2000, "prof": "S1", "scen": "E1", "bright": 0.5, "lux": 120, "cct": 5500, "dist": 60, "age": 18, "dur": 2.0},
    "Red Dead Redemption 2 (Gün Batımı)": {"r": 240, "g": 120, "b": 40, "x": 0.5, "y": 0.5, "theta": 30.0, "hours": 5000, "prof": "S2", "scen": "E1", "bright": 0.5, "lux": 25, "cct": 2400, "dist": 65, "age": 28, "dur": 2.5},
    "Valorant (Icebox B Bölgesi)": {"r": 140, "g": 210, "b": 255, "x": 0.5, "y": 0.5, "theta": 15.0, "hours": 7000, "prof": "S3", "scen": "E1", "bright": 0.8, "lux": 60, "cct": 6500, "dist": 50, "age": 21, "dur": 4.5}
}

# ----------------- HESAPLAMA MOTORLARI -----------------

def compute_prism_theta(r: float, g: float, b: float, x: float, y: float, theta: float):
    t0 = time.perf_counter_ns()
    r_val = float(np.clip(r, 0, 255))
    g_val = float(np.clip(g, 0, 255))
    b_val = float(np.clip(b, 0, 255))
    x_val = float(np.clip(x, 0.0, 1.0))
    y_val = float(np.clip(y, 0.0, 1.0))
    theta_deg = float(np.clip(theta, 0.0, 85.0))
    
    # 1. PRISM [R,G,B,x,y] Tahmini
    if session is not None:
        input_tensor = np.array([[r_val / 255.0, g_val / 255.0, b_val / 255.0, x_val, y_val]], dtype=np.float32)
        out_log = session.run([output_name], {input_name: input_tensor})[0][0]
        base_spd = np.maximum(0.0, np.exp(out_log) - 1e-4)
    else:
        b_p = (b_val / 255.0) * 0.95 * np.exp(-0.5 * ((wavelengths - 452) / 13) ** 2)
        g_p = (g_val / 255.0) * 0.75 * np.exp(-0.5 * ((wavelengths - 530) / 28) ** 2)
        r_p = (r_val / 255.0) * 0.85 * np.exp(-0.5 * ((wavelengths - 625) / 18) ** 2)
        base_spd = 0.015 + b_p + g_p + r_p

    # 2. Açısal Modülasyon (Theta Entegrasyonu)
    theta_rad = math.radians(theta_deg)
    angular_scale = math.cos(theta_rad)
    angle_shift_nm = math.sin(theta_rad) * 4.5
    
    shifted_wl = wavelengths - angle_shift_nm
    theta_spd = np.interp(shifted_wl, wavelengths, base_spd) * angular_scale
    theta_spd = np.maximum(0.0, theta_spd)
    
    t1 = time.perf_counter_ns()
    latency_ms = (t1 - t0) / 1_000_000.0
    
    peak_idx = int(np.argmax(theta_spd))
    peak_wl = float(wavelengths[peak_idx])
    peak_power = float(theta_spd[peak_idx])
    
    return {
        "wavelengths": wavelengths.tolist(),
        "spd_0": theta_spd.tolist(),
        "peak_wavelength": peak_wl,
        "peak_power": peak_power,
        "angular_scale": round(angular_scale, 4),
        "angle_shift_nm": round(angle_shift_nm, 2),
        "latency_ms": round(latency_ms, 3)
    }

def compute_display_aging(spd_0: Optional[List[float]], calendar_hours: float, profile_id: str = "S2", scenario_id: str = "E1", brightness: float = 0.5):
    t0 = time.perf_counter_ns()
    prof = PROFILES.get(profile_id, PROFILES["S2"])
    scen = SCENARIOS.get(scenario_id, SCENARIOS["E1"])
    
    tau = prof["tau"]
    beta = prof["beta"]
    k_env = scen["k_env"]
    b_ref = 0.5
    eta = 1.5
    
    b_val = float(np.clip(brightness, 0.1, 1.0))
    t_val = float(max(0.0, calendar_hours))
    
    t_eff_tot = k_env * t_val * ((b_val / b_ref) ** eta)
    g_t = float(np.exp(-((t_eff_tot / tau) ** beta))) if tau > 0 else 1.0
    
    blue_decay = np.exp(-0.5 * ((wavelengths - 450) / 35) ** 2)
    spec_gt = g_t * (1.0 - 0.25 * (1.0 - g_t) * blue_decay)
    
    if spd_0 and len(spd_0) == 401:
        base_spd = np.array(spd_0, dtype=np.float64)
    else:
        b_p = 0.95 * np.exp(-0.5 * ((wavelengths - 452) / 13) ** 2)
        g_p = 0.75 * np.exp(-0.5 * ((wavelengths - 530) / 28) ** 2)
        r_p = 0.85 * np.exp(-0.5 * ((wavelengths - 625) / 18) ** 2)
        base_spd = 0.02 + b_p + g_p + r_p

    spd_aging = np.maximum(0.0, base_spd * spec_gt)
    
    t1 = time.perf_counter_ns()
    latency_ms = (t1 - t0) / 1_000_000.0
    
    blue_idx = (wavelengths >= 430) & (wavelengths <= 470)
    initial_blue = float(np.mean(base_spd[blue_idx]))
    aged_blue = float(np.mean(spd_aging[blue_idx]))
    blue_retention_pct = round((aged_blue / (initial_blue + 1e-9)) * 100.0, 1)

    return {
        "wavelengths": wavelengths.tolist(),
        "spd_initial": base_spd.tolist(),
        "spd_aging": spd_aging.tolist(),
        "g_t": round(g_t, 4),
        "t_eff_tot": round(t_eff_tot, 1),
        "blue_retention_pct": blue_retention_pct,
        "latency_ms": round(latency_ms, 3)
    }

def generate_ambient_spd(lux: float, cct: float):
    lux_val = float(max(0.0, lux))
    if lux_val <= 1e-4:
        return np.zeros_like(wavelengths)
    cct_val = float(np.clip(cct, 2000, 6500))
    h = 6.626e-34
    c = 3.0e8
    k = 1.381e-23
    wl_m = wavelengths * 1e-9
    c1 = 3.7418e-16
    c2 = 1.4388e-2
    planck = (c1 / (wl_m ** 5)) / (np.exp(c2 / (wl_m * cct_val)) - 1.0)
    denom = 683.002 * np.sum(planck * v_lambda)
    k_norm = lux_val / denom if denom > 0 else 0.0
    return planck * k_norm * 0.015

def get_lens_transmittance(age: float):
    age_clamped = float(np.clip(age, 15.0, 75.0))
    tl32 = 0.017 + 0.983 * np.exp(-((400.0 / wavelengths) ** 4.0))
    aging_factor = 1.0 + 0.015 * (age_clamped - 32.0)
    tl_age = tl32 ** max(0.2, aging_factor)
    return np.clip(tl_age, 0.05, 1.0)

def compute_ambient_cornea(spd_screen: Optional[List[float]], ambient_lux: float, ambient_cct: float, distance_cm: float, observer_age: float, duration_hours: float):
    t0 = time.perf_counter_ns()
    dist_val = float(np.clip(distance_cm, 30.0, 100.0))
    dist_scale = (50.0 / dist_val) ** 2
    
    if spd_screen and len(spd_screen) == 401:
        screen_base = np.array(spd_screen, dtype=np.float64)
    else:
        b_p = 0.95 * np.exp(-0.5 * ((wavelengths - 452) / 13) ** 2)
        g_p = 0.75 * np.exp(-0.5 * ((wavelengths - 530) / 28) ** 2)
        r_p = 0.85 * np.exp(-0.5 * ((wavelengths - 625) / 18) ** 2)
        screen_base = 0.02 + b_p + g_p + r_p

    screen_corneal = screen_base * dist_scale * 0.01
    amb_spd = generate_ambient_spd(ambient_lux, ambient_cct)
    corneal_raw = screen_corneal + amb_spd
    t_lens = get_lens_transmittance(observer_age)
    corneal_spd = np.maximum(0.0, corneal_raw * t_lens)

    photopic_lux = float(683.002 * np.sum(corneal_spd * v_lambda))
    melanopic_edi = float(np.sum(corneal_spd * edi_weights))
    mp_ratio = float(melanopic_edi / photopic_lux) if photopic_lux > 1e-4 else 0.0
    
    cla = 1.548 * melanopic_edi
    cs = float(0.7 * (1.0 - 1.0 / (1.0 + (cla / 355.7) ** 1.1026))) if cla > 0 else 0.0
    
    dur_val = float(max(0.5, duration_hours))
    i50 = 30.0 + 70.0 / (dur_val ** 0.5)
    suppression_pct = float(100.0 / (1.0 + (i50 / (melanopic_edi + 1e-6)) ** 1.0))
    phase_shift_h = float(3.0 * (melanopic_edi ** 0.5) / (melanopic_edi ** 0.5 + 60.0 ** 0.5)) if melanopic_edi > 0 else 0.0
    
    # Brown et al. (2022) Risk Seviyeleri
    if melanopic_edi < 10.0:
        risk_level = "GÜVENLİ (Düşük)"
        risk_color = "#22c55e"
        risk_desc = "Melatonin salgılanması korunur; gece oturumları için biyolojik olarak güvenlidir."
    elif melanopic_edi < 50.0:
        risk_level = "ORTA (Akşam Eşiği)"
        risk_color = "#eab308"
        risk_desc = "Gece saatlerinde hafif melatonin gecikmesi oluşturabilir. Yatmadan önce parlaklığı kısmanız önerilir."
    elif melanopic_edi < 150.0:
        risk_level = "YÜKSEK (Melatonin Baskısı)"
        risk_color = "#f97316"
        risk_desc = "Kısa dalga boylu spektrum maruziyeti. Gece saatlerinde melatonin üretimini kuvvetle baskılar."
    else:
        risk_level = "KRİTİK (Faz Kayması)"
        risk_color = "#ef4444"
        risk_desc = "ipRGC hücreleri maksimum uyarılmaktadır. Biyolojik saatin 2-3 saat gecikmesine yol açar."

    t1 = time.perf_counter_ns()
    latency_ms = (t1 - t0) / 1_000_000.0

    return {
        "wavelengths": wavelengths.tolist(),
        "corneal_spd": corneal_spd.tolist(),
        "ambient_spd": amb_spd.tolist(),
        "screen_corneal": screen_corneal.tolist(),
        "photopic_lux": round(photopic_lux, 1),
        "melanopic_edi": round(melanopic_edi, 1),
        "mp_ratio": round(mp_ratio, 3),
        "circadian_stimulus_cs": round(cs, 3),
        "melatonin_suppression_pct": round(suppression_pct, 1),
        "phase_shift_hours": round(phase_shift_h, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_desc": risk_desc,
        "latency_ms": round(latency_ms, 3)
    }

def run_full_pipeline(r, g, b, x, y, theta, hours, prof, scen, bright, lux, cct, dist, age, dur):
    t0 = time.perf_counter_ns()
    m1 = compute_prism_theta(r, g, b, x, y, theta)
    m2 = compute_display_aging(m1["spd_0"], hours, prof, scen, bright)
    m3 = compute_ambient_cornea(m2["spd_aging"], lux, cct, dist, age, dur)
    t1 = time.perf_counter_ns()
    
    return {
        "wavelengths": wavelengths.tolist(),
        "spd_0": m1["spd_0"],
        "spd_aging": m2["spd_aging"],
        "corneal_spd": m3["corneal_spd"],
        "peak_wavelength": m1["peak_wavelength"],
        "angular_scale": m1["angular_scale"],
        "g_t": m2["g_t"],
        "t_eff_tot": m2["t_eff_tot"],
        "blue_retention_pct": m2["blue_retention_pct"],
        "photopic_lux": m3["photopic_lux"],
        "melanopic_edi": m3["melanopic_edi"],
        "circadian_stimulus_cs": m3["circadian_stimulus_cs"],
        "phase_shift_hours": m3["phase_shift_hours"],
        "melatonin_suppression_pct": m3["melatonin_suppression_pct"],
        "risk_level": m3["risk_level"],
        "risk_color": m3["risk_color"],
        "risk_desc": m3["risk_desc"],
        "latency_ms": round((t1 - t0) / 1_000_000.0, 3)
    }

# ----------------- GRAFİK ÜRETİCİLERİ -----------------

def create_plot_m1(data, theta):
    spd = data["spd_0"]
    wl = data["wavelengths"]
    max_y = max(0.15, float(np.max(spd)))
    
    fig = go.Figure()
    spectral_slices = [
        (380, 435, "rgba(76, 29, 149, 0.45)"),
        (435, 485, "rgba(37, 99, 235, 0.55)"),
        (485, 510, "rgba(6, 182, 212, 0.65)"),
        (510, 565, "rgba(34, 197, 94, 0.55)"),
        (565, 590, "rgba(234, 179, 8, 0.50)"),
        (590, 625, "rgba(249, 115, 22, 0.45)"),
        (625, 781, "rgba(239, 68, 68, 0.40)")
    ]
    for start_wl, end_wl, fill_c in spectral_slices:
        mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
        fig.add_trace(go.Scatter(
            x=wavelengths[mask].tolist(), y=np.array(spd)[mask].tolist(),
            mode="lines", line=dict(width=0.5, color=fill_c), fill="tozeroy", fillcolor=fill_c, showlegend=False, hoverinfo="skip"
        ))
    
    fig.add_trace(go.Scatter(
        x=wl, y=spd, mode="lines", line=dict(color="#38bdf8", width=3.0),
        name=f"SPD₀ (θ={theta}°)", hovertemplate="λ: %{x} nm<br>SPD: %{y:.4f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=[data["peak_wavelength"]], y=[data["peak_power"]], mode="markers+text",
        marker=dict(color="#facc15", size=9, line=dict(color="#ffffff", width=2)),
        text=[f"Pik: {int(data['peak_wavelength'])} nm"], textposition="top center", name="Tepe Dalga Boyu"
    ))
    
    fig.update_layout(
        title=f"<b>Model 1: PRISM-Theta Çıktısı SPD₀(λ)</b> | Açı: θ={theta}° (×{data['angular_scale']})",
        xaxis_title="Dalga Boyu λ (nm)", yaxis_title="Spektral Güç [a.u.]",
        template="plotly_dark", plot_bgcolor="#090d16", paper_bgcolor="#090d16",
        xaxis=dict(range=[380, 780], autorange=False, gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(range=[0, max_y * 1.25], autorange=False, gridcolor="rgba(255, 255, 255, 0.05)"),
        margin=dict(l=45, r=20, t=45, b=40)
    )
    return fig

def create_plot_m2(data, hours):
    wl = data["wavelengths"]
    spd_init = data["spd_initial"]
    spd_aged = data["spd_aging"]
    max_y = max(0.15, float(np.max(spd_init)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wl, y=spd_init, mode="lines", line=dict(color="rgba(148, 163, 184, 0.6)", width=1.8, dash="dot"),
        name="0 Saat (Referans Başlangıç)", hovertemplate="λ: %{x} nm<br>SPD: %{y:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=wl, y=spd_aged, mode="lines", line=dict(color="#f59e0b", width=3.0),
        fill="tonexty", fillcolor="rgba(245, 158, 11, 0.12)",
        name=f"{int(hours):,} Saat Sonrası", hovertemplate="λ: %{x} nm<br>Aged SPD: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"<b>Model 2: Ekran Yaşlanması ve Spektral Aşınma</b> | g(t): {data['g_t']} | Mavi Korunumu: %{data['blue_retention_pct']}",
        xaxis_title="Dalga Boyu λ (nm)", yaxis_title="Spektral Güç [a.u.]",
        template="plotly_dark", plot_bgcolor="#090d16", paper_bgcolor="#090d16",
        xaxis=dict(range=[380, 780], autorange=False, gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(range=[0, max_y * 1.25], autorange=False, gridcolor="rgba(255, 255, 255, 0.05)"),
        margin=dict(l=45, r=20, t=45, b=40)
    )
    return fig

def create_plot_m3(data):
    wl = data["wavelengths"]
    corneal = data["corneal_spd"]
    max_y = max(0.01, float(np.max(corneal)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wl, y=corneal, mode="lines", line=dict(color="#10b981", width=3.0),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.20)",
        name="Korneal Spektrum SPD(λ)", hovertemplate="λ: %{x} nm<br>Kornea: %{y:.5f}<extra></extra>"
    ))
    
    mel_scaled = (s_mel_curve * (max_y * 0.85)).tolist()
    fig.add_trace(go.Scatter(
        x=wl, y=mel_scaled, mode="lines", line=dict(color="#06b6d4", width=2.0, dash="dash"),
        name="CIE S 026 ipRGC (490 nm)", opacity=0.85
    ))
    
    fig.update_layout(
        title=f"<b>Model 3: Korneal Spektrum & ipRGC Uyarımı</b> | m-EDI: {data['melanopic_edi']} lx",
        xaxis_title="Dalga Boyu λ (nm)", yaxis_title="Korneal Işınım [W/m²·nm]",
        template="plotly_dark", plot_bgcolor="#090d16", paper_bgcolor="#090d16",
        xaxis=dict(range=[380, 780], autorange=False, gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(range=[0, max_y * 1.3], autorange=False, gridcolor="rgba(255, 255, 255, 0.05)"),
        margin=dict(l=45, r=20, t=45, b=40)
    )
    return fig

def create_plot_unified(res):
    wl = res["wavelengths"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=res["spd_0"], mode="lines", line=dict(color="#38bdf8", width=1.5, dash="dot"), name="1. Ham SPD₀(θ)"))
    fig.add_trace(go.Scatter(x=wl, y=res["spd_aging"], mode="lines", line=dict(color="#f59e0b", width=2.0), name="2. Yaşlanmış SPD"))
    fig.add_trace(go.Scatter(x=wl, y=res["corneal_spd"], mode="lines", line=dict(color="#10b981", width=3.0), fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.15)", name="3. Kornea SPD (Ortam + Yaş)"))
    
    fig.update_layout(
        title=f"<b>Aetheris Uçtan-Uca Spektral Boru Hattı</b> | m-EDI: {res['melanopic_edi']} lx | Gecikme: {res['latency_ms']} ms",
        xaxis_title="Dalga Boyu λ (nm)", yaxis_title="Rölatif Güç / Spektral Akı",
        template="plotly_dark", plot_bgcolor="#090d16", paper_bgcolor="#090d16",
        xaxis=dict(range=[380, 780], autorange=False, gridcolor="rgba(255, 255, 255, 0.05)"),
        margin=dict(l=45, r=20, t=45, b=40)
    )
    return fig

# ----------------- GRADIO OLAY YÖNETİCİLERİ -----------------

@spaces.GPU
def ui_predict_m1(r, g, b, x, y, theta):
    res = compute_prism_theta(r, g, b, x, y, theta)
    fig = create_plot_m1(res, theta)
    info = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-family: monospace;">
        <div style="background: #111827; padding: 8px; border-radius: 6px; border: 1px solid #374151;">
            <div style="color: #9ca3af; font-size: 11px;">AÇISAL ÖLÇEK</div>
            <div style="color: #38bdf8; font-size: 16px; font-weight: bold;">×{res['angular_scale']}</div>
        </div>
        <div style="background: #111827; padding: 8px; border-radius: 6px; border: 1px solid #374151;">
            <div style="color: #9ca3af; font-size: 11px;">TEPE DALGA BOYU</div>
            <div style="color: #facc15; font-size: 16px; font-weight: bold;">{int(res['peak_wavelength'])} nm</div>
        </div>
        <div style="background: #111827; padding: 8px; border-radius: 6px; border: 1px solid #374151;">
            <div style="color: #9ca3af; font-size: 11px;">GECİKME</div>
            <div style="color: #22c55e; font-size: 16px; font-weight: bold;">{res['latency_ms']} ms</div>
        </div>
    </div>
    """
    return fig, info, json.dumps(res)

@spaces.GPU
def ui_predict_m2(hours, bright, prof, scen):
    res = compute_display_aging(None, hours, prof, scen, bright)
    fig = create_plot_m2(res, hours)
    info = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-family: monospace;">
        <div style="background: #111827; padding: 8px; border-radius: 6px; border: 1px solid #374151;">
            <div style="color: #9ca3af; font-size: 11px;">AŞINMA FAKTÖRÜ g(t)</div>
            <div style="color: #f59e0b; font-size: 16px; font-weight: bold;">{res['g_t']}</div>
        </div>
        <div style="background: #111827; padding: 8px; border-radius: 6px; border: 1px solid #374151;">
            <div style="color: #9ca3af; font-size: 11px;">EFEKTİF ÇALIŞMA</div>
            <div style="color: #38bdf8; font-size: 16px; font-weight: bold;">{int(res['t_eff_tot']):,} saat</div>
        </div>
        <div style="background: #111827; padding: 8px; border-radius: 6px; border: 1px solid #374151;">
            <div style="color: #9ca3af; font-size: 11px;">MAVİ KORUNUMU</div>
            <div style="color: #22c55e; font-size: 16px; font-weight: bold;">%{res['blue_retention_pct']}</div>
        </div>
    </div>
    """
    return fig, info, json.dumps(res)

@spaces.GPU
def ui_predict_m3(lux, cct, dist, age, dur):
    res = compute_ambient_cornea(None, lux, cct, dist, age, dur)
    fig = create_plot_m3(res)
    info = f"""
    <div style="background: #111827; padding: 10px; border-radius: 8px; border-left: 4px solid {res['risk_color']}; margin-bottom: 8px;">
        <div style="color: {res['risk_color']}; font-weight: bold; font-size: 13px;">BİYOLOJİK RİSK: {res['risk_level']}</div>
        <div style="color: #9ca3af; font-size: 11px;">{res['risk_desc']}</div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; font-family: monospace;">
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">m-EDI</div>
            <div style="color:#38bdf8; font-size:15px; font-weight:bold;">{res['melanopic_edi']} lx</div>
        </div>
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">CS</div>
            <div style="color:#a855f7; font-size:15px; font-weight:bold;">{res['circadian_stimulus_cs']}</div>
        </div>
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">BASKILANMA</div>
            <div style="color:#f97316; font-size:15px; font-weight:bold;">%{res['melatonin_suppression_pct']}</div>
        </div>
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">FAZ KAYMASI</div>
            <div style="color:#ef4444; font-size:15px; font-weight:bold;">+{res['phase_shift_hours']} h</div>
        </div>
    </div>
    """
    return fig, info, json.dumps(res)

@spaces.GPU
def ui_predict_unified(r, g, b, x, y, theta, hours, prof, scen, bright, lux, cct, dist, age, dur):
    res = run_full_pipeline(r, g, b, x, y, theta, hours, prof, scen, bright, lux, cct, dist, age, dur)
    fig = create_plot_unified(res)
    info = f"""
    <div style="background: #111827; padding: 10px; border-radius: 8px; border-left: 4px solid {res['risk_color']}; margin-bottom: 8px;">
        <div style="color: {res['risk_color']}; font-weight: bold; font-size: 13px;">UÇTAN UCA SİRKADİYEN RİSK: {res['risk_level']}</div>
        <div style="color: #9ca3af; font-size: 11px;">{res['risk_desc']}</div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; font-family: monospace;">
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">m-EDI</div>
            <div style="color:#38bdf8; font-size:15px; font-weight:bold;">{res['melanopic_edi']} lx</div>
        </div>
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">CS</div>
            <div style="color:#a855f7; font-size:15px; font-weight:bold;">{res['circadian_stimulus_cs']}</div>
        </div>
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">FAZ KAYMASI</div>
            <div style="color:#ef4444; font-size:15px; font-weight:bold;">+{res['phase_shift_hours']} h</div>
        </div>
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">AŞINMA g(t)</div>
            <div style="color:#f59e0b; font-size:15px; font-weight:bold;">{res['g_t']}</div>
        </div>
        <div style="background: #111827; padding: 6px; border-radius: 6px; text-align: center;">
            <div style="color:#9ca3af; font-size:10px;">GECİKME</div>
            <div style="color:#22c55e; font-size:15px; font-weight:bold;">{res['latency_ms']} ms</div>
        </div>
    </div>
    """
    return fig, info, json.dumps(res)

def apply_preset(preset_name):
    p = GAME_PRESETS.get(preset_name, GAME_PRESETS["Cyberpunk 2077 (Neon Gece Şehri)"])
    return (
        p["r"], p["g"], p["b"], p["x"], p["y"], p["theta"],
        p["hours"], p["prof"], p["scen"], p["bright"],
        p["lux"], p["cct"], p["dist"], p["age"], p["dur"]
    )

# ----------------- GRADIO ARAYÜZÜ (TABS) -----------------

custom_css = """
body, .gradio-container { background-color: #030712 !important; color: #f3f4f6 !important; }
.tab-nav button { font-weight: bold !important; font-size: 13px !important; }
"""

with gr.Blocks(title="PRISM-Spectra", css=custom_css, theme=gr.themes.Base()) as demo:
    with gr.Tabs():
        # SEKME 1: MODEL 1
        with gr.Tab("📐 Model 1: PRISM-Theta"):
            gr.Markdown("### 📐 Model 1: Açısal ve Kromatik Spektral Kestirim $[R, G, B, x, y, \\theta] \\to SPD_0(\\lambda)$")
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        m1_r = gr.Slider(0, 255, value=15, step=1, label="Kırmızı (R)")
                        m1_g = gr.Slider(0, 255, value=140, step=1, label="Yeşil (G)")
                        m1_b = gr.Slider(0, 255, value=255, step=1, label="Mavi (B)")
                    with gr.Row():
                        m1_x = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Uzamsal X")
                        m1_y = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Uzamsal Y")
                    m1_theta = gr.Slider(0.0, 80.0, value=0.0, step=1.0, label="İzleme Açısı θ (°)")
                    m1_btn = gr.Button("🔮 Spektrumu Hesapla (Run Model 1)", variant="primary")
                with gr.Column(scale=6):
                    m1_plot = gr.Plot(label="Başlangıç Spektral Güç Dağılımı SPD₀(λ)")
                    m1_info = gr.HTML(label="Metrikler")
                    m1_json = gr.Textbox(visible=False)
            
            m1_inputs = [m1_r, m1_g, m1_b, m1_x, m1_y, m1_theta]
            m1_btn.click(fn=ui_predict_m1, inputs=m1_inputs, outputs=[m1_plot, m1_info, m1_json], api_name=False)
            for inp in m1_inputs:
                inp.change(fn=ui_predict_m1, inputs=m1_inputs, outputs=[m1_plot, m1_info, m1_json], show_progress="hidden")

        # SEKME 2: MODEL 2
        with gr.Tab("⏳ Model 2: MODULATOR-Aging"):
            gr.Markdown("### ⏳ Model 2: Ekran Yaşlanması ve Spektral Aşınma $[SPD_0, t, \\text{Profil, Senaryo}, B] \\to SPD_{\\text{aging}}(\\lambda)$")
            with gr.Row():
                with gr.Column(scale=4):
                    m2_hours = gr.Slider(0, 25000, value=5000, step=500, label="Kullanım / Çalışma Saati (t)")
                    m2_bright = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Ekran Parlaklığı (B)")
                    with gr.Row():
                        m2_prof = gr.Dropdown(["S1", "S2", "S3"], value="S2", label="Oyuncu Profili (S1-S3)")
                        m2_scen = gr.Dropdown(["E1", "E2", "E3"], value="E1", label="Ortam Senaryosu (E1-E3)")
                    m2_btn = gr.Button("⏳ Yaşlanmayı Hesapla (Run Model 2)", variant="primary")
                with gr.Column(scale=6):
                    m2_plot = gr.Plot(label="Aşınmış Spektrum Kıyaslaması")
                    m2_info = gr.HTML(label="Yaşlanma Metrikleri")
                    m2_json = gr.Textbox(visible=False)
            
            m2_inputs = [m2_hours, m2_bright, m2_prof, m2_scen]
            m2_btn.click(fn=ui_predict_m2, inputs=m2_inputs, outputs=[m2_plot, m2_info, m2_json], api_name=False)
            for inp in m2_inputs:
                inp.change(fn=ui_predict_m2, inputs=m2_inputs, outputs=[m2_plot, m2_info, m2_json], show_progress="hidden")

        # SEKME 3: MODEL 3
        with gr.Tab("💡 Model 3: MODULATOR-Ambient"):
            gr.Markdown("### 💡 Model 3: Ortam Aydınlatması ve Korneal Sirkadiyen Yük $[SPD_{\\text{aging}}, E_{\\text{amb}}, CCT, d, \\text{Yaş}] \\to SPD_{\\text{cornea}}$")
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        m3_lux = gr.Slider(0, 500, value=30, step=5, label="Ortam Işığı (Lux)")
                        m3_cct = gr.Slider(2000, 6500, value=3000, step=100, label="Ortam Renk Sıcaklığı (CCT K)")
                    with gr.Row():
                        m3_dist = gr.Slider(30, 100, value=50, step=5, label="İzleme Mesafesi (cm)")
                        m3_age = gr.Slider(15, 75, value=25, step=1, label="Gözlemci Yaşı (Yıl)")
                    m3_dur = gr.Slider(0.5, 8.0, value=3.0, step=0.5, label="Maruz Kalma Süresi (Saat)")
                    m3_btn = gr.Button("💡 Korneal Spektrum ve Sirkadiyen Yükü Hesapla (Run Model 3)", variant="primary")
                with gr.Column(scale=6):
                    m3_plot = gr.Plot(label="Korneal Spektrum & ipRGC Eğrisi")
                    m3_info = gr.HTML(label="Sirkadiyen Metrikler & Risk")
                    m3_json = gr.Textbox(visible=False)
            
            m3_inputs = [m3_lux, m3_cct, m3_dist, m3_age, m3_dur]
            m3_btn.click(fn=ui_predict_m3, inputs=m3_inputs, outputs=[m3_plot, m3_info, m3_json], api_name=False)
            for inp in m3_inputs:
                inp.change(fn=ui_predict_m3, inputs=m3_inputs, outputs=[m3_plot, m3_info, m3_json], show_progress="hidden")

        # SEKME 4: BİRLEŞİK BORU HATTI
        with gr.Tab("🌈 Uçtan Uca Birleşik Boru Hattı"):
            gr.Markdown("### 🔗 3 Modelli Uçtan Uca Entegre Akış: Pikselden Göze Sirkadiyen Analiz")
            with gr.Row():
                preset_dropdown = gr.Dropdown(list(GAME_PRESETS.keys()), label="🎮 Hızlı Oyun Sahnesi Seçin (Presets)", value="Cyberpunk 2077 (Neon Gece Şehri)")
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("#### 1. Piksel & Bakış Açısı (PRISM)")
                    with gr.Row():
                        u_r = gr.Slider(0, 255, value=255, step=1, label="R")
                        u_g = gr.Slider(0, 255, value=20, step=1, label="G")
                        u_b = gr.Slider(0, 255, value=147, step=1, label="B")
                    with gr.Row():
                        u_x = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="x")
                        u_y = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="y")
                    u_theta = gr.Slider(0.0, 80.0, value=25.0, step=1.0, label="Açı θ (°)")
                    
                    gr.Markdown("#### 2. Panel Yaşlanması (MODULATOR-Aging)")
                    with gr.Row():
                        u_hours = gr.Slider(0, 25000, value=8000, step=500, label="Saat (t)")
                        u_bright = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Parlaklık")
                    with gr.Row():
                        u_prof = gr.Dropdown(["S1", "S2", "S3"], value="S3", label="Profil")
                        u_scen = gr.Dropdown(["E1", "E2", "E3"], value="E2", label="Senaryo")
                        
                    gr.Markdown("#### 3. Ortam & Sirkadiyen Biyoloji (MODULATOR-Ambient)")
                    with gr.Row():
                        u_lux = gr.Slider(0, 500, value=15, step=5, label="Ortam Lux")
                        u_cct = gr.Slider(2000, 6500, value=2700, step=100, label="CCT K")
                    with gr.Row():
                        u_dist = gr.Slider(30, 100, value=45, step=5, label="Mesafe (cm)")
                        u_age = gr.Slider(15, 75, value=22, step=1, label="Yaş")
                    u_dur = gr.Slider(0.5, 8.0, value=4.0, step=0.5, label="Süre (Saat)")
                    u_btn = gr.Button("🚀 Uçtan Uca Boru Hattını Çalıştır (Run Pipeline)", variant="primary")
                    
                with gr.Column(scale=6):
                    u_plot = gr.Plot(label="3 Aşamalı Spektrum Kıyaslaması: SPD₀ → SPD_aging → SPD_cornea")
                    u_info = gr.HTML(label="Sirkadiyen Biyolojik Analiz Özeti")
                    u_json = gr.Textbox(visible=False)

            u_inputs = [u_r, u_g, u_b, u_x, u_y, u_theta, u_hours, u_prof, u_scen, u_bright, u_lux, u_cct, u_dist, u_age, u_dur]
            preset_dropdown.change(fn=apply_preset, inputs=preset_dropdown, outputs=u_inputs)
            u_btn.click(fn=ui_predict_unified, inputs=u_inputs, outputs=[u_plot, u_info, u_json], api_name="predict_pipeline")
            for inp in u_inputs:
                inp.change(fn=ui_predict_unified, inputs=u_inputs, outputs=[u_plot, u_info, u_json], show_progress="hidden")

    # Sayfa ilk açıldığında varsayılan olarak Model 1 ve Birleşik Boru Hattını çiz
    demo.load(fn=ui_predict_m1, inputs=m1_inputs, outputs=[m1_plot, m1_info, m1_json], api_name=False)
    demo.load(fn=ui_predict_m2, inputs=m2_inputs, outputs=[m2_plot, m2_info, m2_json], api_name=False)
    demo.load(fn=ui_predict_m3, inputs=m3_inputs, outputs=[m3_plot, m3_info, m3_json], api_name=False)
    demo.load(fn=ui_predict_unified, inputs=u_inputs, outputs=[u_plot, u_info, u_json], api_name=False)

# ----------------- REST API VE CORS ENTEGRASYONU -----------------

class PrismThetaRequest(BaseModel):
    r: float = 15.0
    g: float = 140.0
    b: float = 255.0
    x: float = 0.5
    y: float = 0.5
    theta: float = 0.0

class AgingRequest(BaseModel):
    spd_0: Optional[List[float]] = None
    calendar_hours: float = 5000.0
    profile: str = "S2"
    scenario: str = "E1"
    brightness: float = 0.5

class AmbientRequest(BaseModel):
    spd_screen: Optional[List[float]] = None
    ambient_lux: float = 30.0
    ambient_cct: float = 3000.0
    distance_cm: float = 50.0
    observer_age: float = 25.0
    exposure_duration_hours: float = 3.0

class FullPipelineRequest(BaseModel):
    r: float = 15.0
    g: float = 140.0
    b: float = 255.0
    x: float = 0.5
    y: float = 0.5
    theta: float = 0.0
    calendar_hours: float = 5000.0
    profile: str = "S2"
    scenario: str = "E1"
    brightness: float = 0.5
    ambient_lux: float = 30.0
    ambient_cct: float = 3000.0
    distance_cm: float = 50.0
    observer_age: float = 25.0
    duration_hours: float = 3.0

demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@demo.app.post("/predict")
async def api_predict(req: PrismThetaRequest):
    return compute_prism_theta(req.r, req.g, req.b, req.x, req.y, req.theta)

@demo.app.post("/predict/theta")
async def api_theta(req: PrismThetaRequest):
    return compute_prism_theta(req.r, req.g, req.b, req.x, req.y, req.theta)

@demo.app.post("/predict/aging")
async def api_aging(req: AgingRequest):
    return compute_display_aging(req.spd_0, req.calendar_hours, req.profile, req.scenario, req.brightness)

@demo.app.post("/predict/ambient")
async def api_ambient(req: AmbientRequest):
    return compute_ambient_cornea(req.spd_screen, req.ambient_lux, req.ambient_cct, req.distance_cm, req.observer_age, req.exposure_duration_hours)

@demo.app.post("/predict/pipeline")
async def api_pipeline(req: FullPipelineRequest):
    return run_full_pipeline(
        req.r, req.g, req.b, req.x, req.y, req.theta,
        req.calendar_hours, req.profile, req.scenario, req.brightness,
        req.ambient_lux, req.ambient_cct, req.distance_cm, req.observer_age, req.duration_hours
    )

@demo.app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Aetheris-Unified-Modular-Pipeline",
        "models": ["PRISM-Theta", "MODULATOR-Aging", "MODULATOR-Ambient", "Unified-Pipeline"],
        "onnx_active": session is not None
    }

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
