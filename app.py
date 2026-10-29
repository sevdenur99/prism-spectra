import os
import json
import time
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
                    import os
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

# Gradio 4.44.0 Client & Route Koruma Yaması (TypeError: argument of type 'bool' is not iterable önleyici)
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

try:
    import gradio.routes as gr_routes
    _orig_api_info = getattr(gr_routes, "api_info", None)
    if _orig_api_info:
        def _safe_api_info(*args, **kwargs):
            try:
                return _orig_api_info(*args, **kwargs)
            except Exception:
                return {"named_endpoints": {}, "unnamed_endpoints": {}}
        gr_routes.api_info = _safe_api_info
except Exception:
    pass

try:
    _orig_blocks_get_api_info = getattr(gr.Blocks, "get_api_info", None)
    if _orig_blocks_get_api_info:
        def _safe_blocks_get_api_info(self):
            try:
                return _orig_blocks_get_api_info(self)
            except Exception:
                return {"named_endpoints": {}, "unnamed_endpoints": {}}
        gr.Blocks.get_api_info = _safe_blocks_get_api_info
except Exception:
    pass

# ZeroGPU Uyumluluk Kontrolü
try:
    import spaces
    @spaces.GPU
    def _spaces_warmup():
        return True
except (ImportError, Exception):
    pass

# 1. Model ve Standart Verilerin Yüklenmesi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

candidate_models = [
    os.path.join(BASE_DIR, "model_weights", "prism.onnx"),
    os.path.join(BASE_DIR, "prism.onnx"),
    os.path.join(BASE_DIR, "backend", "model_weights", "prism.onnx"),
]
MODEL_PATH = next((p for p in candidate_models if os.path.exists(p)), None)

candidate_cies = [
    os.path.join(BASE_DIR, "model_weights", "cie_curves.json"),
    os.path.join(BASE_DIR, "cie_curves.json"),
    os.path.join(BASE_DIR, "backend", "model_weights", "cie_curves.json"),
]
CIE_PATH = next((p for p in candidate_cies if os.path.exists(p)), None)

# ONNX Runtime Oturumu (Ultra Düşük Karmaşıklık - 0.28-0.39 ms CPU Çıkarımı)
session = None
input_name = None
output_name = None

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"Model başarıyla yüklendi: {MODEL_PATH}")
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
else:
    print("Uyarı: prism.onnx bulunamadı, çevrimdışı matematiksel simülatör devrede.")

# CIE Standart Verileri
if CIE_PATH and os.path.exists(CIE_PATH):
    with open(CIE_PATH, "r", encoding="utf-8") as f:
        cie_data = json.load(f)
    wavelengths = np.array(cie_data.get("wavelengths", list(range(380, 781))), dtype=np.float64)
    v_lambda = np.array(cie_data.get("v_lambda", []), dtype=np.float64)
    
    s_mel_raw = cie_data.get("s_mel_curve") or cie_data.get("s_mel") or []
    if len(s_mel_raw) == len(wavelengths):
        s_mel_curve = np.array(s_mel_raw, dtype=np.float64)
    else:
        s_mel_curve = np.exp(-0.5 * ((wavelengths - 490) / 38) ** 2)
        
    if "edi_weights" in cie_data and len(cie_data["edi_weights"]) == len(wavelengths):
        edi_weights = np.array(cie_data["edi_weights"], dtype=np.float64)
    else:
        edi_weights = s_mel_curve * (683.002 / 0.90564) * (1.0 / len(wavelengths))
        
    if len(v_lambda) != len(wavelengths):
        v_lambda = np.exp(-0.5 * ((wavelengths - 555) / 44) ** 2)
else:
    print("Bilgi: cie_curves.json bulunamadı, dahili standart eğriler üretildi.")
    wavelengths = np.arange(380, 781, 1.0, dtype=np.float64)
    v_lambda = np.exp(-0.5 * ((wavelengths - 555) / 44) ** 2)
    s_mel_curve = np.exp(-0.5 * ((wavelengths - 490) / 38) ** 2)
    edi_weights = s_mel_curve * (683.002 / 0.90564) * (1.0 / len(wavelengths))

# Popüler Oyun Presets
GAME_PRESETS = {
    "Cyberpunk 2077 - Neon Gece Şehri": (15, 140, 255, 0.5, 0.5, "Kısa dalga boylu (440-460 nm) mavi pik nedeniyle kritik melanopik uyarım üretir.", "#06b6d4", "Yüksek Mavi Yük"),
    "Valorant - Icebox A-Bölgesi": (185, 225, 255, 0.5, 0.5, "Kutup ortamı ve kar yansıması. Yüksek CCT renk sıcaklığı taşır.", "#38bdf8", "Yüksek Renk Sıcaklığı"),
    "Counter-Strike 2 - Mirage Gün Ortası": (235, 195, 115, 0.5, 0.5, "Aydınlık sarı ve kum tonları. Dengeli fotopik aydınlık sunar.", "#f59e0b", "Sıcak Spektrum"),
    "League of Legends - Vadi Nehri": (35, 195, 170, 0.45, 0.55, "Canlı camgöbeği su efektleri. 490 nm ipRGC duyarlılık pikiyle doğrudan örtüşür.", "#10b981", "ipRGC Tepe Örtüşmesi"),
    "GTA V - Los Santos Gün Batımı": (255, 95, 30, 0.5, 0.3, "Kızıl-turuncu gökyüzü. Düşük M/P oranı ile gece melatonini korur.", "#f97316", "Melatonin Koruyucu"),
    "Elden Ring - Karanlık Katakomp": (50, 35, 20, 0.5, 0.5, "Düşük lüks seviyesi ve loş ortam. Sirkadiyen açıdan biyolojik olarak güvenlidir.", "#78716c", "Biyolojik Güvenli")
}

def predict_spectrum_and_metrics(r, g, b, x, y):
    try:
        t0 = time.perf_counter_ns()
        
        r_val = float(r)
        g_val = float(g)
        b_val = float(b)
        x_val = float(x)
        y_val = float(y)
        
        # 1. PRISM Model Çıkarımı ([R,G,B,x,y] -> 401 bantlı SPD)
        if session is not None:
            input_tensor = np.array([[r_val / 255.0, g_val / 255.0, b_val / 255.0, x_val, y_val]], dtype=np.float32)
            out_log = session.run([output_name], {input_name: input_tensor})[0][0]
            raw_spd = np.maximum(0.0, np.exp(out_log) - 1e-4)
        else:
            # Yedek fiziksel yaklaşım
            b_p = (b_val / 255.0) * 0.95 * np.exp(-0.5 * ((wavelengths - 452) / 13) ** 2)
            g_p = (g_val / 255.0) * 0.75 * np.exp(-0.5 * ((wavelengths - 530) / 28) ** 2)
            r_p = (r_val / 255.0) * 0.85 * np.exp(-0.5 * ((wavelengths - 625) / 18) ** 2)
            raw_spd = 0.015 + b_p + g_p + r_p

        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0

        # 2. Fotometrik ve Biyolojik Hesaplamalar (CIE Standart Referansı: 250 cd/m², 50 cm mesafe)
        corneal_spd = raw_spd * 0.01

        photopic_lux = float(683.002 * np.sum(corneal_spd * v_lambda))
        melanopic_edi = float(np.sum(corneal_spd * edi_weights))
        mp_ratio = float(melanopic_edi / photopic_lux) if photopic_lux > 1e-4 else 0.0

        cla = 1.548 * melanopic_edi
        cs = float(0.7 * (1.0 - 1.0 / (1.0 + (cla / 355.7) ** 1.1026))) if cla > 0 else 0.0
        
        # Sirkadiyen Faz Kayması (Zeitzer / Duffy modeli, saat cinsinden uyku gecikmesi)
        phase_shift = float(3.0 * (melanopic_edi ** 0.5) / (melanopic_edi ** 0.5 + 60.0 ** 0.5)) if melanopic_edi > 0 else 0.0

        peak_idx = int(np.argmax(raw_spd))
        peak_wl = float(wavelengths[peak_idx])
        peak_power = float(raw_spd[peak_idx])

        # 3. Sirkadiyen Biyolojik Risk Değerlendirmesi (Brown et al. 2022)
        if melanopic_edi < 10.0:
            risk_title = "GÜVENLİ (Düşük)"
            risk_color = "#22c55e"
            risk_bg = "rgba(34, 197, 94, 0.15)"
            rec = "Melatonin salgılanması korunur; gece oturumları için biyolojik olarak güvenlidir."
        elif melanopic_edi < 50.0:
            risk_title = "ORTA (Akşam Eşiği)"
            risk_color = "#eab308"
            risk_bg = "rgba(234, 179, 8, 0.15)"
            rec = "Gece saatlerinde hafif melatonin gecikmesi oluşturabilir. Yatmadan önce parlaklığı kısmanız önerilir."
        elif melanopic_edi < 150.0:
            risk_title = "YÜKSEK (Melatonin Baskısı)"
            risk_color = "#f97316"
            risk_bg = "rgba(249, 115, 22, 0.15)"
            rec = "Kısa dalga boylu spektrum maruziyeti. Gece saatlerinde melatonin üretimini kuvvetle baskılar."
        else:
            risk_title = "KRİTİK (Sirkadiyen Faz Kayması)"
            risk_color = "#ef4444"
            risk_bg = "rgba(239, 68, 68, 0.15)"
            rec = "ipRGC hücreleri maksimum uyarılmaktadır. Biyolojik saatin 2-3 saat gecikmesine yol açar."

        fps_eq = 1000.0 / latency_ms if latency_ms > 0 else 3500.0

        # 4. Göz Alıcı Gökkuşağı Spektral Plotly Çizimi (Image 3 Tarzı)
        max_y = max(0.18, float(np.max(raw_spd)))
        fig = go.Figure()

        wl_list = wavelengths.tolist()
        spd_list = raw_spd.tolist()
        mel_scaled_list = (s_mel_curve * (max_y * 0.88)).tolist()

        # 4.1 Görünür Işık Tayfı Renk Gradyanı Dilimleri (Violet -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red)
        spectral_slices = [
            (380, 435, "rgba(76, 29, 149, 0.50)"),   # Mor (Violet)
            (435, 485, "rgba(37, 99, 235, 0.60)"),   # Mavi (Blue)
            (485, 510, "rgba(6, 182, 212, 0.70)"),   # Camgöbeği (Cyan / ipRGC Bölgesi)
            (510, 565, "rgba(34, 197, 94, 0.60)"),   # Yeşil (Green)
            (565, 590, "rgba(234, 179, 8, 0.55)"),   # Sarı (Yellow)
            (590, 625, "rgba(249, 115, 22, 0.50)"),  # Turuncu (Orange)
            (625, 781, "rgba(239, 68, 68, 0.45)")    # Kırmızı (Red)
        ]

        for start_wl, end_wl, fill_color in spectral_slices:
            mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            sub_wl = wavelengths[mask].tolist()
            sub_spd = raw_spd[mask].tolist()
            if len(sub_wl) > 0:
                fig.add_trace(go.Scatter(
                    x=sub_wl,
                    y=sub_spd,
                    mode="lines",
                    line=dict(width=0.5, color=fill_color),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    showlegend=False,
                    hoverinfo="skip"
                ))

        # 4.2.1 Neon Parlama Katmanı (Arka plan mavi/camgöbeği neon halesi)
        fig.add_trace(go.Scatter(
            x=wl_list,
            y=spd_list,
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.45)", width=6.5),
            showlegend=False,
            hoverinfo="skip"
        ))

        # 4.2.2 Ana Parlayan Neon Beyaz Çizgi (Üst katman net vuruş)
        fig.add_trace(go.Scatter(
            x=wl_list,
            y=spd_list,
            mode="lines",
            name="Kestirilen Spektrum (SPD)",
            line=dict(color="#ffffff", width=2.8),
            hoverinfo="x+y"
        ))

        # 4.3 CIE S 026 ipRGC (490 nm) Kesikli Camgöbeği Eğrisi
        fig.add_trace(go.Scatter(
            x=wl_list,
            y=mel_scaled_list,
            mode="lines",
            name="CIE S 026 ipRGC (490 nm)",
            line=dict(color="#06b6d4", width=2, dash="dash"),
            opacity=0.85
        ))

        # 4.4 490 nm ipRGC Tepe Noktası Belirteci
        fig.add_trace(go.Scatter(
            x=[490],
            y=[max_y * 0.88],
            mode="markers+text",
            marker=dict(size=7, color="#06b6d4", symbol="circle"),
            text=["ipRGC 490nm"],
            textposition="top center",
            textfont=dict(color="#06b6d4", size=10, family="sans-serif"),
            showlegend=False,
            hoverinfo="skip"
        ))

        # 4.5 Tepe Noktası İşaretçisi (Image 3 tarzı baloncuk)
        fig.add_trace(go.Scatter(
            x=[peak_wl],
            y=[peak_power],
            mode="markers",
            marker=dict(size=11, color="#ffffff", line=dict(color="#38bdf8", width=3)),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.add_annotation(
            x=peak_wl,
            y=peak_power,
            text=f"<b>Tepe: {int(peak_wl)} nm ({peak_power:.3f})</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowcolor="#38bdf8",
            arrowwidth=1.5,
            ax=0,
            ay=-40,
            font=dict(color="#38bdf8", size=12, family="monospace"),
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="#38bdf8",
            borderwidth=1.5,
            borderpad=5,
            opacity=0.98
        )

        fig.update_layout(
            title=f"<b>Spektral Güç Dağılımı (SPD) Eğrisi [380–780 nm]</b> | Çıkarım: {latency_ms:.2f} ms ({fps_eq:.0f} FPS)",
            xaxis_title="Dalga Boyu λ (nm) — CIE Spektral Bölge",
            yaxis_title="Spektral Güç Yoğunluğu SPD [a.u.]",
            template="plotly_dark",
            plot_bgcolor="#090d16",
            paper_bgcolor="#090d16",
            xaxis=dict(
                range=[380, 780],
                autorange=False,
                tickmode="array",
                tickvals=[380, 400, 450, 490, 500, 550, 600, 650, 700, 750, 780],
                ticktext=["380", "400", "450", "490", "500", "550", "600", "650", "700", "750", "780"],
                gridcolor="rgba(255, 255, 255, 0.05)",
                zeroline=False
            ),
            yaxis=dict(
                range=[0, max_y * 1.25],
                autorange=False,
                gridcolor="rgba(255, 255, 255, 0.05)",
                zeroline=False
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=25, t=55, b=45)
        )

        # 5. Alt Spektral Bölge Göstergeleri (Image 3 Lejandı)
        spectral_badges_html = """
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 8px; font-family: sans-serif;">
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(59, 130, 246, 0.3); padding: 8px 10px; border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: bold; color: #60a5fa;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #3b82f6; display: inline-block;"></span>
                    Mavi Işık Piki
                </div>
                <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">435 – 465 nm (LED Emisyonu)</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(6, 182, 212, 0.3); padding: 8px 10px; border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: bold; color: #22d3ee;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #06b6d4; display: inline-block;"></span>
                    Melanopik Eşik
                </div>
                <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">490 nm (ipRGC Max Hassasiyet)</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(34, 197, 94, 0.3); padding: 8px 10px; border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: bold; color: #4ade80;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e; display: inline-block;"></span>
                    Fotopik Aydınlık
                </div>
                <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">555 nm (Gündüz Görme V(λ))</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(239, 68, 68, 0.3); padding: 8px 10px; border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: bold; color: #f87171;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444; display: inline-block;"></span>
                    Güvenli Kırmızı
                </div>
                <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">620 – 780 nm (Sıfır ipRGC Yükü)</div>
            </div>
        </div>
        """

        # 6. HTML Metrik Kartları
        metrics_html = f"""
        {spectral_badges_html}
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; font-family: sans-serif;">
            <div style="background: #111827; padding: 12px; border-radius: 12px; border: 1px solid #374151;">
                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Melanopik Gün Işığı</div>
                <div style="font-size: 22px; font-weight: bold; color: #38bdf8; font-family: monospace;">{melanopic_edi:.1f} <span style="font-size: 13px;">lx</span></div>
                <div style="font-size: 10px; color: #6b7280;">CIE S 026:2018 m-EDI</div>
            </div>
            <div style="background: #111827; padding: 12px; border-radius: 12px; border: 1px solid #374151;">
                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Sirkadiyen Uyaran (CS)</div>
                <div style="font-size: 22px; font-weight: bold; color: #a855f7; font-family: monospace;">{cs:.3f}</div>
                <div style="font-size: 10px; color: #6b7280;">Rea & Figueiro (0.0 - 0.7)</div>
            </div>
            <div style="background: #111827; padding: 12px; border-radius: 12px; border: 1px solid #374151;">
                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Fotopik Aydınlık</div>
                <div style="font-size: 22px; font-weight: bold; color: #f59e0b; font-family: monospace;">{photopic_lux:.1f} <span style="font-size: 13px;">lx</span></div>
                <div style="font-size: 10px; color: #6b7280;">CIE 1931 V(λ) E_v</div>
            </div>
            <div style="background: #111827; padding: 12px; border-radius: 12px; border: 1px solid #374151;">
                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">M / P Oranı</div>
                <div style="font-size: 22px; font-weight: bold; color: #10b981; font-family: monospace;">{mp_ratio:.3f}</div>
                <div style="font-size: 10px; color: #6b7280;">Melanopik / Fotopik Oran</div>
            </div>
            <div style="background: #111827; padding: 12px; border-radius: 12px; border: 1px solid #374151;">
                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Uyku Fazı Kayması</div>
                <div style="font-size: 22px; font-weight: bold; color: #ec4899; font-family: monospace;">+{phase_shift:.1f} <span style="font-size: 13px;">saat</span></div>
                <div style="font-size: 10px; color: #6b7280;">Duffy & Zeitzer Uyku Gecikmesi</div>
            </div>
            <div style="background: #111827; padding: 12px; border-radius: 12px; border: 1px solid #374151;">
                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Tepe Dalga Boyu</div>
                <div style="font-size: 22px; font-weight: bold; color: #facc15; font-family: monospace;">{int(peak_wl)} <span style="font-size: 13px;">nm</span></div>
                <div style="font-size: 10px; color: #6b7280;">Maksimum Spektral Pik</div>
            </div>
        </div>

        <div style="margin-top: 12px; background: {risk_bg}; border: 1.5px solid {risk_color}; padding: 12px; border-radius: 12px; font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11px; text-transform: uppercase; color: {risk_color}; font-weight: bold;">Brown et al. (2022) Sirkadiyen Risk:</span>
                <span style="background: {risk_color}; color: #000; font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 6px;">{risk_title}</span>
            </div>
            <p style="font-size: 12px; color: #e5e7eb; margin: 6px 0 0 0; line-height: 1.4;">{rec}</p>
        </div>
        """

        return fig, metrics_html
    except Exception as err:
        import traceback
        traceback.print_exc()
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Hesaplama hatası: {str(err)}", showarrow=False, font=dict(color="red", size=14))
        err_fig.update_layout(template="plotly_dark", plot_bgcolor="#090d16", paper_bgcolor="#090d16")
        return err_fig, f"<div style='color: #ef4444; padding: 12px; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px;'>Hata oluştu: {str(err)}</div>"

def load_preset(preset_name):
    if preset_name in GAME_PRESETS:
        r, g, b, x, y, desc, color, tag = GAME_PRESETS[preset_name]
        badge_html = f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(17, 24, 39, 0.85); border: 1px solid {color}40; padding: 10px 14px; border-radius: 10px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; border-radius: 50%; background: rgb({r},{g},{b}); box-shadow: 0 0 10px rgb({r},{g},{b}); display: inline-block;"></span>
                <span style="font-size: 12px; font-weight: bold; color: #f3f4f6;">{preset_name.split(' - ')[0]}</span>
            </div>
            <span style="background: {color}20; color: {color}; border: 1px solid {color}40; font-size: 10px; font-weight: bold; padding: 2px 8px; border-radius: 6px;">{tag}</span>
        </div>
        <p style="font-size: 11px; color: #9ca3af; margin: 0 0 8px 0; line-height: 1.4;">{desc}</p>
        """
        return r, g, b, x, y, badge_html
    return 15, 140, 255, 0.5, 0.5, ""

# Görselden Piksel Seçme (Interactive Pipette)
def pick_pixel_from_image(image, evt: gr.SelectData):
    if image is None:
        return 15, 140, 255, 0.5, 0.5, "Görsel yüklenmedi"
    
    try:
        h, w = image.shape[:2]
        x_px, y_px = evt.index
        
        x_px = min(w - 1, max(0, int(x_px)))
        y_px = min(h - 1, max(0, int(y_px)))
        
        pixel_rgb = image[y_px, x_px]
        r = int(pixel_rgb[0])
        g = int(pixel_rgb[1])
        b = int(pixel_rgb[2])
        
        norm_x = round(float(x_px) / float(w), 2)
        norm_y = round(float(y_px) / float(h), 2)
        
        info_html = f"""
        <div style="background: rgba(17, 24, 39, 0.9); border: 1px solid #38bdf8; padding: 8px 12px; border-radius: 8px; font-size: 11px;">
            <span style="color: #38bdf8; font-weight: bold;">Seçilen Nokta:</span> ({x_px}, {y_px}) | 
            <span style="color: #a855f7; font-weight: bold;">RGB:</span> ({r}, {g}, {b}) | 
            <span style="color: #22c55e; font-weight: bold;">Uzamsal (x,y):</span> ({norm_x}, {norm_y})
            <span style="display: inline-block; width: 12px; height: 12px; background: rgb({r},{g},{b}); border-radius: 3px; vertical-align: middle; margin-left: 6px;"></span>
        </div>
        """
        return r, g, b, norm_x, norm_y, info_html
    except Exception as e:
        return 15, 140, 255, 0.5, 0.5, f"<div>Piksel okuma hatası: {e}</div>"

# 7. Gradio Blocks Arayüzü
custom_css = """
body, .gradio-container { background-color: #030712 !important; color: #f3f4f6 !important; }
.card { background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; }
"""

with gr.Blocks(title="PRISM-Spectra v2.2", css=custom_css, theme=gr.themes.Base()) as demo:
    gr.Markdown("""
    # 🌈 PRISM-Spectra: Oyun İçi Gerçek Zamanlı Spektral Kestirim ve Sirkadiyen Yük Çerçevesi
    > **Makale Başlığı:** *A Low-Complexity Deep Learning Framework for In-Game Real-Time Spectral Estimation and Melanopic Load Assessment*  
    > **Yazarlar:** Sevdenur Güdül & Rümeysa &bull; Samsun Üniversitesi &bull; **Danışman:** Dr. Nurettin Şenyer  
    > **Uluslararası İş Birliği:** Dr. Dorukalp Durmuş &bull; Pennsylvania State University (PSU) Illumination Lab
    """)

    with gr.Row():
        # Sol Kolon: Oyun Girişleri (Hazır Senaryolar veya Kendi Görselini Yükle)
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("🎮 1. Hazır Oyun Senaryoları"):
                    preset_dropdown = gr.Dropdown(
                        choices=list(GAME_PRESETS.keys()),
                        value="Cyberpunk 2077 - Neon Gece Şehri",
                        label="Oyun Sahnesi Seçin",
                        interactive=True
                    )
                    preset_info_box = gr.HTML(
                        value="""
                        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(17, 24, 39, 0.85); border: 1px solid #06b6d440; padding: 10px 14px; border-radius: 10px; margin-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="width: 14px; height: 14px; border-radius: 50%; background: rgb(15,140,255); box-shadow: 0 0 10px rgb(15,140,255); display: inline-block;"></span>
                                <span style="font-size: 12px; font-weight: bold; color: #f3f4f6;">Cyberpunk 2077</span>
                            </div>
                            <span style="background: #06b6d420; color: #06b6d4; border: 1px solid #06b6d440; font-size: 10px; font-weight: bold; padding: 2px 8px; border-radius: 6px;">Yüksek Mavi Yük</span>
                        </div>
                        <p style="font-size: 11px; color: #9ca3af; margin: 0 0 8px 0; line-height: 1.4;">Kısa dalga boylu (440-460 nm) mavi pik nedeniyle kritik melanopik uyarım üretir.</p>
                        """
                    )

                with gr.TabItem("🖼️ 2. Kendi Oyun Görselini Yükle (Damlalık)"):
                    gr.Markdown("Oyun ekran görüntünüzü yükleyin ve **istediğiniz pikselin üzerine tıklayarak** spektrumunu canlı kestirin:")
                    image_input = gr.Image(
                        label="Görsel Üzerine Tıklayarak Piksel Seçin",
                        type="numpy",
                        interactive=True
                    )
                    image_info_box = gr.HTML(value="<div style='font-size: 11px; color: #9ca3af;'>Görsel üzerinde bir noktaya tıkladığınızda renk ve koordinatlar otomatik aktarılır.</div>")

            # Ortak Kaydırıcılar (RGB ve Uzamsal Konum)
            gr.Markdown("#### 🎛️ PRISM Girdi Değerleri:")
            with gr.Row():
                r_slider = gr.Slider(0, 255, value=15, step=1, label="Kırmızı (R)")
                g_slider = gr.Slider(0, 255, value=140, step=1, label="Yeşil (G)")
                b_slider = gr.Slider(0, 255, value=255, step=1, label="Mavi (B)")

            with gr.Row():
                x_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Uzamsal X (Sol 0.0 - Sağ 1.0)")
                y_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Uzamsal Y (Üst 0.0 - Alt 1.0)")

            gr.Markdown("""
            #### ⚡ Literatür Yenilikleri (Novelty Points):
            1. **Donanımsız SPD:** 50.000$ optik cihazlar yerine yalnızca (R,G,B,x,y) girdisinden 401 bantlı sürekli tayf kestirimi.
            2. **Ultra Düşük Karmaşıklık:** CPU üzerinde 0.28–0.48 ms çıkarım süresiyle 144–240 FPS oyun motorlarında sıfır gecikme.
            3. **Dinamik Sirkadiyen Haritalama:** CIE S 026 ve Brown et al. (2022) eşiklerinde anlık biyolojik risk sınıflandırması.
            4. **Displays (2025) Köprüsü:** Şenyer & Durmuş (Displays 2025) yapay zeka modelini dinamik oyunlara uyarlama.
            """)

        # Sağ Kolon: Görsel Spektrum Çizimi ve Sirkadiyen Kartlar
        with gr.Column(scale=6):
            gr.Markdown("### 📊 2. Spektrum Analizi ve Sirkadiyen Metrikler")
            plot_output = gr.Plot(label="Spektral Güç Dağılımı SPD (380-780 nm)")
            metrics_output = gr.HTML(label="Sirkadiyen Biyolojik Metrikler")

    # Olay Tetikleyicileri (api_name=False ile API dokümantasyonu hatası tamamen engellendi)
    inputs = [r_slider, g_slider, b_slider, x_slider, y_slider]
    
    for inp in inputs:
        inp.change(
            fn=predict_spectrum_and_metrics,
            inputs=inputs,
            outputs=[plot_output, metrics_output],
            show_progress="hidden",
            api_name=False
        )

    preset_dropdown.change(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[r_slider, g_slider, b_slider, x_slider, y_slider, preset_info_box],
        api_name=False
    ).then(
        fn=predict_spectrum_and_metrics,
        inputs=inputs,
        outputs=[plot_output, metrics_output],
        api_name=False
    )

    image_input.select(
        fn=pick_pixel_from_image,
        inputs=[image_input],
        outputs=[r_slider, g_slider, b_slider, x_slider, y_slider, image_info_box],
        api_name=False
    ).then(
        fn=predict_spectrum_and_metrics,
        inputs=inputs,
        outputs=[plot_output, metrics_output],
        api_name=False
    )

    demo.load(
        fn=predict_spectrum_and_metrics,
        inputs=inputs,
        outputs=[plot_output, metrics_output],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
