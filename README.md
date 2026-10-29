---
title: PRISM Spectra
emoji: 🌈
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: In-Game Real-Time Spectral & Circadian Load
---

# 🌈 PRISM-Spectra: In-Game Real-Time Spectral Estimation & Melanopic Load Assessment

> **Research Paper:** *A Low-Complexity Deep Learning Framework for In-Game Real-Time Spectral Estimation and Melanopic Load Assessment*  
> **Authors:** Sevdenur Güdül & Rümeysa  
> **Affiliation:** Samsun University, Department of Computer Engineering  
> **Advisor:** Dr. Nurettin Şenyer  
> **International Collaboration:** Dr. Dorukalp Durmuş (Pennsylvania State University Illumination Lab)  

---

## 🔬 Scientific Overview

**PRISM-Spectra** eliminates the need for expensive and cumbersome laboratory optical spectroradiometers ($50,000–$80,000) by inferring the 401-band continuous **Spectral Power Distribution (SPD, 380–780 nm)** directly from in-game RGB pixel values and spatial coordinates $(x, y)$.

### ⚡ Key Novelties (4 Novelty Points)
1. **Hardware-less In-Situ SPD Estimation:** Predicts the continuous 401-band spectrum purely in software with $R^2 > 0.95$.
2. **Zero In-Game Overhead (Ultra Low-Complexity):** Single-frame CPU inference latency of **0.28–0.48 ms** (1,100–3,500 FPS), introducing zero frame-rate drop on 144–240 Hz competitive gaming engines.
3. **Dynamic Gaming Scene Melanopic Assessment:** Instant calculation of CIE S 026:2018 melanopic equivalent daylight illuminance ($m\text{-EDI}$), Circadian Stimulus ($CS$), and Brown et al. (2022) biological risk classification.
4. **Elevating Displays (2025) to Real-Time Interactive Engines:** Translates the theoretical foundation of Şenyer & Durmuş (*Displays*, 2025) into a lightweight engine for real-time video game pipelines.
