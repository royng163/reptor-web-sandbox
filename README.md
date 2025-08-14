# FitnessCoach Web Sandbox

Fast, web-first sandbox to iterate on pose ingestion, preprocessing, rep/phase detection, and learned quality scoring. Mirrors the React Native pipeline and exports models via ONNX for on-device inference.

- Framework: React + Vite + TypeScript
- Pose: MediaPipe Tasks (Pose Landmarker) in the browser
- Inference: onnxruntime-web (WebGL/WebGPU)
- Parity: Shared TypeScript feature pipeline used in both web and RN
- Scope: Single person in frame; known exercise provided by the workout plan

## Why this sandbox

- Rapid iteration on pose features, rules, DTW, rep/phase logic, and feedback without dealing with native build loops.
- Lock feature ordering and normalization in Python and load JSON constants here to ensure parity with RN.
- Validate on-device performance targets early (FPS, latency, feedback timing).

---

## Quick start

1. Prerequisites

- Node.js ≥ 18
- A modern browser (Chrome/Edge recommended for WebGL/WebGPU)
- HTTPS is required for camera on some browsers; Vite’s localhost works without HTTPS

2. Install

```bash
npm install
```

3. Run dev server

```bash
npm run dev
```

---

## Pose pipeline overview

1. Pose ingestion (MediaPipe Tasks)

   - Initialize Pose Landmarker with video stream (front/side camera angles supported).
   - Stream world or normalized landmarks to the pipeline at target FPS.

2. Preprocessing/features (shared TS)

   - Landmark smoothing and gap handling
   - Angle/velocity/ratio features (exercise-specific subsets)
   - Normalization using Python-exported `feature_norms.json` (lock order/constants)

3. Rep/phase detection

   - Rule-based state machine driven by key features and thresholds
   - Optional DTW against reference traces for phase alignment/validation

4. Quality scoring (ONNX)

   - Load `quality_scorer.onnx` via onnxruntime-web (WebGL/WebGPU > WASM)
   - Feed normalized feature windows
   - Confidence-gated feedback to avoid spam; phase-aware cues

5. Feedback UI
   - Real-time overlay and text/audio cues timed to phases
   - Respect debounce and minimum confidence thresholds
