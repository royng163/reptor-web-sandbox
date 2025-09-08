# FitnessCoach Web Sandbox

Fast, web-first sandbox to iterate on pose ingestion, preprocessing, rep/phase detection, and learned quality scoring. Mirrors the React Native pipeline and exports models via ONNX for on-device inference.

- Framework: React + Vite + TypeScript
- Pose: Pose Landmarker in the browser
- Inference: onnxruntime-web (WebGL/WebGPU)
- Parity: Shared TypeScript feature pipeline used in both web and RN
- Scope: Single person in frame; known exercise provided by the workout plan

---

## Quick start

1. Prerequisites

- Node.js ≥ 18
- A modern browser (Chrome/Edge recommended for WebGL/WebGPU)

2. Install

```bash
npm install
```

3. Run dev server

```bash
npm run dev
```
