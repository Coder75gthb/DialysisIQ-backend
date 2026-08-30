---
title: DialysisIQ Backend
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DialysisIQ Backend API

FastAPI backend for the DialysisIQ clinical decision support system.

## Modules

- **Module 1**: Blood flow rate (Qb) prediction
- **Module 2**: Intradialytic hypotension risk classification
- **Module 3**: Interruption-event classification
- **Module 4**: Dry weight drift detection
- **Module 5**: Morning briefing & unit-wide risk assessment

## API Endpoints

- `GET /` — Health check
- `GET /patients` — List patients
- `POST /sessions` — Create dialysis session
- `GET /module1/predict/{session_id}` — Qb prediction
- `GET /module2/predict/{session_id}` — Hypotension risk
- `POST /module3/predict` — Event classification
- `GET /module4/predict/{pid}` — Drift detection
- `GET /module5/predict` — Morning briefing
