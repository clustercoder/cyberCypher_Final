# Deployment Guide

Split deployment: FastAPI backend on Render, Vite React frontend on Vercel.

## Backend — Render

1. Create a new **Web Service** on [render.com](https://render.com), pointing at this repo.
2. Render auto-detects `render.yaml` — no manual settings needed.
3. Add the required environment variable in the Render dashboard:
   - `OPENAI_API_KEY` — your OpenAI API key
4. The service exposes a `/health` endpoint used by Render's health check.
5. WebSocket connections are served at `/ws` on the same service.

**Start command** (from `render.yaml`):
```
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

> Note: `stable-baselines3`, `torch`, and `torch-geometric` increase build time significantly.
> Use the **Standard** plan or above for sufficient build memory.

---

## Frontend — Vercel

1. Import this repo on [vercel.com](https://vercel.com).
2. In project settings, set **Root Directory** to `src/ui/dashboard`.
3. Framework preset: **Vite** (auto-detected).
4. Add environment variables:
   - `VITE_API_URL` — your Render service URL, e.g. `https://cybercypher-api.onrender.com`
   - `VITE_WS_URL` — *(optional)* WebSocket URL. If omitted, derived automatically from `VITE_API_URL` (`https` → `wss`, appends `/ws`).
5. `vercel.json` in `src/ui/dashboard/` handles SPA rewrites — no extra config needed.

---

## Local Development

Backend:
```bash
cp .env.example .env
# fill in OPENAI_API_KEY
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

Frontend:
```bash
cd src/ui/dashboard
cp .env.example .env.local
# set VITE_API_URL=http://localhost:8000 (or leave blank; vite.config.js proxies /api and /ws)
npm install
npm run dev
```

---

## RL Training (optional, improves DeciderAgent)

```bash
python train_rl_synthetic.py --timesteps 50000 --output models/rl_traffic_engineer
```

The model is auto-loaded by `DeciderAgent` at startup from `models/rl_traffic_engineer.zip`.

## LLM Fine-tuning (optional, domain-specific incident reasoning)

```bash
# Step 1 — generate synthetic incident dataset
python src/models/llm_finetune/synthetic_incident_generator.py --count 500 --output data/incidents.jsonl

# Step 2 — fine-tune with LoRA
python src/models/llm_finetune/train_lora.py --dataset data/incidents.jsonl
```
