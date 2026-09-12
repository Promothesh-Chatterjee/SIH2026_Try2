# Cloud Deployment Guide: Render (Backend) + Vercel (Frontend)

This guide walks you through deploying the **Cognitive EW SmartScan Backend** to **Render** and connecting it to your **Frontend** already running on **Vercel** (`https://sih-2026-try2.vercel.app/`).

---

## 1. Commit and Push to GitHub

Ensure all new deployment configurations and optimizations are pushed to GitHub:

```bash
git add .
git commit -m "feat(deploy): add render.yaml, requirements-render.txt, vercel config, and cloud connection panel"
git push origin main
```

---

## 2. Deploy Backend on Render

### Method A: Blueprint (1-Click Automated Setup — Recommended)
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** (top right) $\rightarrow$ **Blueprint**.
3. Select your GitHub repository: `Promothesh-Chatterjee/SIH2026_Try2`.
4. Render will automatically read `render.yaml` from the repository:
   - **Service Name**: `smartscan-backend`
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install --upgrade pip && pip install -r cognitive_ew_smart_scan/requirements-render.txt`
   - **Start Command**: `python -m uvicorn src.deployment.api:app --app-dir cognitive_ew_smart_scan --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
   - **Plan**: Free
5. Click **Apply**.
6. Wait 1–2 minutes for the build to complete. Once finished, Render will display your service URL (e.g., `https://smartscan-backend-xxxx.onrender.com`).

---

### Method B: Manual Web Service Setup
If you prefer creating the Web Service manually:
1. Go to [Render Dashboard](https://dashboard.render.com) $\rightarrow$ **New +** $\rightarrow$ **Web Service**.
2. Select your repository: `Promothesh-Chatterjee/SIH2026_Try2`.
3. Configure the following settings:
   - **Name**: `smartscan-backend`
   - **Region**: Any (e.g., Oregon or Frankfurt)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank or set to repository root)*
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r cognitive_ew_smart_scan/requirements-render.txt
     ```
   - **Start Command**:
     ```bash
     python -m uvicorn src.deployment.api:app --app-dir cognitive_ew_smart_scan --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free`
4. Under **Advanced**:
   - Add Environment Variable:
     - `PYTHON_VERSION` = `3.11.9`
     - `DEVICE` = `cpu`
     - `CORS_ORIGINS` = `*`
   - **Health Check Path**: `/health`
5. Click **Create Web Service**.

---

## 3. Verify Backend Deployment

Once deployed on Render, verify the service is healthy by opening the URL in your browser:
```text
https://<your-service-name>.onrender.com/health
```
You will receive a JSON response confirming that all neural models and the mission controller are online:
```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": {
    "deinterleaver": true,
    "scheduler": true,
    "memory": true
  },
  "dimension_check_passed": true,
  "normalization_hash_match": true,
  "hidden_state_ready": true,
  "mission_controller_ready": true
}
```

---

## 4. Connect Backend to Vercel Frontend

### Option 1: Permanent Environment Variable in Vercel (Recommended)
1. Go to your [Vercel Dashboard](https://vercel.com) and open the `sih-2026-try2` project.
2. Go to **Settings** $\rightarrow$ **Environment Variables**.
3. Add a new variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://<your-service-name>.onrender.com` *(no trailing slash)*
4. Go to the **Deployments** tab in Vercel, click on the **...** menu on the latest deployment, and click **Redeploy**.
5. Once rebuilt, the Vercel app will permanently point to your live Render backend for both REST and secure WebSockets (`wss://`).

---

### Option 2: Instant Connection via UI (No Rebuild Required!)
1. Open your live app: [https://sih-2026-try2.vercel.app/](https://sih-2026-try2.vercel.app/)
2. In the sidebar, click on **System Config** (`/system`).
3. Locate the panel: **00 // LIVE BACKEND CLOUD CONNECTION (RENDER / VERCEL)**.
4. Paste your Render backend URL (e.g. `https://<your-service-name>.onrender.com`).
5. Click **Connect & Save**.
6. The app will immediately test the connection, verify the active frozen neural models (`Gate-25k-R4.2-alpha020`), and switch all live telemetry and WebSocket streams to your Render backend!

---

## 5. End-to-End Operational Checklist

| Subsystem | Endpoint | Verification |
| :--- | :--- | :--- |
| **Liveness & Neural Health** | `GET /health` | Status 200 with `models_loaded: true` |
| **Telemetry History** | `GET /telemetry/latest` | Real FoM metrics from receiver |
| **Live Telemetry Stream** | `WSS /ws/state` | Continuous 5 Hz telemetry broadcasting |
| **Live Mission Controller** | `POST /mission/stream/start` | Closed-loop 180-action DRQN radar scheduler |
| **Dynamic Benchmark** | `POST /benchmark/evaluate` | Dynamic multi-scheduler comparison |
