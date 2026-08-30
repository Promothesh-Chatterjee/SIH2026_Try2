## 🚀 Quick Start — Cognitive EW SmartScan Dashboard

### 30-Second Setup

**Terminal 1 — Backend:**
```bash
cd cognitive_ew_smart_scan
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8080 --reload
```

**Terminal 2 — Frontend:**
```bash
cd cognitive_ew_smart_scan/frontend
npm run dev
```

**Open Browser:**
```
http://localhost:5173
```

---

### What to Expect

✅ **WebSocket: ONLINE** (green indicator, top-right)  
✅ Four navigation tabs: **OVERVIEW | SPECTRUM | METRICS | PDWS**  
✅ Empty panels (normal) until training starts  
✅ Real-time update every 250ms (4Hz)

---

### Test the Connection

```bash
# Verify backend health
curl http://localhost:8080/health

# Test WebSocket (optional, requires wscat)
npm install -g wscat
wscat -c ws://localhost:8080/ws/state
```

---

### Production Deployment

```bash
# Build frontend
cd frontend && npm run build

# Run production (backend serves dashboard)
cd .. && uvicorn src.deployment.api:app --port 8080

# Dashboard: http://localhost:8080
```

---

### Live Training Example

When running training, inject data into STATE to see live updates:

```python
from src.deployment.api import STATE

# In your training loop after env.step():
STATE["current_band"] = int(action)
STATE["epsilon"] = float(eps)
STATE["fom"] = metrics_object  # FiguresOfMerit instance
STATE["latest_pdws"] = [...pdw_list...]
```

---

### Files Created

```
cognitive_ew_smart_scan/
├── frontend/                          # ← New React app
│   ├── src/
│   │   └── App.jsx                    # ← 900+ line dashboard
│   ├── package.json
│   └── node_modules/                  # ← 39 packages installed
├── FRONTEND_DEPLOYMENT_GUIDE.md       # ← Full documentation
└── FRONTEND_SETUP_COMPLETE.md         # ← This setup status
```

---

### Key Resources

| Document | Purpose |
|----------|---------|
| [FRONTEND_DEPLOYMENT_GUIDE.md](FRONTEND_DEPLOYMENT_GUIDE.md) | Complete setup, prod config, troubleshooting |
| [FRONTEND_SETUP_COMPLETE.md](FRONTEND_SETUP_COMPLETE.md) | Status checklist, what was implemented |
| [src/deployment/api.py](src/deployment/api.py#L508) | WebSocket endpoint (line 508) |
| [frontend/src/App.jsx](frontend/src/App.jsx) | Dashboard component (903 lines) |

---

### Troubleshooting at a Glance

| Issue | Fix |
|-------|-----|
| WebSocket shows OFFLINE | Verify backend running on port 8080: `curl http://localhost:8080/health` |
| No data in panels | Training must be running; inject STATE data (see example above) |
| Canvas shows nothing | Normal until data arrives; check browser console for errors (F12) |
| Port already in use | Change port: `uvicorn ... --port 8081` and update frontend wsUrl |

---

**✅ All backend fixes are in place.** You can now run the dashboard!

Questions? See [FRONTEND_DEPLOYMENT_GUIDE.md](FRONTEND_DEPLOYMENT_GUIDE.md) Step 8.
